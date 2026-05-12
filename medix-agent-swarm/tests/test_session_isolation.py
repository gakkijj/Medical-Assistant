"""Session isolation checks for API and memory layers."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import app as api_app_module
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory


def fresh_short_term_memory() -> ShortTermMemory:
    memory = ShortTermMemory(storage_type="memory")
    memory.sessions.clear()
    return memory


def test_short_term_memory_is_bucketed_by_session_id():
    memory = fresh_short_term_memory()

    memory.add_message("session-A", "user", "我的名字叫张三")
    memory.add_message("session-B", "user", "我的名字叫什么？")

    history_a = memory.get_history("session-A", limit=10)
    history_b = memory.get_history("session-B", limit=10)

    assert history_a != history_b
    assert any("张三" in item["content"] for item in history_a)
    assert not any("张三" in item["content"] for item in history_b)


def test_short_term_memory_empty_session_does_not_return_all_history():
    memory = fresh_short_term_memory()
    memory.add_message("session-A", "user", "我的名字叫张三")

    assert memory.get_history("", limit=10) == []
    assert memory.get_recent_messages("", limit=10) == []


def test_api_generates_session_id_when_missing(monkeypatch):
    async def fake_process(message, session_id):
        return {
            "session_id": session_id,
            "answer": "ok",
            "swarm_enabled": False,
            "suggestions": [],
            "disclaimer": None,
        }

    class FakeCoordinator:
        async def process(self, message, session_id=None):
            return await fake_process(message, session_id)

    async def fake_get_coordinator():
        return FakeCoordinator(), 0.0

    monkeypatch.setattr(api_app_module, "_get_coordinator", fake_get_coordinator)
    client = TestClient(api_app_module.app)

    response = client.post("/api/chat", json={"message": "你好"})

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"]
    assert data["answer"] == "ok"


def test_api_passes_request_session_id(monkeypatch):
    seen = {}

    class FakeCoordinator:
        async def process(self, message, session_id=None):
            seen["session_id"] = session_id
            return {
                "session_id": session_id,
                "answer": "ok",
                "swarm_enabled": False,
                "suggestions": [],
                "disclaimer": None,
            }

    async def fake_get_coordinator():
        return FakeCoordinator(), 0.0

    monkeypatch.setattr(api_app_module, "_get_coordinator", fake_get_coordinator)
    client = TestClient(api_app_module.app)

    response = client.post(
        "/api/chat",
        json={"message": "你好", "session_id": "session-A"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] == "session-A"
    assert seen["session_id"] == "session-A"


def test_long_term_memory_uses_session_as_user_id_by_default():
    memory = object.__new__(LongTermMemory)
    memory.enabled = True
    memory.entropy_manager = None
    calls = {}

    class FakeMem0:
        def add(self, **kwargs):
            calls["add"] = kwargs
            return {"id": "memory-1"}

        def search(self, **kwargs):
            calls["search"] = kwargs
            return []

    memory.mem0 = FakeMem0()

    memory.add_session_summary(
        session_id="session-A",
        question="我的名字叫张三",
        answer="已记录",
    )
    memory.search_similar_sessions("我的名字叫什么？", user_id="session-B")

    assert calls["add"]["user_id"] == "session-A"
    assert calls["search"]["user_id"] == "session-B"
