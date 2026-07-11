"""Session isolation checks for API and memory layers."""
import asyncio
import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import app as api_app_module
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from core.routing import AdaptiveRouter


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


def test_api_rejects_path_like_session_id():
    client = TestClient(api_app_module.app)
    response = client.post(
        "/api/chat",
        json={"message": "你好", "session_id": "../../outside"},
    )
    assert response.status_code == 422


def test_api_exposes_route_citations_and_zero_cost_metrics(monkeypatch):
    class FakeCoordinator:
        async def process(self, message, session_id=None):
            return {
                "session_id": session_id,
                "answer": "ok",
                "route": {"mode": "single", "reason_codes": ["simple_request"]},
                "citations": [{"title": "测试资料", "source": "本地知识库"}],
            }

    async def fake_get_coordinator():
        return FakeCoordinator(), 0.0

    monkeypatch.setattr(api_app_module, "_get_coordinator", fake_get_coordinator)
    response = TestClient(api_app_module.app).post("/api/chat", json={"message": "你好"})
    data = response.json()

    assert response.status_code == 200
    assert data["route"]["mode"] == "single"
    assert data["citations"][0]["source"] == "本地知识库"
    assert data["llm_call_count"] == 0
    assert data["total_tokens"] == 0


def test_api_passes_forced_routing_mode_for_ablation(monkeypatch):
    seen = {}

    class FakeCoordinator:
        async def process(self, message, session_id=None, context=None):
            seen["context"] = context
            return {"session_id": session_id, "answer": "ok"}

    async def fake_get_coordinator():
        return FakeCoordinator(), 0.0

    monkeypatch.setattr(api_app_module, "_get_coordinator", fake_get_coordinator)
    response = TestClient(api_app_module.app).post(
        "/api/chat",
        json={"message": "你好", "routing_mode": "swarm"},
    )

    assert response.status_code == 200
    assert seen["context"] == {"_routing_mode": "swarm"}


def test_simple_request_skips_lead_agent_planning():
    coordinator = object.__new__(api_app_module.SwarmCoordinator)
    coordinator.enable_swarm = True
    coordinator.router = AdaptiveRouter()

    class FakeLeadAgent:
        async def assess_and_decompose(self, question, context):
            raise AssertionError("simple requests must not call LeadAgent")

    class FakeAgent:
        async def process(self, input_data):
            return {"answer": "ok", "agent_id": "consultation_agent"}

    class FakeShortMemory:
        def get_recent_messages(self, session_id, limit):
            return []

    class FakeLongMemory:
        def search_similar_sessions(self, query, limit, user_id):
            return []

        def add_session_summary(self, **kwargs):
            return None

    coordinator.lead_agent = FakeLeadAgent()
    coordinator.consultation_agent = FakeAgent()
    coordinator.diagnostic_agent = FakeAgent()
    coordinator.research_agent = FakeAgent()
    coordinator.short_term_memory = FakeShortMemory()
    coordinator.long_term_memory = FakeLongMemory()

    result = asyncio.run(coordinator.process("感冒了怎么办？", session_id="test-simple"))

    assert result["answer"] == "ok"
    assert result["route"]["mode"] == "single"
    assert result["route"]["lead_planning_required"] is False
