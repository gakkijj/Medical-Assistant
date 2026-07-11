"""FastAPI application for the MediX multi-agent assistant."""
import asyncio
import os
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swarm import SwarmCoordinator  # noqa: E402
from core.request_metrics import (  # noqa: E402
    get_request_metrics,
    reset_request_metrics,
    start_request_metrics,
)
from .schemas import ChatRequest, ChatResponse  # noqa: E402


app = FastAPI(
    title="MediX Medical Multi-Agent Assistant",
    description="FastAPI backend for the terminal-based MediX multi-agent project.",
    version="0.1.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "MEDIX_CORS_ORIGINS",
        "http://localhost:8000,http://127.0.0.1:8000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_coordinator: SwarmCoordinator | None = None
_coordinator_init_time: float | None = None
_coordinator_lock = asyncio.Lock()
_chat_lock = asyncio.Lock()


async def _get_coordinator() -> tuple[SwarmCoordinator, float]:
    """Create the coordinator once per FastAPI process and reuse it."""
    global _coordinator, _coordinator_init_time

    if _coordinator is not None:
        return _coordinator, 0.0

    async with _coordinator_lock:
        if _coordinator is not None:
            return _coordinator, 0.0

        start = time.perf_counter()
        _coordinator = SwarmCoordinator(enable_swarm=True)
        _coordinator_init_time = time.perf_counter() - start
        logger.info(f"SwarmCoordinator initialized for API in {_coordinator_init_time:.2f}s")
        return _coordinator, _coordinator_init_time


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str) and value.strip():
        return [value]
    return None


def _adapt_chat_result(
    result: dict[str, Any],
    fallback_session_id: str,
    elapsed: float,
    timings: dict[str, float],
    metrics: dict[str, Any],
) -> ChatResponse:
    """Normalize the existing swarm result without changing core logic."""
    agents_involved = _as_string_list(result.get("agents_involved"))
    if not agents_involved and result.get("agent_id"):
        agents_involved = [str(result["agent_id"])]

    total_time = result.get("total_time")
    if total_time is None:
        total_time = elapsed

    return ChatResponse(
        session_id=str(result.get("session_id") or fallback_session_id),
        answer=str(result.get("answer") or "抱歉，系统暂时没有返回有效回答。"),
        suggestions=_as_string_list(result.get("suggestions")),
        disclaimer=result.get("disclaimer"),
        agents_involved=agents_involved,
        swarm_enabled=bool(result.get("swarm_enabled", False)),
        total_time=float(total_time) if total_time is not None else None,
        total_elapsed_time=round(elapsed, 4),
        llm_total_time=round(float(metrics.get("llm_total_time", 0.0)), 4),
        llm_call_count=int(metrics.get("llm_call_count", 0)),
        prompt_tokens=int(metrics.get("prompt_tokens", 0)),
        completion_tokens=int(metrics.get("completion_tokens", 0)),
        total_tokens=int(metrics.get("total_tokens", 0)),
        tool_call_count=int(metrics.get("tool_call_count", 0)),
        route=result.get("route") or metrics.get("route"),
        citations=result.get("citations") or [],
        trace=metrics.get("events") or [],
        timings=timings,
        raw=result,
    )


@app.get("/")
async def index():
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend page not found")
    return FileResponse(index_path)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "coordinator_ready": _coordinator is not None,
        "coordinator_init_time": _coordinator_init_time,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    session_id = request.session_id or str(uuid4())
    start = time.perf_counter()
    metrics_token = start_request_metrics()
    logger.info(f"chat request session={session_id[:8]} message_length={len(message)}")

    try:
        coordinator, init_time = await _get_coordinator()
        process_start = time.perf_counter()

        # The current AgentLoop instances keep mutable per-run state, so serialize
        # requests while reusing the coordinator.
        async with _chat_lock:
            process_kwargs = {"session_id": session_id}
            if request.routing_mode != "auto":
                process_kwargs["context"] = {"_routing_mode": request.routing_mode}
            result = await coordinator.process(message, **process_kwargs)

        elapsed = time.perf_counter() - start
        metrics = get_request_metrics()
        llm_total_time = round(metrics["llm_total_time"], 4)
        timings = {
            "agent_process_time": round(time.perf_counter() - process_start, 4),
            "api_total_time": round(elapsed, 4),
            "llm_total_time": llm_total_time,
            "tool_total_time": round(metrics["tool_total_time"], 4),
        }

        if not isinstance(result, dict):
            result = {"answer": str(result), "session_id": session_id}
        return _adapt_chat_result(
            result,
            session_id,
            elapsed,
            timings,
            metrics=metrics,
        )
    except Exception as exc:
        logger.error(f"Failed to process chat request session={session_id[:8]}: {exc.__class__.__name__}")
        raise HTTPException(
            status_code=500,
            detail="系统处理请求时出现错误，请稍后重试。",
        ) from exc
    finally:
        reset_request_metrics(metrics_token)
