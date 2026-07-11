"""FastAPI application for the MediX multi-agent assistant."""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
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
from core.service_metrics import service_metrics  # noqa: E402
from .schemas import ChatRequest, ChatResponse  # noqa: E402
from .security import inspect_message  # noqa: E402


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


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response

WEB_DIR = PROJECT_ROOT / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_coordinator: SwarmCoordinator | None = None
_coordinator_init_time: float | None = None
_coordinator_lock = asyncio.Lock()
_max_concurrency = max(1, int(os.getenv("MEDIX_MAX_CONCURRENCY", "8")))
_request_timeout = max(1.0, float(os.getenv("MEDIX_REQUEST_TIMEOUT_SECONDS", "120")))
_expose_raw_response = os.getenv("MEDIX_EXPOSE_RAW_RESPONSE", "false").lower() == "true"
_chat_slots = asyncio.Semaphore(_max_concurrency)


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
    request_id: str | None = None,
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
        request_id=request_id,
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
        raw=result if _expose_raw_response else None,
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
        "max_concurrency": _max_concurrency,
        "request_timeout_seconds": _request_timeout,
    }


@app.get("/api/metrics", response_class=PlainTextResponse)
async def metrics():
    return service_metrics.prometheus()


async def _process_chat(payload: ChatRequest, request_id: str) -> ChatResponse:
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")
    safety = inspect_message(message)
    if not safety.allowed:
        logger.warning(f"blocked unsafe input request={request_id[:8]} reasons={safety.reason_codes}")
        raise HTTPException(status_code=400, detail="请求包含不支持的控制指令。")

    session_id = payload.session_id or str(uuid4())
    start = time.perf_counter()
    outcome = "error"
    route_mode = None
    service_metrics.start_request()
    metrics_token = start_request_metrics()
    logger.info(f"chat request session={session_id[:8]} message_length={len(message)}")

    try:
        coordinator, init_time = await _get_coordinator()
        process_start = time.perf_counter()

        # Agent execution state is request-local. The semaphore protects the
        # external LLM and memory backends without serializing all traffic.
        async with _chat_slots:
            process_kwargs = {"session_id": session_id}
            if payload.routing_mode != "auto":
                process_kwargs["context"] = {"_routing_mode": payload.routing_mode}
            result = await asyncio.wait_for(
                coordinator.process(message, **process_kwargs),
                timeout=_request_timeout,
            )

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
        response = _adapt_chat_result(
            result,
            session_id,
            elapsed,
            timings,
            metrics=metrics,
            request_id=request_id,
        )
        route_mode = (response.route or {}).get("mode")
        outcome = "success"
        return response
    except asyncio.TimeoutError as exc:
        outcome = "timeout"
        logger.warning(f"chat timeout request={request_id[:8]} session={session_id[:8]}")
        raise HTTPException(status_code=504, detail="请求处理超时，请缩短问题后重试。") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"chat failed request={request_id[:8]} session={session_id[:8]} "
            f"error={exc.__class__.__name__}"
        )
        raise HTTPException(
            status_code=500,
            detail="系统处理请求时出现错误，请稍后重试。",
        ) from exc
    finally:
        service_metrics.finish_request(outcome, time.perf_counter() - start, route_mode)
        reset_request_metrics(metrics_token)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, http_request: Request):
    request_id = http_request.headers.get("X-Request-ID") or str(uuid4())
    return await _process_chat(payload, request_id)


@app.post("/api/chat/stream")
async def chat_stream(payload: ChatRequest, http_request: Request):
    """Stream request lifecycle events as SSE; the final event contains the answer."""
    request_id = http_request.headers.get("X-Request-ID") or str(uuid4())

    async def event_stream():
        accepted = json.dumps({"request_id": request_id, "status": "accepted"}, ensure_ascii=False)
        yield f"event: accepted\ndata: {accepted}\n\n"
        try:
            response = await _process_chat(payload, request_id)
            if hasattr(response, "model_dump"):
                data = response.model_dump()
            else:
                data = response.dict()
            yield f"event: complete\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        except HTTPException as exc:
            error = json.dumps(
                {"request_id": request_id, "status": exc.status_code, "detail": exc.detail},
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {error}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
