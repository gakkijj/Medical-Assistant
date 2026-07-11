"""Request-scoped performance metrics."""
from contextvars import ContextVar
from copy import deepcopy
from time import perf_counter
from typing import Any


_metrics: ContextVar[dict[str, Any] | None] = ContextVar("request_metrics", default=None)


def start_request_metrics():
    """Start a fresh metrics bucket for the current async context."""
    return _metrics.set({
        "llm_call_count": 0,
        "llm_total_time": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tool_call_count": 0,
        "tool_total_time": 0.0,
        "route": None,
        "events": [],
        "started_at": perf_counter(),
    })


def reset_request_metrics(token) -> None:
    """Restore the previous metrics context."""
    _metrics.reset(token)


def record_llm_call(
    duration: float,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    success: bool = True,
    operation: str = "chat",
) -> None:
    """Record one LLM API call duration for the current request."""
    current = _metrics.get()
    if current is None:
        return
    current["llm_call_count"] += 1
    current["llm_total_time"] += duration
    current["prompt_tokens"] += prompt_tokens
    current["completion_tokens"] += completion_tokens
    current["total_tokens"] += total_tokens or prompt_tokens + completion_tokens
    record_event(
        "llm_call",
        duration,
        operation=operation,
        success=success,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )


def record_tool_call(tool_name: str, duration: float, *, success: bool = True) -> None:
    """Record a function/skill invocation without storing its sensitive payload."""
    current = _metrics.get()
    if current is None:
        return
    current["tool_call_count"] += 1
    current["tool_total_time"] += duration
    record_event("tool_call", duration, tool_name=tool_name, success=success)


def record_route_decision(decision: dict[str, Any], duration: float) -> None:
    """Record the explainable router output for the current request."""
    current = _metrics.get()
    if current is None:
        return
    current["route"] = deepcopy(decision)
    record_event(
        "routing",
        duration,
        mode=decision.get("mode"),
        primary_agent=decision.get("primary_agent"),
        complexity_score=decision.get("complexity_score"),
    )


def record_event(name: str, duration: float, **metadata: Any) -> None:
    """Append a sanitized trace event to the current request."""
    current = _metrics.get()
    if current is None:
        return
    current["events"].append({
        "name": name,
        "duration": round(max(duration, 0.0), 6),
        "metadata": metadata,
    })


def get_request_metrics() -> dict[str, Any]:
    """Return a copy of current request metrics."""
    current = _metrics.get() or {}
    return {
        "llm_call_count": int(current.get("llm_call_count", 0)),
        "llm_total_time": float(current.get("llm_total_time", 0.0)),
        "prompt_tokens": int(current.get("prompt_tokens", 0)),
        "completion_tokens": int(current.get("completion_tokens", 0)),
        "total_tokens": int(current.get("total_tokens", 0)),
        "tool_call_count": int(current.get("tool_call_count", 0)),
        "tool_total_time": float(current.get("tool_total_time", 0.0)),
        "route": deepcopy(current.get("route")),
        "events": deepcopy(current.get("events", [])),
    }
