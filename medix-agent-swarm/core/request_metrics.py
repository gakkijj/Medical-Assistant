"""Request-scoped performance metrics."""
from contextvars import ContextVar
from typing import Any


_metrics: ContextVar[dict[str, Any] | None] = ContextVar("request_metrics", default=None)


def start_request_metrics():
    """Start a fresh metrics bucket for the current async context."""
    return _metrics.set({
        "llm_call_count": 0,
        "llm_total_time": 0.0,
    })


def reset_request_metrics(token) -> None:
    """Restore the previous metrics context."""
    _metrics.reset(token)


def record_llm_call(duration: float) -> None:
    """Record one LLM API call duration for the current request."""
    current = _metrics.get()
    if current is None:
        return
    current["llm_call_count"] += 1
    current["llm_total_time"] += duration


def get_request_metrics() -> dict[str, Any]:
    """Return a copy of current request metrics."""
    current = _metrics.get() or {}
    return {
        "llm_call_count": int(current.get("llm_call_count", 0)),
        "llm_total_time": float(current.get("llm_total_time", 0.0)),
    }
