"""Pydantic schemas for the Web API."""
from __future__ import annotations
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(..., min_length=1, max_length=8000, description="User message")
    session_id: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
        description="Opaque conversation session ID",
    )
    routing_mode: Literal["auto", "single", "swarm"] = Field(
        default="auto",
        description="Ablation control; auto is recommended for normal use",
    )


class ChatResponse(BaseModel):
    """Normalized chat response for the frontend."""

    session_id: str
    request_id: Optional[str] = None
    answer: str
    suggestions: Optional[list[str]] = None
    disclaimer: Optional[str] = None
    agents_involved: Optional[list[str]] = None
    swarm_enabled: Optional[bool] = None
    total_time: Optional[float] = None
    total_elapsed_time: Optional[float] = None
    llm_total_time: Optional[float] = None
    llm_call_count: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tool_call_count: Optional[int] = None
    route: Optional[dict[str, Any]] = None
    citations: Optional[list[dict[str, Any]]] = None
    trace: Optional[list[dict[str, Any]]] = None
    timings: Optional[dict[str, float]] = None
    raw: Optional[dict[str, Any]] = None
