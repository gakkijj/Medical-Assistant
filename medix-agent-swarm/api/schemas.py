"""Pydantic schemas for the Web API."""
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    message: str = Field(..., min_length=1, description="User message")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID")


class ChatResponse(BaseModel):
    """Normalized chat response for the frontend."""

    session_id: str
    answer: str
    suggestions: Optional[list[str]] = None
    disclaimer: Optional[str] = None
    agents_involved: Optional[list[str]] = None
    swarm_enabled: Optional[bool] = None
    total_time: Optional[float] = None
    total_elapsed_time: Optional[float] = None
    llm_total_time: Optional[float] = None
    llm_call_count: Optional[int] = None
    timings: Optional[dict[str, float]] = None
    raw: Optional[dict[str, Any]] = None
