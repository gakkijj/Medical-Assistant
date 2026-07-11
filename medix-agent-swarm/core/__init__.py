"""Core package with lightweight routing imports and lazy runtime modules."""
from importlib import import_module
from typing import Dict, Tuple

from .routing import AdaptiveRouter, RouteDecision


_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "LLMClient": (".llm_client", "LLMClient"),
    "ToolCall": (".llm_client", "ToolCall"),
    "LLMResponse": (".llm_client", "LLMResponse"),
    "AgentLoop": (".agent_loop", "AgentLoop"),
    "AgentState": (".state_manager", "AgentState"),
    "TaskStatus": (".state_manager", "TaskStatus"),
    "SkillRegistry": (".skill_registry", "SkillRegistry"),
    "SkillParameter": (".skill_registry", "SkillParameter"),
}


def __getattr__(name: str):
    """Load network/model-heavy modules only when the runtime asks for them."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


__all__ = [
    "AdaptiveRouter",
    "RouteDecision",
    *_LAZY_IMPORTS.keys(),
]
