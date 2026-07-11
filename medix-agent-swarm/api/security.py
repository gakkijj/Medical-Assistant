"""Small, deterministic API safety checks and log redaction helpers."""
import re
from dataclasses import dataclass
from typing import Tuple


INJECTION_PATTERNS = (
    re.compile(r"忽略.{0,8}(系统|之前|以上).{0,12}(提示|指令)"),
    re.compile(r"(输出|显示|泄露).{0,12}(system prompt|系统提示词|api.?key|密钥)", re.I),
    re.compile(r"pretend you are.{0,30}(developer|system)", re.I),
)
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|token|secret)(\s*[=:]\s*)([^\s,;]{6,})"
)
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class InputSafetyResult:
    allowed: bool
    reason_codes: Tuple[str, ...]


def inspect_message(message: str) -> InputSafetyResult:
    reasons = []
    if CONTROL_PATTERN.search(message):
        reasons.append("control_characters")
    if any(pattern.search(message) for pattern in INJECTION_PATTERNS):
        reasons.append("prompt_injection")
    return InputSafetyResult(allowed=not reasons, reason_codes=tuple(reasons))


def redact_secrets(value: str) -> str:
    """Redact common credential assignments before text reaches logs/traces."""
    return SECRET_PATTERN.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", value)

