from __future__ import annotations

import re

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|passwd|bearer)\b\s*[:=]\s*['\"]?([A-Za-z0-9._~+/=-]{8,})"
)
_OPENAI_STYLE_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")
_AWS_ACCESS_KEY = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_CARD_LIKE_NUMBER = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


def redact_sensitive_text(text: str) -> str:
    redacted = _SECRET_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    redacted = _OPENAI_STYLE_KEY.sub("[REDACTED_API_KEY]", redacted)
    redacted = _AWS_ACCESS_KEY.sub("[REDACTED_AWS_KEY]", redacted)
    redacted = _JWT.sub("[REDACTED_TOKEN]", redacted)
    redacted = _EMAIL.sub("[REDACTED_EMAIL]", redacted)
    return _CARD_LIKE_NUMBER.sub("[REDACTED_NUMBER]", redacted)
