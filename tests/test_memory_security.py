from __future__ import annotations

from memory_store.security import redact_sensitive_text


def test_redact_sensitive_text_removes_common_secret_patterns() -> None:
    text = (
        "password = supersecret123\n"
        "OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456\n"
        "email person@example.com\n"
        "card 4242 4242 4242 4242"
    )

    redacted = redact_sensitive_text(text)

    assert "supersecret123" not in redacted
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in redacted
    assert "person@example.com" not in redacted
    assert "4242 4242 4242 4242" not in redacted
    assert "[REDACTED" in redacted
