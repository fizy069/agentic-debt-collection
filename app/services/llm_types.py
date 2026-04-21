"""Shared types and utilities for all LLM provider backends."""

from __future__ import annotations

from dataclasses import dataclass


class LLMServiceError(RuntimeError):
    """Structured error for predictable activity retry behavior.

    Error codes are provider-agnostic:
      - ``rate_limited``
      - ``request_failed``
      - ``context_overflow``
      - ``empty_response``
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


# Keep the old name importable so downstream code that catches it still works.
AnthropicServiceError = LLMServiceError


@dataclass
class LLMResult:
    text: str
    model: str
    used_fallback: bool = False


def is_context_overflow_message(msg: str) -> bool:
    """Return True if *msg* looks like a context-length error from any provider."""
    lowered = msg.lower()
    return any(
        kw in lowered
        for kw in (
            "too many tokens",
            "context length",
            "context_length_exceeded",
            "maximum context length",
            "max tokens",
            "prompt is too long",
        )
    )
