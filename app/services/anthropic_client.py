from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class AnthropicServiceError(RuntimeError):
    """Structured error for predictable activity retry behavior."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass
class LLMResult:
    text: str
    model: str
    used_fallback: bool = False


def _is_context_overflow(exc: APIError) -> bool:
    """Detect whether an API error is specifically a context-length overflow."""
    status = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if status == 400 and any(
        kw in msg for kw in ("too many tokens", "context length", "max tokens", "prompt is too long")
    ):
        return True
    return False


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self._client = AsyncAnthropic(api_key=self._api_key) if self._api_key else None
        logger.info(
            "AnthropicClient init  model=%s  key_present=%s  key_prefix=%s",
            self._model,
            bool(self._api_key),
            (self._api_key or "")[:12] + "..." if self._api_key else "none",
        )

    def _extract_text(self, response: object) -> str:
        """Pull text content from an Anthropic Messages response."""
        text_blocks = []
        for content_block in response.content:  # type: ignore[attr-defined]
            block_text = getattr(content_block, "text", None)
            if block_text:
                text_blocks.append(block_text.strip())
        return "\n".join(text_blocks).strip()

    def _log_rate_limit_headers(self, response: object) -> None:
        resp_headers = getattr(response, "_raw_response", None)
        if resp_headers is None:
            return
        h = resp_headers.headers
        logger.info(
            "LLM rate-limit headers  "
            "requests_remaining=%s  requests_limit=%s  requests_reset=%s  "
            "tokens_remaining=%s  tokens_limit=%s  tokens_reset=%s",
            h.get("anthropic-ratelimit-requests-remaining"),
            h.get("anthropic-ratelimit-requests-limit"),
            h.get("anthropic-ratelimit-requests-reset"),
            h.get("anthropic-ratelimit-tokens-remaining"),
            h.get("anthropic-ratelimit-tokens-limit"),
            h.get("anthropic-ratelimit-tokens-reset"),
        )

    def _handle_api_error(self, exc: APIError) -> None:
        """Raise the appropriate AnthropicServiceError for an API failure."""
        if isinstance(exc, RateLimitError):
            headers = getattr(exc, "response", None)
            retry_after = remaining = limit = reset = None
            if headers is not None:
                h = headers.headers
                retry_after = h.get("retry-after")
                remaining = h.get("anthropic-ratelimit-requests-remaining")
                limit = h.get("anthropic-ratelimit-requests-limit")
                reset = h.get("anthropic-ratelimit-requests-reset")
            logger.error(
                "RATE LIMITED  status=%s  retry_after=%s  "
                "remaining=%s  limit=%s  reset=%s  detail=%s",
                getattr(exc, "status_code", "?"),
                retry_after, remaining, limit, reset, exc,
            )
            raise AnthropicServiceError("anthropic_rate_limited", str(exc)) from exc

        if isinstance(exc, (APIConnectionError, APITimeoutError)):
            logger.error("LLM connection/timeout error: %s", exc)
            raise AnthropicServiceError("anthropic_request_failed", str(exc)) from exc

        if _is_context_overflow(exc):
            logger.error(
                "LLM context overflow  status=%s  detail=%s",
                getattr(exc, "status_code", "?"), exc,
            )
            raise AnthropicServiceError("anthropic_context_overflow", str(exc)) from exc

        status_code = getattr(exc, "status_code", "?")
        logger.error(
            "LLM API error  status=%s  type=%s  detail=%s",
            status_code, type(exc).__name__, exc,
        )
        raise AnthropicServiceError("anthropic_request_failed", str(exc)) from exc

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
    ) -> LLMResult:
        if not self._client:
            logger.warning(
                "LLM call skipped – no API key, returning stub.  "
                "env ANTHROPIC_API_KEY=%s",
                "set" if os.getenv("ANTHROPIC_API_KEY") else "MISSING",
            )
            return LLMResult(
                text=(
                    "Stub response: live Anthropic key not configured. "
                    f"Handled prompt excerpt -> {user_prompt[:140]}"
                ),
                model="stub",
                used_fallback=True,
            )

        logger.info(
            "LLM request  model=%s  max_tokens=%d  system_len=%d  user_len=%d",
            self._model, max_tokens, len(system_prompt), len(user_prompt),
        )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=0.2,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except APIError as exc:
            self._handle_api_error(exc)

        self._log_rate_limit_headers(response)

        logger.info(
            "LLM response  model=%s  stop_reason=%s  input_tokens=%s  output_tokens=%s",
            response.model,
            response.stop_reason,
            getattr(response.usage, "input_tokens", "?"),
            getattr(response.usage, "output_tokens", "?"),
        )

        text = self._extract_text(response)
        if not text:
            logger.error("LLM returned empty text content")
            raise AnthropicServiceError(
                "anthropic_empty_response",
                "Anthropic returned no text content.",
            )

        return LLMResult(text=text, model=self._model, used_fallback=False)

    async def summarize(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 200,
    ) -> LLMResult:
        """Dedicated summarization call with small output cap.

        Used for overflow compression and handoff enrichment.
        Falls back to a stub when no API key is configured.
        """
        if not self._client:
            logger.warning("Summarize call skipped – no API key, returning stub")
            return LLMResult(
                text=user_prompt[:300],
                model="stub",
                used_fallback=True,
            )

        logger.info(
            "LLM summarize request  model=%s  max_tokens=%d  system_len=%d  user_len=%d",
            self._model, max_tokens, len(system_prompt), len(user_prompt),
        )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except APIError as exc:
            self._handle_api_error(exc)

        self._log_rate_limit_headers(response)

        logger.info(
            "LLM summarize response  model=%s  stop_reason=%s  "
            "input_tokens=%s  output_tokens=%s",
            response.model,
            response.stop_reason,
            getattr(response.usage, "input_tokens", "?"),
            getattr(response.usage, "output_tokens", "?"),
        )

        text = self._extract_text(response)
        if not text:
            logger.warning("Summarize returned empty text, using truncated input as fallback")
            return LLMResult(
                text=user_prompt[:300],
                model=self._model,
                used_fallback=True,
            )

        try:
            json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Summarize response is not valid JSON, using raw text")

        return LLMResult(text=text, model=self._model, used_fallback=False)
