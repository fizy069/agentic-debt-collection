"""Anthropic provider backend for the LLM client."""

from __future__ import annotations

import logging

from anthropic import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncAnthropic,
    RateLimitError,
)

from app.services.llm_types import LLMResult, LLMServiceError, is_context_overflow_message

logger = logging.getLogger(__name__)


def create_client(api_key: str) -> AsyncAnthropic:
    return AsyncAnthropic(api_key=api_key)


def extract_text(response: object) -> str:
    """Pull text content from an Anthropic Messages response."""
    text_blocks: list[str] = []
    for content_block in response.content:  # type: ignore[attr-defined]
        block_text = getattr(content_block, "text", None)
        if block_text:
            text_blocks.append(block_text.strip())
    return "\n".join(text_blocks).strip()


def log_rate_limit_headers(response: object) -> None:
    raw = getattr(response, "_raw_response", None)
    if raw is None:
        return
    h = raw.headers
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


def handle_api_error(exc: APIError) -> None:
    """Classify an Anthropic SDK error and raise a uniform LLMServiceError."""
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
        raise LLMServiceError("rate_limited", str(exc)) from exc

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        logger.error("LLM connection/timeout error: %s", exc)
        raise LLMServiceError("request_failed", str(exc)) from exc

    status = getattr(exc, "status_code", None)
    if status == 400 and is_context_overflow_message(str(exc)):
        logger.error(
            "LLM context overflow  status=%s  detail=%s", status, exc,
        )
        raise LLMServiceError("context_overflow", str(exc)) from exc

    logger.error(
        "LLM API error  status=%s  type=%s  detail=%s",
        status, type(exc).__name__, exc,
    )
    raise LLMServiceError("request_failed", str(exc)) from exc


async def generate(
    client: AsyncAnthropic,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> LLMResult:
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except APIError as exc:
        handle_api_error(exc)

    log_rate_limit_headers(response)

    logger.info(
        "LLM response  provider=anthropic  model=%s  stop_reason=%s  "
        "input_tokens=%s  output_tokens=%s",
        response.model,
        response.stop_reason,
        getattr(response.usage, "input_tokens", "?"),
        getattr(response.usage, "output_tokens", "?"),
    )
    in_tok = getattr(response.usage, "input_tokens", 0) or 0
    out_tok = getattr(response.usage, "output_tokens", 0) or 0
    return LLMResult(
        text=extract_text(response),
        model=model,
        used_fallback=False,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )
