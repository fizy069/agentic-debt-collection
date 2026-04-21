"""OpenAI provider backend for the LLM client (temporary)."""

from __future__ import annotations

import logging

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.services.llm_types import LLMResult, LLMServiceError, is_context_overflow_message

logger = logging.getLogger(__name__)


def create_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=api_key)


def extract_text(response: object) -> str:
    """Pull text from an OpenAI ChatCompletion response."""
    choices = getattr(response, "choices", None) or []
    parts: list[str] = []
    for choice in choices:
        msg = getattr(choice, "message", None)
        if msg:
            text = getattr(msg, "content", None)
            if text:
                parts.append(text.strip())
    return "\n".join(parts).strip()


def handle_api_error(exc: APIError) -> None:
    """Classify an OpenAI SDK error and raise a uniform LLMServiceError."""
    if isinstance(exc, RateLimitError):
        logger.error("OpenAI RATE LIMITED  detail=%s", exc)
        raise LLMServiceError("rate_limited", str(exc)) from exc

    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        logger.error("OpenAI connection/timeout error: %s", exc)
        raise LLMServiceError("request_failed", str(exc)) from exc

    status = getattr(exc, "status_code", None)
    if status == 400 and is_context_overflow_message(str(exc)):
        logger.error("OpenAI context overflow  detail=%s", exc)
        raise LLMServiceError("context_overflow", str(exc)) from exc

    logger.error(
        "OpenAI API error  status=%s  type=%s  detail=%s",
        status, type(exc).__name__, exc,
    )
    raise LLMServiceError("request_failed", str(exc)) from exc


async def generate(
    client: AsyncOpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> LLMResult:
    try:
        response = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except APIError as exc:
        handle_api_error(exc)

    logger.info(
        "LLM response  provider=openai  model=%s  finish_reason=%s  "
        "prompt_tokens=%s  completion_tokens=%s",
        getattr(response, "model", model),
        getattr(response.choices[0], "finish_reason", "?") if response.choices else "?",
        getattr(getattr(response, "usage", None), "prompt_tokens", "?"),
        getattr(getattr(response, "usage", None), "completion_tokens", "?"),
    )
    return LLMResult(text=extract_text(response), model=model, used_fallback=False)
