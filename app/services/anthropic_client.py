"""LLM client facade with Anthropic-first, OpenAI-fallback provider selection.

All provider-specific logic lives in ``llm_anthropic`` and ``llm_openai``;
this module exposes a single ``AnthropicClient`` class (name kept for backward
compatibility) and re-exports shared types so existing imports keep working.
"""

from __future__ import annotations

import json
import logging
import os

from app.services import llm_anthropic, llm_openai
from app.services.llm_types import (
    AnthropicServiceError,
    LLMResult,
    LLMServiceError,
    is_context_overflow_message,
)
from app.services.token_budget import configure_encoding

__all__ = [
    "AnthropicClient",
    "AnthropicServiceError",
    "LLMServiceError",
    "LLMResult",
    "is_context_overflow_message",
]

logger = logging.getLogger(__name__)


class AnthropicClient:
    """Thin facade that routes ``generate`` / ``summarize`` to the active provider.

    Provider precedence at init:
      1. Anthropic — if ``ANTHROPIC_API_KEY`` is available.
      2. OpenAI   — if ``OPENAI_API_KEY`` is available (temporary).
      3. Stub     — no live key configured.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        anthropic_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        self._provider: str
        self._anthropic = None
        self._openai = None

        if anthropic_key:
            self._provider = "anthropic"
            self._api_key = anthropic_key
            self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
            self._anthropic = llm_anthropic.create_client(self._api_key)
            configure_encoding()
        elif openai_key:
            self._provider = "openai"
            self._api_key = openai_key
            self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self._openai = llm_openai.create_client(self._api_key)
            configure_encoding(self._model)
        else:
            self._provider = "stub"
            self._api_key = None
            self._model = model or "stub"
            configure_encoding()

        logger.info(
            "LLMClient init  provider=%s  model=%s  key_present=%s  key_prefix=%s",
            self._provider,
            self._model,
            bool(self._api_key),
            (self._api_key or "")[:12] + "..." if self._api_key else "none",
        )

    @property
    def _client(self) -> object | None:
        """Backward-compatible property used by the cached-client check in agents.py."""
        return self._anthropic or self._openai

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    async def _call_provider(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float,
    ) -> LLMResult:
        if self._provider == "openai":
            return await llm_openai.generate(
                self._openai,
                model=self._model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return await llm_anthropic.generate(
            self._anthropic,
            model=self._model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
    ) -> LLMResult:
        if self._provider == "stub":
            logger.warning(
                "LLM call skipped – no API key, returning stub.  "
                "env ANTHROPIC_API_KEY=%s  OPENAI_API_KEY=%s",
                "set" if os.getenv("ANTHROPIC_API_KEY") else "MISSING",
                "set" if os.getenv("OPENAI_API_KEY") else "MISSING",
            )
            return LLMResult(
                text=(
                    "Stub response: no LLM key configured. "
                    f"Handled prompt excerpt -> {user_prompt[:140]}"
                ),
                model="stub",
                used_fallback=True,
            )

        logger.info(
            "LLM request  provider=%s  model=%s  max_tokens=%d  system_len=%d  user_len=%d",
            self._provider, self._model, max_tokens, len(system_prompt), len(user_prompt),
        )

        result = await self._call_provider(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.2,
        )

        if not result.text:
            logger.error("LLM returned empty text content")
            raise LLMServiceError("empty_response", "LLM returned no text content.")

        return result

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
        if self._provider == "stub":
            logger.warning("Summarize call skipped – no API key, returning stub")
            return LLMResult(text=user_prompt[:300], model="stub", used_fallback=True)

        logger.info(
            "LLM summarize request  provider=%s  model=%s  max_tokens=%d  "
            "system_len=%d  user_len=%d",
            self._provider, self._model, max_tokens,
            len(system_prompt), len(user_prompt),
        )

        result = await self._call_provider(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        if not result.text:
            logger.warning("Summarize returned empty text, using truncated input as fallback")
            return LLMResult(text=user_prompt[:300], model=self._model, used_fallback=True)

        try:
            json.loads(result.text)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Summarize response is not valid JSON, using raw text")

        return result
