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

_ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5"
_OPENAI_DEFAULT_MODEL = "gpt-4o-mini"


def _looks_like_anthropic_model(name: str) -> bool:
    return name.lower().startswith(("claude",))


def _looks_like_openai_model(name: str) -> bool:
    lower = name.lower()
    return lower.startswith(("gpt-", "gpt_", "o1", "o3", "o4", "chatgpt"))


class AnthropicClient:
    """Thin facade that routes ``generate`` / ``summarize`` to the active provider.

    Provider precedence at init:
      1. Anthropic — if ``ANTHROPIC_API_KEY`` is available.
      2. OpenAI   — if ``OPENAI_API_KEY`` is available (temporary).
      3. Stub     — no live key configured.

    Model selection is provider-aware: if the caller (or an env override)
    passes a model name that clearly belongs to the *other* provider — e.g.
    ``claude-haiku-4-5`` when only ``OPENAI_API_KEY`` is configured — the
    client ignores the mismatch and falls back to the active provider's
    default model.  This prevents the eval harness and judges from hard-
    coding Anthropic model names that 404 when the OpenAI backend is in use.
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
            self._model = self._resolve_model(
                requested=model,
                env_var="ANTHROPIC_MODEL",
                default=_ANTHROPIC_DEFAULT_MODEL,
                mismatch_predicate=_looks_like_openai_model,
            )
            self._anthropic = llm_anthropic.create_client(self._api_key)
            configure_encoding()
        elif openai_key:
            self._provider = "openai"
            self._api_key = openai_key
            self._model = self._resolve_model(
                requested=model,
                env_var="OPENAI_MODEL",
                default=_OPENAI_DEFAULT_MODEL,
                mismatch_predicate=_looks_like_anthropic_model,
            )
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

    def _resolve_model(
        self,
        *,
        requested: str | None,
        env_var: str,
        default: str,
        mismatch_predicate,
    ) -> str:
        """Pick a sensible model name for the active provider.

        Precedence:
          1. Caller-supplied ``requested`` model (CLI flag / code).
          2. ``env_var`` from the environment.
          3. Hard-coded per-provider default.

        If a candidate clearly belongs to the *other* provider (detected by
        ``mismatch_predicate``) it is dropped with a warning and the next
        source is tried.  This keeps the eval harness working when only one
        provider key is configured but judges/simulators hard-code the other
        provider's model name.
        """
        candidates: list[tuple[str, str | None]] = [
            ("argument", requested),
            (f"env:{env_var}", os.getenv(env_var)),
        ]
        for source, value in candidates:
            if not value:
                continue
            if mismatch_predicate(value):
                logger.warning(
                    "LLMClient model mismatch  provider=%s  source=%s  requested=%s  "
                    "falling_back_to=%s",
                    self._provider, source, value, default,
                )
                continue
            return value
        return default

    @property
    def _client(self) -> object | None:
        """Backward-compatible property used by the cached-client check in agents.py."""
        return self._anthropic or self._openai

    @property
    def model(self) -> str:
        """Resolved model id for the active provider (e.g. ``claude-haiku-4-5`` or ``gpt-4o-mini``)."""
        return self._model

    @property
    def provider(self) -> str:
        """Active provider: ``anthropic``, ``openai``, or ``stub``."""
        return self._provider

    @property
    def model(self) -> str:
        """Public, read-only view of the resolved model name."""
        return self._model

    @property
    def provider(self) -> str:
        """Public, read-only view of the active provider: ``anthropic`` / ``openai`` / ``stub``."""
        return self._provider

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
