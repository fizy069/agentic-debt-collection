from __future__ import annotations

import os
from dataclasses import dataclass

from anthropic import APIConnectionError, APIError, APITimeoutError, AsyncAnthropic


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


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
        self._client = AsyncAnthropic(api_key=self._api_key) if self._api_key else None

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
    ) -> LLMResult:
        if not self._client:
            return LLMResult(
                text=(
                    "Stub response: live Anthropic key not configured. "
                    f"Handled prompt excerpt -> {user_prompt[:140]}"
                ),
                model="stub",
                used_fallback=True,
            )

        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=0.2,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except (APIConnectionError, APITimeoutError, APIError) as exc:
            raise AnthropicServiceError("anthropic_request_failed", str(exc)) from exc

        text_blocks = []
        for content_block in response.content:
            block_text = getattr(content_block, "text", None)
            if block_text:
                text_blocks.append(block_text.strip())

        text = "\n".join(text_blocks).strip()
        if not text:
            raise AnthropicServiceError(
                "anthropic_empty_response",
                "Anthropic returned no text content.",
            )

        return LLMResult(text=text, model=self._model, used_fallback=False)
