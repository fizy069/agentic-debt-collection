"""Tests for AnthropicClient error classification and stub behavior."""

from __future__ import annotations

import pytest

from app.services.anthropic_client import (
    AnthropicClient,
    AnthropicServiceError,
    _is_context_overflow,
)


class _FakeAPIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(message)


class TestContextOverflowDetection:
    def test_detects_too_many_tokens(self):
        exc = _FakeAPIError(400, "Request too large: too many tokens in prompt")
        assert _is_context_overflow(exc) is True

    def test_detects_context_length(self):
        exc = _FakeAPIError(400, "context length exceeded")
        assert _is_context_overflow(exc) is True

    def test_detects_prompt_too_long(self):
        exc = _FakeAPIError(400, "prompt is too long for the model")
        assert _is_context_overflow(exc) is True

    def test_ignores_non_400(self):
        exc = _FakeAPIError(500, "too many tokens")
        assert _is_context_overflow(exc) is False

    def test_ignores_unrelated_400(self):
        exc = _FakeAPIError(400, "invalid request format")
        assert _is_context_overflow(exc) is False


class TestClientStubBehavior:
    @pytest.mark.asyncio
    async def test_generate_stub_when_no_key(self):
        client = AnthropicClient(api_key=None)
        result = await client.generate(
            system_prompt="test system",
            user_prompt="test user",
        )
        assert result.used_fallback is True
        assert result.model == "stub"

    @pytest.mark.asyncio
    async def test_summarize_stub_when_no_key(self):
        client = AnthropicClient(api_key=None)
        result = await client.summarize(
            system_prompt="compress this",
            user_prompt="long content here",
        )
        assert result.used_fallback is True
        assert result.model == "stub"
        assert len(result.text) > 0
