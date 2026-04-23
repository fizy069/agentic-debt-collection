"""Tests for LLM client facade: error classification, stub behavior, and provider selection."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.services.anthropic_client import AnthropicClient, AnthropicServiceError, LLMServiceError
from app.services.llm_types import is_context_overflow_message
from app.services.token_budget import configure_encoding, get_active_encoding_name


class _FakeAPIError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        super().__init__(message)


# ------------------------------------------------------------------
# Shared overflow-message detection
# ------------------------------------------------------------------

class TestContextOverflowDetection:
    """Covers the unified overflow-message helper used by both providers."""

    def test_detects_too_many_tokens(self):
        assert is_context_overflow_message("Request too large: too many tokens in prompt") is True

    def test_detects_context_length(self):
        assert is_context_overflow_message("context length exceeded") is True

    def test_detects_prompt_too_long(self):
        assert is_context_overflow_message("prompt is too long for the model") is True

    def test_detects_maximum_context_length(self):
        assert is_context_overflow_message("maximum context length is 128000 tokens") is True

    def test_detects_context_length_exceeded_code(self):
        assert is_context_overflow_message("context_length_exceeded") is True

    def test_ignores_unrelated_message(self):
        assert is_context_overflow_message("invalid request format") is False

    def test_ignores_invalid_model(self):
        assert is_context_overflow_message("invalid model") is False


# ------------------------------------------------------------------
# Error type backward compatibility
# ------------------------------------------------------------------

class TestErrorTypeCompat:
    def test_llm_service_error_is_anthropic_service_error(self):
        err = LLMServiceError("rate_limited", "test")
        assert isinstance(err, AnthropicServiceError)

    def test_anthropic_service_error_has_code_and_detail(self):
        err = AnthropicServiceError("request_failed", "connection reset")
        assert err.code == "request_failed"
        assert err.detail == "connection reset"


# ------------------------------------------------------------------
# Stub behavior
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Provider selection
# ------------------------------------------------------------------

class TestProviderSelection:
    """Verify the Anthropic -> OpenAI -> stub precedence logic."""

    def test_anthropic_selected_when_key_provided(self):
        client = AnthropicClient(api_key="sk-ant-test-key")
        assert client._provider == "anthropic"
        assert client._anthropic is not None
        assert client._openai is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    def test_anthropic_wins_over_openai_env(self):
        client = AnthropicClient(api_key="sk-ant-test-key")
        assert client._provider == "anthropic"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-test-openai", "OPENAI_MODEL": "gpt-4o"},
        clear=False,
    )
    def test_openai_selected_when_no_anthropic_key(self):
        client = AnthropicClient(api_key=None)
        assert client._provider == "openai"
        assert client._openai is not None
        assert client._anthropic is None
        assert client._model == "gpt-4o"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    def test_openai_default_model(self):
        client = AnthropicClient(api_key=None)
        assert client._model == "gpt-4o-mini"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    def test_openai_rejects_claude_model_argument(self):
        """Regression: judges used to hardcode ``claude-haiku-4-5`` which 404'd
        when only OpenAI was configured.  The client must now detect the
        cross-provider mismatch and fall back to the OpenAI default."""
        client = AnthropicClient(api_key=None, model="claude-haiku-4-5")
        assert client._provider == "openai"
        assert client._model == "gpt-4o-mini"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-test-openai", "OPENAI_MODEL": "gpt-4o"},
        clear=False,
    )
    def test_openai_rejects_claude_model_but_honors_env_override(self):
        client = AnthropicClient(api_key=None, model="claude-opus-4")
        assert client._provider == "openai"
        assert client._model == "gpt-4o"

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-test-openai", "OPENAI_MODEL": "claude-haiku-4-5"},
        clear=False,
    )
    def test_openai_rejects_claude_env_override(self):
        """``OPENAI_MODEL`` set to a Claude name (common copy/paste mistake)
        must also be ignored in favour of the OpenAI default."""
        client = AnthropicClient(api_key=None)
        assert client._provider == "openai"
        assert client._model == "gpt-4o-mini"

    def test_anthropic_rejects_openai_model_argument(self):
        client = AnthropicClient(api_key="sk-ant-test-key", model="gpt-4o-mini")
        assert client._provider == "anthropic"
        assert client._model == "claude-haiku-4-5"

    @patch.dict("os.environ", {"ANTHROPIC_MODEL": "gpt-4o"}, clear=False)
    def test_anthropic_rejects_openai_env_override(self):
        client = AnthropicClient(api_key="sk-ant-test-key")
        assert client._provider == "anthropic"
        assert client._model == "claude-haiku-4-5"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    def test_public_model_property(self):
        client = AnthropicClient(api_key=None)
        assert client.model == client._model == "gpt-4o-mini"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    def test_public_provider_property(self):
        client = AnthropicClient(api_key=None)
        assert client.provider == "openai"

    def test_stub_when_no_keys(self):
        client = AnthropicClient(api_key=None)
        assert client._provider == "stub"
        assert client._client is None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    def test_client_property_truthy_for_openai(self):
        client = AnthropicClient(api_key=None)
        assert client._client is not None

    def test_client_property_truthy_for_anthropic(self):
        client = AnthropicClient(api_key="sk-ant-test-key")
        assert client._client is not None


# ------------------------------------------------------------------
# Token encoding configuration
# ------------------------------------------------------------------

class TestTokenEncodingOnInit:
    """Verify that provider selection configures the right tiktoken encoding."""

    def teardown_method(self):
        configure_encoding()

    @patch.dict(
        "os.environ",
        {"OPENAI_API_KEY": "sk-test-openai", "OPENAI_MODEL": "gpt-4o"},
        clear=False,
    )
    def test_openai_gpt4o_sets_o200k(self):
        AnthropicClient(api_key=None)
        assert get_active_encoding_name() == "o200k_base"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    def test_openai_default_model_sets_o200k(self):
        AnthropicClient(api_key=None)
        assert get_active_encoding_name() == "o200k_base"

    def test_anthropic_keeps_default_encoding(self):
        AnthropicClient(api_key="sk-ant-test-key")
        assert get_active_encoding_name() == "cl100k_base"

    def test_stub_keeps_default_encoding(self):
        AnthropicClient(api_key=None)
        assert get_active_encoding_name() == "cl100k_base"


# ------------------------------------------------------------------
# OpenAI dispatch (mocked)
# ------------------------------------------------------------------

def _fake_openai_response(content: str = "hello from openai"):
    return type("R", (), {
        "choices": [type("C", (), {
            "message": type("M", (), {"content": content})(),
            "finish_reason": "stop",
        })()],
        "model": "gpt-4o-mini",
        "usage": type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})(),
    })()


class TestOpenAIDispatch:
    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    @pytest.mark.asyncio
    async def test_generate_routes_to_openai(self):
        client = AnthropicClient(api_key=None)
        assert client._provider == "openai"

        async def _fake_create(**kwargs):
            return _fake_openai_response()

        client._openai.chat.completions.create = _fake_create  # type: ignore[union-attr]

        result = await client.generate(system_prompt="sys", user_prompt="usr")
        assert result.text == "hello from openai"
        assert result.model == "gpt-4o-mini"
        assert result.used_fallback is False

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-openai"}, clear=False)
    @pytest.mark.asyncio
    async def test_summarize_routes_to_openai(self):
        client = AnthropicClient(api_key=None)

        async def _fake_create(**kwargs):
            return _fake_openai_response("summary text")

        client._openai.chat.completions.create = _fake_create  # type: ignore[union-attr]

        result = await client.summarize(system_prompt="compress", user_prompt="long text")
        assert result.text == "summary text"
        assert result.used_fallback is False
