from __future__ import annotations

from unittest.mock import patch

import pytest

from app.activities.agents import (
    _ASSESSMENT_OPENING_DISCLOSURE,
    _run_stage_turn,
)
from app.services.llm_types import LLMResult


class _FakeClient:
    def __init__(self, text: str) -> None:
        self._text = text

    async def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 300,
    ) -> LLMResult:
        return LLMResult(
            text=self._text,
            model="test-model",
            used_fallback=False,
        )


def _make_payload(*, turn_index: int) -> dict:
    return {
        "borrower": {
            "borrower_id": "b-disclosure-test",
            "account_reference": "acct-4321",
            "debt_amount": 1500.00,
            "currency": "USD",
            "days_past_due": 30,
            "borrower_message": "I received a notice and need help.",
            "notes": None,
        },
        "stage": "assessment",
        "borrower_message": "I received a notice and need help.",
        "transcript": [],
        "collected_fields": {},
        "turn_index": turn_index,
        "completed_stages": [],
    }


@pytest.mark.asyncio
async def test_first_assessment_turn_prepends_opening_disclosure():
    fake_client = _FakeClient("Please confirm the last four digits on the account.")

    with patch("app.activities.agents._get_anthropic_client", return_value=fake_client):
        result = await _run_stage_turn(_make_payload(turn_index=1))

    assert result["assistant_reply"].startswith(_ASSESSMENT_OPENING_DISCLOSURE)
    assert result["assistant_reply"].endswith(
        "Please confirm the last four digits on the account."
    )
    assert result["summary"].startswith(_ASSESSMENT_OPENING_DISCLOSURE)


@pytest.mark.asyncio
async def test_follow_up_assessment_turn_does_not_prepend_opening_disclosure():
    fake_client = _FakeClient("Please confirm the last four digits on the account.")

    with patch("app.activities.agents._get_anthropic_client", return_value=fake_client):
        result = await _run_stage_turn(_make_payload(turn_index=2))

    assert result["assistant_reply"] == "Please confirm the last four digits on the account."


@pytest.mark.asyncio
async def test_first_assessment_turn_does_not_duplicate_existing_disclosure():
    reply = (
        f"{_ASSESSMENT_OPENING_DISCLOSURE} "
        "Please confirm the last four digits on the account."
    )
    fake_client = _FakeClient(reply)

    with patch("app.activities.agents._get_anthropic_client", return_value=fake_client):
        result = await _run_stage_turn(_make_payload(turn_index=1))

    assert result["assistant_reply"] == reply
