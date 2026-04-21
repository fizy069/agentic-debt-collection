"""Integration tests for the overflow summarization path in _run_stage_turn.

These tests use stub mode (no API key) to verify the full assembly,
overflow detection, and fallback paths produce valid outputs within budget.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.activities.agents import _run_stage_turn
from app.services.token_budget import (
    MAX_BORROWER_MESSAGE_TOKENS,
    MAX_CONTEXT_TOKENS,
    MAX_HANDOFF_TOKENS,
    OVERSIZED_MESSAGE_REPLY,
    count_tokens,
)


def _make_payload(
    stage: str = "resolution",
    num_transcript_pairs: int = 3,
    msg_len: int = 80,
    completed_stages: list | None = None,
):
    transcript = []
    for i in range(num_transcript_pairs):
        transcript.append({
            "role": "borrower",
            "stage": "assessment",
            "text": f"Borrower turn {i} " + "x" * msg_len,
            "timestamp": "2026-04-01T00:00:00",
        })
        transcript.append({
            "role": "agent",
            "stage": "assessment",
            "text": f"Agent turn {i} " + "y" * msg_len,
            "timestamp": "2026-04-01T00:00:01",
        })

    if completed_stages is None:
        completed_stages = [{
            "stage": "assessment",
            "collected_fields": {
                "identity_confirmed": True,
                "debt_acknowledged": True,
                "ability_to_pay_discussed": True,
            },
            "transition_reason": "fields_collected",
            "turns": 3,
        }]

    return {
        "borrower": {
            "borrower_id": "b-overflow-test",
            "account_reference": "acct-9999",
            "debt_amount": 2500.00,
            "currency": "USD",
            "days_past_due": 45,
            "borrower_message": "I want to discuss my options for the plan.",
            "notes": None,
        },
        "stage": stage,
        "borrower_message": "I want to discuss my options for the plan.",
        "transcript": transcript,
        "collected_fields": {},
        "turn_index": 1,
        "completed_stages": completed_stages,
    }


class TestOverflowPath:
    """Verify the overflow compression path triggers and stays within budget."""

    @pytest.mark.asyncio
    async def test_short_transcript_no_overflow(self):
        payload = _make_payload(num_transcript_pairs=2, msg_len=20)
        result = await _run_stage_turn(payload)

        assert result["stage"] == "resolution"
        meta = result["metadata"]
        assert "overflow_detected" in meta
        assert "total_tokens" in meta

    @pytest.mark.asyncio
    async def test_long_transcript_triggers_overflow(self):
        """Build a payload large enough to exceed 2000 tokens total context.

        The 8-item recent transcript window is the main knob: each message
        at ~160 tokens * 8 = ~1300, plus system (~240) + snapshot + meta +
        handoff pushes us well past 2000.
        """
        long_transcript = []
        for i in range(20):
            long_transcript.append({
                "role": "borrower",
                "stage": "assessment",
                "text": f"Borrower message {i}: " + "detailed explanation " * 80,
                "timestamp": "2026-04-01T00:00:00",
            })
            long_transcript.append({
                "role": "agent",
                "stage": "assessment",
                "text": f"Agent response {i}: " + "comprehensive reply " * 80,
                "timestamp": "2026-04-01T00:00:01",
            })

        payload = {
            "borrower": {
                "borrower_id": "b-overflow-test",
                "account_reference": "acct-9999",
                "debt_amount": 2500.00,
                "currency": "USD",
                "days_past_due": 45,
                "borrower_message": "I want to discuss options " + "context " * 50,
                "notes": "Important notes " + "detail " * 50,
            },
            "stage": "resolution",
            "borrower_message": "I want to discuss options " + "context " * 50,
            "transcript": long_transcript,
            "collected_fields": {},
            "turn_index": 1,
            "completed_stages": [{
                "stage": "assessment",
                "collected_fields": {
                    "identity_confirmed": True,
                    "debt_acknowledged": True,
                    "ability_to_pay_discussed": True,
                },
                "transition_reason": "fields_collected",
                "turns": 3,
            }],
        }
        result = await _run_stage_turn(payload)

        meta = result["metadata"]
        assert meta["overflow_detected"] is True
        assert meta["post_overflow_tokens"] <= MAX_CONTEXT_TOKENS

    @pytest.mark.asyncio
    async def test_final_notice_two_stage_handoff(self):
        completed = [
            {
                "stage": "assessment",
                "collected_fields": {
                    "identity_confirmed": True,
                    "debt_acknowledged": True,
                    "ability_to_pay_discussed": True,
                },
                "transition_reason": "fields_collected",
                "turns": 3,
            },
            {
                "stage": "resolution",
                "collected_fields": {
                    "options_reviewed": True,
                    "borrower_position_known": True,
                    "commitment_or_disposition": True,
                },
                "transition_reason": "resolution_terms_captured",
                "turns": 3,
            },
        ]
        payload = _make_payload(
            stage="final_notice",
            num_transcript_pairs=15,
            msg_len=150,
            completed_stages=completed,
        )
        result = await _run_stage_turn(payload)

        meta = result["metadata"]
        assert "handoff_tokens" in meta
        assert meta["handoff_tokens"] <= MAX_HANDOFF_TOKENS

    @pytest.mark.asyncio
    async def test_assessment_no_handoff_no_overflow(self):
        payload = _make_payload(
            stage="assessment",
            num_transcript_pairs=1,
            msg_len=20,
            completed_stages=[],
        )
        result = await _run_stage_turn(payload)

        meta = result["metadata"]
        assert meta["overflow_detected"] is False
        assert meta["handoff_tokens"] == 0

    @pytest.mark.asyncio
    async def test_budget_metadata_always_present(self):
        payload = _make_payload(num_transcript_pairs=2, msg_len=30)
        result = await _run_stage_turn(payload)

        meta = result["metadata"]
        required_keys = [
            "budget_limit", "total_tokens", "overflow_detected",
            "overflow_summary_used", "overflow_fallback_used",
            "handoff_tokens", "sections",
        ]
        for key in required_keys:
            assert key in meta, f"Missing metadata key: {key}"


def _make_oversized_payload(stage: str = "assessment"):
    """Build a payload whose borrower_message exceeds MAX_BORROWER_MESSAGE_TOKENS."""
    oversized_msg = "word " * 2300
    assert count_tokens(oversized_msg) > MAX_BORROWER_MESSAGE_TOKENS

    return {
        "borrower": {
            "borrower_id": "b-oversized-test",
            "account_reference": "acct-1234",
            "debt_amount": 1000.00,
            "currency": "USD",
            "days_past_due": 30,
            "borrower_message": "I need help.",
            "notes": None,
        },
        "stage": stage,
        "borrower_message": oversized_msg,
        "transcript": [],
        "collected_fields": {},
        "turn_index": 1,
        "completed_stages": [],
    }


class TestOversizedBorrowerMessageGuard:
    """Verify oversized borrower messages get a fixed reply with no LLM call."""

    @pytest.mark.asyncio
    async def test_oversized_message_returns_fixed_reply(self):
        payload = _make_oversized_payload()
        result = await _run_stage_turn(payload)

        assert result["assistant_reply"] == OVERSIZED_MESSAGE_REPLY
        assert result["summary"] == OVERSIZED_MESSAGE_REPLY

    @pytest.mark.asyncio
    async def test_oversized_message_metadata_flags(self):
        payload = _make_oversized_payload()
        result = await _run_stage_turn(payload)

        meta = result["metadata"]
        assert meta["borrower_message_oversized"] is True
        assert meta["borrower_message_tokens"] > MAX_BORROWER_MESSAGE_TOKENS
        assert meta["model"] == "none"

    @pytest.mark.asyncio
    async def test_oversized_message_keeps_stage_incomplete(self):
        payload = _make_oversized_payload()
        result = await _run_stage_turn(payload)

        assert result["stage_complete"] is False
        assert result["decision"] == "borrower_message_oversized"
        assert result["transition_reason"] == "borrower_message_too_long"
        assert result["next_stage"] == payload["stage"]

    @pytest.mark.asyncio
    async def test_oversized_guard_preserves_collected_fields(self):
        payload = _make_oversized_payload(stage="resolution")
        payload["collected_fields"] = {"options_reviewed": True}
        result = await _run_stage_turn(payload)

        assert result["collected_fields"] == {"options_reviewed": True}

    @pytest.mark.asyncio
    async def test_normal_message_not_blocked(self):
        payload = _make_payload(
            stage="assessment",
            num_transcript_pairs=1,
            msg_len=20,
            completed_stages=[],
        )
        result = await _run_stage_turn(payload)

        assert result["assistant_reply"] != OVERSIZED_MESSAGE_REPLY
        assert result["metadata"].get("borrower_message_oversized") is None
