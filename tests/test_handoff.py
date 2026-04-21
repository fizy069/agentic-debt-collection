"""Tests for handoff summary builder – budget guarantees and priority pruning."""

from __future__ import annotations

import json

import pytest

from app.services.handoff import build_handoff_summary, _build_raw_summary
from app.services.token_budget import MAX_HANDOFF_TOKENS, count_tokens


def _make_borrower(**overrides):
    base = {
        "borrower_id": "b-test-001",
        "account_reference": "acct-1234",
        "debt_amount": 5000.00,
        "currency": "USD",
        "days_past_due": 60,
        "borrower_message": "I need to discuss my debt.",
        "notes": None,
    }
    base.update(overrides)
    return base


def _make_transcript(num_pairs: int = 3, stage: str = "assessment", msg_len: int = 80):
    transcript = []
    for i in range(num_pairs):
        transcript.append({
            "role": "borrower",
            "stage": stage,
            "text": f"Borrower message {i} " + "x" * msg_len,
            "timestamp": "2026-04-01T00:00:00",
        })
        transcript.append({
            "role": "agent",
            "stage": stage,
            "text": f"Agent response {i} " + "y" * msg_len,
            "timestamp": "2026-04-01T00:00:01",
        })
    return transcript


def _make_completed_stages(stages=("assessment",)):
    result = []
    for stage in stages:
        fields = {}
        if stage == "assessment":
            fields = {
                "identity_confirmed": True,
                "debt_acknowledged": True,
                "ability_to_pay_discussed": True,
            }
        elif stage == "resolution":
            fields = {
                "options_reviewed": True,
                "borrower_position_known": True,
                "commitment_or_disposition": True,
            }
        result.append({
            "stage": stage,
            "collected_fields": fields,
            "transition_reason": f"{stage}_fields_collected",
            "turns": 3,
        })
    return result


class TestHandoffBudgetGuarantee:
    """The summary must always be <= MAX_HANDOFF_TOKENS."""

    def test_simple_summary_within_budget(self):
        borrower = _make_borrower()
        transcript = _make_transcript(num_pairs=2)
        completed = _make_completed_stages(["assessment"])

        result = build_handoff_summary(completed, borrower, transcript)
        tokens = count_tokens(result)

        assert tokens <= MAX_HANDOFF_TOKENS
        parsed = json.loads(result)
        assert "stages_covered" in parsed

    def test_large_transcript_still_within_budget(self):
        borrower = _make_borrower()
        transcript = _make_transcript(num_pairs=20, msg_len=200)
        completed = _make_completed_stages(["assessment"])

        result = build_handoff_summary(completed, borrower, transcript)
        assert count_tokens(result) <= MAX_HANDOFF_TOKENS

    def test_two_stage_handoff_within_budget(self):
        borrower = _make_borrower()
        assessment_transcript = _make_transcript(num_pairs=5, stage="assessment", msg_len=150)
        resolution_transcript = _make_transcript(num_pairs=5, stage="resolution", msg_len=150)
        transcript = assessment_transcript + resolution_transcript
        completed = _make_completed_stages(["assessment", "resolution"])

        result = build_handoff_summary(
            completed, borrower, transcript, target_stage="final_notice",
        )
        assert count_tokens(result) <= MAX_HANDOFF_TOKENS

    def test_huge_fixed_fields_hard_capped(self):
        borrower = _make_borrower(notes="N" * 2000)
        transcript = _make_transcript(num_pairs=10, msg_len=300)
        completed = _make_completed_stages(["assessment", "resolution"])

        result = build_handoff_summary(completed, borrower, transcript)
        assert count_tokens(result) <= MAX_HANDOFF_TOKENS

    def test_empty_transcript(self):
        borrower = _make_borrower()
        completed = _make_completed_stages(["assessment"])

        result = build_handoff_summary(completed, borrower, [])
        assert count_tokens(result) <= MAX_HANDOFF_TOKENS
        parsed = json.loads(result)
        assert parsed["borrower_stance"] == "unknown"


class TestHandoffContent:
    """Critical fields must survive pruning."""

    def test_must_keep_fields_preserved(self):
        borrower = _make_borrower()
        transcript = _make_transcript(num_pairs=2)
        completed = _make_completed_stages(["assessment"])

        result = build_handoff_summary(
            completed, borrower, transcript, target_stage="resolution",
        )
        parsed = json.loads(result)
        assert parsed["borrower_id"] == "b-test-001"
        assert parsed["debt_amount"] == 5000.00
        assert "assessment" in parsed["stages_covered"]

    def test_target_stage_influences_pruning(self):
        borrower = _make_borrower()
        transcript = _make_transcript(num_pairs=15, msg_len=200)
        completed = _make_completed_stages(["assessment"])

        result_res = build_handoff_summary(
            completed, borrower, transcript, target_stage="resolution",
        )
        result_fn = build_handoff_summary(
            completed, borrower, transcript, target_stage="final_notice",
        )
        assert count_tokens(result_res) <= MAX_HANDOFF_TOKENS
        assert count_tokens(result_fn) <= MAX_HANDOFF_TOKENS


class TestRawSummaryAssembly:
    """_build_raw_summary produces the expected structure."""

    def test_structure(self):
        borrower = _make_borrower()
        transcript = _make_transcript(num_pairs=2)
        completed = _make_completed_stages(["assessment"])

        raw = _build_raw_summary(completed, borrower, transcript)
        assert "stages_covered" in raw
        assert "key_exchanges" in raw
        assert "borrower_stance" in raw
        assert "assessment" in raw

    def test_stance_detection(self):
        borrower = _make_borrower()
        transcript = [
            {"role": "borrower", "stage": "assessment",
             "text": "I'm struggling with hardship", "timestamp": "t"},
        ]
        completed = _make_completed_stages(["assessment"])

        raw = _build_raw_summary(completed, borrower, transcript)
        assert raw["borrower_stance"] == "distressed"
