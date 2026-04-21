"""Tests for summarization policies, priority pruning, and overflow prompts."""

from __future__ import annotations

import json

import pytest

from app.services.summarization import (
    STAGE_POLICIES,
    build_overflow_prompt,
    get_policy_for_stage,
    prioritize_handoff_fields,
    prune_to_budget,
)
from app.services.token_budget import MAX_HANDOFF_TOKENS, count_tokens


class TestPolicies:
    """Stage policies exist and declare meaningful keep/remove signals."""

    def test_resolution_policy_exists(self):
        policy = get_policy_for_stage("resolution")
        assert policy is not None
        assert policy.stage == "assessment"
        assert len(policy.keep_signals) > 0
        assert len(policy.remove_signals) > 0

    def test_final_notice_policy_exists(self):
        policy = get_policy_for_stage("final_notice")
        assert policy is not None
        assert policy.stage == "resolution"
        assert len(policy.keep_signals) > 0

    def test_assessment_has_no_policy(self):
        assert get_policy_for_stage("assessment") is None

    def test_all_policies_have_system_instruction(self):
        for stage, policy in STAGE_POLICIES.items():
            assert len(policy.system_instruction) > 20, f"Policy for {stage} has no instruction"


class TestFieldPrioritization:
    def test_must_keep_includes_borrower_id(self):
        summary = {
            "stages_covered": ["assessment"],
            "borrower_id": "b-001",
            "debt_amount": 1000,
            "currency": "USD",
            "days_past_due": 30,
            "key_exchanges": [{"s": "assessment", "t": 1, "b": "hi", "a": "hello"}],
            "borrower_stance": "cooperative",
        }
        policy = get_policy_for_stage("resolution")
        prioritized = prioritize_handoff_fields(summary, policy)

        must_keep = {fp.key for fp in prioritized if fp.priority == "must_keep"}
        assert "borrower_id" in must_keep
        assert "debt_amount" in must_keep
        assert "stages_covered" in must_keep

    def test_key_exchanges_are_optional(self):
        summary = {
            "stages_covered": ["assessment"],
            "borrower_id": "b-001",
            "key_exchanges": [],
        }
        prioritized = prioritize_handoff_fields(summary, None)
        exchange_priority = next(fp for fp in prioritized if fp.key == "key_exchanges")
        assert exchange_priority.priority == "optional"


class TestPruneToBudget:
    def test_small_summary_unchanged(self):
        summary = {
            "stages_covered": ["assessment"],
            "borrower_id": "b-001",
            "debt_amount": 1000,
        }
        result = prune_to_budget(summary, None, max_tokens=500)
        assert json.loads(result) == summary

    def test_prune_drops_optional_first(self):
        summary = {
            "stages_covered": ["assessment"],
            "borrower_id": "b-001",
            "debt_amount": 1000,
            "currency": "USD",
            "days_past_due": 30,
            "key_exchanges": [
                {"s": "assessment", "t": i, "b": "B" * 100, "a": "A" * 100}
                for i in range(10)
            ],
            "borrower_stance": "cooperative",
        }
        result = prune_to_budget(summary, None, max_tokens=100)
        parsed = json.loads(result)
        assert "key_exchanges" not in parsed
        assert count_tokens(result) <= 100

    def test_hard_truncation_guarantees_budget(self):
        summary = {
            "stages_covered": ["assessment"],
            "borrower_id": "b-" + "x" * 500,
            "debt_amount": 1000,
        }
        result = prune_to_budget(summary, None, max_tokens=30)
        assert count_tokens(result) <= 30

    def test_policy_aware_pruning_keeps_fields(self):
        policy = get_policy_for_stage("resolution")
        summary = {
            "stages_covered": ["assessment"],
            "borrower_id": "b-001",
            "debt_amount": 5000,
            "currency": "USD",
            "days_past_due": 45,
            "assessment": {
                "identity_confirmed": True,
                "debt_acknowledged": True,
                "ability_to_pay_discussed": True,
                "turns": 3,
                "reason": "fields_collected",
            },
            "key_exchanges": [
                {"s": "assessment", "t": i, "b": "B" * 80, "a": "A" * 80}
                for i in range(8)
            ],
            "borrower_stance": "cooperative",
        }
        result = prune_to_budget(summary, policy, max_tokens=MAX_HANDOFF_TOKENS)
        parsed = json.loads(result)
        assert "borrower_id" in parsed
        assert "debt_amount" in parsed
        assert count_tokens(result) <= MAX_HANDOFF_TOKENS


class TestOverflowPrompt:
    def test_prompt_contains_keep_and_remove(self):
        policy = get_policy_for_stage("resolution")
        sys_prompt, usr_prompt = build_overflow_prompt(
            policy, "some content to compress", target_tokens=200,
        )
        assert "KEEP" in usr_prompt
        assert "REMOVE" in usr_prompt
        assert "200" in usr_prompt
        assert "JSON" in sys_prompt

    def test_prompt_includes_stage_instruction(self):
        policy = get_policy_for_stage("final_notice")
        sys_prompt, usr_prompt = build_overflow_prompt(
            policy, "content", target_tokens=300,
        )
        assert "resolution" in usr_prompt.lower() or "final" in usr_prompt.lower()
