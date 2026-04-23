"""Tests for compliance guardrails (problem_statement.md §9, rules 2-5, 7-8).

Covers:
  Rule 2 — No false threats (only allowed consequences).
  Rule 3 — Stop-contact detection and pipeline termination.
  Rule 4 — Offers within policy-defined ranges.
  Rule 5 — Hardship/crisis detection and flagging.
  Rule 7 — Professional language / abusive borrower handling.
  Rule 8 — Privacy / PII redaction.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.activities.agents import _run_stage_turn
from app.models.pipeline import ComplianceFlags
from app.services.compliance import (
    ABUSIVE_CLOSE_REPLY,
    ALLOWED_CONSEQUENCES,
    PII_REDACTION_MARKER,
    STOP_CONTACT_REPLY,
    allowed_consequences_directive,
    check_false_threats,
    detect_abusive,
    detect_hardship,
    detect_stop_contact,
    redact_pii,
)
from app.services.llm_types import LLMResult


# ---- Helpers ---------------------------------------------------------------


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
        return LLMResult(text=self._text, model="test-model", used_fallback=False)

    async def summarize(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 200,
    ) -> LLMResult:
        return LLMResult(text="summary", model="test-model", used_fallback=False)


def _base_payload(
    *,
    stage: str = "assessment",
    borrower_message: str = "I received a notice and need help.",
    turn_index: int = 1,
    compliance_flags: dict | None = None,
) -> dict:
    return {
        "borrower": {
            "borrower_id": "b-compliance-test",
            "account_reference": "acct-5678",
            "date_of_birth": "1982-06-20",
            "debt_amount": 2000.00,
            "currency": "USD",
            "days_past_due": 45,
            "borrower_message": "I received a notice and need help.",
            "notes": None,
        },
        "stage": stage,
        "borrower_message": borrower_message,
        "transcript": [],
        "collected_fields": {},
        "turn_index": turn_index,
        "completed_stages": [],
        "compliance_flags": compliance_flags or {},
    }


# ---- Rule 8: PII redaction ------------------------------------------------


class TestPIIRedaction:
    def test_redacts_ssn_dashes(self):
        text, redacted = redact_pii("My SSN is 123-45-6789")
        assert redacted
        assert "123-45-6789" not in text
        assert PII_REDACTION_MARKER in text

    def test_redacts_ssn_no_dashes(self):
        text, redacted = redact_pii("SSN: 123456789")
        assert redacted
        assert "123456789" not in text

    def test_redacts_long_account_number(self):
        text, redacted = redact_pii("account number 1234567890")
        assert redacted
        assert "1234567890" not in text

    def test_redacts_credit_card(self):
        text, redacted = redact_pii("card: 4111-1111-1111-1111")
        assert redacted
        assert "4111-1111-1111-1111" not in text

    def test_preserves_safe_text(self):
        original = "My last four are 1234 and DOB is 1990-01-01"
        text, redacted = redact_pii(original)
        assert not redacted
        assert text == original

    def test_partial_account_not_redacted(self):
        text, redacted = redact_pii("the last four digits are 5678")
        assert not redacted

    @pytest.mark.asyncio
    async def test_output_pii_redacted_in_stage_turn(self):
        reply_with_ssn = "Your SSN 123-45-6789 is on file."
        fake_client = _FakeClient(reply_with_ssn)
        payload = _base_payload(borrower_message="hello", turn_index=2)

        with patch("app.activities.agents._get_anthropic_client", return_value=fake_client):
            result = await _run_stage_turn(payload)

        assert "123-45-6789" not in result["assistant_reply"]
        assert result["metadata"].get("compliance_output_pii_redacted") is True


# ---- Rule 3: Stop-contact detection and pipeline halt ---------------------


class TestStopContact:
    @pytest.mark.parametrize("phrase", [
        "stop contacting me",
        "please do not contact me again",
        "cease contact immediately",
        "leave me alone please",
        "no more calls, stop calling",
    ])
    def test_detects_stop_contact(self, phrase: str):
        assert detect_stop_contact(phrase)

    def test_normal_message_not_stop_contact(self):
        assert not detect_stop_contact("I want to discuss my payment options")

    @pytest.mark.asyncio
    async def test_stop_contact_terminates_stage(self):
        payload = _base_payload(borrower_message="Stop contacting me immediately")

        with patch("app.activities.agents._get_anthropic_client"):
            result = await _run_stage_turn(payload)

        assert result["decision"] == "stop_contact_requested"
        assert result["stage_complete"] is True
        assert result["assistant_reply"] == STOP_CONTACT_REPLY
        assert result["compliance_flags"]["stop_contact_requested"] is True
        assert result["metadata"]["compliance_stop_contact"] is True

    @pytest.mark.asyncio
    async def test_stop_contact_sets_next_stage_none(self):
        payload = _base_payload(borrower_message="Do not contact me")

        with patch("app.activities.agents._get_anthropic_client"):
            result = await _run_stage_turn(payload)

        assert result["next_stage"] is None


# ---- Rule 5: Hardship/crisis detection ------------------------------------


class TestHardshipDetection:
    @pytest.mark.parametrize("phrase", [
        "I'm facing financial hardship",
        "I lost my job last month",
        "I cannot afford to pay anything",
        "medical emergency wiped out my savings",
        "I'm going through a crisis",
    ])
    def test_detects_hardship(self, phrase: str):
        assert detect_hardship(phrase)

    def test_normal_message_not_hardship(self):
        assert not detect_hardship("I'd like to set up a payment plan")

    @pytest.mark.asyncio
    async def test_hardship_sets_flag_but_continues(self):
        fake_client = _FakeClient("I understand. Let me tell you about our hardship program.")
        payload = _base_payload(borrower_message="I lost my job and cannot afford this")

        with patch("app.activities.agents._get_anthropic_client", return_value=fake_client):
            result = await _run_stage_turn(payload)

        assert result["compliance_flags"]["hardship_detected"] is True
        assert result["metadata"].get("compliance_hardship_detected") is True
        assert result["decision"] != "stop_contact_requested"

    @pytest.mark.asyncio
    async def test_hardship_directive_injected_into_prompt(self):
        captured_prompts = {}

        class _CapturingClient:
            async def generate(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> LLMResult:
                captured_prompts["user"] = user_prompt
                return LLMResult(text="Understood.", model="test", used_fallback=False)

        payload = _base_payload(borrower_message="I'm facing financial hardship")

        with patch("app.activities.agents._get_anthropic_client", return_value=_CapturingClient()):
            await _run_stage_turn(payload)

        assert "Hardship detected" in captured_prompts["user"]
        assert "Do not pressure" in captured_prompts["user"]


# ---- Rule 7: Abusive language detection and close -------------------------


class TestAbusiveLanguage:
    @pytest.mark.parametrize("phrase", [
        "fuck you and your company",
        "you piece of shit",
        "I'll kill you",
    ])
    def test_detects_abusive(self, phrase: str):
        assert detect_abusive(phrase)

    def test_normal_angry_not_abusive(self):
        assert not detect_abusive("I'm very frustrated with this process")

    @pytest.mark.asyncio
    async def test_abusive_terminates_stage(self):
        payload = _base_payload(borrower_message="Fuck this, go to hell")

        with patch("app.activities.agents._get_anthropic_client"):
            result = await _run_stage_turn(payload)

        assert result["decision"] == "abusive_borrower_close"
        assert result["stage_complete"] is True
        assert result["assistant_reply"] == ABUSIVE_CLOSE_REPLY
        assert result["compliance_flags"]["abusive_borrower"] is True


# ---- Rule 2: No false threats / allowed consequences ----------------------


class TestFalseThreats:
    def test_detects_arrest_threat(self):
        threats = check_false_threats("If you don't pay, you will be under arrest.")
        assert "arrest" in threats

    def test_detects_garnishment_threat(self):
        threats = check_false_threats("We will garnish your wages immediately.")
        assert "garnish your wages" in threats

    def test_detects_jail_threat(self):
        threats = check_false_threats("You could go to jail for this.")
        assert "jail" in threats

    def test_allowed_consequence_not_flagged(self):
        for consequence in ALLOWED_CONSEQUENCES:
            threats = check_false_threats(f"The next step is {consequence}.")
            assert not threats, f"'{consequence}' should not be flagged"

    def test_clean_text_no_threats(self):
        threats = check_false_threats(
            "Your account may be referred to a collections agency if payment is not received."
        )
        assert not threats

    def test_substring_false_positives_rejected(self):
        """Regression: substring matching used to flag benign text.

        Before the word-boundary fix, 'issue you' substring-matched
        'sue you', producing compliance false positives in the eval
        harness. These must all be clean with the new regex.
        """
        benign = [
            "a billing issue you noticed",
            "this is a sensitive issue you should address",
            "reassurance about your account",
        ]
        for text in benign:
            assert not check_false_threats(text), (
                f"unexpected false positive for {text!r}: "
                f"{check_false_threats(text)}"
            )

    def test_verb_stems_still_detected(self):
        """Regression: we must not lose detection of common inflections."""
        assert "arrest" in check_false_threats("You could be arrested for this.")
        assert "jail" in check_false_threats("You will be jailed if unpaid.")
        assert "sue you" in check_false_threats("We are suing you today.")

    def test_allowed_consequences_directive_lists_all(self):
        directive = allowed_consequences_directive()
        for consequence in ALLOWED_CONSEQUENCES:
            assert consequence in directive

    @pytest.mark.asyncio
    async def test_false_threats_logged_in_metadata(self):
        reply_with_threat = "Pay now or you will be under arrest immediately."
        fake_client = _FakeClient(reply_with_threat)
        payload = _base_payload(borrower_message="hello", turn_index=2)

        with patch("app.activities.agents._get_anthropic_client", return_value=fake_client):
            result = await _run_stage_turn(payload)

        assert "compliance_false_threats" in result["metadata"]
        assert "arrest" in result["metadata"]["compliance_false_threats"]


# ---- Policy directives injected into resolution/final_notice prompts ------


class TestPolicyDirectiveInjection:
    @pytest.mark.asyncio
    async def test_resolution_prompt_includes_policy_bounds(self):
        captured = {}

        class _CapturingClient:
            async def generate(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> LLMResult:
                captured["user"] = user_prompt
                return LLMResult(text="Here are your options.", model="test", used_fallback=False)

        payload = _base_payload(
            stage="resolution",
            borrower_message="What can I do?",
            turn_index=1,
        )

        with patch("app.activities.agents._get_anthropic_client", return_value=_CapturingClient()):
            await _run_stage_turn(payload)

        assert "Settlement offers must be between" in captured["user"]
        assert "consequences you may reference" in captured["user"]

    @pytest.mark.asyncio
    async def test_assessment_prompt_does_not_include_offer_bounds(self):
        captured = {}

        class _CapturingClient:
            async def generate(self, *, system_prompt: str, user_prompt: str, max_tokens: int = 300) -> LLMResult:
                captured["user"] = user_prompt
                return LLMResult(text="Let me verify.", model="test", used_fallback=False)

        payload = _base_payload(
            stage="assessment",
            borrower_message="I got a notice",
            turn_index=1,
        )

        with patch("app.activities.agents._get_anthropic_client", return_value=_CapturingClient()):
            await _run_stage_turn(payload)

        assert "Settlement offers must be between" not in captured["user"]


# ---- ComplianceFlags model tests ------------------------------------------


class TestComplianceFlagsModel:
    def test_defaults_all_false(self):
        flags = ComplianceFlags()
        assert not flags.stop_contact_requested
        assert not flags.hardship_detected
        assert not flags.abusive_borrower

    def test_any_terminal_stop_contact(self):
        flags = ComplianceFlags(stop_contact_requested=True)
        assert flags.any_terminal()

    def test_any_terminal_abusive(self):
        flags = ComplianceFlags(abusive_borrower=True)
        assert flags.any_terminal()

    def test_hardship_not_terminal(self):
        flags = ComplianceFlags(hardship_detected=True)
        assert not flags.any_terminal()

    def test_roundtrip_serialization(self):
        flags = ComplianceFlags(
            stop_contact_requested=True,
            hardship_detected=True,
            abusive_borrower=False,
        )
        data = flags.model_dump(mode="json")
        restored = ComplianceFlags.model_validate(data)
        assert restored == flags
