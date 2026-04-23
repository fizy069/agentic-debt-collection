"""Tests for Layer 2 compliance judge, vector store, and activity integration.

Covers:
  - Judge service: prompt construction, JSON parsing, env-disable, error fallback.
  - Vector store: upsert, similarity query, empty-collection handling.
  - Activity integration: judge metadata enrichment, vector metadata enrichment,
    judge skip when disabled, existing hard-stop behaviour preserved.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from app.activities.agents import _run_stage_turn
from app.services.compliance_judge import (
    JudgeFindings,
    JudgeViolation,
    _parse_judge_response,
    build_judge_prompt,
    findings_to_metadata,
    is_judge_enabled,
    run_judge,
)
from app.services.compliance_vector_store import (
    ViolationRecord,
    VectorSearchResult,
    get_client,
    query_similar_violations,
    reset_client,
    upsert_violations,
    vector_hits_to_metadata,
)
from app.services.llm_types import LLMResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal stub matching the AnthropicClient interface."""

    def __init__(self, text: str = "OK") -> None:
        self._text = text

    async def generate(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int = 300
    ) -> LLMResult:
        return LLMResult(text=self._text, model="test-model", used_fallback=False)

    async def summarize(
        self, *, system_prompt: str, user_prompt: str, max_tokens: int = 200
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
            "borrower_id": "b-judge-test",
            "account_reference": "acct-9999",
            "date_of_birth": "1975-02-28",
            "debt_amount": 3000.00,
            "currency": "USD",
            "days_past_due": 60,
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


# ====================================================================
# Judge Service — unit tests
# ====================================================================


class TestIsJudgeEnabled:
    def test_defaults_to_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("COMPLIANCE_JUDGE_ENABLED", None)
            assert not is_judge_enabled()

    def test_enabled_true(self):
        with patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "true"}):
            assert is_judge_enabled()

    def test_enabled_yes(self):
        with patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "yes"}):
            assert is_judge_enabled()

    def test_enabled_1(self):
        with patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "1"}):
            assert is_judge_enabled()

    def test_disabled_false(self):
        with patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "false"}):
            assert not is_judge_enabled()


class TestBuildJudgePrompt:
    def test_contains_stage_and_turn(self):
        prompt = build_judge_prompt(
            stage="resolution",
            turn_index=2,
            assistant_reply="Pay 60% now.",
            transcript_excerpt="borrower: what options?",
        )
        assert "Stage: resolution" in prompt
        assert "Turn: 2" in prompt
        assert "Pay 60% now." in prompt
        assert "what options?" in prompt


class TestParseJudgeResponse:
    def test_parses_clean_json(self):
        raw = json.dumps(
            {
                "violations": [
                    {
                        "rule": "2",
                        "label": "false_threat",
                        "confidence": 0.9,
                        "excerpt": "you will be arrested",
                    }
                ],
                "overall_risk": "high",
            }
        )
        findings = _parse_judge_response(raw)
        assert len(findings.violations) == 1
        assert findings.violations[0].rule == "2"
        assert findings.violations[0].label == "false_threat"
        assert findings.overall_risk == "high"
        assert findings.judge_error is None

    def test_parses_json_with_markdown_fences(self):
        raw = '```json\n{"violations": [], "overall_risk": "low"}\n```'
        findings = _parse_judge_response(raw)
        assert findings.violations == []
        assert findings.overall_risk == "low"

    def test_handles_invalid_json(self):
        findings = _parse_judge_response("not json at all")
        assert findings.judge_error is not None
        assert "json_parse_failed" in findings.judge_error

    def test_handles_empty_violations_list(self):
        raw = json.dumps({"violations": [], "overall_risk": "low"})
        findings = _parse_judge_response(raw)
        assert findings.violations == []
        assert findings.overall_risk == "low"

    def test_skips_malformed_violation_entries(self):
        raw = json.dumps(
            {
                "violations": [
                    {"rule": "4", "label": "offer_bounds", "confidence": 0.8, "excerpt": "ok"},
                    {"bad": "entry"},
                ],
                "overall_risk": "medium",
            }
        )
        findings = _parse_judge_response(raw)
        assert len(findings.violations) == 2


class TestFindingsToMetadata:
    def test_no_violations(self):
        findings = JudgeFindings(overall_risk="low")
        meta = findings_to_metadata(findings)
        assert meta == {"judge_risk": "low"}

    def test_with_violations(self):
        findings = JudgeFindings(
            violations=[
                JudgeViolation(rule="2", label="false_threat", confidence=0.95, excerpt="arrest")
            ],
            overall_risk="high",
        )
        meta = findings_to_metadata(findings)
        assert meta["judge_risk"] == "high"
        assert len(meta["judge_violations"]) == 1
        assert meta["judge_violations"][0]["rule"] == "2"

    def test_with_error(self):
        findings = JudgeFindings(judge_error="something broke")
        meta = findings_to_metadata(findings)
        assert meta == {"judge_error": "something broke"}


class TestRunJudge:
    @pytest.mark.asyncio
    async def test_returns_empty_when_disabled(self):
        with patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "false"}):
            findings = await run_judge(
                stage="assessment", turn_index=1,
                assistant_reply="hello", transcript_excerpt="none",
            )
        assert findings.violations == []
        assert findings.judge_error is None

    @pytest.mark.asyncio
    async def test_calls_llm_when_enabled(self):
        judge_response = json.dumps({
            "violations": [{"rule": "7", "label": "unprofessional", "confidence": 0.7, "excerpt": "rude"}],
            "overall_risk": "medium",
        })
        fake = _FakeClient(judge_response)

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "true"}),
            patch("app.services.compliance_judge._get_judge_client", return_value=fake),
        ):
            findings = await run_judge(
                stage="resolution", turn_index=2,
                assistant_reply="rude reply", transcript_excerpt="prior exchange",
            )

        assert len(findings.violations) == 1
        assert findings.violations[0].label == "unprofessional"

    @pytest.mark.asyncio
    async def test_swallows_exception(self):
        async def _explode(**kwargs):
            raise RuntimeError("boom")

        mock_client = AsyncMock()
        mock_client.generate = _explode

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "true"}),
            patch("app.services.compliance_judge._get_judge_client", return_value=mock_client),
        ):
            findings = await run_judge(
                stage="assessment", turn_index=1,
                assistant_reply="safe", transcript_excerpt="ok",
            )

        assert findings.judge_error is not None
        assert "judge_call_failed" in findings.judge_error


# ====================================================================
# Vector Store — unit tests
# ====================================================================


@pytest.fixture()
def _temp_vector_db(tmp_path):
    """Point vector store at a disposable temp directory."""
    db_path = str(tmp_path / "test_vectors")
    with patch.dict(os.environ, {"COMPLIANCE_VECTOR_DB_PATH": db_path}):
        reset_client()
        yield db_path
        reset_client()


class TestVectorUpsert:
    def test_upsert_returns_count(self, _temp_vector_db):
        count = upsert_violations(
            workflow_id="wf-1",
            stage="assessment",
            turn_index=1,
            violations=[
                {"rule": "2", "label": "false_threat", "confidence": 0.9, "excerpt": "arrest"},
            ],
        )
        assert count == 1

    def test_upsert_empty_list(self, _temp_vector_db):
        assert upsert_violations(workflow_id="wf-1", stage="assessment", turn_index=1, violations=[]) == 0

    def test_upsert_multiple(self, _temp_vector_db):
        violations = [
            {"rule": "2", "label": "false_threat", "confidence": 0.9, "excerpt": "arrest"},
            {"rule": "4", "label": "offer_bounds", "confidence": 0.8, "excerpt": "10%"},
        ]
        assert upsert_violations(workflow_id="wf-2", stage="resolution", turn_index=1, violations=violations) == 2


class TestVectorQuery:
    def test_query_empty_collection(self, _temp_vector_db):
        result = query_similar_violations("any text")
        assert result.records == []
        assert result.error is None

    def test_query_returns_results_after_upsert(self, _temp_vector_db):
        upsert_violations(
            workflow_id="wf-q",
            stage="resolution",
            turn_index=1,
            violations=[
                {"rule": "2", "label": "false_threat", "confidence": 0.95, "excerpt": "you will be arrested"},
            ],
        )
        result = query_similar_violations("arrested for debt", n_results=2)
        assert len(result.records) >= 1
        assert result.records[0].rule == "2"

    def test_query_filters_by_stage(self, _temp_vector_db):
        upsert_violations(
            workflow_id="wf-s",
            stage="assessment",
            turn_index=1,
            violations=[{"rule": "8", "label": "pii_leak", "confidence": 0.8, "excerpt": "SSN shown"}],
        )
        upsert_violations(
            workflow_id="wf-s",
            stage="resolution",
            turn_index=1,
            violations=[{"rule": "2", "label": "false_threat", "confidence": 0.9, "excerpt": "arrest"}],
        )
        result = query_similar_violations("SSN leak", stage="assessment")
        assert all(r.stage == "assessment" for r in result.records)


class TestVectorHitsToMetadata:
    def test_empty_results(self):
        assert vector_hits_to_metadata(VectorSearchResult()) == {}

    def test_with_error(self):
        meta = vector_hits_to_metadata(VectorSearchResult(error="db down"))
        assert meta == {"vector_lookup_error": "db down"}

    def test_with_records(self):
        records = [
            ViolationRecord(
                violation_id="v1", rule="2", label="false_threat",
                confidence=0.9, excerpt="arrest", stage="resolution", turn_index=1,
            )
        ]
        meta = vector_hits_to_metadata(VectorSearchResult(records=records))
        assert len(meta["vector_similar_violations"]) == 1


# ====================================================================
# Activity integration — judge + vector wired into _run_stage_turn
# ====================================================================


class TestActivityJudgeIntegration:
    @pytest.mark.asyncio
    async def test_judge_skipped_when_disabled(self):
        fake = _FakeClient("I can help you with your account.")
        payload = _base_payload(borrower_message="hello", turn_index=2)

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "false"}, clear=False),
            patch("app.activities.agents._get_anthropic_client", return_value=fake),
            patch("app.activities.agents.query_similar_violations", return_value=VectorSearchResult()),
        ):
            result = await _run_stage_turn(payload)

        assert "judge_violations" not in result["metadata"]
        assert "judge_error" not in result["metadata"]

    @pytest.mark.asyncio
    async def test_judge_findings_in_metadata_when_enabled(self):
        fake_main = _FakeClient("Pay up or face arrest.")
        judge_response = json.dumps({
            "violations": [{"rule": "2", "label": "false_threat", "confidence": 0.92, "excerpt": "face arrest"}],
            "overall_risk": "high",
        })
        fake_judge = _FakeClient(judge_response)

        payload = _base_payload(borrower_message="what happens if I don't pay?", turn_index=2)

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "true"}, clear=False),
            patch("app.activities.agents._get_anthropic_client", return_value=fake_main),
            patch("app.services.compliance_judge._get_judge_client", return_value=fake_judge),
            patch("app.activities.agents.query_similar_violations", return_value=VectorSearchResult()),
            patch("app.activities.agents.upsert_violations", return_value=1) as mock_upsert,
        ):
            result = await _run_stage_turn(payload)

        assert result["metadata"]["judge_risk"] == "high"
        assert len(result["metadata"]["judge_violations"]) == 1
        mock_upsert.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_hits_in_metadata(self):
        fake = _FakeClient("Let me review your options.")
        prior_records = [
            ViolationRecord(
                violation_id="old-1", rule="4", label="offer_bounds",
                confidence=0.85, excerpt="5% settlement", stage="resolution", turn_index=1,
            )
        ]
        payload = _base_payload(borrower_message="options please", turn_index=2)

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "false"}, clear=False),
            patch("app.activities.agents._get_anthropic_client", return_value=fake),
            patch(
                "app.activities.agents.query_similar_violations",
                return_value=VectorSearchResult(records=prior_records),
            ),
        ):
            result = await _run_stage_turn(payload)

        assert "vector_similar_violations" in result["metadata"]
        assert result["metadata"]["vector_similar_violations"][0]["rule"] == "4"

    @pytest.mark.asyncio
    async def test_hard_stop_unaffected_by_judge(self):
        """Stop-contact still terminates immediately; judge/vector never called."""
        payload = _base_payload(borrower_message="Stop contacting me immediately")

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "true"}, clear=False),
            patch("app.activities.agents._get_anthropic_client"),
            patch("app.activities.agents.run_judge") as mock_judge,
            patch("app.activities.agents.query_similar_violations") as mock_vector,
        ):
            result = await _run_stage_turn(payload)

        assert result["decision"] == "stop_contact_requested"
        assert result["compliance_flags"]["stop_contact_requested"] is True
        mock_judge.assert_not_called()
        mock_vector.assert_not_called()

    @pytest.mark.asyncio
    async def test_abusive_hard_stop_unaffected(self):
        payload = _base_payload(borrower_message="fuck you")

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "true"}, clear=False),
            patch("app.activities.agents._get_anthropic_client"),
            patch("app.activities.agents.run_judge") as mock_judge,
            patch("app.activities.agents.query_similar_violations") as mock_vector,
        ):
            result = await _run_stage_turn(payload)

        assert result["decision"] == "abusive_borrower_close"
        mock_judge.assert_not_called()
        mock_vector.assert_not_called()

    @pytest.mark.asyncio
    async def test_judge_error_does_not_block_pipeline(self):
        """If the judge call fails, the main reply is still returned."""
        fake_main = _FakeClient("Here are your options.")
        payload = _base_payload(borrower_message="help", turn_index=2)

        async def _boom(**kwargs):
            raise RuntimeError("judge down")

        mock_judge_client = AsyncMock()
        mock_judge_client.generate = _boom

        with (
            patch.dict(os.environ, {"COMPLIANCE_JUDGE_ENABLED": "true"}, clear=False),
            patch("app.activities.agents._get_anthropic_client", return_value=fake_main),
            patch("app.services.compliance_judge._get_judge_client", return_value=mock_judge_client),
            patch("app.activities.agents.query_similar_violations", return_value=VectorSearchResult()),
        ):
            result = await _run_stage_turn(payload)

        assert result["assistant_reply"] == "Here are your options."
        assert "judge_error" in result["metadata"]
