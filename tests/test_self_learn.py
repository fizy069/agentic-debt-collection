"""Tests for the self-learning loop components:
cost ledger, audit trail, failure miner, proposer guard, and A/B decision gate.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.eval.ab_harness import ABHarness, ABResult, _compliance_pass_rate
from app.eval.audit_trail import AuditTrail
from app.eval.cost_ledger import BudgetExceededError, CostLedger
from app.eval.failure_miner import FailureDigest, mine_failures
from app.eval.models import (
    BorrowerPersona,
    ConversationRecord,
    ConversationScores,
    CostReport,
    EvalConfig,
    EvalRunResult,
    JudgeScore,
    MetricResult,
    PersonaType,
    Scenario,
    StageRecord,
)
from app.eval.proposer import PromptProposer, ProposalRejectedError
from app.services.prompt_registry import PromptRegistry, reset_prompt_registry


# ======================================================================
# Cost Ledger
# ======================================================================


class TestCostLedger:
    def test_record_accumulates_cost(self, tmp_path: Path):
        ledger = CostLedger(
            run_id="test1",
            budget_usd=10.0,
            base_dir=str(tmp_path),
        )
        ledger.record(role="sim", model="claude-haiku-4-5", input_tokens=1000, output_tokens=500)
        assert ledger.total_calls == 1
        assert ledger.total_input_tokens == 1000
        assert ledger.total_output_tokens == 500
        assert ledger.total_cost_usd > 0

    def test_cost_calculation_haiku(self, tmp_path: Path):
        ledger = CostLedger(run_id="test2", budget_usd=10.0, base_dir=str(tmp_path))
        ledger.record(role="sim", model="claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
        assert abs(ledger.total_cost_usd - 0.80) < 0.001

    def test_cost_calculation_output_tokens(self, tmp_path: Path):
        ledger = CostLedger(run_id="test3", budget_usd=10.0, base_dir=str(tmp_path))
        ledger.record(role="judge", model="claude-haiku-4-5", input_tokens=0, output_tokens=1_000_000)
        assert abs(ledger.total_cost_usd - 4.00) < 0.001

    def test_budget_enforcement(self, tmp_path: Path):
        ledger = CostLedger(run_id="test4", budget_usd=0.001, base_dir=str(tmp_path))
        ledger.record(role="sim", model="claude-haiku-4-5", input_tokens=10000, output_tokens=5000)
        with pytest.raises(BudgetExceededError):
            ledger.check_budget_or_raise()

    def test_under_budget_passes(self, tmp_path: Path):
        ledger = CostLedger(run_id="test5", budget_usd=100.0, base_dir=str(tmp_path))
        ledger.record(role="sim", model="stub", input_tokens=100, output_tokens=50)
        ledger.check_budget_or_raise()

    def test_remaining_budget(self, tmp_path: Path):
        ledger = CostLedger(run_id="test6", budget_usd=10.0, base_dir=str(tmp_path))
        assert ledger.remaining_budget == 10.0
        ledger.record(role="sim", model="claude-haiku-4-5", input_tokens=1_000_000, output_tokens=0)
        assert abs(ledger.remaining_budget - 9.20) < 0.001

    def test_cost_by_role(self, tmp_path: Path):
        ledger = CostLedger(run_id="test7", budget_usd=100.0, base_dir=str(tmp_path))
        ledger.record(role="sim", model="stub", input_tokens=100, output_tokens=50)
        ledger.record(role="judge", model="stub", input_tokens=200, output_tokens=100)
        by_role = ledger.cost_by_role()
        assert "sim" in by_role
        assert "judge" in by_role

    def test_jsonl_persistence(self, tmp_path: Path):
        ledger = CostLedger(run_id="persist_test", budget_usd=10.0, base_dir=str(tmp_path))
        ledger.record(role="sim", model="claude-haiku-4-5", input_tokens=500, output_tokens=100)
        ledger.record(role="judge", model="claude-haiku-4-5", input_tokens=800, output_tokens=200)

        log_path = tmp_path / "persist_test.jsonl"
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        entry = json.loads(lines[0])
        assert entry["role"] == "sim"
        assert entry["input_tokens"] == 500

    def test_fallback_pricing(self, tmp_path: Path):
        ledger = CostLedger(run_id="fallback", budget_usd=100.0, base_dir=str(tmp_path))
        ledger.record(role="sim", model="unknown-model-xyz", input_tokens=1_000_000, output_tokens=0)
        assert ledger.total_cost_usd > 0


# ======================================================================
# Audit Trail
# ======================================================================


class TestAuditTrail:
    def test_append_and_list_events(self, tmp_path: Path):
        audit = AuditTrail(base_dir=str(tmp_path))
        audit.append_event("run_started", run_id="r1")
        audit.append_event("override", section_key="test:key", old_version="v1", new_version="v2")

        events = audit.list_events()
        assert len(events) == 2
        assert events[0]["event_type"] == "run_started"
        assert events[1]["section_key"] == "test:key"

    def test_filter_by_section(self, tmp_path: Path):
        audit = AuditTrail(base_dir=str(tmp_path))
        audit.append_event("override", section_key="key_a", new_version="v2")
        audit.append_event("override", section_key="key_b", new_version="v3")

        events = audit.list_events(section_key="key_a")
        assert len(events) == 1
        assert events[0]["new_version"] == "v2"

    def test_snapshot_and_load_section(self, tmp_path: Path):
        from app.models.prompt import PromptSection

        audit = AuditTrail(base_dir=str(tmp_path))
        section = PromptSection(
            name="test:section",
            content="Hello world",
            version="v2",
            stage="assessment",
            section_type="system",
        )
        audit.snapshot_section(section)

        loaded = audit.load_section("test:section", "v2")
        assert loaded is not None
        assert loaded["content"] == "Hello world"
        assert loaded["version"] == "v2"

    def test_load_missing_section_returns_none(self, tmp_path: Path):
        audit = AuditTrail(base_dir=str(tmp_path))
        assert audit.load_section("nonexistent", "v1") is None

    def test_latest_adopted_version(self, tmp_path: Path):
        audit = AuditTrail(base_dir=str(tmp_path))
        audit.append_event("candidate_adopted", section_key="s1", new_version="v2")
        audit.append_event("candidate_rejected", section_key="s1", new_version="v3")
        audit.append_event("candidate_adopted", section_key="s1", new_version="v4")

        assert audit.latest_adopted_version("s1") == "v4"

    def test_latest_adopted_version_none(self, tmp_path: Path):
        audit = AuditTrail(base_dir=str(tmp_path))
        assert audit.latest_adopted_version("s1") is None

    def test_save_run_artefact(self, tmp_path: Path):
        audit = AuditTrail(base_dir=str(tmp_path))
        path = audit.save_run_artefact("run123", "summary.json", {"foo": "bar"})
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["foo"] == "bar"


# ======================================================================
# Failure Miner
# ======================================================================


def _make_scenario() -> Scenario:
    return Scenario(
        scenario_id="test_sc",
        persona=BorrowerPersona(
            persona_type=PersonaType.COOPERATIVE,
            system_prompt="test",
        ),
        borrower_id="b1",
        account_reference="ref1",
        date_of_birth="1990-01-01",
        debt_amount=1000.0,
        days_past_due=30,
    )


def _make_scored(
    compliance_overrides: dict[str, float] | None = None,
    quality_overrides: dict[str, float] | None = None,
) -> ConversationScores:
    comp = compliance_overrides or {}
    qual = quality_overrides or {}
    compliance = [
        JudgeScore(rule_id=rid, rule_name=f"rule_{rid}", score=comp.get(rid, 1.0))
        for rid in ("1", "2", "3", "5", "6", "7", "8")
    ]
    quality = [
        JudgeScore(rule_id=rid, rule_name=rid, score=qual.get(rid, 1.0))
        for rid in ("task_completion", "tone_appropriateness", "conciseness", "summarization_quality")
    ]
    handoff = [
        JudgeScore(rule_id=rid, rule_name=rid, score=1.0)
        for rid in ("context_preservation", "summarization_fidelity", "seamless_experience")
    ]
    return ConversationScores(
        scenario_id="test_sc",
        compliance_scores=compliance,
        quality_scores=quality,
        handoff_scores=handoff,
        composite_score=0.5,
    )


class TestFailureMiner:
    def test_no_failures_returns_empty_digest(self):
        result = EvalRunResult(
            config=EvalConfig(),
            scores=[_make_scored()],
        )
        digest = mine_failures(result)
        assert digest.top_section is None
        assert len(digest.sections) == 0

    def test_compliance_failure_maps_to_section(self):
        result = EvalRunResult(
            config=EvalConfig(),
            scores=[_make_scored(compliance_overrides={"1": 0.3})],
        )
        digest = mine_failures(result)
        assert digest.top_section is not None
        assert "system_template:assessment" in digest.top_section

    def test_quality_failure_maps_to_section(self):
        result = EvalRunResult(
            config=EvalConfig(),
            scores=[_make_scored(quality_overrides={"tone_appropriateness": 0.2})],
        )
        digest = mine_failures(result)
        assert digest.top_section is not None
        assert "turn_directives" in digest.top_section

    def test_ranking_by_severity(self):
        result = EvalRunResult(
            config=EvalConfig(),
            scores=[
                _make_scored(
                    compliance_overrides={"1": 0.3},
                    quality_overrides={"tone_appropriateness": 0.2},
                ),
            ],
        )
        digest = mine_failures(result)
        assert len(digest.sections) >= 1
        top = digest.sections[0]
        assert top.weighted_count > 0


# ======================================================================
# Proposer Guard
# ======================================================================


class TestProposerGuard:
    def test_guard_passes_when_phrases_preserved(self):
        original = (
            "You are an ai agent. No false threat allowed. "
            "Stop if asked. Settlement within policy. "
            "Hardship program. Logged and recorded. "
            "Professional language. Privacy respected."
        )
        proposed = (
            "You are an ai agent acting professionally. No false threat tolerated. "
            "Stop if explicitly asked. Settlement offers within policy. "
            "Hardship program offered. Logged and recorded always. "
            "Professional composure. Privacy of data respected."
        )
        PromptProposer._compliance_guard(original, proposed)

    def test_guard_rejects_missing_phrase(self):
        original = "You are an ai agent. No false threat. Stop if asked. Hardship program. Logged and recorded. Professional. Privacy."
        proposed = "You are an agent. No false threat. Stop if asked. Hardship program. Logged and recorded. Professional. Privacy."
        with pytest.raises(ProposalRejectedError, match="ai agent"):
            PromptProposer._compliance_guard(original, proposed)

    def test_guard_only_checks_phrases_present_in_original(self):
        original = "Simple prompt without compliance keywords."
        proposed = "Another simple prompt."
        PromptProposer._compliance_guard(original, proposed)


# ======================================================================
# A/B Decision Gate
# ======================================================================


def _make_eval_run(
    composite_scores: list[float],
    compliance_overrides: dict[str, float] | None = None,
) -> EvalRunResult:
    scores = []
    for i, comp in enumerate(composite_scores):
        cs = _make_scored(compliance_overrides=compliance_overrides)
        cs = cs.model_copy(update={"composite_score": comp, "scenario_id": f"sc_{i}"})
        scores.append(cs)
    return EvalRunResult(config=EvalConfig(), scores=scores)


class TestABDecisionGate:
    @pytest.mark.asyncio
    async def test_rejects_when_not_significant(self, tmp_path: Path):
        baseline = _make_eval_run([0.5, 0.5, 0.5, 0.5])
        candidate = _make_eval_run([0.51, 0.49, 0.50, 0.52])

        ab = ABHarness(config=EvalConfig())
        result = await ab.compare(baseline, candidate)
        assert not result.adopt
        assert "not significant" in result.reason or "effect too small" in result.reason

    @pytest.mark.asyncio
    async def test_adopts_clear_improvement(self, tmp_path: Path):
        baseline = _make_eval_run([0.25, 0.30, 0.28, 0.35, 0.27, 0.32, 0.29, 0.31])
        candidate = _make_eval_run([0.85, 0.90, 0.88, 0.92, 0.87, 0.91, 0.89, 0.93])

        ab = ABHarness(config=EvalConfig())
        result = await ab.compare(baseline, candidate)
        assert result.adopt
        assert result.significant
        assert result.cohen_d > 0.3

    @pytest.mark.asyncio
    async def test_rejects_compliance_regression(self, tmp_path: Path):
        baseline = _make_eval_run(
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
        )
        candidate = _make_eval_run(
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            compliance_overrides={"1": 0.0},
        )

        ab = ABHarness(config=EvalConfig())
        result = await ab.compare(baseline, candidate)
        assert not result.adopt
        assert "compliance regression" in result.reason

    def test_compliance_pass_rate_helper(self):
        scores = [_make_scored(compliance_overrides={"1": 1.0})]
        assert _compliance_pass_rate(scores, "1") == 1.0

        scores = [_make_scored(compliance_overrides={"1": 0.5})]
        assert _compliance_pass_rate(scores, "1") == 0.0


# ======================================================================
# Registry clone
# ======================================================================


class TestRegistryClone:
    def test_clone_is_independent(self):
        reset_prompt_registry()
        from app.services.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        cloned = registry.clone()

        section = registry.get_section("system_template:assessment")
        original_content = section.content

        cloned.override_section("system_template:assessment", "MODIFIED", "v_clone")
        assert registry.get_section("system_template:assessment").content == original_content
        assert cloned.get_section("system_template:assessment").content == "MODIFIED"

        reset_prompt_registry()


# ======================================================================
# LLMResult token fields
# ======================================================================


class TestLLMResultTokens:
    def test_default_tokens_zero(self):
        from app.services.llm_types import LLMResult
        result = LLMResult(text="hello", model="test")
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_tokens_round_trip(self):
        from app.services.llm_types import LLMResult
        result = LLMResult(
            text="hello", model="test",
            input_tokens=150, output_tokens=75,
        )
        assert result.input_tokens == 150
        assert result.output_tokens == 75


# ======================================================================
# CostReport model
# ======================================================================


class TestCostReportModel:
    def test_new_fields_present(self):
        report = CostReport(
            simulation_calls=5,
            evaluation_calls=10,
            total_calls=15,
            input_tokens=5000,
            output_tokens=2000,
            by_role={"sim": 0.01, "judge": 0.05},
            estimated_cost_usd=0.06,
        )
        data = report.model_dump()
        assert data["input_tokens"] == 5000
        assert data["output_tokens"] == 2000
        assert data["by_role"]["sim"] == 0.01

    def test_backward_compatible_defaults(self):
        report = CostReport()
        assert report.input_tokens == 0
        assert report.output_tokens == 0
        assert report.by_role == {}
