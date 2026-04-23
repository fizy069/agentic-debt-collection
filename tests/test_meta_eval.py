"""Tests for the Darwin Godel Machine meta-evaluation components:
meta oracle, meta evaluator, meta A/B harness, and registry integration.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.eval.meta_eval import JudgeFlaw, JudgeReport, MetaEvaluator, MetaReport
from app.eval.meta_oracle import MetaOracle, OracleVerdict
from app.eval.meta_ab import MetaABHarness, MetaABResult
from app.eval.models import (
    BorrowerPersona,
    ConversationRecord,
    ConversationScores,
    CostReport,
    EvalConfig,
    EvalRunResult,
    JudgeScore,
    PersonaType,
    Scenario,
    StageRecord,
    TurnRecord,
)
from app.services.prompt_registry import PromptRegistry, reset_prompt_registry


# ======================================================================
# Helpers
# ======================================================================


def _make_scenario(
    scenario_id: str = "test_sc",
    persona_type: PersonaType = PersonaType.COOPERATIVE,
) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        persona=BorrowerPersona(
            persona_type=persona_type,
            system_prompt="test",
        ),
        borrower_id="b1",
        account_reference="ref1",
        date_of_birth="1990-01-01",
        debt_amount=1000.0,
        days_past_due=30,
    )


def _make_record(
    scenario_id: str = "test_sc",
    persona_type: PersonaType = PersonaType.COOPERATIVE,
    agent_replies: list[str] | None = None,
    borrower_messages: list[str] | None = None,
) -> ConversationRecord:
    replies = agent_replies or [
        "I am an AI agent on behalf of the company. This conversation is logged and recorded.",
        "Thank you for confirming your identity.",
    ]
    messages = borrower_messages or [
        "Hi, I received a notice about my account.",
        "Yes, my date of birth is 1990-01-01.",
    ]
    turns = [
        TurnRecord(
            stage="assessment",
            turn_index=i,
            borrower_message=messages[min(i, len(messages) - 1)],
            agent_reply=replies[min(i, len(replies) - 1)],
            stage_complete=(i == len(replies) - 1),
        )
        for i in range(len(replies))
    ]
    return ConversationRecord(
        scenario=_make_scenario(scenario_id, persona_type),
        stages=[StageRecord(stage="assessment", turns=turns, completed=True)],
        final_outcome="completed",
        total_turns=len(turns),
        all_agent_replies=replies,
        all_borrower_messages=messages,
    )


def _make_scored(
    scenario_id: str = "test_sc",
    compliance_overrides: dict[str, float] | None = None,
    quality_overrides: dict[str, float] | None = None,
    handoff_overrides: dict[str, float] | None = None,
) -> ConversationScores:
    comp = compliance_overrides or {}
    qual = quality_overrides or {}
    hand = handoff_overrides or {}
    compliance = [
        JudgeScore(rule_id=rid, rule_name=f"rule_{rid}", score=comp.get(rid, 1.0))
        for rid in ("1", "2", "3", "5", "6", "7", "8")
    ]
    quality = [
        JudgeScore(rule_id=rid, rule_name=rid, score=qual.get(rid, 1.0))
        for rid in ("task_completion", "tone_appropriateness", "conciseness", "summarization_quality")
    ]
    handoff = [
        JudgeScore(rule_id=rid, rule_name=rid, score=hand.get(rid, 1.0))
        for rid in ("context_preservation", "summarization_fidelity", "seamless_experience")
    ]
    return ConversationScores(
        scenario_id=scenario_id,
        compliance_scores=compliance,
        quality_scores=quality,
        handoff_scores=handoff,
        composite_score=0.8,
    )


def _make_eval_run(
    records: list[ConversationRecord],
    scores: list[ConversationScores],
) -> EvalRunResult:
    return EvalRunResult(
        config=EvalConfig(),
        conversations=records,
        scores=scores,
    )


# ======================================================================
# MetaOracle
# ======================================================================


class TestMetaOracle:
    def test_compliance_oracle_definitive_rules(self):
        record = _make_record()
        scores = _make_scored()
        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        result = oracle.evaluate(record, scores)

        definitive_ids = {v.rule_id for v in result.compliance if v.definitive}
        assert "1" in definitive_ids
        assert "6" in definitive_ids
        assert "2" in definitive_ids

    def test_compliance_oracle_identity_disclosure_pass(self):
        record = _make_record(
            agent_replies=[
                "I am an AI agent on behalf of the company. This conversation is logged and recorded.",
            ],
        )
        scores = _make_scored()
        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        result = oracle.evaluate(record, scores)

        rule1 = [v for v in result.compliance if v.rule_id == "1"]
        assert len(rule1) == 1
        assert rule1[0].expected_score == 1.0

    def test_compliance_oracle_identity_disclosure_fail(self):
        record = _make_record(
            agent_replies=["Hello, how can I help you today?"],
        )
        scores = _make_scored()
        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        result = oracle.evaluate(record, scores)

        rule1 = [v for v in result.compliance if v.rule_id == "1"]
        assert len(rule1) == 1
        assert rule1[0].expected_score == 0.0

    def test_persona_implied_rule_distressed(self):
        record = _make_record(
            persona_type=PersonaType.DISTRESSED,
            borrower_messages=["I just lost my job and can't pay anything."],
        )
        scores = _make_scored()
        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        result = oracle.evaluate(record, scores)

        persona_verdicts = [v for v in result.compliance if v.source == "persona"]
        rule_ids = {v.rule_id for v in persona_verdicts}
        assert "5" in rule_ids

    def test_gold_labels_loaded(self, tmp_path: Path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        label_data = {
            "quality": [
                {"rule_id": "task_completion", "expected_score": 0.75, "reason": "test"},
            ],
            "handoff": [],
        }
        (labels_dir / "test_sc.json").write_text(json.dumps(label_data), encoding="utf-8")

        oracle = MetaOracle(gold_dir=tmp_path)
        assert oracle.has_gold_labels
        assert "test_sc" in oracle.gold_scenario_ids

    def test_gold_quality_verdicts(self, tmp_path: Path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        label_data = {
            "quality": [
                {"rule_id": "task_completion", "expected_score": 0.75, "reason": "test"},
            ],
            "handoff": [],
        }
        (labels_dir / "test_sc.json").write_text(json.dumps(label_data), encoding="utf-8")

        oracle = MetaOracle(gold_dir=tmp_path)
        record = _make_record(scenario_id="test_sc")
        scores = _make_scored(scenario_id="test_sc")
        result = oracle.evaluate(record, scores)

        assert len(result.quality) == 1
        assert result.quality[0].expected_score == 0.75
        assert result.quality[0].source == "gold"

    def test_no_gold_dir_returns_empty(self):
        oracle = MetaOracle(gold_dir=Path("/nonexistent/path"))
        record = _make_record()
        scores = _make_scored()
        result = oracle.evaluate(record, scores)
        assert len(result.quality) == 0
        assert len(result.handoff) == 0


# ======================================================================
# MetaEvaluator
# ======================================================================


class TestMetaEvaluator:
    def test_perfect_judge_not_flagged(self):
        records = [_make_record(scenario_id=f"sc_{i}") for i in range(10)]
        scores = [_make_scored(scenario_id=f"sc_{i}") for i in range(10)]
        run_result = _make_eval_run(records, scores)

        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        evaluator = MetaEvaluator(oracle, min_samples=3)
        report = evaluator.evaluate(run_result)

        compliance_jr = report.judges[0]
        assert compliance_jr.judge_name == "compliance"
        assert not compliance_jr.flagged_for_revision

    def test_inaccurate_judge_flagged(self):
        records = [
            _make_record(
                scenario_id=f"sc_{i}",
                agent_replies=["I am an AI agent on behalf of the company. "
                               "This conversation is logged and recorded."],
            )
            for i in range(10)
        ]
        scores = [
            _make_scored(
                scenario_id=f"sc_{i}",
                compliance_overrides={"1": 0.0, "6": 0.0},
            )
            for i in range(10)
        ]
        run_result = _make_eval_run(records, scores)

        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        evaluator = MetaEvaluator(oracle, min_samples=3, accuracy_threshold=0.85)
        report = evaluator.evaluate(run_result)

        compliance_jr = report.judges[0]
        assert compliance_jr.flagged_for_revision
        assert any(f.flaw_type == "low_accuracy" for f in compliance_jr.flaws)

    def test_too_few_samples_not_flagged(self):
        """With min_samples=20 and only 1 record (~6 definitive verdicts),
        the evaluator should not flag because there is not enough data."""
        records = [_make_record(scenario_id="sc_0")]
        scores = [_make_scored(scenario_id="sc_0", compliance_overrides={"1": 0.0})]
        run_result = _make_eval_run(records, scores)

        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        evaluator = MetaEvaluator(oracle, min_samples=20)
        report = evaluator.evaluate(run_result)

        compliance_jr = report.judges[0]
        assert not compliance_jr.flagged_for_revision

    def test_blind_spot_detection(self):
        records = []
        scores_list = []
        for i in range(6):
            records.append(_make_record(
                scenario_id=f"sc_{i}",
                persona_type=PersonaType.DISTRESSED,
                borrower_messages=["I lost my job and can't pay"],
                agent_replies=[
                    "I am an AI agent on behalf of the company. "
                    "This conversation is logged and recorded."
                ],
            ))
            scores_list.append(_make_scored(
                scenario_id=f"sc_{i}",
                compliance_overrides={
                    "5": 1.0,
                },
            ))
            scores_list[-1].compliance_scores = [
                s if s.rule_id != "5" else JudgeScore(
                    rule_id="5", rule_name="hardship_handling",
                    score=1.0, confidence=0.9, explanation="not_triggered",
                )
                for s in scores_list[-1].compliance_scores
            ]

        run_result = _make_eval_run(records, scores_list)
        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        evaluator = MetaEvaluator(oracle, min_samples=3)
        report = evaluator.evaluate(run_result)

        compliance_jr = report.judges[0]
        blind_spots = [f for f in compliance_jr.flaws if f.flaw_type == "blind_spot"]
        assert len(blind_spots) > 0
        assert any(f.rule_id == "5" for f in blind_spots)

    def test_meta_report_serialization(self):
        report = MetaReport(
            judges=[
                JudgeReport(
                    judge_name="compliance",
                    section_key="eval_compliance_judge",
                    n_evaluated=10,
                    oracle_accuracy=0.80,
                    flaws=[
                        JudgeFlaw(
                            flaw_type="low_accuracy",
                            rule_id=None,
                            description="Test flaw",
                        ),
                    ],
                    flagged_for_revision=True,
                ),
            ],
            total_oracle_comparisons=10,
        )
        d = report.to_dict()
        assert d["any_flagged"] is True
        assert d["judges"][0]["oracle_accuracy"] == 0.80
        assert len(d["judges"][0]["flaws"]) == 1


# ======================================================================
# MetaABHarness
# ======================================================================


class TestMetaABHarness:
    def test_rejects_when_no_improvement(self):
        records = [
            _make_record(scenario_id=f"sc_{i}")
            for i in range(10)
        ]
        scores = [_make_scored(scenario_id=f"sc_{i}") for i in range(10)]

        baseline = _make_eval_run(records, scores)
        candidate = _make_eval_run(records, scores)

        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        harness = MetaABHarness(oracle)
        result = harness.compare("compliance", "eval_compliance_judge", baseline, candidate)

        assert not result.adopt

    def test_adopts_clear_accuracy_improvement(self):
        records = [
            _make_record(
                scenario_id=f"sc_{i}",
                agent_replies=[
                    "I am an AI agent on behalf of the company. "
                    "This conversation is logged and recorded."
                ],
            )
            for i in range(10)
        ]
        baseline_scores = [
            _make_scored(
                scenario_id=f"sc_{i}",
                compliance_overrides={
                    "1": 0.1 * (i % 3),
                    "6": 0.15 * (i % 4),
                },
            )
            for i in range(10)
        ]
        candidate_scores = [
            _make_scored(
                scenario_id=f"sc_{i}",
                compliance_overrides={
                    "1": 0.9 + 0.01 * i,
                    "6": 0.95 + 0.005 * i,
                },
            )
            for i in range(10)
        ]

        baseline = _make_eval_run(records, baseline_scores)
        candidate = _make_eval_run(records, candidate_scores)

        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        harness = MetaABHarness(oracle, min_effect=0.1)
        result = harness.compare("compliance", "eval_compliance_judge", baseline, candidate)

        assert result.adopt
        assert result.candidate_accuracy > result.baseline_accuracy

    def test_rejects_critical_rule_regression(self):
        records = [
            _make_record(
                scenario_id=f"sc_{i}",
                agent_replies=[
                    "I am an AI agent on behalf of the company. "
                    "This conversation is logged and recorded."
                ],
            )
            for i in range(10)
        ]
        baseline_scores = [_make_scored(scenario_id=f"sc_{i}") for i in range(10)]
        candidate_scores = [
            _make_scored(
                scenario_id=f"sc_{i}",
                compliance_overrides={"1": 0.0},
            )
            for i in range(10)
        ]

        baseline = _make_eval_run(records, baseline_scores)
        candidate = _make_eval_run(records, candidate_scores)

        oracle = MetaOracle(gold_dir=Path(tempfile.mkdtemp()))
        harness = MetaABHarness(oracle)
        result = harness.compare("compliance", "eval_compliance_judge", baseline, candidate)

        assert not result.adopt
        assert result.critical_rule_regression


# ======================================================================
# Registry integration
# ======================================================================


class TestRegistryEvalJudgeKeys:
    def test_eval_judge_sections_exist(self):
        reset_prompt_registry()
        from app.services.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()

        for key in ("eval_compliance_judge", "eval_quality_judge", "eval_handoff_judge"):
            section = registry.get_section(key)
            assert section is not None
            assert section.version == "v1"
            assert len(section.content) > 50

        reset_prompt_registry()

    def test_eval_judge_sections_in_snapshot(self):
        reset_prompt_registry()
        from app.services.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        snap = registry.snapshot()

        assert "eval_compliance_judge" in snap
        assert "eval_quality_judge" in snap
        assert "eval_handoff_judge" in snap

        reset_prompt_registry()

    def test_eval_judge_override_and_rollback(self):
        reset_prompt_registry()
        from app.services.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()

        original = registry.get_section("eval_compliance_judge").content
        registry.override_section("eval_compliance_judge", "MODIFIED JUDGE", "v2")
        assert registry.get_section("eval_compliance_judge").content == "MODIFIED JUDGE"
        assert registry.get_section("eval_compliance_judge").version == "v2"

        registry.rollback("eval_compliance_judge", "v1")
        assert registry.get_section("eval_compliance_judge").content == original
        assert registry.get_section("eval_compliance_judge").version == "v1"

        reset_prompt_registry()

    def test_eval_judge_clone_independence(self):
        reset_prompt_registry()
        from app.services.prompt_registry import get_prompt_registry
        registry = get_prompt_registry()
        cloned = registry.clone()

        original_content = registry.get_section("eval_compliance_judge").content
        cloned.override_section("eval_compliance_judge", "CLONED CHANGE", "v_clone")

        assert registry.get_section("eval_compliance_judge").content == original_content
        assert cloned.get_section("eval_compliance_judge").content == "CLONED CHANGE"

        reset_prompt_registry()
