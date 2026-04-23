"""A/B harness for judge prompt candidates.

Compares a baseline vs candidate judge prompt by measuring oracle-agreement
accuracy instead of composite score.  Reuses the same statistical tests as
the agent-prompt :class:`ABHarness` but with a different primary metric.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from app.eval.meta_eval import MetaEvaluator, MetaReport
from app.eval.meta_oracle import MetaOracle, OracleVerdict
from app.eval.models import ConversationScores, EvalRunResult, JudgeScore
from app.eval.stats import compute_effect_size, compute_summary, is_significant_improvement

if TYPE_CHECKING:
    from app.eval.audit_trail import AuditTrail
    from app.eval.cost_ledger import CostLedger

logger = logging.getLogger(__name__)


def _find_judge(report: MetaReport, judge_name: str) -> "JudgeReport | None":
    """Look up a judge report by name."""
    from app.eval.meta_eval import JudgeReport as _JR

    for jr in report.judges:
        if jr.judge_name == judge_name:
            return jr
    return None


_CRITICAL_COMPLIANCE_RULES = ("1", "2", "3", "6", "8")
_DEFAULT_ALPHA = 0.05
_DEFAULT_MIN_EFFECT = 0.2


@dataclass
class MetaABResult:
    """Outcome of comparing baseline vs candidate judge prompt."""

    judge_name: str
    section_key: str
    baseline_accuracy: float = 0.0
    candidate_accuracy: float = 0.0
    p_value: float = 1.0
    cohen_d: float = 0.0
    significant: bool = False
    critical_rule_regression: bool = False
    adopt: bool = False
    reason: str = ""
    baseline_meta_report: MetaReport | None = None
    candidate_meta_report: MetaReport | None = None


class MetaABHarness:
    """Compare baseline vs candidate eval judge prompts on oracle accuracy."""

    def __init__(
        self,
        oracle: MetaOracle,
        *,
        alpha: float = _DEFAULT_ALPHA,
        min_effect: float = _DEFAULT_MIN_EFFECT,
        ledger: "CostLedger | None" = None,
        audit: "AuditTrail | None" = None,
    ) -> None:
        self._oracle = oracle
        self._alpha = alpha
        self._min_effect = min_effect
        self._ledger = ledger
        self._audit = audit

    def compare(
        self,
        judge_name: str,
        section_key: str,
        baseline_result: EvalRunResult,
        candidate_result: EvalRunResult,
    ) -> MetaABResult:
        """Compare oracle accuracy of baseline vs candidate judge."""
        baseline_meta = MetaEvaluator(self._oracle).evaluate(baseline_result)
        candidate_meta = MetaEvaluator(self._oracle).evaluate(candidate_result)

        baseline_jr = _find_judge(baseline_meta, judge_name)
        candidate_jr = _find_judge(candidate_meta, judge_name)

        if baseline_jr is None or candidate_jr is None:
            return MetaABResult(
                judge_name=judge_name,
                section_key=section_key,
                reason="Judge not found in meta report",
            )

        base_errors = self._per_scenario_errors(
            baseline_result, judge_name, section_key,
        )
        cand_errors = self._per_scenario_errors(
            candidate_result, judge_name, section_key,
        )

        base_accuracies = [1.0 - e for e in base_errors]
        cand_accuracies = [1.0 - e for e in cand_errors]

        significant, p_value = is_significant_improvement(
            base_accuracies, cand_accuracies,
        )
        effect = compute_effect_size(base_accuracies, cand_accuracies)

        critical_regression = self._check_critical_rule_regression(
            baseline_result, candidate_result, judge_name,
        )

        passes_stats = significant and effect >= self._min_effect
        adopt = passes_stats and not critical_regression

        if adopt:
            reason = (
                f"Adopted: p={p_value:.4f} d={effect:.3f} "
                f"baseline_acc={baseline_jr.oracle_accuracy:.3f} "
                f"candidate_acc={candidate_jr.oracle_accuracy:.3f}"
            )
        else:
            parts: list[str] = []
            if not significant:
                parts.append(f"not significant (p={p_value:.4f})")
            if effect < self._min_effect:
                parts.append(f"effect too small (d={effect:.3f})")
            if critical_regression:
                parts.append("critical compliance rule accuracy regressed")
            reason = "Rejected: " + "; ".join(parts) if parts else "Rejected"

        logger.info("meta_ab_result  judge=%s  adopt=%s  reason=%s", judge_name, adopt, reason)

        return MetaABResult(
            judge_name=judge_name,
            section_key=section_key,
            baseline_accuracy=baseline_jr.oracle_accuracy,
            candidate_accuracy=candidate_jr.oracle_accuracy,
            p_value=p_value,
            cohen_d=effect,
            significant=significant,
            critical_rule_regression=critical_regression,
            adopt=adopt,
            reason=reason,
            baseline_meta_report=baseline_meta,
            candidate_meta_report=candidate_meta,
        )

    def _per_scenario_errors(
        self,
        run_result: EvalRunResult,
        judge_name: str,
        section_key: str,
    ) -> list[float]:
        """Per-scenario mean absolute error vs oracle."""
        oracle_results = self._oracle.evaluate_batch(
            run_result.conversations, run_result.scores,
        )
        scores_by_id = {s.scenario_id: s for s in run_result.scores}

        score_attr_map = {
            "compliance": "compliance_scores",
            "quality": "quality_scores",
            "handoff": "handoff_scores",
        }
        score_attr = score_attr_map.get(judge_name, "compliance_scores")
        errors: list[float] = []

        for oracle_res in oracle_results:
            verdicts: list[OracleVerdict] = getattr(oracle_res, judge_name, [])
            conv_scores = scores_by_id.get(oracle_res.scenario_id)
            if conv_scores is None:
                continue

            judge_scores: list[JudgeScore] = getattr(conv_scores, score_attr, [])
            js_by_rule = {s.rule_id: s for s in judge_scores}

            scenario_errors: list[float] = []
            for v in verdicts:
                if not v.definitive or v.expected_score < 0:
                    continue
                js = js_by_rule.get(v.rule_id)
                if js is None:
                    continue
                scenario_errors.append(abs(js.score - v.expected_score))

            if scenario_errors:
                errors.append(sum(scenario_errors) / len(scenario_errors))

        return errors

    def _check_critical_rule_regression(
        self,
        baseline_result: EvalRunResult,
        candidate_result: EvalRunResult,
        judge_name: str,
    ) -> bool:
        """Candidate must not score worse on critical compliance rules."""
        if judge_name != "compliance":
            return False

        for rule_id in _CRITICAL_COMPLIANCE_RULES:
            base_acc = self._rule_accuracy(baseline_result, rule_id)
            cand_acc = self._rule_accuracy(candidate_result, rule_id)
            if cand_acc < base_acc - 0.01:
                logger.warning(
                    "meta_ab  critical rule %s regressed: %.3f -> %.3f",
                    rule_id, base_acc, cand_acc,
                )
                return True
        return False

    def _rule_accuracy(self, run_result: EvalRunResult, rule_id: str) -> float:
        """Oracle accuracy for a single compliance rule."""
        oracle_results = self._oracle.evaluate_batch(
            run_result.conversations, run_result.scores,
        )
        scores_by_id = {s.scenario_id: s for s in run_result.scores}
        correct = 0
        total = 0

        for oracle_res in oracle_results:
            for v in oracle_res.compliance:
                if v.rule_id != rule_id or not v.definitive or v.expected_score < 0:
                    continue
                conv_scores = scores_by_id.get(oracle_res.scenario_id)
                if conv_scores is None:
                    continue
                for js in conv_scores.compliance_scores:
                    if js.rule_id == rule_id:
                        total += 1
                        if abs(js.score - v.expected_score) < 0.5:
                            correct += 1
                        break

        return correct / total if total > 0 else 1.0
