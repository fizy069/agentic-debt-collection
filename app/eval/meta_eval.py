"""Meta-evaluation: score the evaluation judges themselves.

The Darwin Godel Machine requirement: the system must evaluate its own
evaluation framework and detect incorrect metrics, poor thresholds,
and compliance blind spots.

``MetaEvaluator`` compares each judge's scores against the oracle and
produces a ``MetaReport`` with per-judge accuracy, calibration, blind
spots, and a flag indicating whether the judge should be revised.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from app.eval.meta_oracle import MetaOracle, OracleResult, OracleVerdict
from app.eval.models import (
    ConversationRecord,
    ConversationScores,
    EvalRunResult,
    JudgeScore,
)

logger = logging.getLogger(__name__)

_MIN_SAMPLES = 5
_ACCURACY_THRESHOLD = 0.85
_BOOTSTRAP_CI_THRESHOLD = 0.90
_BOOTSTRAP_ITERATIONS = 1000
_VACUOUS_SCORE_THRESHOLD = 0.95
_BLIND_SPOT_RATE_THRESHOLD = 0.50


@dataclass
class JudgeFlaw:
    """A specific flaw detected in a judge."""

    flaw_type: str  # "low_accuracy", "blind_spot", "miscalibration"
    rule_id: str | None
    description: str
    evidence: dict = field(default_factory=dict)


@dataclass
class JudgeReport:
    """Per-judge meta-evaluation report."""

    judge_name: str
    section_key: str
    n_evaluated: int = 0
    oracle_accuracy: float = 1.0
    accuracy_ci_lower: float = 1.0
    accuracy_ci_upper: float = 1.0
    mean_absolute_error: float = 0.0
    calibration_error: float = 0.0
    flaws: list[JudgeFlaw] = field(default_factory=list)
    flagged_for_revision: bool = False


@dataclass
class MetaReport:
    """Aggregate meta-evaluation report across all judges."""

    judges: list[JudgeReport] = field(default_factory=list)
    total_oracle_comparisons: int = 0

    @property
    def any_flagged(self) -> bool:
        return any(j.flagged_for_revision for j in self.judges)

    @property
    def flagged_judges(self) -> list[JudgeReport]:
        return [j for j in self.judges if j.flagged_for_revision]

    def to_dict(self) -> dict:
        result: dict = {
            "total_oracle_comparisons": self.total_oracle_comparisons,
            "any_flagged": self.any_flagged,
            "judges": [],
        }
        for jr in self.judges:
            jd: dict = {
                "judge_name": jr.judge_name,
                "section_key": jr.section_key,
                "n_evaluated": jr.n_evaluated,
                "oracle_accuracy": round(jr.oracle_accuracy, 4),
                "accuracy_ci_lower": round(jr.accuracy_ci_lower, 4),
                "accuracy_ci_upper": round(jr.accuracy_ci_upper, 4),
                "mean_absolute_error": round(jr.mean_absolute_error, 4),
                "calibration_error": round(jr.calibration_error, 4),
                "flagged_for_revision": jr.flagged_for_revision,
                "flaws": [
                    {
                        "flaw_type": f.flaw_type,
                        "rule_id": f.rule_id,
                        "description": f.description,
                        "evidence": f.evidence,
                    }
                    for f in jr.flaws
                ],
            }
            result["judges"].append(jd)
        return result


class MetaEvaluator:
    """Scores eval judges against the oracle to detect flaws."""

    def __init__(
        self,
        oracle: MetaOracle,
        *,
        min_samples: int = _MIN_SAMPLES,
        accuracy_threshold: float = _ACCURACY_THRESHOLD,
        bootstrap_ci_threshold: float = _BOOTSTRAP_CI_THRESHOLD,
    ) -> None:
        self._oracle = oracle
        self._min_samples = min_samples
        self._accuracy_threshold = accuracy_threshold
        self._bootstrap_ci_threshold = bootstrap_ci_threshold

    def evaluate(self, run_result: EvalRunResult) -> MetaReport:
        """Run meta-evaluation on a completed eval run."""
        oracle_results = self._oracle.evaluate_batch(
            run_result.conversations, run_result.scores,
        )

        scores_by_id = {s.scenario_id: s for s in run_result.scores}
        report = MetaReport()

        report.judges.append(self._evaluate_judge(
            "compliance",
            "eval_compliance_judge",
            oracle_results,
            scores_by_id,
            score_attr="compliance_scores",
        ))
        report.judges.append(self._evaluate_judge(
            "quality",
            "eval_quality_judge",
            oracle_results,
            scores_by_id,
            score_attr="quality_scores",
        ))
        report.judges.append(self._evaluate_judge(
            "handoff",
            "eval_handoff_judge",
            oracle_results,
            scores_by_id,
            score_attr="handoff_scores",
        ))

        report.total_oracle_comparisons = sum(j.n_evaluated for j in report.judges)

        for jr in report.judges:
            logger.info(
                "meta_eval  judge=%s  accuracy=%.3f  ci=[%.3f, %.3f]  "
                "n=%d  flaws=%d  flagged=%s",
                jr.judge_name, jr.oracle_accuracy,
                jr.accuracy_ci_lower, jr.accuracy_ci_upper,
                jr.n_evaluated, len(jr.flaws), jr.flagged_for_revision,
            )

        return report

    def _evaluate_judge(
        self,
        judge_name: str,
        section_key: str,
        oracle_results: list[OracleResult],
        scores_by_id: dict[str, ConversationScores],
        score_attr: str,
    ) -> JudgeReport:
        jr = JudgeReport(judge_name=judge_name, section_key=section_key)

        errors: list[float] = []
        rule_errors: dict[str, list[float]] = {}
        blind_spot_candidates: dict[str, dict] = {}

        for oracle_res in oracle_results:
            verdicts: list[OracleVerdict] = getattr(oracle_res, judge_name, [])
            conv_scores = scores_by_id.get(oracle_res.scenario_id)
            if conv_scores is None:
                continue

            judge_scores: list[JudgeScore] = getattr(conv_scores, score_attr, [])
            js_by_rule = {s.rule_id: s for s in judge_scores}

            for verdict in verdicts:
                if not verdict.definitive:
                    continue
                if verdict.expected_score < 0:
                    continue

                js = js_by_rule.get(verdict.rule_id)
                if js is None:
                    continue

                error = abs(js.score - verdict.expected_score)
                errors.append(error)
                rule_errors.setdefault(verdict.rule_id, []).append(error)

            self._check_blind_spots(
                verdicts, judge_scores, oracle_res.scenario_id, blind_spot_candidates,
            )

        jr.n_evaluated = len(errors)

        if jr.n_evaluated < self._min_samples:
            jr.oracle_accuracy = 1.0
            jr.accuracy_ci_lower = 0.0
            jr.accuracy_ci_upper = 1.0
            return jr

        jr.mean_absolute_error = sum(errors) / len(errors)
        jr.oracle_accuracy = 1.0 - jr.mean_absolute_error

        accuracies = [1.0 - e for e in errors]
        ci_lo, ci_hi = self._bootstrap_ci(accuracies)
        jr.accuracy_ci_lower = ci_lo
        jr.accuracy_ci_upper = ci_hi

        if jr.oracle_accuracy < self._accuracy_threshold:
            jr.flaws.append(JudgeFlaw(
                flaw_type="low_accuracy",
                rule_id=None,
                description=(
                    f"Oracle accuracy {jr.oracle_accuracy:.3f} is below "
                    f"threshold {self._accuracy_threshold}"
                ),
                evidence={
                    "accuracy": jr.oracle_accuracy,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                    "n": jr.n_evaluated,
                },
            ))

        for rule_id, errs in rule_errors.items():
            rule_acc = 1.0 - (sum(errs) / len(errs))
            if rule_acc < self._accuracy_threshold and len(errs) >= 3:
                jr.flaws.append(JudgeFlaw(
                    flaw_type="low_accuracy",
                    rule_id=rule_id,
                    description=(
                        f"Rule {rule_id} accuracy {rule_acc:.3f} below threshold"
                    ),
                    evidence={"rule_accuracy": rule_acc, "n": len(errs)},
                ))

        for rule_id, info in blind_spot_candidates.items():
            total = info["total"]
            vacuous_count = info["vacuous"]
            if total >= 3 and vacuous_count / total > _BLIND_SPOT_RATE_THRESHOLD:
                jr.flaws.append(JudgeFlaw(
                    flaw_type="blind_spot",
                    rule_id=rule_id,
                    description=(
                        f"Rule {rule_id}: {vacuous_count}/{total} scored vacuously "
                        f"despite persona-implied triggering"
                    ),
                    evidence={
                        "vacuous_rate": vacuous_count / total,
                        "scenarios": info.get("scenarios", []),
                    },
                ))

        should_flag = (
            jr.oracle_accuracy < self._accuracy_threshold
            and ci_hi < self._bootstrap_ci_threshold
        ) or len(jr.flaws) > 0

        jr.flagged_for_revision = should_flag
        return jr

    def _check_blind_spots(
        self,
        verdicts: list[OracleVerdict],
        judge_scores: list[JudgeScore],
        scenario_id: str,
        candidates: dict[str, dict],
    ) -> None:
        """Detect rules the judge marks vacuous/perfect when the oracle
        indicates the triggering condition should be present."""
        js_by_rule = {s.rule_id: s for s in judge_scores}

        for v in verdicts:
            if v.source != "persona":
                continue

            js = js_by_rule.get(v.rule_id)
            if js is None:
                continue

            info = candidates.setdefault(v.rule_id, {
                "total": 0, "vacuous": 0, "scenarios": [],
            })
            info["total"] += 1

            is_vacuous = (
                js.score >= _VACUOUS_SCORE_THRESHOLD
                and "not_triggered" in (js.explanation or "").lower()
            )
            if is_vacuous:
                info["vacuous"] += 1
                info["scenarios"].append(scenario_id)

    @staticmethod
    def _bootstrap_ci(
        values: Sequence[float],
        n_iter: int = _BOOTSTRAP_ITERATIONS,
        ci: float = 0.95,
    ) -> tuple[float, float]:
        """Percentile bootstrap confidence interval for the mean."""
        n = len(values)
        if n < 2:
            return (0.0, 1.0)

        rng = random.Random(42)
        means: list[float] = []
        for _ in range(n_iter):
            sample = [rng.choice(values) for _ in range(n)]
            means.append(sum(sample) / n)

        means.sort()
        alpha = (1.0 - ci) / 2.0
        lo_idx = max(0, int(math.floor(alpha * n_iter)))
        hi_idx = min(n_iter - 1, int(math.ceil((1.0 - alpha) * n_iter)) - 1)
        return (means[lo_idx], means[hi_idx])
