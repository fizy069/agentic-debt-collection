"""A/B evaluation harness for prompt candidate comparison.

Runs the same seeded scenarios against two prompt registries (baseline vs
candidate), applies statistical tests, and enforces a per-compliance-rule
non-regression gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

from app.eval.harness import EvalHarness
from app.eval.models import ConversationScores, EvalConfig, EvalRunResult
from app.eval.stats import compute_effect_size, compute_summary, is_significant_improvement

if TYPE_CHECKING:
    from app.eval.audit_trail import AuditTrail
    from app.eval.cost_ledger import CostLedger

logger = logging.getLogger(__name__)

_COMPLIANCE_RULE_IDS = ("1", "2", "3", "5", "6", "7", "8")

_DEFAULT_ALPHA = 0.05
_DEFAULT_MIN_EFFECT = 0.3
_DEFAULT_COMPLIANCE_TOLERANCE = 0.0


@dataclass
class RuleComparison:
    rule_id: str
    baseline_pass_rate: float
    candidate_pass_rate: float
    delta: float
    regressed: bool


@dataclass
class ABResult:
    """Full comparison outcome."""
    baseline_composite_mean: float
    candidate_composite_mean: float
    p_value: float
    cohen_d: float
    significant: bool
    compliance_regressions: list[RuleComparison] = field(default_factory=list)
    adopt: bool = False
    reason: str = ""

    baseline_result: EvalRunResult | None = None
    candidate_result: EvalRunResult | None = None


class ABHarness:
    """Compare baseline vs candidate prompt registries on identical scenarios."""

    def __init__(
        self,
        config: EvalConfig,
        ledger: CostLedger | None = None,
        audit: AuditTrail | None = None,
        alpha: float = _DEFAULT_ALPHA,
        min_effect: float = _DEFAULT_MIN_EFFECT,
        compliance_tolerance: float = _DEFAULT_COMPLIANCE_TOLERANCE,
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._audit = audit
        self._alpha = alpha
        self._min_effect = min_effect
        self._compliance_tol = compliance_tolerance

    async def compare(
        self,
        baseline_result: EvalRunResult,
        candidate_result: EvalRunResult,
    ) -> ABResult:
        """Run statistical comparison and compliance-regression checks."""
        base_composites = [s.composite_score for s in baseline_result.scores]
        cand_composites = [s.composite_score for s in candidate_result.scores]

        base_summary = compute_summary(base_composites)
        cand_summary = compute_summary(cand_composites)

        significant, p_value = is_significant_improvement(base_composites, cand_composites)
        effect = compute_effect_size(base_composites, cand_composites)

        regressions = self._check_compliance_regression(
            baseline_result.scores, candidate_result.scores,
        )

        has_regression = any(r.regressed for r in regressions)
        passes_stats = significant and effect >= self._min_effect
        adopt = passes_stats and not has_regression

        if adopt:
            reason = (
                f"Adopted: p={p_value:.4f} d={effect:.3f} "
                f"baseline_mean={base_summary['mean']:.3f} "
                f"candidate_mean={cand_summary['mean']:.3f}"
            )
        else:
            parts: list[str] = []
            if not significant:
                parts.append(f"not significant (p={p_value:.4f})")
            if effect < self._min_effect:
                parts.append(f"effect too small (d={effect:.3f})")
            if has_regression:
                reg_rules = [r.rule_id for r in regressions if r.regressed]
                parts.append(f"compliance regression on rules {reg_rules}")
            reason = "Rejected: " + "; ".join(parts)

        logger.info("ab_harness_result  adopt=%s  reason=%s", adopt, reason)

        return ABResult(
            baseline_composite_mean=base_summary["mean"],
            candidate_composite_mean=cand_summary["mean"],
            p_value=p_value,
            cohen_d=effect,
            significant=significant,
            compliance_regressions=regressions,
            adopt=adopt,
            reason=reason,
            baseline_result=baseline_result,
            candidate_result=candidate_result,
        )

    def _check_compliance_regression(
        self,
        baseline_scores: Sequence[ConversationScores],
        candidate_scores: Sequence[ConversationScores],
    ) -> list[RuleComparison]:
        """Per-rule non-regression: candidate must be >= baseline - tolerance."""
        comparisons: list[RuleComparison] = []

        for rule_id in _COMPLIANCE_RULE_IDS:
            base_rate = _compliance_pass_rate(baseline_scores, rule_id)
            cand_rate = _compliance_pass_rate(candidate_scores, rule_id)
            delta = cand_rate - base_rate
            regressed = cand_rate < base_rate - self._compliance_tol

            comparisons.append(RuleComparison(
                rule_id=rule_id,
                baseline_pass_rate=base_rate,
                candidate_pass_rate=cand_rate,
                delta=delta,
                regressed=regressed,
            ))

        return comparisons


def _compliance_pass_rate(
    scores: Sequence[ConversationScores],
    rule_id: str,
    threshold: float = 0.9,
) -> float:
    """Fraction of conversations where the given rule scored >= threshold."""
    hits = 0
    total = 0
    for cs in scores:
        for js in cs.compliance_scores:
            if js.rule_id == rule_id:
                total += 1
                if js.score >= threshold:
                    hits += 1
                break
    return hits / total if total > 0 else 1.0
