"""Metric definitions for the evaluation harness.

Each metric implements a ``compute`` method that takes scored conversations
and returns a ``MetricResult`` with value, sample size, std, and CI bounds.
"""

from __future__ import annotations

import math
from typing import Protocol, Sequence

from app.eval.models import ConversationScores, JudgeScore, MetricResult


class Metric(Protocol):
    """Protocol for all evaluation metrics."""

    name: str

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult: ...


def _ci_95(mean: float, std: float, n: int) -> tuple[float, float]:
    """Approximate 95% CI using z=1.96 for the mean."""
    if n < 2:
        return (mean, mean)
    se = std / math.sqrt(n)
    return (mean - 1.96 * se, mean + 1.96 * se)


def _scores_for_rule(
    scored: Sequence[ConversationScores],
    source: str,
    rule_id: str,
) -> list[float]:
    """Extract score values for a specific rule across conversations."""
    values: list[float] = []
    for cs in scored:
        pool = getattr(cs, source, [])
        for js in pool:
            if js.rule_id == rule_id:
                values.append(js.score)
                break
    return values


# ------------------------------------------------------------------
# Compliance metrics
# ------------------------------------------------------------------

class CompliancePassRate:
    """Percentage of conversations where all 8 compliance rules scored >= 0.9."""

    name = "compliance_pass_rate"
    _threshold = 0.9

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult:
        passes: list[float] = []
        for cs in scored:
            all_pass = all(s.score >= self._threshold for s in cs.compliance_scores)
            passes.append(1.0 if all_pass else 0.0)

        n = len(passes)
        mean = sum(passes) / n if n else 0.0
        std = _std(passes)
        lo, hi = _ci_95(mean, std, n)
        return MetricResult(
            name=self.name, value=mean, sample_size=n,
            std=std, ci_lower=lo, ci_upper=hi,
        )


class ComplianceScore:
    """Mean compliance score across all 8 rules and all conversations."""

    name = "compliance_score"

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult:
        values: list[float] = []
        for cs in scored:
            if cs.compliance_scores:
                values.append(
                    sum(s.score for s in cs.compliance_scores) / len(cs.compliance_scores)
                )
        n = len(values)
        mean = sum(values) / n if n else 0.0
        std = _std(values)
        lo, hi = _ci_95(mean, std, n)
        return MetricResult(
            name=self.name, value=mean, sample_size=n,
            std=std, ci_lower=lo, ci_upper=hi,
        )


# ------------------------------------------------------------------
# Quality metrics
# ------------------------------------------------------------------

class TaskCompletionRate:
    """Percentage of conversations completing all 3 stages."""

    name = "task_completion_rate"

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult:
        values = _scores_for_rule(scored, "quality_scores", "task_completion")
        n = len(values)
        mean = sum(values) / n if n else 0.0
        std = _std(values)
        lo, hi = _ci_95(mean, std, n)
        return MetricResult(
            name=self.name, value=mean, sample_size=n,
            std=std, ci_lower=lo, ci_upper=hi,
        )


class ToneScore:
    """Mean tone-appropriateness score across conversations."""

    name = "tone_score"

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult:
        values = _scores_for_rule(scored, "quality_scores", "tone_appropriateness")
        n = len(values)
        mean = sum(values) / n if n else 0.0
        std = _std(values)
        lo, hi = _ci_95(mean, std, n)
        return MetricResult(
            name=self.name, value=mean, sample_size=n,
            std=std, ci_lower=lo, ci_upper=hi,
        )


class SummarizationScore:
    """Mean summarisation-quality score from the quality judge."""

    name = "summarization_score"

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult:
        values = _scores_for_rule(scored, "quality_scores", "summarization_quality")
        n = len(values)
        mean = sum(values) / n if n else 0.0
        std = _std(values)
        lo, hi = _ci_95(mean, std, n)
        return MetricResult(
            name=self.name, value=mean, sample_size=n,
            std=std, ci_lower=lo, ci_upper=hi,
        )


# ------------------------------------------------------------------
# Handoff metrics
# ------------------------------------------------------------------

class HandoffFidelityScore:
    """Mean handoff-fidelity score across conversations."""

    name = "handoff_fidelity_score"

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult:
        values: list[float] = []
        for cs in scored:
            if cs.handoff_scores:
                values.append(
                    sum(s.score for s in cs.handoff_scores) / len(cs.handoff_scores)
                )
        n = len(values)
        mean = sum(values) / n if n else 0.0
        std = _std(values)
        lo, hi = _ci_95(mean, std, n)
        return MetricResult(
            name=self.name, value=mean, sample_size=n,
            std=std, ci_lower=lo, ci_upper=hi,
        )


# ------------------------------------------------------------------
# Composite
# ------------------------------------------------------------------

class CompositeScore:
    """Weighted composite with compliance gating.

    If compliance mean < 0.8, composite is forced to 0.
    """

    name = "composite_score"

    def compute(self, scored: Sequence[ConversationScores]) -> MetricResult:
        values = [cs.composite_score for cs in scored]
        n = len(values)
        mean = sum(values) / n if n else 0.0
        std = _std(values)
        lo, hi = _ci_95(mean, std, n)
        return MetricResult(
            name=self.name, value=mean, sample_size=n,
            std=std, ci_lower=lo, ci_upper=hi,
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

ALL_METRICS: list[Metric] = [
    CompliancePassRate(),
    ComplianceScore(),
    TaskCompletionRate(),
    ToneScore(),
    SummarizationScore(),
    HandoffFidelityScore(),
    CompositeScore(),
]


def compute_all_metrics(scored: Sequence[ConversationScores]) -> list[MetricResult]:
    return [m.compute(scored) for m in ALL_METRICS]


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
