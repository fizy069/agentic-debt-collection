"""Statistical utilities for the evaluation harness.

Provides summary statistics, significance testing, and effect-size
calculation so the self-learning loop can prove that improvements are
real and not noise.
"""

from __future__ import annotations

import math
from typing import Sequence


def compute_summary(values: Sequence[float]) -> dict[str, float]:
    """Descriptive statistics for a list of scores.

    Returns mean, std, median, min, max, p5, p95, and sample_size.
    """
    n = len(values)
    if n == 0:
        return {
            "mean": 0.0, "std": 0.0, "median": 0.0,
            "min": 0.0, "max": 0.0, "p5": 0.0, "p95": 0.0,
            "sample_size": 0,
        }

    sorted_v = sorted(values)
    mean = sum(sorted_v) / n
    variance = sum((v - mean) ** 2 for v in sorted_v) / max(n - 1, 1)
    std = math.sqrt(variance)

    median = (
        sorted_v[n // 2]
        if n % 2 == 1
        else (sorted_v[n // 2 - 1] + sorted_v[n // 2]) / 2
    )

    return {
        "mean": mean,
        "std": std,
        "median": median,
        "min": sorted_v[0],
        "max": sorted_v[-1],
        "p5": _percentile(sorted_v, 5),
        "p95": _percentile(sorted_v, 95),
        "sample_size": float(n),
    }


def is_significant_improvement(
    baseline: Sequence[float],
    candidate: Sequence[float],
    alpha: float = 0.05,
) -> tuple[bool, float]:
    """Welch's t-test for whether *candidate* > *baseline*.

    Returns ``(is_significant, p_value)``.  Uses a one-sided test
    (candidate mean > baseline mean).

    Falls back to ``(False, 1.0)`` if sample sizes are too small.
    """
    n1, n2 = len(baseline), len(candidate)
    if n1 < 2 or n2 < 2:
        return False, 1.0

    m1 = sum(baseline) / n1
    m2 = sum(candidate) / n2
    s1 = _var(baseline, m1)
    s2 = _var(candidate, m2)

    se = math.sqrt(s1 / n1 + s2 / n2)
    if se == 0:
        return (m2 > m1, 0.0 if m2 > m1 else 1.0)

    t_stat = (m2 - m1) / se

    # Welch-Satterthwaite degrees of freedom
    num = (s1 / n1 + s2 / n2) ** 2
    denom = ((s1 / n1) ** 2) / (n1 - 1) + ((s2 / n2) ** 2) / (n2 - 1)
    df = num / denom if denom > 0 else 1.0

    p_value = _t_cdf_upper(t_stat, df)
    return (p_value < alpha, p_value)


def compute_effect_size(
    baseline: Sequence[float],
    candidate: Sequence[float],
) -> float:
    """Cohen's d effect size (pooled standard deviation)."""
    n1, n2 = len(baseline), len(candidate)
    if n1 < 2 or n2 < 2:
        return 0.0

    m1 = sum(baseline) / n1
    m2 = sum(candidate) / n2
    s1 = _var(baseline, m1)
    s2 = _var(candidate, m2)

    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    pooled_std = math.sqrt(pooled_var)
    if pooled_std == 0:
        return 0.0
    return (m2 - m1) / pooled_std


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _var(values: Sequence[float], mean: float) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    return sum((v - mean) ** 2 for v in values) / (n - 1)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear interpolation percentile on already-sorted data."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    k = (pct / 100) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def _t_cdf_upper(t: float, df: float) -> float:
    """Approximate one-sided upper-tail p-value for Student's t-distribution.

    Uses the Abramowitz & Stegun normal approximation which is well-behaved
    for df >= 1 and avoids complex continued-fraction numerics.
    """
    if df <= 0:
        return 0.5

    # Transform t to a standard normal deviate via A&S 26.7.5
    g = math.log(1 + t * t / df) * (df - 0.5)
    z = math.sqrt(g)
    if t < 0:
        z = -z

    return _normal_cdf_upper(z)


def _normal_cdf_upper(z: float) -> float:
    """Upper-tail probability of the standard normal distribution.

    Uses Horner-form rational approximation (Abramowitz & Stegun 26.2.17).
    Maximum error < 7.5e-8.
    """
    if z < -8.0:
        return 1.0
    if z > 8.0:
        return 0.0

    abs_z = abs(z)
    t = 1.0 / (1.0 + 0.2316419 * abs_z)
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    t5 = t4 * t

    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    p = pdf * (
        0.319381530 * t
        - 0.356563782 * t2
        + 1.781477937 * t3
        - 1.821255978 * t4
        + 1.330274429 * t5
    )

    if z > 0:
        return p
    return 1.0 - p
