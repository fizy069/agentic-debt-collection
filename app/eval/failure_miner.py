"""Mine evaluation results for the highest-leverage improvement targets.

Groups low-scoring conversations by the prompt section most likely
responsible, ranks sections by failure count * severity, and surfaces
the top transcript excerpts that the proposer will use as evidence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from app.eval.models import ConversationScores, EvalRunResult, JudgeScore

logger = logging.getLogger(__name__)

# Maps judge rule_ids to the registry section key most likely responsible.
_COMPLIANCE_RULE_TO_SECTION: dict[str, str] = {
    "1": "system_template:assessment",
    "2": "compliance_directives:allowed_consequences",
    "3": "system_template:assessment",
    "5": "system_template:assessment",
    "6": "system_template:assessment",
    "7": "turn_directives:default",
    "8": "system_template:assessment",
}

_QUALITY_RULE_TO_SECTION: dict[str, str] = {
    "task_completion": "system_template:assessment",
    "tone_appropriateness": "turn_directives:default",
    "conciseness": "turn_directives:default",
    "summarization_quality": "system_template:assessment",
}

_HANDOFF_RULE_TO_SECTION: dict[str, str] = {
    "context_preservation": "system_template:assessment",
    "summarization_fidelity": "system_template:assessment",
    "seamless_experience": "system_template:assessment",
}

SCORE_THRESHOLD = 0.7

_SEVERITY: dict[str, float] = {
    "compliance_scores": 3.0,
    "quality_scores": 1.0,
    "handoff_scores": 1.5,
}


@dataclass
class FailureExcerpt:
    """A single failing evidence item for the proposer."""
    scenario_id: str
    rule_id: str
    rule_name: str
    score: float
    explanation: str
    source: str


@dataclass
class SectionFailures:
    """Aggregated failures attributed to a single prompt section."""
    section_key: str
    weighted_count: float = 0.0
    excerpts: list[FailureExcerpt] = field(default_factory=list)


@dataclass
class FailureDigest:
    """Ranked list of sections to improve, with evidence."""
    sections: list[SectionFailures] = field(default_factory=list)

    @property
    def top_section(self) -> str | None:
        return self.sections[0].section_key if self.sections else None


def mine_failures(
    result: EvalRunResult,
    threshold: float = SCORE_THRESHOLD,
    max_excerpts_per_section: int = 5,
) -> FailureDigest:
    """Identify the most impactful prompt sections to improve."""
    bucket: dict[str, SectionFailures] = {}

    for cs in result.scores:
        _collect(cs, "compliance_scores", _COMPLIANCE_RULE_TO_SECTION, threshold, bucket)
        _collect(cs, "quality_scores", _QUALITY_RULE_TO_SECTION, threshold, bucket)
        _collect(cs, "handoff_scores", _HANDOFF_RULE_TO_SECTION, threshold, bucket)

    for sf in bucket.values():
        sf.excerpts.sort(key=lambda e: e.score)
        sf.excerpts = sf.excerpts[:max_excerpts_per_section]

    ranked = sorted(bucket.values(), key=lambda s: s.weighted_count, reverse=True)
    digest = FailureDigest(sections=ranked)

    if digest.top_section:
        logger.info(
            "failure_miner  top_section=%s  weighted_failures=%.1f  total_sections=%d",
            digest.top_section, ranked[0].weighted_count, len(ranked),
        )
    else:
        logger.info("failure_miner  no_failures_found")

    return digest


def _collect(
    cs: ConversationScores,
    source_attr: str,
    rule_map: dict[str, str],
    threshold: float,
    bucket: dict[str, SectionFailures],
) -> None:
    scores: list[JudgeScore] = getattr(cs, source_attr, [])
    severity = _SEVERITY.get(source_attr, 1.0)

    for js in scores:
        if js.score >= threshold:
            continue
        section_key = rule_map.get(js.rule_id)
        if section_key is None:
            continue

        if section_key not in bucket:
            bucket[section_key] = SectionFailures(section_key=section_key)

        sf = bucket[section_key]
        sf.weighted_count += severity
        sf.excerpts.append(FailureExcerpt(
            scenario_id=cs.scenario_id,
            rule_id=js.rule_id,
            rule_name=js.rule_name,
            score=js.score,
            explanation=js.explanation,
            source=source_attr,
        ))
