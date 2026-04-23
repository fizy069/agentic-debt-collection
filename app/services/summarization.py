"""Stage-aware summarization policies and LLM-assisted overflow compression.

Defines per-stage keep/remove contracts so each agent receives a summary
that preserves exactly the data it needs while staying within budget.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from app.services.token_budget import MAX_HANDOFF_TOKENS, count_tokens, truncate_to_budget

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummarizationPolicy:
    """Declares what a downstream agent must keep vs. may drop."""

    stage: str
    keep_fields: tuple[str, ...]
    keep_signals: tuple[str, ...]
    remove_signals: tuple[str, ...]
    system_instruction: str
    max_output_tokens: int = MAX_HANDOFF_TOKENS


_ASSESSMENT_TO_RESOLUTION = SummarizationPolicy(
    stage="assessment",
    keep_fields=(
        "identity_confirmed",
        "debt_acknowledged",
        "ability_to_pay_discussed",
    ),
    keep_signals=(
        "identity verification status",
        "debt amount and acknowledgement",
        "ability to pay or hardship indicators",
        "borrower stance",
        "stop-contact requests",
    ),
    remove_signals=(
        "greeting pleasantries",
        "repeated identity prompts already answered",
        "filler acknowledgements with no new information",
    ),
    system_instruction=(
        "Compress the assessment stage into a concise JSON summary for the "
        "resolution agent. Preserve identity status, debt acknowledgement, "
        "ability-to-pay signals, and any hardship or stop-contact flags. "
        "Remove pleasantries, repeated questions, and filler."
    ),
)

_RESOLUTION_TO_FINAL_NOTICE = SummarizationPolicy(
    stage="resolution",
    keep_fields=(
        "options_reviewed",
        "borrower_position_known",
        "commitment_or_disposition",
    ),
    keep_signals=(
        "offers presented and borrower reaction",
        "objections raised",
        "commitment or refusal",
        "hardship referral status",
        "unresolved blockers",
    ),
    remove_signals=(
        "repeated option listings already acknowledged",
        "conversational filler",
        "duplicated stance restatements",
    ),
    system_instruction=(
        "Compress the assessment and resolution stages into a concise JSON "
        "summary for the final-notice agent. Preserve offers shown, borrower "
        "objections, commitment/disposition, hardship status, and unresolved "
        "blockers. Remove duplicated option listings and filler."
    ),
)

STAGE_POLICIES: dict[str, SummarizationPolicy] = {
    "resolution": _ASSESSMENT_TO_RESOLUTION,
    "final_notice": _RESOLUTION_TO_FINAL_NOTICE,
}


@dataclass
class FieldPriority:
    """A handoff field tagged with its pruning priority."""

    key: str
    value: Any
    priority: str = "must_keep"


@dataclass
class SectionBudget:
    """Token accounting for one prompt section."""

    name: str
    text: str
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = count_tokens(self.text)


@dataclass
class BudgetReport:
    """Full token accounting across all prompt sections."""

    sections: list[SectionBudget] = field(default_factory=list)
    total_tokens: int = 0
    overflow_detected: bool = False
    overflow_summary_used: bool = False
    overflow_fallback_used: bool = False
    handoff_tokens: int = 0
    pre_overflow_tokens: int = 0
    post_overflow_tokens: int = 0

    def add_section(self, name: str, text: str) -> SectionBudget:
        section = SectionBudget(name=name, text=text)
        self.sections.append(section)
        self.total_tokens = sum(s.tokens for s in self.sections)
        return section


def get_policy_for_stage(stage: str) -> SummarizationPolicy | None:
    return STAGE_POLICIES.get(stage)


def build_overflow_prompt(
    policy: SummarizationPolicy,
    content_to_compress: str,
    target_tokens: int,
) -> tuple[str, str]:
    """Build a short system+user prompt pair for overflow summarization.

    Returns (system_prompt, user_prompt) designed to be sent to the LLM
    as a dedicated compression call with strict keep/remove instructions.
    The system prompt is loaded from the centralized prompt registry.
    """
    from app.services.prompt_assembler import assemble_overflow_prompt as _assemble
    from app.services.prompt_registry import get_prompt_registry

    from app.models.pipeline import PipelineStage
    stage = PipelineStage(policy.stage) if policy.stage in {s.value for s in PipelineStage} else PipelineStage.RESOLUTION
    registry = get_prompt_registry()
    config = registry.get_overflow_config(stage)
    assembled = _assemble(
        config,
        policy_system_instruction=policy.system_instruction,
        keep_signals=policy.keep_signals,
        remove_signals=policy.remove_signals,
        content_to_compress=content_to_compress,
        target_tokens=target_tokens,
    )
    return assembled.system_prompt, assembled.user_prompt


def prioritize_handoff_fields(
    summary: dict[str, Any],
    policy: SummarizationPolicy | None,
) -> list[FieldPriority]:
    """Tag each top-level handoff field with a pruning priority.

    Priority levels:
    - must_keep: borrower identifiers, stages_covered, collected_fields
    - should_keep: stance, transition reasons
    - optional: key_exchanges, verbose metadata
    """
    must_keep_keys = {"stages_covered", "borrower_id", "debt_amount", "currency", "days_past_due"}
    if policy:
        for stage_key in policy.keep_fields:
            must_keep_keys.add(stage_key)
    optional_keys = {"key_exchanges"}

    result: list[FieldPriority] = []
    for key, value in summary.items():
        if key in must_keep_keys:
            priority = "must_keep"
        elif key in optional_keys:
            priority = "optional"
        elif isinstance(value, dict):
            priority = "must_keep"
        else:
            priority = "should_keep"
        result.append(FieldPriority(key=key, value=value, priority=priority))
    return result


def prune_to_budget(
    summary: dict[str, Any],
    policy: SummarizationPolicy | None,
    max_tokens: int = MAX_HANDOFF_TOKENS,
) -> str:
    """Deterministic priority-based pruning that guarantees <= max_tokens.

    Drops fields from lowest priority first (optional -> should_keep),
    then truncates exchange text, then as a last resort truncates the
    serialized result at the token boundary.
    """
    prioritized = prioritize_handoff_fields(summary, policy)

    working = {fp.key: fp.value for fp in prioritized}
    result = json.dumps(working, separators=(",", ":"))
    if count_tokens(result) <= max_tokens:
        return result

    for priority_level in ("optional", "should_keep"):
        keys_at_level = [fp.key for fp in prioritized if fp.priority == priority_level]
        for key in keys_at_level:
            if key in working:
                del working[key]
                logger.info(
                    "prune_to_budget  dropped=%s  priority=%s  tokens_now=%d  limit=%d",
                    key, priority_level, count_tokens(json.dumps(working, separators=(",", ":"))),
                    max_tokens,
                )
                result = json.dumps(working, separators=(",", ":"))
                if count_tokens(result) <= max_tokens:
                    return result

    for key in list(working):
        if isinstance(working[key], dict):
            trimmed = {}
            for sub_key, sub_val in working[key].items():
                if isinstance(sub_val, str) and len(sub_val) > 40:
                    trimmed[sub_key] = sub_val[:40]
                else:
                    trimmed[sub_key] = sub_val
            working[key] = trimmed
            logger.info(
                "prune_to_budget  truncated_subfields=%s  tokens_now=%d  limit=%d",
                key, count_tokens(json.dumps(working, separators=(",", ":"))), max_tokens,
            )
            result = json.dumps(working, separators=(",", ":"))
            if count_tokens(result) <= max_tokens:
                return result

    result = json.dumps(working, separators=(",", ":"))
    if count_tokens(result) > max_tokens:
        logger.warning(
            "Hard-truncating handoff summary from %d to %d tokens",
            count_tokens(result), max_tokens,
        )
        result = truncate_to_budget(result, max_tokens)

    return result
