"""Prompt assembler — composes registry sections + runtime context into final prompts.

Each ``assemble_*`` function takes a ``PromptConfig`` from the registry
together with runtime data and returns an ``AssembledPrompt`` that is
ready for the LLM client.  Metadata emitted alongside each prompt lets
the self-learning loop trace exactly which section versions contributed
to a given turn.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.pipeline import (
    BorrowerRequest,
    ComplianceFlags,
    ConversationMessage,
    PipelineStage,
)
from app.models.prompt import AssembledPrompt, PromptConfig
from app.services.compliance import (
    ALLOWED_CONSEQUENCES,
    OFFER_POLICY_BOUNDS,
)
from app.services.handoff import build_handoff_summary
from app.services.token_budget import count_tokens

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Shared formatters (pure functions, no LLM calls)
# ------------------------------------------------------------------

def _borrower_snapshot(
    borrower: BorrowerRequest,
    *,
    identity_confirmed: bool = False,
    stage: PipelineStage | None = None,
) -> str:
    """Build the snapshot injected into agent prompts.

    During the assessment stage, account details are labelled as
    system-internal until the borrower's identity has been verified.
    Post-assessment (or once identity is confirmed), the snapshot is
    presented as verified.
    """
    pre_verification = stage == PipelineStage.ASSESSMENT and not identity_confirmed
    masked_reference = borrower.account_reference[-4:]

    if pre_verification:
        header = "SYSTEM ACCOUNT RECORD (UNVERIFIED — do NOT disclose to caller until identity is confirmed):"
    else:
        header = "VERIFIED BORROWER ACCOUNT:"

    return (
        f"{header}\n"
        f"Borrower ID: {borrower.borrower_id}\n"
        f"Account Ref (last4): {masked_reference}\n"
        f"Date of Birth (on file): {borrower.date_of_birth}\n"
        f"Debt: {borrower.debt_amount:.2f} {borrower.currency}\n"
        f"Days past due: {borrower.days_past_due}\n"
        f"Borrower message: {borrower.borrower_message}\n"
        f"Notes: {borrower.notes or 'None provided'}"
    )


def _format_recent_transcript(
    transcript: list[ConversationMessage],
    max_items: int = 8,
) -> str:
    if not transcript:
        return "No prior transcript."
    recent = transcript[-max_items:]
    lines = []
    for item in recent:
        stage = item.stage.value if item.stage else "none"
        lines.append(f"{item.role.value}@{stage}: {item.text}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Directive rendering
# ------------------------------------------------------------------

def _render_turn_directives(
    config: PromptConfig,
    final_notice_expiry: str | None,
) -> str:
    """Render the turn-directives section, substituting runtime values."""
    section = config.sections.get("turn_directives")
    if section is None:
        return ""
    template = section.content
    if final_notice_expiry and "{expiry}" in template:
        return template.format(expiry=final_notice_expiry)
    return template


def _render_closing_turn_directive(config: PromptConfig) -> str:
    """Render the closing_turn directive that replaces normal turn directives."""
    section = config.sections.get("closing_turn")
    if section is None:
        return (
            "This is your final turn for this stage. Produce ONLY a closing "
            "response in 2-3 sentences. Do NOT ask any question. Do NOT end "
            "with a question mark."
        )
    return section.content


def _render_compliance_directives(config: PromptConfig) -> str:
    """Render offer-policy and allowed-consequences directives."""
    parts: list[str] = []

    offer_section = config.sections.get("offer_policy")
    if offer_section:
        parts.append(offer_section.content.format(**OFFER_POLICY_BOUNDS))

    consequences_section = config.sections.get("allowed_consequences")
    if consequences_section:
        items = ", ".join(ALLOWED_CONSEQUENCES)
        parts.append(consequences_section.content.format(consequences=items))

    return " ".join(parts)


# ------------------------------------------------------------------
# Section-version metadata helper
# ------------------------------------------------------------------

def _section_versions(config: PromptConfig) -> dict[str, str]:
    return {name: s.version for name, s in config.sections.items()}


# ------------------------------------------------------------------
# Main agent assembler
# ------------------------------------------------------------------

def assemble_agent_prompt(
    config: PromptConfig,
    *,
    borrower: BorrowerRequest,
    borrower_message: str,
    stage: PipelineStage,
    turn_index: int,
    prior_fields: dict[str, bool],
    transcript: list[ConversationMessage],
    completed_stages: list[dict[str, Any]],
    flags: ComplianceFlags,
    final_notice_expiry: str | None,
    closing_turn: bool = False,
) -> AssembledPrompt:
    """Assemble the system + user prompt pair for a main agent turn."""
    system_template = config.sections["system_template"]
    system_prompt = system_template.content

    identity_confirmed = prior_fields.get("identity_confirmed", False)
    snapshot_section = (
        f"Borrower snapshot:\n"
        f"{_borrower_snapshot(borrower, identity_confirmed=identity_confirmed, stage=stage)}"
    )

    turn_meta_lines = [
        f"Current stage: {stage.value}",
        f"Turn index in stage: {turn_index}",
        f"Borrower message: {borrower_message}",
        f"Collected fields so far: {prior_fields}",
    ]
    if flags.hardship_detected:
        turn_meta_lines.append(
            "COMPLIANCE: Hardship detected \u2014 offer hardship referral route. "
            "Do not pressure the borrower."
        )
    if final_notice_expiry:
        turn_meta_lines.append(f"Hard expiry date: {final_notice_expiry}")
    turn_meta_section = "\n".join(turn_meta_lines)

    transcript_section = (
        "Recent transcript:\n"
        f"{_format_recent_transcript(transcript)}"
    )

    if closing_turn:
        directives_section = _render_closing_turn_directive(config)
    else:
        directives_section = _render_turn_directives(config, final_notice_expiry)
        compliance_dir = _render_compliance_directives(config)
        if compliance_dir:
            directives_section += f" {compliance_dir}"

    handoff_section = ""
    if completed_stages:
        handoff_section = build_handoff_summary(
            completed_stages,
            borrower.model_dump(mode="json"),
            [msg.model_dump(mode="json") for msg in transcript],
            target_stage=stage.value,
        )

    user_prompt_parts = [
        "Continue the debt-collection conversation for the current stage.\n",
    ]
    if handoff_section:
        user_prompt_parts.append(f"Prior stage context:\n{handoff_section}\n")
    user_prompt_parts.extend([
        f"{snapshot_section}\n",
        f"{turn_meta_section}\n",
        f"{transcript_section}\n",
        directives_section,
    ])
    user_prompt = "\n".join(user_prompt_parts)

    metadata: dict[str, Any] = {
        "prompt_config_version": config.version,
        "section_versions": _section_versions(config),
        "section_tokens": {
            "system_prompt": count_tokens(system_prompt),
            "snapshot": count_tokens(snapshot_section),
            "turn_meta": count_tokens(turn_meta_section),
            "transcript": count_tokens(transcript_section),
            "directives": count_tokens(directives_section),
        },
    }
    if closing_turn:
        metadata["closing_turn"] = True
    if handoff_section:
        metadata["section_tokens"]["handoff"] = count_tokens(handoff_section)
        metadata["handoff_section"] = handoff_section

    return AssembledPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata=metadata,
    )


def assemble_overflow_user_prompt(
    *,
    compressed: str,
    turn_meta_section: str,
    directives_section: str,
) -> str:
    """Re-assemble the user prompt after overflow compression."""
    return (
        "Continue the debt-collection conversation for the current stage.\n\n"
        f"{compressed}\n\n"
        f"{turn_meta_section}\n\n"
        f"{directives_section}"
    )


# ------------------------------------------------------------------
# Judge assembler
# ------------------------------------------------------------------

def assemble_judge_prompt(
    config: PromptConfig,
    *,
    stage: str,
    turn_index: int,
    assistant_reply: str,
    transcript_excerpt: str,
) -> AssembledPrompt:
    """Assemble system + user prompts for the compliance judge."""
    system_prompt = config.sections["judge_system"].content
    user_prompt = (
        f"Stage: {stage}\n"
        f"Turn: {turn_index}\n\n"
        f"--- Transcript excerpt ---\n{transcript_excerpt}\n\n"
        f"--- Assistant reply under review ---\n{assistant_reply}"
    )
    return AssembledPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata={
            "prompt_config_version": config.version,
            "section_versions": _section_versions(config),
        },
    )


# ------------------------------------------------------------------
# Overflow-compression assembler
# ------------------------------------------------------------------

def assemble_overflow_prompt(
    config: PromptConfig,
    *,
    policy_system_instruction: str,
    keep_signals: tuple[str, ...],
    remove_signals: tuple[str, ...],
    content_to_compress: str,
    target_tokens: int,
) -> AssembledPrompt:
    """Assemble system + user prompts for overflow compression."""
    system_prompt = config.sections["overflow_system"].content
    user_prompt = (
        f"{policy_system_instruction}\n\n"
        "KEEP these data points:\n"
        + "\n".join(f"- {s}" for s in keep_signals)
        + "\n\nREMOVE:\n"
        + "\n".join(f"- {s}" for s in remove_signals)
        + f"\n\nTarget: fit within {target_tokens} tokens.\n\n"
        f"INPUT:\n{content_to_compress}"
    )
    return AssembledPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        metadata={
            "prompt_config_version": config.version,
            "section_versions": _section_versions(config),
        },
    )
