from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import logging
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from temporalio import activity

logger = logging.getLogger(__name__)

from app.models.pipeline import (
    AgentChannel,
    BorrowerRequest,
    ComplianceFlags,
    ConversationMessage,
    PipelineStage,
    StageTurnInput,
    StageTurnOutput,
)
from app.services.anthropic_client import AnthropicClient
from app.services.compliance import (
    ABUSIVE_CLOSE_REPLY,
    STOP_CONTACT_REPLY,
    allowed_consequences_directive,
    check_false_threats,
    check_offer_bounds,
    detect_abusive,
    detect_hardship,
    detect_stop_contact,
    offer_policy_directive,
    redact_pii,
)
from app.services.handoff import build_handoff_summary
from app.services.summarization import (
    build_overflow_prompt,
    get_policy_for_stage,
)
from app.services.token_budget import (
    MAX_CONTEXT_TOKENS,
    OVERSIZED_MESSAGE_REPLY,
    ContextBudgetReport,
    count_tokens,
    enforce_context_budget,
    is_borrower_message_oversized,
    truncate_to_budget,
)


_cached_client: AnthropicClient | None = None

_ASSESSMENT_OPENING_DISCLOSURE = (
    "I am an AI agent acting on behalf of the company, and this conversation "
    "is being logged and recorded."
)


def _get_anthropic_client() -> AnthropicClient:
    global _cached_client
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    if _cached_client is not None and _cached_client._client is not None:
        return _cached_client

    if _cached_client is not None and _cached_client._client is None:
        logger.warning("Cached client had no API key – rebuilding")

    _cached_client = AnthropicClient()
    return _cached_client


def _trim_summary(text: str, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


def _prepend_assessment_opening_disclosure(
    text: str,
    *,
    stage: PipelineStage,
    turn_index: int,
) -> str:
    reply = text.strip()
    if stage != PipelineStage.ASSESSMENT or turn_index != 1:
        return reply

    normalized = " ".join(reply.lower().split())
    has_ai_disclosure = "ai agent" in normalized and "company" in normalized
    has_recording_disclosure = "logged and recorded" in normalized or (
        "logged" in normalized and "recorded" in normalized
    )
    if has_ai_disclosure and has_recording_disclosure:
        return reply

    return f"{_ASSESSMENT_OPENING_DISCLOSURE} {reply}".strip()


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _borrower_snapshot(borrower: BorrowerRequest) -> str:
    masked_reference = borrower.account_reference[-4:]
    return (
        f"Borrower ID: {borrower.borrower_id}\n"
        f"Account Ref (last4): {masked_reference}\n"
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


def _assessment_logic(
    *,
    borrower_message: str,
    collected_fields: dict[str, bool],
    turn_index: int,
) -> tuple[dict[str, bool], bool, str, str]:
    updated = {
        "identity_confirmed": bool(collected_fields.get("identity_confirmed")),
        "debt_acknowledged": bool(collected_fields.get("debt_acknowledged")),
        "ability_to_pay_discussed": bool(
            collected_fields.get("ability_to_pay_discussed")
        ),
    }
    lowered = borrower_message.lower()

    identity_regex_hit = bool(re.search(r"\b\d{4}\b", lowered))
    updated["identity_confirmed"] = updated["identity_confirmed"] or (
        _contains_any(lowered, ("date of birth", "dob", "last four", "ssn", "identity"))
        and identity_regex_hit
    )
    updated["debt_acknowledged"] = updated["debt_acknowledged"] or _contains_any(
        lowered, ("debt", "balance", "owe", "i owe", "amount due", "yes")
    )
    updated["ability_to_pay_discussed"] = updated[
        "ability_to_pay_discussed"
    ] or _contains_any(
        lowered, ("pay", "income", "salary", "monthly", "installment", "plan", "afford")
    )

    complete_by_fields = all(updated.values())
    complete_by_turn_cap = turn_index >= 3
    stage_complete = complete_by_fields or complete_by_turn_cap

    transition_reason = (
        "required_assessment_fields_collected"
        if complete_by_fields
        else "assessment_max_turns_reached"
        if complete_by_turn_cap
        else "awaiting_assessment_details"
    )
    decision = "assessment_completed" if stage_complete else "assessment_follow_up"
    return updated, stage_complete, transition_reason, decision


def _resolution_logic(
    *,
    borrower_message: str,
    collected_fields: dict[str, bool],
    turn_index: int,
) -> tuple[dict[str, bool], bool, str, str]:
    updated = {
        "options_reviewed": bool(collected_fields.get("options_reviewed")),
        "borrower_position_known": bool(collected_fields.get("borrower_position_known")),
        "commitment_or_disposition": bool(
            collected_fields.get("commitment_or_disposition")
        ),
    }
    lowered = borrower_message.lower()

    updated["options_reviewed"] = updated["options_reviewed"] or _contains_any(
        lowered, ("option", "lump", "plan", "hardship", "discount")
    )
    updated["borrower_position_known"] = updated["borrower_position_known"] or _contains_any(
        lowered,
        (
            "i choose",
            "choose",
            "prefer",
            "agree",
            "yes",
            "no",
            "cannot",
            "can't",
            "wont",
            "won't",
            "hardship",
        ),
    )
    updated["commitment_or_disposition"] = updated[
        "commitment_or_disposition"
    ] or _contains_any(
        lowered,
        (
            "agree",
            "commit",
            "i can pay",
            "schedule",
            "refuse",
            "decline",
            "hardship",
        ),
    )

    complete_by_fields = all(updated.values())
    complete_by_turn_cap = turn_index >= 3
    stage_complete = complete_by_fields or complete_by_turn_cap

    transition_reason = (
        "resolution_terms_captured"
        if complete_by_fields
        else "resolution_max_turns_reached"
        if complete_by_turn_cap
        else "awaiting_resolution_position"
    )
    decision = "resolution_attempted" if stage_complete else "resolution_follow_up"
    return updated, stage_complete, transition_reason, decision


def _final_notice_logic(
    *,
    borrower_message: str,
    collected_fields: dict[str, bool],
    turn_index: int,
) -> tuple[dict[str, bool], bool, str, str]:
    updated = {
        "final_notice_acknowledged": bool(
            collected_fields.get("final_notice_acknowledged")
        ),
        "borrower_response_recorded": bool(
            collected_fields.get("borrower_response_recorded")
        ),
    }
    lowered = borrower_message.lower()

    updated["final_notice_acknowledged"] = updated[
        "final_notice_acknowledged"
    ] or _contains_any(
        lowered,
        ("understand", "acknowledge", "received", "got it", "okay", "ok"),
    )
    updated["borrower_response_recorded"] = updated[
        "borrower_response_recorded"
    ] or len(lowered.strip()) > 0

    complete_by_fields = all(updated.values())
    complete_by_turn_cap = turn_index >= 2
    stage_complete = complete_by_fields or complete_by_turn_cap

    transition_reason = (
        "final_notice_acknowledged"
        if complete_by_fields
        else "final_notice_max_turns_reached"
        if complete_by_turn_cap
        else "awaiting_final_notice_acknowledgement"
    )
    decision = "final_notice_sent" if stage_complete else "final_notice_follow_up"
    return updated, stage_complete, transition_reason, decision


def _evaluate_stage_turn(
    *,
    stage: PipelineStage,
    borrower_message: str,
    collected_fields: dict[str, bool],
    turn_index: int,
) -> tuple[dict[str, bool], bool, str, str]:
    if stage == PipelineStage.ASSESSMENT:
        return _assessment_logic(
            borrower_message=borrower_message,
            collected_fields=collected_fields,
            turn_index=turn_index,
        )
    if stage == PipelineStage.RESOLUTION:
        return _resolution_logic(
            borrower_message=borrower_message,
            collected_fields=collected_fields,
            turn_index=turn_index,
        )
    return _final_notice_logic(
        borrower_message=borrower_message,
        collected_fields=collected_fields,
        turn_index=turn_index,
    )


_PROMPT_FILES = {
    PipelineStage.ASSESSMENT: "assessment.txt",
    PipelineStage.RESOLUTION: "resolution.txt",
    PipelineStage.FINAL_NOTICE: "final_notice.txt",
}


@lru_cache(maxsize=3)
def _load_prompt(stage: PipelineStage) -> str:
    """Load system prompt text from ``app/prompts/<stage>.txt``."""
    filename = _PROMPT_FILES[stage]
    path = Path(__file__).resolve().parent.parent / "prompts" / filename
    return path.read_text(encoding="utf-8").strip()


def _stage_defaults(
    stage: PipelineStage,
) -> tuple[AgentChannel, PipelineStage | None, str]:
    system_prompt = _load_prompt(stage)
    if stage == PipelineStage.ASSESSMENT:
        return AgentChannel.CHAT, PipelineStage.RESOLUTION, system_prompt
    if stage == PipelineStage.RESOLUTION:
        return AgentChannel.VOICE_STUB, PipelineStage.FINAL_NOTICE, system_prompt
    return AgentChannel.CHAT, None, system_prompt


def _final_notice_expiry_date(
    *,
    base_date: date | None = None,
    window_days: int = 7,
) -> str:
    anchor = base_date or datetime.now(timezone.utc).date()
    return (anchor + timedelta(days=window_days)).isoformat()


def _build_turn_directives(
    *,
    stage: PipelineStage,
    stage_complete: bool,
    final_notice_expiry: str | None = None,
) -> str:
    if stage == PipelineStage.FINAL_NOTICE:
        expiry = final_notice_expiry or _final_notice_expiry_date()
        completion_instruction = (
            "Record the borrower's final response and close the conversation in a "
            "declarative statement. Do not ask a follow-up question."
            if stage_complete
            else "Ask one concrete acknowledgement question and wait for the "
            "borrower's response."
        )
        return (
            "Respond in 3-6 concise sentences. "
            f"Use this exact hard expiry date: {expiry}. "
            "Never use placeholders (for example, '[insert hard expiry date]'). "
            f"{completion_instruction}"
        )

    return (
        "Respond in 3-6 concise sentences. "
        "If stage is incomplete, ask one concrete follow-up question. "
        "If complete, provide a brief transition-ready response."
    )


async def _compress_overflow(
    client: AnthropicClient,
    stage: PipelineStage,
    content: str,
    target_tokens: int,
) -> tuple[str, bool, bool]:
    """Attempt LLM-assisted overflow compression with deterministic fallback.

    Returns (compressed_text, used_llm_summary, used_fallback).
    """
    policy = get_policy_for_stage(stage.value)
    if not policy:
        logger.info(
            "overflow_summarize  stage=%s  action=deterministic_truncation  "
            "reason=no_policy  input_tokens=%d  target=%d",
            stage.value, count_tokens(content), target_tokens,
        )
        return truncate_to_budget(content, target_tokens), False, True

    sys_prompt, usr_prompt = build_overflow_prompt(policy, content, target_tokens)

    try:
        logger.info(
            "overflow_summarize  stage=%s  action=llm_call  "
            "input_tokens=%d  target=%d  policy_stage=%s",
            stage.value, count_tokens(content), target_tokens, policy.stage,
        )
        result = await client.summarize(
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            max_tokens=min(target_tokens, 200),
        )
        compressed = result.text
        compressed_tokens = count_tokens(compressed)
        if compressed_tokens <= target_tokens:
            logger.info(
                "overflow_summarize  stage=%s  action=llm_accepted  "
                "output_tokens=%d  target=%d  used_fallback=%s",
                stage.value, compressed_tokens, target_tokens, result.used_fallback,
            )
            return compressed, True, result.used_fallback
        logger.warning(
            "overflow_summarize  stage=%s  action=llm_over_budget  "
            "output_tokens=%d  target=%d  applying_hard_truncation",
            stage.value, compressed_tokens, target_tokens,
        )
        return truncate_to_budget(compressed, target_tokens), True, True
    except Exception:
        logger.exception(
            "overflow_summarize  stage=%s  action=llm_failed  "
            "falling_back_to_truncation",
            stage.value,
        )
        return truncate_to_budget(content, target_tokens), False, True


async def _run_stage_turn(payload: dict[str, Any]) -> dict[str, Any]:
    turn_input = StageTurnInput.model_validate(payload)
    logger.info(
        "stage_turn_start  stage=%s  turn=%d  borrower=%s",
        turn_input.stage.value, turn_input.turn_index, turn_input.borrower.borrower_id,
    )

    msg_tokens = count_tokens(turn_input.borrower_message)
    if is_borrower_message_oversized(turn_input.borrower_message):
        logger.warning(
            "borrower_message_oversized  stage=%s  turn=%d  tokens=%d  "
            "action=hardcoded_reply",
            turn_input.stage.value, turn_input.turn_index, msg_tokens,
        )
        channel, next_stage, _ = _stage_defaults(turn_input.stage)
        return StageTurnOutput(
            stage=turn_input.stage,
            channel=channel,
            assistant_reply=OVERSIZED_MESSAGE_REPLY,
            summary=OVERSIZED_MESSAGE_REPLY,
            decision="borrower_message_oversized",
            stage_complete=False,
            collected_fields=turn_input.collected_fields,
            transition_reason="borrower_message_too_long",
            next_stage=turn_input.stage,
            metadata={
                "model": "none",
                "used_fallback": False,
                "turn_index": turn_input.turn_index,
                "borrower_message_oversized": True,
                "borrower_message_tokens": msg_tokens,
            },
        ).model_dump(mode="json")

    # --- Compliance pre-checks on borrower input ---
    flags = ComplianceFlags(**turn_input.compliance_flags.model_dump())
    channel, next_stage, _ = _stage_defaults(turn_input.stage)

    borrower_message, pii_redacted = redact_pii(turn_input.borrower_message)
    if pii_redacted:
        logger.info(
            "compliance_pii_redacted  stage=%s  turn=%d",
            turn_input.stage.value, turn_input.turn_index,
        )

    if detect_stop_contact(borrower_message):
        flags.stop_contact_requested = True
        logger.info(
            "compliance_stop_contact  stage=%s  turn=%d",
            turn_input.stage.value, turn_input.turn_index,
        )
        return StageTurnOutput(
            stage=turn_input.stage,
            channel=channel,
            assistant_reply=STOP_CONTACT_REPLY,
            summary=STOP_CONTACT_REPLY,
            decision="stop_contact_requested",
            stage_complete=True,
            collected_fields=turn_input.collected_fields,
            transition_reason="borrower_requested_stop_contact",
            next_stage=None,
            compliance_flags=flags,
            metadata={
                "model": "none",
                "used_fallback": False,
                "turn_index": turn_input.turn_index,
                "compliance_stop_contact": True,
            },
        ).model_dump(mode="json")

    if detect_abusive(borrower_message):
        flags.abusive_borrower = True
        logger.info(
            "compliance_abusive_borrower  stage=%s  turn=%d",
            turn_input.stage.value, turn_input.turn_index,
        )
        return StageTurnOutput(
            stage=turn_input.stage,
            channel=channel,
            assistant_reply=ABUSIVE_CLOSE_REPLY,
            summary=ABUSIVE_CLOSE_REPLY,
            decision="abusive_borrower_close",
            stage_complete=True,
            collected_fields=turn_input.collected_fields,
            transition_reason="abusive_language_detected",
            next_stage=None,
            compliance_flags=flags,
            metadata={
                "model": "none",
                "used_fallback": False,
                "turn_index": turn_input.turn_index,
                "compliance_abusive_close": True,
            },
        ).model_dump(mode="json")

    if detect_hardship(borrower_message):
        flags.hardship_detected = True
        logger.info(
            "compliance_hardship_detected  stage=%s  turn=%d",
            turn_input.stage.value, turn_input.turn_index,
        )

    channel, next_stage, system_prompt = _stage_defaults(turn_input.stage)

    updated_fields, stage_complete, transition_reason, decision = _evaluate_stage_turn(
        stage=turn_input.stage,
        borrower_message=borrower_message,
        collected_fields=turn_input.collected_fields,
        turn_index=turn_input.turn_index,
    )

    missing_fields = [key for key, value in updated_fields.items() if not value]
    final_notice_expiry = (
        _final_notice_expiry_date()
        if turn_input.stage == PipelineStage.FINAL_NOTICE
        else None
    )
    report = ContextBudgetReport(limit=MAX_CONTEXT_TOKENS)

    report.add("system_prompt", system_prompt)

    snapshot_section = f"Borrower snapshot:\n{_borrower_snapshot(turn_input.borrower)}"
    turn_meta_lines = [
        f"Current stage: {turn_input.stage.value}",
        f"Turn index in stage: {turn_input.turn_index}",
        f"Borrower message: {borrower_message}",
        f"Collected fields: {updated_fields}",
        f"Missing fields: {missing_fields if missing_fields else 'none'}",
        f"Transition reason: {transition_reason}",
        f"Stage complete this turn: {stage_complete}",
    ]
    if flags.hardship_detected:
        turn_meta_lines.append(
            "COMPLIANCE: Hardship detected — offer hardship referral route. "
            "Do not pressure the borrower."
        )
    if final_notice_expiry:
        turn_meta_lines.append(f"Hard expiry date: {final_notice_expiry}")
    turn_meta_section = "\n".join(turn_meta_lines)
    transcript_section = (
        "Recent transcript:\n"
        f"{_format_recent_transcript(turn_input.transcript)}"
    )
    directives_section = _build_turn_directives(
        stage=turn_input.stage,
        stage_complete=stage_complete,
        final_notice_expiry=final_notice_expiry,
    )
    if turn_input.stage in (PipelineStage.RESOLUTION, PipelineStage.FINAL_NOTICE):
        directives_section += f" {offer_policy_directive()}"
        directives_section += f" {allowed_consequences_directive()}"

    handoff_section = ""
    if turn_input.completed_stages:
        handoff_section = build_handoff_summary(
            turn_input.completed_stages,
            turn_input.borrower.model_dump(mode="json"),
            [msg.model_dump(mode="json") for msg in turn_input.transcript],
            target_stage=turn_input.stage.value,
        )
        report.handoff_tokens = report.add("handoff", handoff_section)

    report.add("snapshot", snapshot_section)
    report.add("turn_meta", turn_meta_section)
    report.add("transcript", transcript_section)
    report.add("directives", directives_section)
    report.pre_overflow_tokens = report.total_tokens

    if report.total_tokens > MAX_CONTEXT_TOKENS:
        report.overflow_detected = True
        logger.info(
            "Context overflow detected: %d > %d tokens, attempting compression",
            report.total_tokens, MAX_CONTEXT_TOKENS,
        )

        system_tokens = count_tokens(system_prompt)
        available = MAX_CONTEXT_TOKENS - system_tokens
        directives_tokens = count_tokens(directives_section)
        turn_meta_tokens = count_tokens(turn_meta_section)
        reserved = directives_tokens + turn_meta_tokens
        compressible_budget = max(50, available - reserved)

        compressible_content = ""
        if handoff_section:
            compressible_content += f"Prior stage context:\n{handoff_section}\n\n"
        compressible_content += f"{snapshot_section}\n\n{transcript_section}"

        client = _get_anthropic_client()
        compressed, used_llm, used_fallback = await _compress_overflow(
            client, turn_input.stage, compressible_content, compressible_budget,
        )
        report.overflow_summary_used = used_llm
        report.overflow_fallback_used = used_fallback
        logger.info(
            "overflow_complete  stage=%s  turn=%d  "
            "pre_tokens=%d  compressed_tokens=%d  "
            "used_llm=%s  used_fallback=%s",
            turn_input.stage.value, turn_input.turn_index,
            report.pre_overflow_tokens, count_tokens(compressed),
            used_llm, used_fallback,
        )

        user_prompt = (
            "Continue the debt-collection conversation for the current stage.\n\n"
            f"{compressed}\n\n"
            f"{turn_meta_section}\n\n"
            f"{directives_section}"
        )

        report.sections.clear()
        report.add("system_prompt", system_prompt)
        report.add("compressed_context", compressed)
        report.add("turn_meta", turn_meta_section)
        report.add("directives", directives_section)
        report.post_overflow_tokens = report.total_tokens
    else:
        user_prompt_parts = ["Continue the debt-collection conversation for the current stage.\n"]
        if handoff_section:
            user_prompt_parts.append(f"Prior stage context:\n{handoff_section}\n")
        user_prompt_parts.extend([
            f"{snapshot_section}\n",
            f"{turn_meta_section}\n",
            f"{transcript_section}\n",
            directives_section,
        ])
        user_prompt = "\n".join(user_prompt_parts)
        report.post_overflow_tokens = report.total_tokens

    system_prompt, user_prompt = enforce_context_budget(
        system_prompt, user_prompt,
    )

    llm_result = await _get_anthropic_client().generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    assistant_reply = _prepend_assessment_opening_disclosure(
        llm_result.text,
        stage=turn_input.stage,
        turn_index=turn_input.turn_index,
    )

    # --- Compliance post-checks on assistant output ---
    assistant_reply, output_pii_redacted = redact_pii(assistant_reply)
    if output_pii_redacted:
        logger.warning(
            "compliance_output_pii_redacted  stage=%s  turn=%d",
            turn_input.stage.value, turn_input.turn_index,
        )

    false_threats = check_false_threats(assistant_reply)
    if false_threats:
        logger.warning(
            "compliance_false_threats_detected  stage=%s  turn=%d  threats=%s",
            turn_input.stage.value, turn_input.turn_index, false_threats,
        )

    offer_violations = check_offer_bounds(assistant_reply)
    if offer_violations:
        logger.warning(
            "compliance_offer_violations  stage=%s  turn=%d  violations=%s",
            turn_input.stage.value, turn_input.turn_index, offer_violations,
        )

    compliance_metadata: dict[str, Any] = {}
    if output_pii_redacted:
        compliance_metadata["compliance_output_pii_redacted"] = True
    if false_threats:
        compliance_metadata["compliance_false_threats"] = false_threats
    if offer_violations:
        compliance_metadata["compliance_offer_violations"] = offer_violations
    if flags.hardship_detected:
        compliance_metadata["compliance_hardship_detected"] = True

    budget_metadata = report.to_metadata()
    logger.info(
        "stage_turn_budget  stage=%s  turn=%d  %s",
        turn_input.stage.value, turn_input.turn_index, budget_metadata,
    )

    turn_output = StageTurnOutput(
        stage=turn_input.stage,
        channel=channel,
        assistant_reply=assistant_reply,
        summary=_trim_summary(assistant_reply),
        decision=decision,
        stage_complete=stage_complete,
        collected_fields=updated_fields,
        transition_reason=transition_reason,
        next_stage=next_stage if stage_complete else turn_input.stage,
        compliance_flags=flags,
        metadata={
            "model": llm_result.model,
            "used_fallback": llm_result.used_fallback,
            "turn_index": turn_input.turn_index,
            **budget_metadata,
            **compliance_metadata,
        },
    )
    logger.info(
        "stage_turn_end  stage=%s  turn=%d  complete=%s  decision=%s  model=%s",
        turn_input.stage.value, turn_input.turn_index, stage_complete,
        decision, llm_result.model,
    )
    return turn_output.model_dump(mode="json")


@activity.defn(name="assessment_agent")
async def assessment_agent(payload: dict[str, Any]) -> dict[str, Any]:
    return await _run_stage_turn(payload)


@activity.defn(name="resolution_agent")
async def resolution_agent(payload: dict[str, Any]) -> dict[str, Any]:
    return await _run_stage_turn(payload)


@activity.defn(name="final_notice_agent")
async def final_notice_agent(payload: dict[str, Any]) -> dict[str, Any]:
    return await _run_stage_turn(payload)
