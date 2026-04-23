from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from temporalio import activity

logger = logging.getLogger(__name__)

from app.models.pipeline import (
    AgentChannel,
    BorrowerRequest,
    ComplianceFlags,
    ConversationMessage,
    LLMStageResponse,
    PipelineStage,
    StageTurnInput,
    StageTurnOutput,
)
from app.services.anthropic_client import AnthropicClient
from app.services.compliance import (
    ABUSIVE_CLOSE_REPLY,
    STOP_CONTACT_REPLY,
    check_false_threats,
    check_offer_bounds,
    detect_abusive,
    detect_hardship,
    detect_stop_contact,
    redact_pii,
)
from app.services.compliance_judge import (
    findings_to_metadata,
    is_judge_enabled,
    run_judge,
)
from app.services.compliance_vector_store import (
    query_similar_violations,
    upsert_violations,
    vector_hits_to_metadata,
)
from app.services.handoff import build_handoff_summary
from app.services.prompt_assembler import (
    _format_recent_transcript,
    _render_compliance_directives,
    _render_turn_directives,
    assemble_agent_prompt,
    assemble_overflow_user_prompt,
)
from app.services.prompt_registry import get_prompt_registry
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


_TURN_CAPS: dict[PipelineStage, int] = {
    PipelineStage.ASSESSMENT: 3,
    PipelineStage.RESOLUTION: 3,
    PipelineStage.FINAL_NOTICE: 2,
}

_STAGE_FIELD_KEYS: dict[PipelineStage, list[str]] = {
    PipelineStage.ASSESSMENT: [
        "identity_confirmed",
        "debt_acknowledged",
        "ability_to_pay_discussed",
    ],
    PipelineStage.RESOLUTION: [
        "options_reviewed",
        "borrower_position_known",
        "commitment_or_disposition",
    ],
    PipelineStage.FINAL_NOTICE: [
        "final_notice_acknowledged",
        "borrower_response_recorded",
    ],
}


def _parse_stage_response(
    raw: str,
    *,
    stage: PipelineStage,
    collected_fields: dict[str, bool],
) -> LLMStageResponse:
    """Parse the LLM's structured JSON response with safe fallback.

    On any parse failure, returns a safe default that keeps the stage open
    and uses the raw text as the assistant reply.
    """
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else 3
        cleaned = cleaned[first_newline:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        return LLMStageResponse.model_validate(data)
    except (json.JSONDecodeError, ValueError, Exception) as exc:
        logger.warning(
            "stage_response_parse_failed  stage=%s  error=%s  raw=%s",
            stage.value, exc, raw[:300],
        )
        field_keys = _STAGE_FIELD_KEYS.get(stage, [])
        fallback_fields = {k: collected_fields.get(k, False) for k in field_keys}
        return LLMStageResponse(
            assistant_reply=raw,
            stage_complete=False,
            collected_fields=fallback_fields,
            transition_reason="llm_response_parse_failed",
            decision=f"{stage.value}_follow_up",
        )


def _borrower_snapshot_for_overflow(
    borrower: BorrowerRequest,
    *,
    identity_confirmed: bool = False,
    stage: PipelineStage | None = None,
) -> str:
    """Rebuild borrower snapshot for overflow compression input."""
    pre_verification = stage == PipelineStage.ASSESSMENT and not identity_confirmed
    masked_reference = borrower.account_reference[-4:]

    if pre_verification:
        header = "SYSTEM ACCOUNT RECORD (UNVERIFIED):"
    else:
        header = "VERIFIED BORROWER ACCOUNT:"

    return (
        f"{header}\n"
        f"Borrower ID: {borrower.borrower_id}\n"
        f"Account Ref (last4): {masked_reference}\n"
        f"Debt: {borrower.debt_amount:.2f} {borrower.currency}\n"
        f"Days past due: {borrower.days_past_due}\n"
        f"Borrower message: {borrower.borrower_message}\n"
        f"Notes: {borrower.notes or 'None provided'}"
    )


def _stage_routing(
    stage: PipelineStage,
) -> tuple[AgentChannel, PipelineStage | None]:
    """Return (channel, next_stage) for the given pipeline stage."""
    if stage == PipelineStage.ASSESSMENT:
        return AgentChannel.CHAT, PipelineStage.RESOLUTION
    if stage == PipelineStage.RESOLUTION:
        return AgentChannel.VOICE_STUB, PipelineStage.FINAL_NOTICE
    return AgentChannel.CHAT, None


def _final_notice_expiry_date(
    *,
    base_date: date | None = None,
    window_days: int = 7,
) -> str:
    anchor = base_date or datetime.now(timezone.utc).date()
    return (anchor + timedelta(days=window_days)).isoformat()


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
        channel, next_stage = _stage_routing(turn_input.stage)
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
    channel, next_stage = _stage_routing(turn_input.stage)

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

    channel, next_stage = _stage_routing(turn_input.stage)

    field_keys = _STAGE_FIELD_KEYS.get(turn_input.stage, [])
    prior_fields = {k: turn_input.collected_fields.get(k, False) for k in field_keys}

    final_notice_expiry = (
        _final_notice_expiry_date()
        if turn_input.stage == PipelineStage.FINAL_NOTICE
        else None
    )

    # --- Assemble prompt via registry + assembler ---
    registry = get_prompt_registry()
    config = registry.get_agent_config(turn_input.stage)
    assembled = assemble_agent_prompt(
        config,
        borrower=turn_input.borrower,
        borrower_message=borrower_message,
        stage=turn_input.stage,
        turn_index=turn_input.turn_index,
        prior_fields=prior_fields,
        transcript=turn_input.transcript,
        completed_stages=turn_input.completed_stages,
        flags=flags,
        final_notice_expiry=final_notice_expiry,
    )
    system_prompt = assembled.system_prompt
    user_prompt = assembled.user_prompt

    report = ContextBudgetReport(limit=MAX_CONTEXT_TOKENS)
    report.add("system_prompt", system_prompt)
    report.add("user_prompt", user_prompt)
    report.pre_overflow_tokens = report.total_tokens

    if report.total_tokens > MAX_CONTEXT_TOKENS:
        report.overflow_detected = True
        logger.info(
            "Context overflow detected: %d > %d tokens, attempting compression",
            report.total_tokens, MAX_CONTEXT_TOKENS,
        )

        st = assembled.metadata.get("section_tokens", {})
        system_tokens = st.get("system_prompt", count_tokens(system_prompt))
        directives_tokens = st.get("directives", 0)
        turn_meta_tokens = st.get("turn_meta", 0)
        available = MAX_CONTEXT_TOKENS - system_tokens
        reserved = directives_tokens + turn_meta_tokens
        compressible_budget = max(50, available - reserved)

        snapshot_section = (
            f"Borrower snapshot:\n"
            f"{_borrower_snapshot_for_overflow(turn_input.borrower, identity_confirmed=prior_fields.get('identity_confirmed', False), stage=turn_input.stage)}"
        )
        transcript_section = (
            "Recent transcript:\n"
            f"{_format_recent_transcript(turn_input.transcript)}"
        )
        handoff_section = ""
        if turn_input.completed_stages:
            handoff_section = build_handoff_summary(
                turn_input.completed_stages,
                turn_input.borrower.model_dump(mode="json"),
                [msg.model_dump(mode="json") for msg in turn_input.transcript],
                target_stage=turn_input.stage.value,
            )

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

        turn_meta_lines = [
            f"Current stage: {turn_input.stage.value}",
            f"Turn index in stage: {turn_input.turn_index}",
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

        directives_section = _render_turn_directives(config, final_notice_expiry)
        compliance_dir = _render_compliance_directives(config)
        if compliance_dir:
            directives_section += f" {compliance_dir}"

        user_prompt = assemble_overflow_user_prompt(
            compressed=compressed,
            turn_meta_section=turn_meta_section,
            directives_section=directives_section,
        )

        report.sections.clear()
        report.add("system_prompt", system_prompt)
        report.add("compressed_context", compressed)
        report.add("turn_meta", turn_meta_section)
        report.add("directives", directives_section)
        report.post_overflow_tokens = report.total_tokens
    else:
        report.post_overflow_tokens = report.total_tokens

    system_prompt, user_prompt = enforce_context_budget(
        system_prompt, user_prompt,
    )

    llm_result = await _get_anthropic_client().generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=500,
    )

    # --- Parse structured LLM response ---
    parsed = _parse_stage_response(
        llm_result.text,
        stage=turn_input.stage,
        collected_fields=prior_fields,
    )

    assistant_reply = _prepend_assessment_opening_disclosure(
        parsed.assistant_reply,
        stage=turn_input.stage,
        turn_index=turn_input.turn_index,
    )

    updated_fields = parsed.collected_fields
    stage_complete = parsed.stage_complete
    transition_reason = parsed.transition_reason
    decision = parsed.decision

    # --- Turn-cap safety net: force completion at the hard limit ---
    turn_cap = _TURN_CAPS.get(turn_input.stage, 3)
    if not stage_complete and turn_input.turn_index >= turn_cap:
        stage_complete = True
        transition_reason = f"{turn_input.stage.value}_max_turns_reached"
        decision = parsed.decision.replace("_follow_up", "_completed")
        logger.info(
            "turn_cap_override  stage=%s  turn=%d  cap=%d",
            turn_input.stage.value, turn_input.turn_index, turn_cap,
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

    # --- Vector DB lookup (Layer 1 reads historical violations) ---
    vector_meta: dict[str, Any] = {}
    try:
        vector_result = query_similar_violations(
            assistant_reply,
            stage=turn_input.stage.value,
        )
        vector_meta = vector_hits_to_metadata(vector_result)
        if vector_result.records:
            logger.info(
                "vector_lookup_hits  stage=%s  turn=%d  count=%d",
                turn_input.stage.value, turn_input.turn_index,
                len(vector_result.records),
            )
    except Exception:
        logger.exception(
            "vector_lookup_error  stage=%s  turn=%d",
            turn_input.stage.value, turn_input.turn_index,
        )

    # --- Layer 2 judge (audit-only) ---
    judge_meta: dict[str, Any] = {}
    if is_judge_enabled():
        transcript_excerpt = _format_recent_transcript(
            turn_input.transcript, max_items=4,
        )
        findings = await run_judge(
            stage=turn_input.stage.value,
            turn_index=turn_input.turn_index,
            assistant_reply=assistant_reply,
            transcript_excerpt=transcript_excerpt,
        )
        judge_meta = findings_to_metadata(findings)

        if findings.violations:
            upsert_violations(
                workflow_id=turn_input.borrower.borrower_id,
                stage=turn_input.stage.value,
                turn_index=turn_input.turn_index,
                violations=[
                    {
                        "rule": v.rule,
                        "label": v.label,
                        "confidence": v.confidence,
                        "excerpt": v.excerpt,
                    }
                    for v in findings.violations
                ],
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
            **vector_meta,
            **judge_meta,
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
