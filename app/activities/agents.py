from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from temporalio import activity

from app.models.pipeline import (
    AgentChannel,
    BorrowerRequest,
    ConversationMessage,
    PipelineStage,
    StageTurnInput,
    StageTurnOutput,
)
from app.services.anthropic_client import AnthropicClient


@lru_cache(maxsize=1)
def _get_anthropic_client() -> AnthropicClient:
    # Ensure project .env is loaded before first client construction.
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)
    return AnthropicClient()


def _trim_summary(text: str, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return f"{compact[: max_chars - 3]}..."


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


def _stage_defaults(
    stage: PipelineStage,
) -> tuple[AgentChannel, PipelineStage | None, str]:
    if stage == PipelineStage.ASSESSMENT:
        return (
            AgentChannel.CHAT,
            PipelineStage.RESOLUTION,
            (
                "You are Agent 1 (Assessment). Keep a professional and clinical tone. "
                "Collect missing identity and ability-to-pay details. "
                "Do not negotiate terms in this stage."
            ),
        )
    if stage == PipelineStage.RESOLUTION:
        return (
            AgentChannel.VOICE_STUB,
            PipelineStage.FINAL_NOTICE,
            (
                "You are Agent 2 (Resolution) acting as voice-style text. "
                "Present policy-bounded options, handle objections, and seek commitment."
            ),
        )
    return (
        AgentChannel.CHAT,
        None,
        (
            "You are Agent 3 (Final Notice). "
            "Provide a clear final option with expiry and documented next steps."
        ),
    )


async def _run_stage_turn(payload: dict[str, Any]) -> dict[str, Any]:
    turn_input = StageTurnInput.model_validate(payload)
    channel, next_stage, system_prompt = _stage_defaults(turn_input.stage)

    updated_fields, stage_complete, transition_reason, decision = _evaluate_stage_turn(
        stage=turn_input.stage,
        borrower_message=turn_input.borrower_message,
        collected_fields=turn_input.collected_fields,
        turn_index=turn_input.turn_index,
    )

    missing_fields = [key for key, value in updated_fields.items() if not value]
    user_prompt = (
        "Continue the debt-collection conversation for the current stage.\n\n"
        f"Borrower snapshot:\n{_borrower_snapshot(turn_input.borrower)}\n\n"
        f"Current stage: {turn_input.stage.value}\n"
        f"Turn index in stage: {turn_input.turn_index}\n"
        f"Borrower message: {turn_input.borrower_message}\n"
        f"Collected fields: {updated_fields}\n"
        f"Missing fields: {missing_fields if missing_fields else 'none'}\n"
        f"Transition reason: {transition_reason}\n\n"
        "Recent transcript:\n"
        f"{_format_recent_transcript(turn_input.transcript)}\n\n"
        "Respond in 3-6 concise sentences. "
        "If stage is incomplete, ask one concrete follow-up question. "
        "If complete, provide a brief transition-ready response."
    )

    llm_result = await _get_anthropic_client().generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    turn_output = StageTurnOutput(
        stage=turn_input.stage,
        channel=channel,
        assistant_reply=llm_result.text,
        summary=_trim_summary(llm_result.text),
        decision=decision,
        stage_complete=stage_complete,
        collected_fields=updated_fields,
        transition_reason=transition_reason,
        next_stage=next_stage if stage_complete else turn_input.stage,
        metadata={
            "model": llm_result.model,
            "used_fallback": llm_result.used_fallback,
            "turn_index": turn_input.turn_index,
        },
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
