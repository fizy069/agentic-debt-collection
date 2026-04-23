"""Conversation runner -- drives full multi-stage conversations for evaluation.

Mirrors the ``BorrowerWorkflow`` logic but calls ``_run_stage_turn``
directly, bypassing Temporal.  This lets us evaluate the *exact* same
code path without requiring a running Temporal worker or server.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.eval.borrower_sim import BorrowerSimulator
from app.eval.models import (
    ConversationRecord,
    Scenario,
    StageRecord,
    TurnRecord,
)
from app.models.pipeline import (
    BorrowerRequest,
    ComplianceFlags,
    ConversationMessage,
    ConversationRole,
    PipelineStage,
    StageTurnInput,
    StageTurnOutput,
)
from app.activities.agents import _run_stage_turn

logger = logging.getLogger(__name__)

_STAGE_ORDER = [
    PipelineStage.ASSESSMENT,
    PipelineStage.RESOLUTION,
    PipelineStage.FINAL_NOTICE,
]


def _scenario_to_borrower_request(scenario: Scenario) -> BorrowerRequest:
    return BorrowerRequest(
        borrower_id=scenario.borrower_id,
        account_reference=scenario.account_reference,
        debt_amount=scenario.debt_amount,
        currency=scenario.currency,
        days_past_due=scenario.days_past_due,
        borrower_message=scenario.borrower_message,
        notes=scenario.notes,
    )


async def run_conversation(
    scenario: Scenario,
    simulator: BorrowerSimulator,
) -> ConversationRecord:
    """Execute a full 3-stage conversation and return the trace.

    The runner replicates the state management of ``BorrowerWorkflow``
    (transcript, collected fields, compliance flags, completed stages)
    so that ``_run_stage_turn`` sees identical inputs to production.
    """
    borrower = _scenario_to_borrower_request(scenario)
    transcript: list[ConversationMessage] = []
    completed_stages: list[dict[str, Any]] = []
    compliance_flags = ComplianceFlags()
    stage_collected_fields: dict[str, dict[str, bool]] = {}

    record = ConversationRecord(scenario=scenario)
    conversation_history: list[dict[str, str]] = []

    borrower_message = scenario.borrower_message

    for stage in _STAGE_ORDER:
        stage_key = stage.value
        stage_collected_fields.setdefault(stage_key, {})
        stage_rec = StageRecord(stage=stage_key)
        turn_index = 0

        while True:
            turn_index += 1
            t0 = time.monotonic()

            transcript.append(
                ConversationMessage(
                    role=ConversationRole.BORROWER,
                    stage=stage,
                    text=borrower_message,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            conversation_history.append({"role": "borrower", "text": borrower_message})
            record.all_borrower_messages.append(borrower_message)

            stage_input = StageTurnInput(
                borrower=borrower,
                stage=stage,
                borrower_message=borrower_message,
                transcript=list(transcript),
                collected_fields=stage_collected_fields[stage_key],
                turn_index=turn_index,
                completed_stages=completed_stages,
                compliance_flags=compliance_flags,
            )

            raw_output = await _run_stage_turn(stage_input.model_dump(mode="json"))
            turn_output = StageTurnOutput.model_validate(raw_output)

            elapsed_ms = (time.monotonic() - t0) * 1000

            compliance_flags = turn_output.compliance_flags
            stage_collected_fields[stage_key] = turn_output.collected_fields

            transcript.append(
                ConversationMessage(
                    role=ConversationRole.AGENT,
                    stage=stage,
                    text=turn_output.assistant_reply,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            conversation_history.append({"role": "agent", "text": turn_output.assistant_reply})
            record.all_agent_replies.append(turn_output.assistant_reply)

            turn_rec = TurnRecord(
                stage=stage_key,
                turn_index=turn_index,
                borrower_message=borrower_message,
                agent_reply=turn_output.assistant_reply,
                stage_complete=turn_output.stage_complete,
                collected_fields=turn_output.collected_fields,
                decision=turn_output.decision,
                transition_reason=turn_output.transition_reason,
                metadata=turn_output.metadata,
                elapsed_ms=elapsed_ms,
            )
            stage_rec.turns.append(turn_rec)
            record.total_turns += 1

            if compliance_flags.any_terminal():
                terminal_reason = (
                    "compliance_stop_contact"
                    if compliance_flags.stop_contact_requested
                    else "compliance_abusive_close"
                )
                logger.info(
                    "eval_conversation_terminated  scenario=%s  stage=%s  reason=%s",
                    scenario.scenario_id, stage_key, terminal_reason,
                )
                stage_rec.completed = True
                stage_rec.transition_reason = terminal_reason
                stage_rec.collected_fields = turn_output.collected_fields
                record.stages.append(stage_rec)
                record.early_termination = True
                record.termination_reason = terminal_reason
                record.final_outcome = terminal_reason
                return record

            if turn_output.stage_complete:
                logger.info(
                    "eval_stage_complete  scenario=%s  stage=%s  turns=%d  reason=%s",
                    scenario.scenario_id, stage_key, turn_index, turn_output.transition_reason,
                )
                completed_stages.append({
                    "stage": stage_key,
                    "collected_fields": turn_output.collected_fields,
                    "transition_reason": turn_output.transition_reason,
                    "turns": turn_index,
                })
                stage_rec.completed = True
                stage_rec.transition_reason = turn_output.transition_reason
                stage_rec.collected_fields = turn_output.collected_fields
                break

            next_borrower = await simulator.generate_reply(
                persona=scenario.persona,
                conversation_history=conversation_history,
                stage=stage_key,
                turn_index=turn_index + 1,
            )
            borrower_message = next_borrower

        record.stages.append(stage_rec)

        if stage != _STAGE_ORDER[-1]:
            next_borrower = await simulator.generate_reply(
                persona=scenario.persona,
                conversation_history=conversation_history,
                stage=_STAGE_ORDER[_STAGE_ORDER.index(stage) + 1].value,
                turn_index=1,
            )
            borrower_message = next_borrower

    record.final_outcome = "final_notice_issued"
    logger.info(
        "eval_conversation_complete  scenario=%s  total_turns=%d",
        scenario.scenario_id, record.total_turns,
    )
    return record
