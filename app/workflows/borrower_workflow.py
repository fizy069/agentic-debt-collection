from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

from app.models.pipeline import (
    STAGE_OPENER_SENTINEL,
    AgentStageOutput,
    BorrowerMessageRequest,
    BorrowerRequest,
    ComplianceFlags,
    ConversationMessage,
    ConversationRole,
    PipelineStage,
    PipelineStatus,
    StageTurnInput,
    StageTurnOutput,
    stage_is_agent_initiated,
)


@workflow.defn
class BorrowerWorkflow:
    def __init__(self) -> None:
        self._status: dict[str, Any] = self._new_status("pending")
        self._pending_messages: list[dict[str, Any]] = []
        self._seen_message_ids: dict[str, bool] = {}
        self._completed_stages: list[dict[str, Any]] = []

    def _new_status(self, workflow_id: str) -> dict[str, Any]:
        return PipelineStatus(
            workflow_id=workflow_id,
            current_stage=PipelineStage.QUEUED,
        ).model_dump(mode="json")

    def _set_stage(self, stage: PipelineStage, *, completed: bool = False) -> None:
        self._status["current_stage"] = stage.value
        self._status["completed"] = completed

    def _append_transcript(
        self,
        *,
        role: ConversationRole,
        stage: PipelineStage | None,
        text: str,
        message_id: str | None = None,
    ) -> None:
        message = ConversationMessage(
            role=role,
            stage=stage,
            text=text,
            timestamp=workflow.now().isoformat(),
            message_id=message_id,
        )
        self._status["transcript"].append(message.model_dump(mode="json"))

    @workflow.query
    def get_status(self) -> dict[str, Any]:
        return self._status

    @workflow.signal
    def add_borrower_message(self, payload: dict[str, Any]) -> None:
        message = BorrowerMessageRequest.model_validate(payload)

        if message.message_id and self._seen_message_ids.get(message.message_id):
            return

        if message.message_id:
            self._seen_message_ids[message.message_id] = True

        self._pending_messages.append(
            {
                "message": message.message,
                "message_id": message.message_id,
            }
        )
        self._status["pending_messages"] = len(self._pending_messages)

    def _activity_name_for_stage(self, stage: PipelineStage) -> str:
        mapping = {
            PipelineStage.ASSESSMENT: "assessment_agent",
            PipelineStage.RESOLUTION: "resolution_agent",
            PipelineStage.FINAL_NOTICE: "final_notice_agent",
        }
        return mapping[stage]

    async def _wait_and_pop_borrower_message(self, stage: PipelineStage) -> dict[str, Any]:
        try:
            await workflow.wait_condition(
                lambda: len(self._pending_messages) > 0,
                timeout=timedelta(minutes=20),
                timeout_summary=f"wait-borrower-message-{stage.value}",
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"No borrower message received during {stage.value} stage."
            ) from exc

        message = self._pending_messages.pop(0)
        self._status["pending_messages"] = len(self._pending_messages)
        return message

    @workflow.run
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        borrower = BorrowerRequest.model_validate(payload["borrower"])
        workflow_id = workflow.info().workflow_id
        outputs: list[dict[str, Any]] = []
        stage_order = [
            PipelineStage.ASSESSMENT,
            PipelineStage.RESOLUTION,
            PipelineStage.FINAL_NOTICE,
        ]

        self._status = self._new_status(workflow_id)
        self._pending_messages = []
        self._seen_message_ids = {}
        self._completed_stages = []

        # Seed first turn from borrower request.
        self.add_borrower_message(
            {
                "message": borrower.borrower_message,
                "message_id": f"{workflow_id}-initial",
            }
        )

        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_attempts=3,
        )
        compliance_flags = ComplianceFlags()

        try:
            for stage in stage_order:
                stage_key = stage.value
                workflow.logger.info("Entering stage %s", stage_key)
                self._set_stage(stage)
                self._status["stage_collected_fields"].setdefault(stage_key, {})
                self._status["stage_turn_counts"].setdefault(stage_key, 0)

                while True:
                    turn_index = int(self._status["stage_turn_counts"][stage_key]) + 1
                    agent_initiated_opener = (
                        stage_is_agent_initiated(stage) and turn_index == 1
                    )

                    if agent_initiated_opener:
                        # Outbound stage on turn 1: we "place the call" or
                        # "send the notice", so the agent speaks first. No
                        # borrower utterance exists yet; feed the activity a
                        # sentinel message and skip appending any borrower
                        # turn to the transcript.
                        borrower_message = STAGE_OPENER_SENTINEL
                        message_id = f"{workflow_id}-{stage_key}-opener"
                        workflow.logger.info(
                            "Opening stage %s with agent-initiated turn %d",
                            stage_key, turn_index,
                        )
                    else:
                        borrower_turn = await self._wait_and_pop_borrower_message(stage)
                        borrower_message = borrower_turn["message"]
                        message_id = borrower_turn.get("message_id")
                        workflow.logger.info(
                            "Executing turn %d for stage %s", turn_index, stage_key,
                        )
                        self._append_transcript(
                            role=ConversationRole.BORROWER,
                            stage=stage,
                            text=borrower_message,
                            message_id=message_id,
                        )

                    stage_input = StageTurnInput(
                        borrower=borrower,
                        stage=stage,
                        borrower_message=borrower_message,
                        transcript=[
                            ConversationMessage.model_validate(message)
                            for message in self._status["transcript"]
                        ],
                        collected_fields=self._status["stage_collected_fields"][stage_key],
                        turn_index=turn_index,
                        completed_stages=self._completed_stages,
                        compliance_flags=compliance_flags,
                    )

                    turn_output_payload = await workflow.execute_activity(
                        self._activity_name_for_stage(stage),
                        stage_input.model_dump(mode="json"),
                        start_to_close_timeout=timedelta(seconds=120),
                        retry_policy=retry_policy,
                    )
                    turn_output = StageTurnOutput.model_validate(turn_output_payload)

                    compliance_flags = turn_output.compliance_flags

                    self._status["compliance_flags"] = compliance_flags.model_dump(
                        mode="json",
                    )

                    if turn_output.metadata.get("borrower_message_oversized"):
                        self._status["transcript"][-1]["text"] = (
                            "[oversized message — borrower asked to reply concisely]"
                        )

                    self._status["stage_turn_counts"][stage_key] = turn_index
                    self._status["stage_collected_fields"][stage_key] = (
                        turn_output.collected_fields
                    )
                    self._status["latest_assistant_reply"] = turn_output.assistant_reply

                    stage_output = AgentStageOutput(
                        stage=turn_output.stage,
                        channel=turn_output.channel,
                        response_text=turn_output.assistant_reply,
                        summary=turn_output.summary,
                        decision=turn_output.decision,
                        stage_complete=turn_output.stage_complete,
                        collected_fields=turn_output.collected_fields,
                        transition_reason=turn_output.transition_reason,
                        next_stage=turn_output.next_stage,
                        metadata=turn_output.metadata,
                    )
                    outputs.append(stage_output.model_dump(mode="json"))
                    self._status["outputs"] = outputs

                    self._append_transcript(
                        role=ConversationRole.AGENT,
                        stage=stage,
                        text=turn_output.assistant_reply,
                    )

                    if compliance_flags.any_terminal():
                        terminal_reason = (
                            "stop_contact"
                            if compliance_flags.stop_contact_requested
                            else "abusive_close"
                        )
                        workflow.logger.info(
                            "Pipeline terminated by compliance: %s at stage %s",
                            terminal_reason, stage_key,
                        )
                        self._status["final_outcome"] = (
                            f"compliance_{terminal_reason}"
                        )
                        self._set_stage(PipelineStage.COMPLETED, completed=True)
                        return self._status

                    if turn_output.stage_complete:
                        workflow.logger.info(
                            "Stage %s complete  reason=%s  turns=%d",
                            stage_key, turn_output.transition_reason, turn_index,
                        )
                        self._completed_stages.append({
                            "stage": stage_key,
                            "collected_fields": turn_output.collected_fields,
                            "transition_reason": turn_output.transition_reason,
                            "turns": turn_index,
                        })
                        break

            self._status["final_outcome"] = "final_notice_issued"
            self._set_stage(PipelineStage.COMPLETED, completed=True)
            workflow.logger.info("Pipeline completed")
            return self._status
        except Exception as exc:
            workflow.logger.error("Pipeline failed: %s", exc)
            self._status["failed"] = True
            self._status["error"] = str(exc)
            self._set_stage(PipelineStage.FAILED, completed=True)
            raise
