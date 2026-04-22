from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

from app.models.pipeline import (
    AgentStageOutput,
    BorrowerMessageRequest,
    BorrowerRequest,
    ComplianceFlags,
    ConversationMessage,
    ConversationRole,
    PipelineStage,
    PipelineStatus,
    ResolutionVoiceCallCreateInput,
    ResolutionVoiceCallCreateOutput,
    ResolutionVoiceEvent,
    ResolutionVoiceFinalizeInput,
    ResolutionVoiceSession,
    ResolutionVoiceSettings,
    StageTurnInput,
    StageTurnOutput,
    VoiceMode,
)


@workflow.defn
class BorrowerWorkflow:
    def __init__(self) -> None:
        self._status: dict[str, Any] = self._new_status("pending")
        self._pending_messages: list[dict[str, Any]] = []
        self._seen_message_ids: dict[str, bool] = {}
        self._completed_stages: list[dict[str, Any]] = []
        self._resolution_voice_event: dict[str, Any] | None = None

    def _new_status(self, workflow_id: str) -> dict[str, Any]:
        return PipelineStatus(
            workflow_id=workflow_id,
            current_stage=PipelineStage.QUEUED,
        ).model_dump(mode="json")

    def _set_stage(self, stage: PipelineStage, *, completed: bool = False) -> None:
        self._status["current_stage"] = stage.value
        self._status["completed"] = completed

    def _voice_mode(self) -> VoiceMode:
        raw = self._status.get("voice_mode", VoiceMode.STUB.value)
        try:
            return VoiceMode(str(raw))
        except ValueError:
            return VoiceMode.STUB

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

        if (
            self._voice_mode() == VoiceMode.VAPI
            and self._status.get("current_stage") == PipelineStage.RESOLUTION.value
        ):
            workflow.logger.info(
                "Ignoring borrower text while Resolution voice call is active."
            )
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

    @workflow.signal
    def update_resolution_voice_event(self, payload: dict[str, Any]) -> None:
        event = ResolutionVoiceEvent.model_validate(payload)
        session_payload = self._status.get("resolution_voice_session")
        session = (
            ResolutionVoiceSession.model_validate(session_payload)
            if session_payload
            else ResolutionVoiceSession(mode=VoiceMode.VAPI, provider=event.provider)
        )

        if event.call_id and session.callId and event.call_id != session.callId:
            workflow.logger.warning(
                "Ignoring voice event for unexpected call id  expected=%s  got=%s",
                session.callId,
                event.call_id,
            )
            return

        if event.call_id:
            session.callId = event.call_id
        if event.call_status:
            session.callStatus = event.call_status
        if event.ended_reason:
            session.endedReason = event.ended_reason
        if event.final_artifact_received:
            session.finalArtifactReceived = True
        if event.metadata:
            merged_metadata = dict(session.metadata)
            merged_metadata.update(event.metadata)
            session.metadata = merged_metadata
        self._status["resolution_voice_session"] = session.model_dump(mode="json")

        event_payload = event.model_dump(mode="json")
        if (
            event.event_type == "end-of-call-report"
            or event.final_artifact_received
            or event.call_status == "ended"
        ):
            self._resolution_voice_event = event_payload

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

    def _record_turn_output(
        self,
        *,
        stage: PipelineStage,
        turn_output: StageTurnOutput,
        turn_index: int,
        outputs: list[dict[str, Any]],
    ) -> ComplianceFlags:
        stage_key = stage.value
        compliance_flags = turn_output.compliance_flags
        self._status["compliance_flags"] = compliance_flags.model_dump(mode="json")

        if turn_output.metadata.get("borrower_message_oversized") and self._status["transcript"]:
            self._status["transcript"][-1]["text"] = (
                "[oversized message — borrower asked to reply concisely]"
            )

        self._status["stage_turn_counts"][stage_key] = turn_index
        self._status["stage_collected_fields"][stage_key] = turn_output.collected_fields
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
        return compliance_flags

    def _handle_terminal_compliance(
        self,
        *,
        compliance_flags: ComplianceFlags,
        stage_key: str,
    ) -> bool:
        if not compliance_flags.any_terminal():
            return False
        terminal_reason = (
            "stop_contact"
            if compliance_flags.stop_contact_requested
            else "abusive_close"
        )
        workflow.logger.info(
            "Pipeline terminated by compliance: %s at stage %s",
            terminal_reason,
            stage_key,
        )
        self._status["final_outcome"] = f"compliance_{terminal_reason}"
        self._set_stage(PipelineStage.COMPLETED, completed=True)
        return True

    async def _run_resolution_voice_stage(
        self,
        *,
        borrower: BorrowerRequest,
        voice_settings: ResolutionVoiceSettings,
        compliance_flags: ComplianceFlags,
        retry_policy: RetryPolicy,
    ) -> tuple[StageTurnOutput, int]:
        stage = PipelineStage.RESOLUTION
        stage_key = stage.value
        turn_index = int(self._status["stage_turn_counts"][stage_key]) + 1

        self._resolution_voice_event = None

        create_input = ResolutionVoiceCallCreateInput(
            workflow_id=workflow.info().workflow_id,
            borrower=borrower,
            stage=stage,
            transcript=[
                ConversationMessage.model_validate(item)
                for item in self._status["transcript"]
            ],
            completed_stages=self._completed_stages,
            voice_settings=voice_settings,
        )
        create_payload = await workflow.execute_activity(
            "create_resolution_voice_call",
            create_input.model_dump(mode="json"),
            start_to_close_timeout=timedelta(seconds=60),
            retry_policy=retry_policy,
        )
        create_output = ResolutionVoiceCallCreateOutput.model_validate(create_payload)

        self._status["resolution_voice_session"] = ResolutionVoiceSession(
            mode=voice_settings.mode,
            provider=create_output.provider,
            callId=create_output.call_id,
            webCallUrl=create_output.web_call_url,
            callStatus=create_output.status or "queued",
            metadata=create_output.metadata,
        ).model_dump(mode="json")
        self._status["latest_assistant_reply"] = (
            "Resolution voice call started. Join the call to continue."
        )

        try:
            await workflow.wait_condition(
                lambda: self._resolution_voice_event is not None,
                timeout=timedelta(minutes=45),
                timeout_summary="wait-resolution-voice-completion",
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                "Resolution voice call did not complete before timeout."
            ) from exc

        event_payload = self._resolution_voice_event or {}
        voice_event = ResolutionVoiceEvent.model_validate(event_payload)
        current_session = ResolutionVoiceSession.model_validate(
            self._status.get("resolution_voice_session")
            or ResolutionVoiceSession(mode=voice_settings.mode).model_dump(mode="json")
        )
        if voice_event.call_status:
            current_session.callStatus = voice_event.call_status
        if voice_event.ended_reason:
            current_session.endedReason = voice_event.ended_reason
        if voice_event.final_artifact_received or voice_event.event_type == "end-of-call-report":
            current_session.finalArtifactReceived = True
        self._status["resolution_voice_session"] = current_session.model_dump(mode="json")

        finalize_input = ResolutionVoiceFinalizeInput(
            borrower=borrower,
            transcript=[
                ConversationMessage.model_validate(item)
                for item in self._status["transcript"]
            ],
            completed_stages=self._completed_stages,
            collected_fields=self._status["stage_collected_fields"][stage_key],
            turn_index=turn_index,
            compliance_flags=compliance_flags,
            call_id=current_session.callId,
            call_status=current_session.callStatus,
            ended_reason=current_session.endedReason,
            artifact=voice_event.artifact,
            messages=voice_event.messages,
            metadata=voice_event.metadata,
        )
        turn_output_payload = await workflow.execute_activity(
            "finalize_resolution_voice_call",
            finalize_input.model_dump(mode="json"),
            start_to_close_timeout=timedelta(seconds=120),
            retry_policy=retry_policy,
        )
        turn_output = StageTurnOutput.model_validate(turn_output_payload)
        borrower_excerpt = turn_output.metadata.get("voice_transcript_excerpt")
        if isinstance(borrower_excerpt, str) and borrower_excerpt.strip():
            self._append_transcript(
                role=ConversationRole.BORROWER,
                stage=PipelineStage.RESOLUTION,
                text=borrower_excerpt,
                message_id=f"{current_session.callId or workflow.info().workflow_id}-voice",
            )
        return turn_output, turn_index

    @workflow.run
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        borrower = BorrowerRequest.model_validate(payload["borrower"])
        voice_settings = ResolutionVoiceSettings.model_validate(
            payload.get("voice_settings", {})
        )
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
        self._resolution_voice_event = None
        self._status["voice_mode"] = voice_settings.mode.value
        if voice_settings.mode == VoiceMode.VAPI:
            self._status["resolution_voice_session"] = ResolutionVoiceSession(
                mode=voice_settings.mode,
                provider=voice_settings.provider,
                metadata={
                    "webhook_base_url": voice_settings.webhook_base_url,
                },
            ).model_dump(mode="json")
        elif voice_settings.fallback_reason:
            self._status["resolution_voice_session"] = ResolutionVoiceSession(
                mode=voice_settings.mode,
                provider=voice_settings.provider,
                metadata={"fallback_reason": voice_settings.fallback_reason},
            ).model_dump(mode="json")

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

                if stage == PipelineStage.RESOLUTION and voice_settings.mode == VoiceMode.VAPI:
                    turn_output, turn_index = await self._run_resolution_voice_stage(
                        borrower=borrower,
                        voice_settings=voice_settings,
                        compliance_flags=compliance_flags,
                        retry_policy=retry_policy,
                    )
                    compliance_flags = self._record_turn_output(
                        stage=stage,
                        turn_output=turn_output,
                        turn_index=turn_index,
                        outputs=outputs,
                    )
                    if self._handle_terminal_compliance(
                        compliance_flags=compliance_flags,
                        stage_key=stage_key,
                    ):
                        return self._status
                    if not turn_output.stage_complete:
                        raise RuntimeError(
                            "Resolution voice path must finalize the stage before proceeding."
                        )
                    self._completed_stages.append({
                        "stage": stage_key,
                        "collected_fields": turn_output.collected_fields,
                        "transition_reason": turn_output.transition_reason,
                        "turns": turn_index,
                    })
                    continue

                while True:
                    borrower_turn = await self._wait_and_pop_borrower_message(stage)
                    borrower_message = borrower_turn["message"]
                    message_id = borrower_turn.get("message_id")

                    turn_index = int(self._status["stage_turn_counts"][stage_key]) + 1
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

                    compliance_flags = self._record_turn_output(
                        stage=stage,
                        turn_output=turn_output,
                        turn_index=turn_index,
                        outputs=outputs,
                    )

                    if self._handle_terminal_compliance(
                        compliance_flags=compliance_flags,
                        stage_key=stage_key,
                    ):
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
