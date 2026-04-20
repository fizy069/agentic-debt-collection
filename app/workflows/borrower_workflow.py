from __future__ import annotations

from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy

from app.models.pipeline import BorrowerRequest, PipelineStage, PipelineStatus


@workflow.defn
class BorrowerWorkflow:
    def __init__(self) -> None:
        self._status: dict[str, Any] = PipelineStatus(
            workflow_id="pending",
            current_stage=PipelineStage.QUEUED,
        ).model_dump(mode="json")

    @workflow.query
    def get_status(self) -> dict[str, Any]:
        return self._status

    def _set_stage(self, stage: PipelineStage, *, completed: bool = False) -> None:
        self._status["current_stage"] = stage.value
        self._status["completed"] = completed

    @workflow.run
    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        borrower = BorrowerRequest.model_validate(payload["borrower"])
        workflow_id = workflow.info().workflow_id
        outputs: list[dict[str, Any]] = []

        self._status = PipelineStatus(
            workflow_id=workflow_id,
            current_stage=PipelineStage.QUEUED,
        ).model_dump(mode="json")

        retry_policy = RetryPolicy(
            initial_interval=timedelta(seconds=1),
            backoff_coefficient=2.0,
            maximum_attempts=3,
        )

        try:
            self._set_stage(PipelineStage.ASSESSMENT)
            assessment_output = await workflow.execute_activity(
                "assessment_agent",
                {
                    "borrower": borrower.model_dump(mode="json"),
                    "prior_outputs": outputs,
                },
                start_to_close_timeout=timedelta(seconds=90),
                retry_policy=retry_policy,
            )
            outputs.append(assessment_output)
            self._status["outputs"] = outputs

            self._set_stage(PipelineStage.RESOLUTION)
            resolution_output = await workflow.execute_activity(
                "resolution_agent",
                {
                    "borrower": borrower.model_dump(mode="json"),
                    "prior_outputs": outputs,
                },
                start_to_close_timeout=timedelta(seconds=90),
                retry_policy=retry_policy,
            )
            outputs.append(resolution_output)
            self._status["outputs"] = outputs

            self._set_stage(PipelineStage.FINAL_NOTICE)
            final_notice_output = await workflow.execute_activity(
                "final_notice_agent",
                {
                    "borrower": borrower.model_dump(mode="json"),
                    "prior_outputs": outputs,
                },
                start_to_close_timeout=timedelta(seconds=90),
                retry_policy=retry_policy,
            )
            outputs.append(final_notice_output)

            self._status["outputs"] = outputs
            self._status["final_outcome"] = "final_notice_issued"
            self._set_stage(PipelineStage.COMPLETED, completed=True)
            return self._status
        except Exception as exc:
            self._status["failed"] = True
            self._status["error"] = str(exc)
            self._set_stage(PipelineStage.FAILED, completed=True)
            raise
