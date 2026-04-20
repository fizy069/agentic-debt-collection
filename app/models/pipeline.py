from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PipelineStage(str, Enum):
    QUEUED = "queued"
    ASSESSMENT = "assessment"
    RESOLUTION = "resolution"
    FINAL_NOTICE = "final_notice"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentChannel(str, Enum):
    CHAT = "chat"
    VOICE_STUB = "voice_stub_chat"


class BorrowerRequest(BaseModel):
    borrower_id: str = Field(min_length=3, max_length=64)
    account_reference: str = Field(
        min_length=3,
        max_length=32,
        description="Internal account ref only, not full sensitive identifiers.",
    )
    debt_amount: float = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    days_past_due: int = Field(ge=1)
    borrower_message: str = Field(
        default=(
            "I received a notice and want to understand my options."
        ),
        max_length=500,
    )
    notes: str | None = Field(default=None, max_length=800)


class PipelineStartRequest(BaseModel):
    borrower: BorrowerRequest
    workflow_id: str | None = Field(
        default=None,
        description="Optional explicit workflow ID. If omitted, API generates one.",
    )


class AgentStageOutput(BaseModel):
    stage: PipelineStage
    channel: AgentChannel
    response_text: str
    summary: str
    decision: str
    next_stage: PipelineStage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineStatus(BaseModel):
    workflow_id: str
    current_stage: PipelineStage
    completed: bool = False
    failed: bool = False
    outputs: list[AgentStageOutput] = Field(default_factory=list)
    final_outcome: str | None = None
    error: str | None = None


class PipelineStartResponse(BaseModel):
    workflow_id: str
    run_id: str
    task_queue: str
