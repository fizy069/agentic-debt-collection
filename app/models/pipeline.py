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


class ConversationRole(str, Enum):
    BORROWER = "borrower"
    AGENT = "agent"


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
    stage_complete: bool = False
    collected_fields: dict[str, bool] = Field(default_factory=dict)
    transition_reason: str | None = None
    next_stage: PipelineStage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    role: ConversationRole
    stage: PipelineStage | None = None
    text: str
    timestamp: str
    message_id: str | None = None


class ComplianceFlags(BaseModel):
    """Structured compliance state tracked across the entire pipeline."""

    stop_contact_requested: bool = False
    hardship_detected: bool = False
    abusive_borrower: bool = False

    def any_terminal(self) -> bool:
        return self.stop_contact_requested or self.abusive_borrower


class StageTurnInput(BaseModel):
    borrower: BorrowerRequest
    stage: PipelineStage
    borrower_message: str = Field(min_length=1, max_length=12000)
    transcript: list[ConversationMessage] = Field(default_factory=list)
    collected_fields: dict[str, bool] = Field(default_factory=dict)
    turn_index: int = Field(ge=1)
    completed_stages: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Metadata from prior completed stages for handoff summary building.",
    )
    compliance_flags: ComplianceFlags = Field(default_factory=ComplianceFlags)


class StageTurnOutput(BaseModel):
    stage: PipelineStage
    channel: AgentChannel
    assistant_reply: str
    summary: str
    decision: str
    stage_complete: bool
    collected_fields: dict[str, bool] = Field(default_factory=dict)
    transition_reason: str
    next_stage: PipelineStage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    compliance_flags: ComplianceFlags = Field(default_factory=ComplianceFlags)


class BorrowerMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    message_id: str | None = Field(
        default=None,
        max_length=64,
        description="Optional idempotency token for the borrower turn.",
    )


class BorrowerMessageResponse(BaseModel):
    workflow_id: str
    accepted: bool
    message_id: str | None = None


class PipelineStatus(BaseModel):
    workflow_id: str
    current_stage: PipelineStage
    completed: bool = False
    failed: bool = False
    outputs: list[AgentStageOutput] = Field(default_factory=list)
    transcript: list[ConversationMessage] = Field(default_factory=list)
    pending_messages: int = 0
    latest_assistant_reply: str | None = None
    stage_turn_counts: dict[str, int] = Field(default_factory=dict)
    stage_collected_fields: dict[str, dict[str, bool]] = Field(default_factory=dict)
    compliance_flags: ComplianceFlags = Field(default_factory=ComplianceFlags)
    final_outcome: str | None = None
    error: str | None = None


class PipelineStartResponse(BaseModel):
    workflow_id: str
    run_id: str
    task_queue: str
