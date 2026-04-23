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


# Sentinel placed in `StageTurnInput.borrower_message` when the agent
# initiates the turn (agent-initiated stages like resolution / final_notice
# on turn 1). These stages are outbound by nature (we call/notify the
# borrower) so there is no real borrower utterance to feed in. The prompt
# templates and the activity code both recognise this marker and treat the
# turn as "open the call / send the notice" rather than "respond to what
# the borrower just said".
STAGE_OPENER_SENTINEL = "[handoff_open]"

# Sentinel placed in `StageTurnInput.borrower_message` when the workflow
# forces a closing turn after hitting the turn cap.  There is no real
# borrower utterance — the activity uses a closing-only sub-prompt to
# produce a handoff-bridge reply without asking further questions.
CLOSING_TURN_SENTINEL = "[closing_turn]"


def stage_is_agent_initiated(stage: "PipelineStage") -> bool:
    """Return True for stages where the agent speaks first on turn 1.

    Assessment is inbound (the borrower contacted us / received a notice
    and is replying) — the borrower speaks first. Resolution is an
    outbound collections call, and Final Notice is an outbound written
    notice — both are agent-initiated.
    """
    return stage in (PipelineStage.RESOLUTION, PipelineStage.FINAL_NOTICE)


class AgentChannel(str, Enum):
    CHAT = "chat"
    VOICE_STUB = "voice_stub_chat"


class ConversationRole(str, Enum):
    BORROWER = "borrower"
    AGENT = "agent"


class AccountRecord(BaseModel):
    """System-side account data — sourced from CRM/database, not from the borrower."""

    borrower_id: str = Field(min_length=3, max_length=64)
    account_reference: str = Field(
        min_length=3,
        max_length=32,
        description="Internal account ref only, not full sensitive identifiers.",
    )
    date_of_birth: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="ISO 8601 (YYYY-MM-DD). Used for identity verification only.",
    )
    debt_amount: float = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    days_past_due: int = Field(ge=1)
    notes: str | None = Field(default=None, max_length=800)


class BorrowerRequest(BaseModel):
    """Combined view passed through the pipeline (account data + conversation context)."""

    borrower_id: str = Field(min_length=3, max_length=64)
    account_reference: str = Field(
        min_length=3,
        max_length=32,
        description="Internal account ref only, not full sensitive identifiers.",
    )
    date_of_birth: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="ISO 8601 (YYYY-MM-DD). Used for identity verification only.",
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

    @classmethod
    def from_account(
        cls,
        account: AccountRecord,
        borrower_message: str = "I received a notice and want to understand my options.",
    ) -> BorrowerRequest:
        return cls(
            borrower_id=account.borrower_id,
            account_reference=account.account_reference,
            date_of_birth=account.date_of_birth,
            debt_amount=account.debt_amount,
            currency=account.currency,
            days_past_due=account.days_past_due,
            borrower_message=borrower_message,
            notes=account.notes,
        )


class PipelineStartRequest(BaseModel):
    borrower_id: str = Field(
        min_length=3,
        max_length=64,
        description="Looked up against the account store to retrieve system-side data.",
    )
    borrower_message: str = Field(
        default="I received a notice and want to understand my options.",
        max_length=500,
    )
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
    closing_turn: bool = Field(
        default=False,
        description="When True the activity uses a closing-only sub-prompt and "
        "forces stage_complete=True regardless of LLM output.",
    )


class LLMStageResponse(BaseModel):
    """Schema the LLM must return as structured JSON from each stage turn."""

    assistant_reply: str
    stage_complete: bool
    collected_fields: dict[str, bool]
    transition_reason: str
    decision: str


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
