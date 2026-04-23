"""Data models for the evaluation harness.

Covers borrower personas, scenario definitions, conversation traces,
judge scores, and aggregate evaluation results.  All models are Pydantic
so they serialise cleanly to JSON for the reproducibility requirement.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Persona & scenario
# ------------------------------------------------------------------

class PersonaType(str, Enum):
    COOPERATIVE = "cooperative"
    COMBATIVE = "combative"
    EVASIVE = "evasive"
    CONFUSED = "confused"
    DISTRESSED = "distressed"


class BorrowerPersona(BaseModel):
    persona_type: PersonaType
    system_prompt: str = Field(
        description="LLM system prompt instructing the model to play this borrower type.",
    )
    description: str = Field(default="", description="Human-readable persona summary.")


class Scenario(BaseModel):
    scenario_id: str
    persona: BorrowerPersona
    borrower_id: str
    account_reference: str
    debt_amount: float = Field(gt=0)
    currency: str = "USD"
    days_past_due: int = Field(ge=1)
    borrower_message: str = "I received a notice and want to understand my options."
    notes: str | None = None
    seed: int = 42


# ------------------------------------------------------------------
# Conversation trace
# ------------------------------------------------------------------

class TurnRecord(BaseModel):
    stage: str
    turn_index: int
    borrower_message: str
    agent_reply: str
    stage_complete: bool
    collected_fields: dict[str, bool] = Field(default_factory=dict)
    decision: str = ""
    transition_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: float = 0.0


class StageRecord(BaseModel):
    stage: str
    turns: list[TurnRecord] = Field(default_factory=list)
    completed: bool = False
    transition_reason: str = ""
    collected_fields: dict[str, bool] = Field(default_factory=dict)
    handoff_summary: str = ""


class ConversationRecord(BaseModel):
    scenario: Scenario
    stages: list[StageRecord] = Field(default_factory=list)
    final_outcome: str = ""
    total_turns: int = 0
    early_termination: bool = False
    termination_reason: str = ""
    all_agent_replies: list[str] = Field(default_factory=list)
    all_borrower_messages: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0


# ------------------------------------------------------------------
# Judge scores
# ------------------------------------------------------------------

class JudgeScore(BaseModel):
    rule_id: str
    rule_name: str
    score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    explanation: str = ""
    deterministic_cross_check: bool | None = None


class ConversationScores(BaseModel):
    scenario_id: str
    compliance_scores: list[JudgeScore] = Field(default_factory=list)
    quality_scores: list[JudgeScore] = Field(default_factory=list)
    handoff_scores: list[JudgeScore] = Field(default_factory=list)
    composite_score: float = 0.0


# ------------------------------------------------------------------
# Aggregate results
# ------------------------------------------------------------------

class MetricResult(BaseModel):
    name: str
    value: float
    sample_size: int
    std: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0


class EvalConfig(BaseModel):
    seed: int = 42
    n_per_persona: int = 2
    agent_model: str | None = None
    sim_model: str | None = None
    judge_model: str | None = None
    output_dir: str = "data/eval_results"


class CostReport(BaseModel):
    simulation_calls: int = 0
    evaluation_calls: int = 0
    prompt_generation_calls: int = 0
    total_calls: int = 0
    estimated_cost_usd: float = 0.0


class EvalRunResult(BaseModel):
    config: EvalConfig
    conversations: list[ConversationRecord] = Field(default_factory=list)
    scores: list[ConversationScores] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    cost: CostReport = Field(default_factory=CostReport)
