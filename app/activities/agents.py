from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from temporalio import activity

from app.models.pipeline import (
    AgentChannel,
    AgentStageOutput,
    BorrowerRequest,
    PipelineStage,
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


def _build_handoff_text(prior_outputs: list[AgentStageOutput]) -> str:
    if not prior_outputs:
        return "No prior stage output available."

    lines = []
    for index, output in enumerate(prior_outputs, start=1):
        lines.append(
            f"{index}. stage={output.stage.value}; decision={output.decision}; "
            f"summary={output.summary}"
        )
    return "\n".join(lines)


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


async def _run_agent_stage(
    *,
    payload: dict[str, Any],
    stage: PipelineStage,
    channel: AgentChannel,
    system_prompt: str,
    decision: str,
    next_stage: PipelineStage | None,
) -> dict[str, Any]:
    borrower = BorrowerRequest.model_validate(payload["borrower"])
    prior_output_payloads = payload.get("prior_outputs", [])
    prior_outputs = [
        AgentStageOutput.model_validate(output_item)
        for output_item in prior_output_payloads
    ]

    user_prompt = (
        "You are continuing a debt collection workflow stage.\n\n"
        "Borrower snapshot:\n"
        f"{_borrower_snapshot(borrower)}\n\n"
        "Prior stage continuity context:\n"
        f"{_build_handoff_text(prior_outputs)}\n\n"
        "Respond as this stage's agent in 4-8 concise sentences."
    )

    llm_result = await _get_anthropic_client().generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    output = AgentStageOutput(
        stage=stage,
        channel=channel,
        response_text=llm_result.text,
        summary=_trim_summary(llm_result.text),
        decision=decision,
        next_stage=next_stage,
        metadata={
            "model": llm_result.model,
            "used_fallback": llm_result.used_fallback,
        },
    )
    return output.model_dump(mode="json")


@activity.defn(name="assessment_agent")
async def assessment_agent(payload: dict[str, Any]) -> dict[str, Any]:
    return await _run_agent_stage(
        payload=payload,
        stage=PipelineStage.ASSESSMENT,
        channel=AgentChannel.CHAT,
        decision="assessment_completed",
        next_stage=PipelineStage.RESOLUTION,
        system_prompt=(
            "You are Agent 1 (Assessment) for post-default collections. "
            "Tone: clinical and direct. "
            "Goal: establish debt context, perform partial identity verification, "
            "and gather ability-to-pay signals. "
            "Do not negotiate or offer settlement terms."
        ),
    )


@activity.defn(name="resolution_agent")
async def resolution_agent(payload: dict[str, Any]) -> dict[str, Any]:
    return await _run_agent_stage(
        payload=payload,
        stage=PipelineStage.RESOLUTION,
        channel=AgentChannel.VOICE_STUB,
        decision="resolution_attempted",
        next_stage=PipelineStage.FINAL_NOTICE,
        system_prompt=(
            "You are Agent 2 (Resolution) representing a voice call in text form. "
            "Tone: transactional negotiator. "
            "Present policy-bounded options only: lump-sum discount, payment plan, "
            "or hardship referral. Restate terms clearly and ask for commitment."
        ),
    )


@activity.defn(name="final_notice_agent")
async def final_notice_agent(payload: dict[str, Any]) -> dict[str, Any]:
    return await _run_agent_stage(
        payload=payload,
        stage=PipelineStage.FINAL_NOTICE,
        channel=AgentChannel.CHAT,
        decision="final_notice_sent",
        next_stage=None,
        system_prompt=(
            "You are Agent 3 (Final Notice). "
            "Tone: professional, consequence-focused closer. "
            "Provide final terms with a clear expiry window and documented next-step "
            "consequences without making false threats."
        ),
    )
