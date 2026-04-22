"""Layer 2 compliance judge — audit-only LLM-as-a-judge for conversation turns.

The judge analyzes a redacted transcript excerpt plus the current assistant reply
against the compliance ruleset and returns structured findings.  It never blocks
or rewrites the debtor-facing reply.

Toggle via ``COMPLIANCE_JUDGE_ENABLED=true`` in the environment.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from app.services.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM_PROMPT = """\
You are a compliance auditor for a regulated debt-collection system.

Given:
  1. A redacted transcript excerpt from the current conversation stage.
  2. The assistant's latest reply that was sent to the borrower.

Evaluate the reply against these rules:
  Rule 2 — No false threats.  Only these consequences are allowed: referral to \
collections agency, credit reporting, legal review, account charge-off.
  Rule 3 — If the borrower asked to stop contact, the agent must comply.
  Rule 4 — Settlement offers must be 40-80 % of the balance; payment plans 3-24 months.
  Rule 5 — Financial hardship or crisis must be acknowledged and routed, never dismissed.
  Rule 7 — The agent must remain professional at all times.
  Rule 8 — No PII (SSNs, full account numbers, credit-card numbers) may appear.

Return ONLY a JSON object (no markdown fences) with this schema:
{
  "violations": [
    {
      "rule": "<rule number, e.g. 2>",
      "label": "<short normalized label, e.g. false_threat>",
      "confidence": <0.0-1.0>,
      "excerpt": "<brief quote from the reply that triggered the finding>"
    }
  ],
  "overall_risk": "<low|medium|high>"
}
If there are no violations return {"violations": [], "overall_risk": "low"}.
"""


@dataclass(frozen=True)
class JudgeViolation:
    rule: str
    label: str
    confidence: float
    excerpt: str


@dataclass(frozen=True)
class JudgeFindings:
    violations: list[JudgeViolation] = field(default_factory=list)
    overall_risk: str = "low"
    raw_response: str = ""
    judge_error: str | None = None


def is_judge_enabled() -> bool:
    return os.getenv("COMPLIANCE_JUDGE_ENABLED", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )


_cached_judge_client: AnthropicClient | None = None


def _get_judge_client() -> AnthropicClient:
    global _cached_judge_client
    env_path = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=env_path, override=True)

    if _cached_judge_client is not None and _cached_judge_client._client is not None:
        return _cached_judge_client

    model = os.getenv("COMPLIANCE_JUDGE_MODEL")
    _cached_judge_client = AnthropicClient(model=model)
    return _cached_judge_client


def build_judge_prompt(
    *,
    stage: str,
    turn_index: int,
    assistant_reply: str,
    transcript_excerpt: str,
) -> str:
    return (
        f"Stage: {stage}\n"
        f"Turn: {turn_index}\n\n"
        f"--- Transcript excerpt ---\n{transcript_excerpt}\n\n"
        f"--- Assistant reply under review ---\n{assistant_reply}"
    )


def _parse_judge_response(raw: str) -> JudgeFindings:
    """Best-effort JSON parse of the judge LLM's output."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.index("\n") if "\n" in cleaned else 3
        cleaned = cleaned[first_newline:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("judge_parse_failed  error=%s  raw=%s", exc, raw[:300])
        return JudgeFindings(
            raw_response=raw,
            judge_error=f"json_parse_failed: {exc}",
        )

    violations: list[JudgeViolation] = []
    for v in data.get("violations", []):
        try:
            violations.append(
                JudgeViolation(
                    rule=str(v.get("rule", "")),
                    label=str(v.get("label", "")),
                    confidence=float(v.get("confidence", 0.0)),
                    excerpt=str(v.get("excerpt", "")),
                )
            )
        except (TypeError, ValueError):
            continue

    return JudgeFindings(
        violations=violations,
        overall_risk=str(data.get("overall_risk", "low")),
        raw_response=raw,
    )


async def run_judge(
    *,
    stage: str,
    turn_index: int,
    assistant_reply: str,
    transcript_excerpt: str,
) -> JudgeFindings:
    """Call the judge LLM and return structured findings.

    Swallows all exceptions so the main pipeline is never blocked.
    """
    if not is_judge_enabled():
        return JudgeFindings()

    user_prompt = build_judge_prompt(
        stage=stage,
        turn_index=turn_index,
        assistant_reply=assistant_reply,
        transcript_excerpt=transcript_excerpt,
    )

    try:
        client = _get_judge_client()
        result = await client.generate(
            system_prompt=_JUDGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=400,
        )
        findings = _parse_judge_response(result.text)
        logger.info(
            "judge_complete  stage=%s  turn=%d  violations=%d  risk=%s  model=%s",
            stage,
            turn_index,
            len(findings.violations),
            findings.overall_risk,
            result.model,
        )
        return findings
    except Exception as exc:
        logger.exception("judge_call_failed  stage=%s  turn=%d", stage, turn_index)
        return JudgeFindings(judge_error=f"judge_call_failed: {exc}")


def findings_to_metadata(findings: JudgeFindings) -> dict:
    """Flatten findings into a dict suitable for StageTurnOutput.metadata."""
    meta: dict = {}
    if findings.judge_error:
        meta["judge_error"] = findings.judge_error
        return meta

    if not findings.violations:
        meta["judge_risk"] = findings.overall_risk
        return meta

    meta["judge_risk"] = findings.overall_risk
    meta["judge_violations"] = [
        {
            "rule": v.rule,
            "label": v.label,
            "confidence": v.confidence,
            "excerpt": v.excerpt,
        }
        for v in findings.violations
    ]
    return meta
