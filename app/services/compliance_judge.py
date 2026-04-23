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
from app.services.prompt_assembler import assemble_judge_prompt
from app.services.prompt_registry import get_prompt_registry

logger = logging.getLogger(__name__)


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
    """Build the user prompt for the compliance judge.

    Kept as a standalone function for backward compatibility and tests.
    The ``run_judge`` function now uses the registry-backed assembler
    which delegates to this same format.
    """
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

    registry = get_prompt_registry()
    judge_config = registry.get_judge_config()
    assembled = assemble_judge_prompt(
        judge_config,
        stage=stage,
        turn_index=turn_index,
        assistant_reply=assistant_reply,
        transcript_excerpt=transcript_excerpt,
    )

    try:
        client = _get_judge_client()
        result = await client.generate(
            system_prompt=assembled.system_prompt,
            user_prompt=assembled.user_prompt,
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
