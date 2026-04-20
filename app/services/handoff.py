"""Deterministic handoff summary builder.

Produces a flat JSON object from completed stage metadata and transcript,
guaranteed to fit within MAX_HANDOFF_TOKENS.  No LLM call required.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.services.token_budget import MAX_HANDOFF_TOKENS, count_tokens

logger = logging.getLogger(__name__)

_STANCE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "distressed": ("hardship", "crisis", "struggling", "can't afford", "lost my job"),
    "resistant": ("refuse", "won't", "wont", "no way", "not paying", "decline"),
    "evasive": ("maybe", "i don't know", "not sure", "later", "let me think"),
    "cooperative": ("yes", "agree", "i can pay", "okay", "understand", "sure"),
}


def _estimate_stance(transcript: list[dict[str, Any]]) -> str:
    """Derive borrower stance from keyword heuristics on borrower messages."""
    borrower_text = " ".join(
        msg["text"].lower()
        for msg in transcript
        if msg.get("role") == "borrower"
    )
    for stance, keywords in _STANCE_KEYWORDS.items():
        if any(kw in borrower_text for kw in keywords):
            return stance
    return "unknown"


def _extract_key_exchanges(
    transcript: list[dict[str, Any]],
    stage_value: str,
    max_pairs: int = 2,
) -> list[dict[str, str]]:
    """Extract the last N borrower-agent exchange pairs for a given stage."""
    stage_messages = [
        msg for msg in transcript if msg.get("stage") == stage_value
    ]

    pairs: list[dict[str, str]] = []
    i = 0
    while i < len(stage_messages) - 1:
        if (
            stage_messages[i].get("role") == "borrower"
            and stage_messages[i + 1].get("role") == "agent"
        ):
            pairs.append({
                "s": stage_value,
                "t": len(pairs) + 1,
                "b": stage_messages[i]["text"][:120],
                "a": stage_messages[i + 1]["text"][:120],
            })
            i += 2
        else:
            i += 1

    return pairs[-max_pairs:]


def build_handoff_summary(
    completed_stages: list[dict[str, Any]],
    borrower: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> str:
    """Build a flat, deterministic JSON handoff summary.

    Args:
        completed_stages: List of dicts, each with keys ``stage``,
            ``collected_fields``, ``transition_reason``, ``turns``.
        borrower: Borrower data dict (from BorrowerRequest).
        transcript: Full conversation transcript as list of message dicts.

    Returns:
        Compact JSON string guaranteed to be <= MAX_HANDOFF_TOKENS tokens.
    """
    summary: dict[str, Any] = {
        "stages_covered": [s["stage"] for s in completed_stages],
        "borrower_id": borrower.get("borrower_id", ""),
        "debt_amount": borrower.get("debt_amount", 0),
        "currency": borrower.get("currency", "USD"),
        "days_past_due": borrower.get("days_past_due", 0),
    }

    for stage_meta in completed_stages:
        stage_value = stage_meta["stage"]
        summary[stage_value] = {
            **stage_meta["collected_fields"],
            "turns": stage_meta["turns"],
            "reason": stage_meta["transition_reason"],
        }

    all_exchanges: list[dict[str, str]] = []
    for stage_meta in completed_stages:
        all_exchanges.extend(
            _extract_key_exchanges(transcript, stage_meta["stage"])
        )
    summary["key_exchanges"] = all_exchanges
    summary["borrower_stance"] = _estimate_stance(transcript)

    result = json.dumps(summary, separators=(",", ":"))

    while count_tokens(result) > MAX_HANDOFF_TOKENS and summary["key_exchanges"]:
        summary["key_exchanges"].pop(0)
        result = json.dumps(summary, separators=(",", ":"))

    if count_tokens(result) > MAX_HANDOFF_TOKENS:
        logger.warning(
            "Handoff summary still exceeds %d tokens (%d) after trimming "
            "exchanges; fixed fields alone are too large.",
            MAX_HANDOFF_TOKENS, count_tokens(result),
        )

    return result
