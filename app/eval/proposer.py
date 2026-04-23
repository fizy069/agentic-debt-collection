"""LLM-driven prompt-candidate generator with compliance safety guard.

Uses a cheap model to read the current prompt section, the failure digest
produced by :mod:`failure_miner`, and generate an improved version. A
deterministic guard rejects proposals that drop any of the 8 compliance
rule keywords.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.eval.failure_miner import FailureDigest, SectionFailures
from app.services.anthropic_client import AnthropicClient
from app.services.prompt_registry import PromptRegistry

if TYPE_CHECKING:
    from app.eval.cost_ledger import CostLedger

logger = logging.getLogger(__name__)

_COMPLIANCE_GUARD_PHRASES: list[str] = [
    "ai agent",
    "false threat",
    "stop",
    "settlement",
    "hardship",
    "logged",
    "recorded",
    "professional",
    "privacy",
]


class ProposalRejectedError(ValueError):
    """Raised when the proposer output fails the compliance guard."""


@dataclass
class ProposedSection:
    """A candidate prompt section proposed by the LLM."""
    section_key: str
    content: str
    rationale: str
    change_summary: str
    version: str


class PromptProposer:
    """Generates candidate prompt improvements via an LLM."""

    def __init__(
        self,
        ledger: CostLedger | None = None,
        model: str | None = None,
    ) -> None:
        requested = model or os.getenv("EVAL_PROPOSER_MODEL")
        self._client = AnthropicClient(model=requested)
        self._ledger = ledger
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def propose(
        self,
        registry: PromptRegistry,
        section_key: str,
        failures: SectionFailures,
        iteration: int = 1,
    ) -> ProposedSection:
        """Generate one candidate for the given section."""
        if self._ledger is not None:
            self._ledger.check_budget_or_raise()

        current_section = registry.get_section(section_key)
        proposer_section = registry.get_section("proposer_system")

        evidence_lines: list[str] = []
        for exc in failures.excerpts[:5]:
            evidence_lines.append(
                f"- [{exc.source}] rule={exc.rule_id} ({exc.rule_name}) "
                f"score={exc.score:.2f}: {exc.explanation}"
            )
        evidence_text = "\n".join(evidence_lines) or "(no specific excerpts)"

        user_prompt = (
            f"SECTION KEY: {section_key}\n"
            f"CURRENT VERSION: {current_section.version}\n\n"
            f"--- CURRENT CONTENT ---\n{current_section.content}\n\n"
            f"--- FAILURE DIGEST ({len(failures.excerpts)} failures, "
            f"weighted={failures.weighted_count:.1f}) ---\n{evidence_text}\n\n"
            "Generate an improved version addressing these failures."
        )

        result = await self._client.generate(
            system_prompt=proposer_section.content,
            user_prompt=user_prompt,
            max_tokens=2000,
        )
        self._call_count += 1

        if self._ledger is not None:
            self._ledger.record(
                role="proposer",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
            )

        parsed = self._parse_response(result.text)
        new_content = parsed["content"]

        self._compliance_guard(current_section.content, new_content)

        new_version = f"v{iteration + 1}"
        return ProposedSection(
            section_key=section_key,
            content=new_content,
            rationale=parsed.get("rationale", ""),
            change_summary=parsed.get("change_summary", ""),
            version=new_version,
        )

    def _parse_response(self, raw: str) -> dict:
        """Extract the JSON proposal from the LLM response."""
        text = raw.strip()
        if text.startswith("```"):
            first_nl = text.find("\n")
            if first_nl != -1:
                text = text[first_nl + 1:]
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3]
            text = text.strip()

        try:
            data = json.loads(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProposalRejectedError(
                f"Proposer returned invalid JSON: {raw[:300]}"
            ) from exc

        if not isinstance(data, dict) or "content" not in data:
            raise ProposalRejectedError(
                f"Proposer JSON missing 'content' key: {list(data.keys())}"
            )
        return data

    @staticmethod
    def _compliance_guard(original: str, proposed: str) -> None:
        """Reject proposals that drop compliance-critical phrases."""
        orig_lower = original.lower()
        prop_lower = proposed.lower()

        missing: list[str] = []
        for phrase in _COMPLIANCE_GUARD_PHRASES:
            if phrase in orig_lower and phrase not in prop_lower:
                missing.append(phrase)

        if missing:
            raise ProposalRejectedError(
                f"Compliance guard: proposed content drops phrases: {missing}"
            )
