"""Token counting and context budget enforcement.

Uses tiktoken with cl100k_base as a proxy for Claude's tokenizer.
cl100k_base slightly overcounts vs Claude, which is the safe direction
for budget enforcement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import tiktoken

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 2000
MAX_HANDOFF_TOKENS = 500
MAX_BORROWER_MESSAGE_TOKENS = 2000

OVERSIZED_MESSAGE_REPLY = "Please reply concisely with the answer."

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the token count for *text*."""
    if not text:
        return 0
    return len(_encoding.encode(text))


def is_borrower_message_oversized(text: str) -> bool:
    """Return True if the borrower message exceeds the input token limit."""
    return count_tokens(text) > MAX_BORROWER_MESSAGE_TOKENS


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate *text* so it fits within *max_tokens*.

    Returns the original string unchanged if it already fits.
    """
    tokens = _encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated = _encoding.decode(tokens[:max_tokens])
    logger.warning(
        "Truncated text from %d to %d tokens", len(tokens), max_tokens,
    )
    return truncated


def enforce_context_budget(
    system_prompt: str,
    user_prompt: str,
    max_total: int = MAX_CONTEXT_TOKENS,
) -> tuple[str, str]:
    """Ensure *system_prompt* + *user_prompt* fit within *max_total* tokens.

    The system prompt is never truncated; the user prompt absorbs all cuts.
    Returns the (possibly truncated) pair.
    """
    system_tokens = count_tokens(system_prompt)
    user_tokens = count_tokens(user_prompt)
    total = system_tokens + user_tokens

    if total <= max_total:
        return system_prompt, user_prompt

    available_for_user = max(0, max_total - system_tokens)
    logger.warning(
        "Context budget exceeded: system=%d user=%d total=%d limit=%d. "
        "Truncating user prompt to %d tokens.",
        system_tokens, user_tokens, total, max_total, available_for_user,
    )
    return system_prompt, truncate_to_budget(user_prompt, available_for_user)


@dataclass
class SectionTokenReport:
    """Token count for a single named prompt section."""

    name: str
    tokens: int


@dataclass
class ContextBudgetReport:
    """Full token accounting for a prompt assembly pass."""

    sections: list[SectionTokenReport] = field(default_factory=list)
    total_tokens: int = 0
    limit: int = MAX_CONTEXT_TOKENS
    overflow_detected: bool = False
    overflow_summary_used: bool = False
    overflow_fallback_used: bool = False
    handoff_tokens: int = 0
    pre_overflow_tokens: int = 0
    post_overflow_tokens: int = 0

    def add(self, name: str, text: str) -> int:
        """Record a section and return its token count."""
        tokens = count_tokens(text)
        self.sections.append(SectionTokenReport(name=name, tokens=tokens))
        self.total_tokens = sum(s.tokens for s in self.sections)
        return tokens

    def to_metadata(self) -> dict[str, object]:
        """Produce a dict suitable for inclusion in stage output metadata."""
        return {
            "budget_limit": self.limit,
            "total_tokens": self.total_tokens,
            "overflow_detected": self.overflow_detected,
            "overflow_summary_used": self.overflow_summary_used,
            "overflow_fallback_used": self.overflow_fallback_used,
            "handoff_tokens": self.handoff_tokens,
            "pre_overflow_tokens": self.pre_overflow_tokens,
            "post_overflow_tokens": self.post_overflow_tokens,
            "sections": {s.name: s.tokens for s in self.sections},
        }
