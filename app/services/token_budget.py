"""Token counting and context budget enforcement.

Uses tiktoken with cl100k_base as a proxy for Claude's tokenizer.
cl100k_base slightly overcounts vs Claude, which is the safe direction
for budget enforcement.
"""

from __future__ import annotations

import logging

import tiktoken

logger = logging.getLogger(__name__)

MAX_CONTEXT_TOKENS = 2000
MAX_HANDOFF_TOKENS = 500

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return the token count for *text*."""
    if not text:
        return 0
    return len(_encoding.encode(text))


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
