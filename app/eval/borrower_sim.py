"""LLM-powered borrower simulator for the evaluation harness.

The simulator takes a persona definition and the current conversation
history, then generates the next borrower reply via the LLM.  A
separate, cheaper model can be used to keep costs within the $20 budget.
"""

from __future__ import annotations

import logging
import os

from app.eval.models import BorrowerPersona
from app.services.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

_DEFAULT_SIM_MODEL = "claude-haiku-4-5"


class BorrowerSimulator:
    """Generates borrower messages by prompting an LLM to role-play a persona."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.getenv("EVAL_SIM_MODEL", _DEFAULT_SIM_MODEL)
        self._client = AnthropicClient(model=self._model)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def generate_reply(
        self,
        persona: BorrowerPersona,
        conversation_history: list[dict[str, str]],
        stage: str,
        turn_index: int,
    ) -> str:
        """Produce the borrower's next message.

        Args:
            persona: The active borrower persona with its system prompt.
            conversation_history: List of ``{"role": "agent"|"borrower", "text": ...}`` dicts.
            stage: Current pipeline stage name.
            turn_index: 1-based turn index within the stage.

        Returns:
            The simulated borrower reply as a plain string.
        """
        history_lines: list[str] = []
        for msg in conversation_history:
            role_label = "Agent" if msg["role"] == "agent" else "You (borrower)"
            history_lines.append(f"{role_label}: {msg['text']}")

        history_text = "\n".join(history_lines) if history_lines else "(conversation just started)"

        user_prompt = (
            f"Stage: {stage} | Turn: {turn_index}\n\n"
            f"Conversation so far:\n{history_text}\n\n"
            "Respond as the borrower in 1-3 sentences. "
            "Stay in character. Do not break the fourth wall or mention "
            "that you are an AI."
        )

        result = await self._client.generate(
            system_prompt=persona.system_prompt,
            user_prompt=user_prompt,
            max_tokens=150,
        )
        self._call_count += 1

        reply = result.text.strip()
        if reply.startswith('"') and reply.endswith('"'):
            reply = reply[1:-1]

        logger.info(
            "borrower_sim  stage=%s  turn=%d  persona=%s  reply_len=%d",
            stage, turn_index, persona.persona_type.value, len(reply),
        )
        return reply
