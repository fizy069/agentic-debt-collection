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


class BorrowerSimulator:
    """Generates borrower messages by prompting an LLM to role-play a persona.

    Model selection precedence:
      1. Explicit ``model`` argument (CLI flag).
      2. ``EVAL_SIM_MODEL`` environment variable.
      3. Provider default resolved by :class:`AnthropicClient`
         (Anthropic -> ``claude-haiku-4-5``; OpenAI -> ``gpt-4o-mini``).
    """

    def __init__(self, model: str | None = None) -> None:
        requested = model or os.getenv("EVAL_SIM_MODEL")
        self._client = AnthropicClient(model=requested)
        self._model = self._client.model
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
        account_facts: dict[str, str] | None = None,
    ) -> str:
        """Produce the borrower's next message.

        Args:
            persona: The active borrower persona with its system prompt.
            conversation_history: List of ``{"role": "agent"|"borrower", "text": ...}`` dicts.
            stage: Current pipeline stage name.
            turn_index: 1-based turn index within the stage.
            account_facts: Optional map of identifiers the borrower honestly
                knows about themselves (e.g. ``{"last4": "9876",
                "date_of_birth": "1985-03-14"}``).  Passed so the simulator
                can truthfully answer identity-verification questions.

        Returns:
            The simulated borrower reply as a plain string.
        """
        history_lines: list[str] = []
        for msg in conversation_history:
            role_label = "Agent" if msg["role"] == "agent" else "You (borrower)"
            history_lines.append(f"{role_label}: {msg['text']}")

        history_text = "\n".join(history_lines) if history_lines else "(conversation just started)"

        facts_block = ""
        if account_facts:
            fact_lines = [f"- {k}: {v}" for k, v in account_facts.items()]
            facts_block = (
                "\nFacts about yourself (use ONLY when the agent asks to verify "
                "your identity; otherwise ignore):\n"
                + "\n".join(fact_lines)
                + "\n"
            )

        user_prompt = (
            f"Stage: {stage} | Turn: {turn_index}\n\n"
            f"Conversation so far:\n{history_text}\n"
            f"{facts_block}\n"
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
