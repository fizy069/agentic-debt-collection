"""Prompt data models for the centralized prompt registry.

These dataclasses represent the atomic units of prompt construction.
Each ``PromptSection`` is independently versioned so the self-learning
loop can propose, track, compare, and roll back changes to individual
parts of any prompt.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PromptSection:
    """One independently-versioned piece of a prompt.

    ``section_type`` is one of:
      - ``"system"``    – goes into the LLM system field
      - ``"user"``      – goes into the LLM user message
      - ``"directive"`` – appended to the user message directives block
    """

    name: str
    content: str
    version: str = "v1"
    stage: str | None = None
    section_type: str = "user"


@dataclass
class PromptConfig:
    """Complete prompt configuration for one LLM call.

    ``version`` is a deterministic hash of every section version so the
    self-learning loop can diff and track composite prompt states.
    """

    stage: str
    call_type: str
    sections: dict[str, PromptSection] = field(default_factory=dict)

    @property
    def version(self) -> str:
        parts = sorted(f"{k}={s.version}" for k, s in self.sections.items())
        digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]
        return f"cfg-{digest}"


@dataclass
class AssembledPrompt:
    """Final output of the prompt assembler, ready for the LLM client."""

    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
