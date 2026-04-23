"""Centralized prompt registry with versioning and override support.

Every prompt section used across all LLM call sites (main agent,
compliance judge, overflow compressor) is loaded here on startup and
exposed through a single registry instance.  The self-learning loop
can call ``override_section`` to swap in a candidate, ``snapshot`` to
capture the full state, and ``rollback`` to revert.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.models.pipeline import PipelineStage
from app.models.prompt import PromptConfig, PromptSection

if TYPE_CHECKING:
    from app.eval.audit_trail import AuditTrail

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_PROMPT_FILES: dict[str, str] = {
    PipelineStage.ASSESSMENT.value: "assessment.txt",
    PipelineStage.RESOLUTION.value: "resolution.txt",
    PipelineStage.FINAL_NOTICE.value: "final_notice.txt",
}


class PromptRegistry:
    """Owns all prompt sections and their version history."""

    def __init__(self, audit_trail: AuditTrail | None = None) -> None:
        self._sections: dict[str, PromptSection] = {}
        self._history: dict[str, list[PromptSection]] = {}
        self._audit: AuditTrail | None = audit_trail
        self._load_system_templates()
        self._load_directives()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_system_templates(self) -> None:
        for stage_val, filename in _PROMPT_FILES.items():
            path = _PROMPTS_DIR / filename
            content = path.read_text(encoding="utf-8").strip()
            key = f"system_template:{stage_val}"
            section = PromptSection(
                name=key,
                content=content,
                version="v1",
                stage=stage_val,
                section_type="system",
            )
            self._register(key, section)

    def _load_directives(self) -> None:
        directives_path = _PROMPTS_DIR / "directives.json"
        raw = json.loads(directives_path.read_text(encoding="utf-8"))

        td = raw["turn_directives"]
        self._register(
            "turn_directives:default",
            PromptSection(
                name="turn_directives:default",
                content=td["default"]["content"],
                version=td["default"]["version"],
                stage=None,
                section_type="directive",
            ),
        )
        self._register(
            "turn_directives:final_notice",
            PromptSection(
                name="turn_directives:final_notice",
                content=td["final_notice"]["content_template"],
                version=td["final_notice"]["version"],
                stage=PipelineStage.FINAL_NOTICE.value,
                section_type="directive",
            ),
        )

        cd = raw["compliance_directives"]
        self._register(
            "compliance_directives:offer_policy",
            PromptSection(
                name="compliance_directives:offer_policy",
                content=cd["offer_policy"]["content_template"],
                version=cd["offer_policy"]["version"],
                stage=None,
                section_type="directive",
            ),
        )
        self._register(
            "compliance_directives:allowed_consequences",
            PromptSection(
                name="compliance_directives:allowed_consequences",
                content=cd["allowed_consequences"]["content_template"],
                version=cd["allowed_consequences"]["version"],
                stage=None,
                section_type="directive",
            ),
        )

        ov = raw["overflow_system"]
        self._register(
            "overflow_system",
            PromptSection(
                name="overflow_system",
                content=ov["content"],
                version=ov["version"],
                stage=None,
                section_type="system",
            ),
        )

        js = raw["judge_system"]
        self._register(
            "judge_system",
            PromptSection(
                name="judge_system",
                content=js["content"],
                version=js["version"],
                stage=None,
                section_type="system",
            ),
        )

        ps = raw.get("proposer_system")
        if ps:
            self._register(
                "proposer_system",
                PromptSection(
                    name="proposer_system",
                    content=ps["content"],
                    version=ps["version"],
                    stage=None,
                    section_type="system",
                ),
            )

    def _register(self, key: str, section: PromptSection) -> None:
        self._sections[key] = section
        self._history.setdefault(key, []).append(section)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_section(self, key: str) -> PromptSection:
        """Return the current version of a named section."""
        return self._sections[key]

    def get_agent_config(self, stage: PipelineStage) -> PromptConfig:
        """Build a ``PromptConfig`` for a main-agent LLM call."""
        stage_val = stage.value
        sections: dict[str, PromptSection] = {}
        sections["system_template"] = self._sections[f"system_template:{stage_val}"]

        if stage == PipelineStage.FINAL_NOTICE:
            sections["turn_directives"] = self._sections["turn_directives:final_notice"]
        else:
            sections["turn_directives"] = self._sections["turn_directives:default"]

        if stage in (PipelineStage.RESOLUTION, PipelineStage.FINAL_NOTICE):
            sections["offer_policy"] = self._sections["compliance_directives:offer_policy"]
            sections["allowed_consequences"] = self._sections[
                "compliance_directives:allowed_consequences"
            ]

        return PromptConfig(stage=stage_val, call_type="agent", sections=sections)

    def get_judge_config(self) -> PromptConfig:
        """Build a ``PromptConfig`` for a compliance-judge LLM call."""
        return PromptConfig(
            stage="all",
            call_type="judge",
            sections={"judge_system": self._sections["judge_system"]},
        )

    def get_overflow_config(self, stage: PipelineStage) -> PromptConfig:
        """Build a ``PromptConfig`` for an overflow-compression LLM call."""
        return PromptConfig(
            stage=stage.value,
            call_type="overflow",
            sections={"overflow_system": self._sections["overflow_system"]},
        )

    # ------------------------------------------------------------------
    # Self-learning mutation
    # ------------------------------------------------------------------

    def override_section(
        self,
        key: str,
        new_content: str,
        new_version: str,
    ) -> PromptSection:
        """Replace a section's content and bump its version.

        The previous version is preserved in history for rollback.
        """
        old = self._sections[key]
        updated = PromptSection(
            name=old.name,
            content=new_content,
            version=new_version,
            stage=old.stage,
            section_type=old.section_type,
        )
        self._register(key, updated)
        logger.info(
            "prompt_override  key=%s  old_version=%s  new_version=%s",
            key, old.version, new_version,
        )
        if self._audit is not None:
            self._audit.snapshot_section(updated)
            self._audit.append_event(
                "override",
                section_key=key,
                old_version=old.version,
                new_version=new_version,
            )
        return updated

    def rollback(self, key: str, target_version: str) -> PromptSection:
        """Restore a section to a specific historical version."""
        current = self._sections.get(key)
        old_version = current.version if current else None

        for entry in reversed(self._history.get(key, [])):
            if entry.version == target_version:
                self._sections[key] = entry
                logger.info(
                    "prompt_rollback  key=%s  restored_version=%s",
                    key, target_version,
                )
                if self._audit is not None:
                    self._audit.append_event(
                        "rollback",
                        section_key=key,
                        old_version=old_version,
                        new_version=target_version,
                    )
                return entry
        raise ValueError(
            f"Version {target_version!r} not found in history for {key!r}"
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Capture the full current state for audit / persistence."""
        return {
            key: {
                "name": s.name,
                "version": s.version,
                "stage": s.stage,
                "section_type": s.section_type,
                "content_length": len(s.content),
            }
            for key, s in self._sections.items()
        }

    def list_versions(self) -> dict[str, list[str]]:
        """Return version strings for every section, oldest first."""
        return {
            key: [entry.version for entry in entries]
            for key, entries in self._history.items()
        }

    def all_section_keys(self) -> list[str]:
        return list(self._sections.keys())

    def clone(self, audit_trail: AuditTrail | None = None) -> PromptRegistry:
        """Deep-copy this registry for A/B comparison without mutating the original."""
        cloned = PromptRegistry.__new__(PromptRegistry)
        cloned._sections = copy.deepcopy(self._sections)
        cloned._history = copy.deepcopy(self._history)
        cloned._audit = audit_trail
        return cloned


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_registry: PromptRegistry | None = None


def get_prompt_registry(
    audit_trail: AuditTrail | None = None,
) -> PromptRegistry:
    """Return (and lazily create) the global prompt registry."""
    global _registry
    if _registry is None:
        _registry = PromptRegistry(audit_trail=audit_trail)
    return _registry


def reset_prompt_registry() -> None:
    """Tear down the singleton (useful in tests)."""
    global _registry
    _registry = None
