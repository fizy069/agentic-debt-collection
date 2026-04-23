"""Persistent audit trail for prompt evolution.

On-disk layout under ``base_dir`` (default ``data/audit``):

    events.jsonl                         — append-only event log
    prompt_versions/<section_key>/<ver>.json  — full content snapshot per version
    runs/<run_id>/                        — per-run artefacts
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from app.models.prompt import PromptSection

logger = logging.getLogger(__name__)

EVENT_TYPES = frozenset({
    "override",
    "rollback",
    "candidate_proposed",
    "candidate_rejected",
    "candidate_adopted",
    "run_started",
    "run_completed",
    "budget_exceeded",
})


def _safe_key(key: str) -> str:
    """Sanitise a section key for use as a directory name."""
    return key.replace(":", "_").replace("/", "_")


class AuditTrail:
    """Append-only audit trail persisted to disk."""

    def __init__(self, base_dir: str | Path = "data/audit") -> None:
        self._base = Path(base_dir)
        self._events_path = self._base / "events.jsonl"
        self._versions_dir = self._base / "prompt_versions"
        self._runs_dir = self._base / "runs"

        for d in (self._base, self._versions_dir, self._runs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def append_event(
        self,
        event_type: str,
        *,
        section_key: str | None = None,
        old_version: str | None = None,
        new_version: str | None = None,
        run_id: str | None = None,
        metrics_before: dict[str, Any] | None = None,
        metrics_after: dict[str, Any] | None = None,
        decision: str | None = None,
        rationale: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a structured event to the JSONL log and return it."""
        event: dict[str, Any] = {
            "event_id": uuid.uuid4().hex[:16],
            "timestamp": time.time(),
            "event_type": event_type,
        }
        if section_key is not None:
            event["section_key"] = section_key
        if old_version is not None:
            event["old_version"] = old_version
        if new_version is not None:
            event["new_version"] = new_version
        if run_id is not None:
            event["run_id"] = run_id
        if metrics_before is not None:
            event["metrics_before"] = metrics_before
        if metrics_after is not None:
            event["metrics_after"] = metrics_after
        if decision is not None:
            event["decision"] = decision
        if rationale is not None:
            event["rationale"] = rationale
        if extra is not None:
            event.update(extra)

        try:
            with open(self._events_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError:
            logger.warning("audit_trail_write_failed  path=%s", self._events_path)

        return event

    def snapshot_section(self, section: PromptSection) -> Path:
        """Persist the full content of a prompt section version to disk."""
        safe = _safe_key(section.name)
        section_dir = self._versions_dir / safe
        section_dir.mkdir(parents=True, exist_ok=True)

        path = section_dir / f"{section.version}.json"
        payload = {
            "name": section.name,
            "version": section.version,
            "stage": section.stage,
            "section_type": section.section_type,
            "content": section.content,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def load_section(self, key: str, version: str) -> dict[str, Any] | None:
        """Load a previously snapshotted section from disk."""
        safe = _safe_key(key)
        path = self._versions_dir / safe / f"{version}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_run_artefact(
        self, run_id: str, filename: str, data: Any,
    ) -> Path:
        """Write an arbitrary JSON artefact into the per-run directory."""
        run_dir = self._runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / filename
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def list_events(
        self, section_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all events, optionally filtered by section key."""
        if not self._events_path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self._events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if section_key is None or evt.get("section_key") == section_key:
                events.append(evt)
        return events

    def latest_adopted_version(self, section_key: str) -> str | None:
        """Return the newest adopted version for a given section, or None."""
        events = self.list_events(section_key=section_key)
        for evt in reversed(events):
            if evt.get("event_type") in ("candidate_adopted", "override"):
                return evt.get("new_version")
        return None
