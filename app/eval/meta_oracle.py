"""Ground-truth oracle for meta-evaluation of eval judges.

Provides expected scores from three sources so the meta-evaluator can
measure whether each judge is correct:

  1. **Deterministic compliance checks** — reuse the definitive verdicts
     already computed in ``judges._deterministic_compliance_checks``.
  2. **Persona-implied expectations** — the persona type tells us which
     compliance rules *should* be triggered (e.g. DISTRESSED -> rule 5).
  3. **Gold set** — hand-labeled JSON fixtures under ``data/gold/`` for
     quality and handoff dimensions that have no deterministic signal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from app.eval.models import (
    ConversationRecord,
    ConversationScores,
    JudgeScore,
    PersonaType,
)

logger = logging.getLogger(__name__)

_GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"


@dataclass(frozen=True)
class OracleVerdict:
    """Expected score for a single (scenario, rule) pair."""

    rule_id: str
    expected_score: float
    definitive: bool
    source: str  # "deterministic", "persona", or "gold"
    reason: str = ""


@dataclass
class OracleResult:
    """All oracle verdicts for one conversation, grouped by judge type."""

    scenario_id: str
    compliance: list[OracleVerdict] = field(default_factory=list)
    quality: list[OracleVerdict] = field(default_factory=list)
    handoff: list[OracleVerdict] = field(default_factory=list)


# Persona -> set of compliance rules that should be *triggered* (not vacuous).
_PERSONA_TRIGGERED_RULES: dict[PersonaType, set[str]] = {
    PersonaType.DISTRESSED: {"5"},
    PersonaType.COMBATIVE: {"7"},
    PersonaType.COOPERATIVE: set(),
    PersonaType.EVASIVE: set(),
    PersonaType.CONFUSED: set(),
}


class MetaOracle:
    """Computes ground-truth expected scores for eval judge outputs."""

    def __init__(self, gold_dir: Path | str | None = None) -> None:
        self._gold_dir = Path(gold_dir) if gold_dir else _GOLD_DIR
        self._gold_labels: dict[str, dict] = {}
        self._load_gold_labels()

    def _load_gold_labels(self) -> None:
        labels_dir = self._gold_dir / "labels"
        if not labels_dir.is_dir():
            logger.info("meta_oracle  no gold labels dir at %s", labels_dir)
            return
        for path in sorted(labels_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            scenario_id = path.stem
            self._gold_labels[scenario_id] = data
            logger.debug("meta_oracle  loaded gold label %s", scenario_id)

    def evaluate(
        self,
        record: ConversationRecord,
        scores: ConversationScores,
    ) -> OracleResult:
        """Produce oracle verdicts for one conversation."""
        result = OracleResult(scenario_id=record.scenario.scenario_id)
        result.compliance = self._compliance_oracle(record, scores)
        result.quality = self._quality_oracle(record, scores)
        result.handoff = self._handoff_oracle(record, scores)
        return result

    def evaluate_batch(
        self,
        records: Sequence[ConversationRecord],
        all_scores: Sequence[ConversationScores],
    ) -> list[OracleResult]:
        scores_by_id = {s.scenario_id: s for s in all_scores}
        results: list[OracleResult] = []
        for record in records:
            cs = scores_by_id.get(record.scenario.scenario_id)
            if cs is None:
                continue
            results.append(self.evaluate(record, cs))
        return results

    # ------------------------------------------------------------------
    # Compliance oracle
    # ------------------------------------------------------------------

    def _compliance_oracle(
        self,
        record: ConversationRecord,
        scores: ConversationScores,
    ) -> list[OracleVerdict]:
        from app.eval.judges import _deterministic_compliance_checks

        det_results = _deterministic_compliance_checks(record)
        verdicts: list[OracleVerdict] = []

        for rule_id, det in det_results.items():
            if det.definitive:
                verdicts.append(OracleVerdict(
                    rule_id=rule_id,
                    expected_score=1.0 if det.passed else 0.0,
                    definitive=True,
                    source="deterministic",
                    reason=det.reason,
                ))
            else:
                verdicts.append(OracleVerdict(
                    rule_id=rule_id,
                    expected_score=1.0 if det.passed else 0.0,
                    definitive=False,
                    source="deterministic",
                    reason=det.reason,
                ))

        persona_triggered = _PERSONA_TRIGGERED_RULES.get(
            record.scenario.persona.persona_type, set()
        )
        for rule_id in persona_triggered:
            existing = {v.rule_id for v in verdicts if v.definitive}
            if rule_id not in existing:
                verdicts.append(OracleVerdict(
                    rule_id=rule_id,
                    expected_score=-1.0,  # sentinel: "should be triggered, not vacuous"
                    definitive=False,
                    source="persona",
                    reason=f"persona {record.scenario.persona.persona_type.value} "
                           f"implies rule {rule_id} should be triggered",
                ))

        return verdicts

    # ------------------------------------------------------------------
    # Quality oracle (gold set only)
    # ------------------------------------------------------------------

    def _quality_oracle(
        self,
        record: ConversationRecord,
        scores: ConversationScores,
    ) -> list[OracleVerdict]:
        return self._gold_verdicts(record.scenario.scenario_id, "quality")

    # ------------------------------------------------------------------
    # Handoff oracle (gold set only)
    # ------------------------------------------------------------------

    def _handoff_oracle(
        self,
        record: ConversationRecord,
        scores: ConversationScores,
    ) -> list[OracleVerdict]:
        return self._gold_verdicts(record.scenario.scenario_id, "handoff")

    # ------------------------------------------------------------------
    # Gold set lookup
    # ------------------------------------------------------------------

    def _gold_verdicts(self, scenario_id: str, judge_type: str) -> list[OracleVerdict]:
        label_data = self._gold_labels.get(scenario_id)
        if label_data is None:
            return []
        judge_labels = label_data.get(judge_type, [])
        verdicts: list[OracleVerdict] = []
        for entry in judge_labels:
            verdicts.append(OracleVerdict(
                rule_id=str(entry["rule_id"]),
                expected_score=float(entry["expected_score"]),
                definitive=True,
                source="gold",
                reason=entry.get("reason", "gold set label"),
            ))
        return verdicts

    @property
    def has_gold_labels(self) -> bool:
        return len(self._gold_labels) > 0

    @property
    def gold_scenario_ids(self) -> list[str]:
        return list(self._gold_labels.keys())
