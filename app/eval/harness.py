"""Evaluation harness orchestrator.

Ties together scenario generation, conversation simulation, judging,
metric computation, and statistics into a single ``run`` entry point.
Tracks real token-level cost via ``CostLedger``.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.eval.borrower_sim import BorrowerSimulator
from app.eval.conversation_runner import run_conversation
from app.eval.cost_ledger import CostLedger
from app.eval.judges import (
    ComplianceJudge,
    HandoffJudge,
    QualityJudge,
    score_conversation,
)
from app.eval.metrics import compute_all_metrics
from app.eval.models import (
    ConversationRecord,
    ConversationScores,
    CostReport,
    EvalConfig,
    EvalRunResult,
)
from app.eval.scenarios import build_scenario_batch
from app.eval.stats import compute_summary

logger = logging.getLogger(__name__)


class EvalHarness:
    """Main orchestrator for a full evaluation run."""

    def __init__(
        self,
        config: EvalConfig,
        ledger: CostLedger | None = None,
    ) -> None:
        self.config = config
        self._ledger = ledger or CostLedger(run_id=uuid.uuid4().hex[:12])
        self._simulator = BorrowerSimulator(model=config.sim_model, ledger=self._ledger)
        self._compliance = ComplianceJudge(model=config.judge_model, ledger=self._ledger)
        self._quality = QualityJudge(model=config.judge_model, ledger=self._ledger)
        self._handoff = HandoffJudge(model=config.judge_model, ledger=self._ledger)

    async def run(self) -> EvalRunResult:
        """Execute the full evaluation pipeline.

        1. Generate scenarios (deterministic from seed).
        2. Run simulated conversations through the real agent code.
        3. Score each conversation with the three judges.
        4. Compute aggregate metrics and statistics.
        5. Build cost report.
        """
        scenarios = build_scenario_batch(
            seed=self.config.seed,
            n_per_persona=self.config.n_per_persona,
        )
        logger.info(
            "eval_harness_start  seed=%d  n_per_persona=%d  total_scenarios=%d",
            self.config.seed, self.config.n_per_persona, len(scenarios),
        )

        conversations: list[ConversationRecord] = []
        for i, scenario in enumerate(scenarios):
            logger.info(
                "eval_running_conversation  %d/%d  scenario=%s  persona=%s",
                i + 1, len(scenarios),
                scenario.scenario_id,
                scenario.persona.persona_type.value,
            )
            try:
                record = await run_conversation(scenario, self._simulator)
                conversations.append(record)
            except Exception:
                logger.exception(
                    "eval_conversation_failed  scenario=%s", scenario.scenario_id,
                )

        scored: list[ConversationScores] = []
        for i, record in enumerate(conversations):
            logger.info(
                "eval_scoring_conversation  %d/%d  scenario=%s",
                i + 1, len(conversations), record.scenario.scenario_id,
            )
            try:
                cs = await score_conversation(
                    record, self._compliance, self._quality, self._handoff,
                )
                scored.append(cs)
            except Exception:
                logger.exception(
                    "eval_scoring_failed  scenario=%s", record.scenario.scenario_id,
                )

        metrics = compute_all_metrics(scored)

        composite_values = [cs.composite_score for cs in scored]
        summary = compute_summary(composite_values)
        logger.info("eval_harness_summary  %s", summary)

        cost = self._build_cost_report()

        result = EvalRunResult(
            config=self.config,
            conversations=conversations,
            scores=scored,
            metrics=metrics,
            cost=cost,
        )
        logger.info(
            "eval_harness_complete  conversations=%d  scored=%d  "
            "composite_mean=%.3f  composite_std=%.3f  total_llm_calls=%d",
            len(conversations), len(scored),
            summary.get("mean", 0), summary.get("std", 0),
            cost.total_calls,
        )
        return result

    def _build_cost_report(self) -> CostReport:
        calls_by_role = self._ledger.calls_by_role()
        cost_by_role = self._ledger.cost_by_role()

        return CostReport(
            simulation_calls=calls_by_role.get("sim", 0),
            evaluation_calls=calls_by_role.get("judge", 0),
            prompt_generation_calls=calls_by_role.get("proposer", 0),
            total_calls=self._ledger.total_calls,
            input_tokens=self._ledger.total_input_tokens,
            output_tokens=self._ledger.total_output_tokens,
            by_role=cost_by_role,
            estimated_cost_usd=round(self._ledger.total_cost_usd, 6),
        )
