"""Self-learning loop driver.

Orchestrates: baseline eval -> failure mining -> LLM proposer ->
A/B comparison -> adopt/reject.  Each step records audit events and
respects the cost-ledger budget cap.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from app.eval.ab_harness import ABHarness, ABResult
from app.eval.audit_trail import AuditTrail
from app.eval.cost_ledger import BudgetExceededError, CostLedger
from app.eval.failure_miner import FailureDigest, SectionFailures, mine_failures
from app.eval.harness import EvalHarness
from app.eval.meta_ab import MetaABHarness, MetaABResult
from app.eval.meta_eval import JudgeReport, MetaEvaluator, MetaReport
from app.eval.meta_oracle import MetaOracle
from app.eval.models import EvalConfig, EvalRunResult
from app.eval.proposer import ProposalRejectedError, PromptProposer, ProposedSection
from app.services.prompt_registry import (
    PromptRegistry,
    get_prompt_registry,
    reset_prompt_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    iteration: int
    section_key: str
    proposed: ProposedSection | None = None
    ab_result: ABResult | None = None
    adopted: bool = False
    reason: str = ""
    skipped: bool = False


@dataclass
class MetaIterationResult:
    """Result of one meta-evaluation judge-improvement attempt."""

    judge_name: str
    section_key: str
    proposed: ProposedSection | None = None
    meta_ab_result: MetaABResult | None = None
    adopted: bool = False
    reason: str = ""
    skipped: bool = False


@dataclass
class SelfLearnConfig:
    eval_config: EvalConfig
    max_iterations: int = 3
    max_sections_per_iteration: int = 1
    budget_usd: float = 20.0
    proposer_model: str | None = None
    output_dir: str = "data/self_learn"


@dataclass
class SelfLearnResult:
    run_id: str
    iterations: list[IterationResult] = field(default_factory=list)
    baseline_result: EvalRunResult | None = None
    final_result: EvalRunResult | None = None
    total_cost_usd: float = 0.0
    stop_reason: str = ""
    meta_report: MetaReport | None = None
    meta_iterations: list[MetaIterationResult] = field(default_factory=list)


class SelfLearningLoop:
    """End-to-end self-learning loop."""

    def __init__(self, config: SelfLearnConfig) -> None:
        self._config = config
        self._run_id = uuid.uuid4().hex[:12]
        self._ledger = CostLedger(
            run_id=self._run_id,
            budget_usd=config.budget_usd,
        )
        self._audit = AuditTrail()
        self._proposer = PromptProposer(
            ledger=self._ledger,
            model=config.proposer_model,
        )

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def ledger(self) -> CostLedger:
        return self._ledger

    @property
    def audit(self) -> AuditTrail:
        return self._audit

    async def run(self) -> SelfLearnResult:
        """Execute the self-learning loop."""
        result = SelfLearnResult(run_id=self._run_id)

        self._audit.append_event(
            "run_started",
            run_id=self._run_id,
            extra={"config": _config_dict(self._config)},
        )

        try:
            baseline_result = await self._run_eval("baseline")
        except BudgetExceededError as exc:
            result.stop_reason = f"budget_exceeded_during_baseline: {exc}"
            self._audit.append_event("budget_exceeded", run_id=self._run_id)
            return result

        result.baseline_result = baseline_result
        self._audit.save_run_artefact(
            self._run_id,
            "baseline_summary.json",
            {"composite_mean": _composite_mean(baseline_result)},
        )

        current_result = baseline_result
        consecutive_no_improvement = 0

        for i in range(1, self._config.max_iterations + 1):
            logger.info("self_learn_iteration  %d/%d", i, self._config.max_iterations)

            if self._ledger.remaining_budget < self._estimate_iteration_cost():
                result.stop_reason = f"budget_exhaustion at iteration {i}"
                self._audit.append_event(
                    "budget_exceeded",
                    run_id=self._run_id,
                    extra={"iteration": i, "remaining": self._ledger.remaining_budget},
                )
                break

            digest = mine_failures(current_result)
            if digest.top_section is None:
                result.stop_reason = "no_failures_to_improve"
                break

            section_key = digest.top_section
            section_failures = digest.sections[0]

            iter_result = IterationResult(iteration=i, section_key=section_key)

            try:
                proposed = await self._proposer.propose(
                    registry=get_prompt_registry(),
                    section_key=section_key,
                    failures=section_failures,
                    iteration=i,
                )
            except (ProposalRejectedError, BudgetExceededError) as exc:
                iter_result.skipped = True
                iter_result.reason = str(exc)
                result.iterations.append(iter_result)
                self._audit.append_event(
                    "candidate_proposed",
                    section_key=section_key,
                    run_id=self._run_id,
                    decision="rejected",
                    rationale=str(exc),
                )
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= 2:
                    result.stop_reason = "consecutive_no_improvement"
                    break
                continue

            iter_result.proposed = proposed
            self._audit.append_event(
                "candidate_proposed",
                section_key=section_key,
                new_version=proposed.version,
                run_id=self._run_id,
                rationale=proposed.rationale,
            )

            try:
                ab_outcome = await self._compare_ab(
                    section_key, proposed, current_result,
                )
            except BudgetExceededError as exc:
                iter_result.skipped = True
                iter_result.reason = f"budget_exceeded: {exc}"
                result.iterations.append(iter_result)
                result.stop_reason = "budget_exceeded_during_ab"
                self._audit.append_event("budget_exceeded", run_id=self._run_id)
                break

            iter_result.ab_result = ab_outcome

            if ab_outcome.adopt:
                registry = get_prompt_registry()
                registry.override_section(
                    section_key,
                    proposed.content,
                    proposed.version,
                )
                iter_result.adopted = True
                iter_result.reason = ab_outcome.reason
                current_result = ab_outcome.candidate_result  # type: ignore[assignment]
                consecutive_no_improvement = 0

                self._audit.append_event(
                    "candidate_adopted",
                    section_key=section_key,
                    new_version=proposed.version,
                    run_id=self._run_id,
                    metrics_before={"composite_mean": ab_outcome.baseline_composite_mean},
                    metrics_after={"composite_mean": ab_outcome.candidate_composite_mean},
                    decision="adopted",
                    rationale=ab_outcome.reason,
                )
            else:
                iter_result.adopted = False
                iter_result.reason = ab_outcome.reason
                consecutive_no_improvement += 1

                self._audit.append_event(
                    "candidate_rejected",
                    section_key=section_key,
                    new_version=proposed.version,
                    run_id=self._run_id,
                    metrics_before={"composite_mean": ab_outcome.baseline_composite_mean},
                    metrics_after={"composite_mean": ab_outcome.candidate_composite_mean},
                    decision="rejected",
                    rationale=ab_outcome.reason,
                )

            result.iterations.append(iter_result)

            if consecutive_no_improvement >= 2:
                result.stop_reason = "consecutive_no_improvement"
                break
        else:
            result.stop_reason = "max_iterations_reached"

        # -- Darwin Godel Machine: meta-evaluation phase --
        meta_report, meta_iters = await self._run_meta_phase(current_result)
        result.meta_report = meta_report
        result.meta_iterations = meta_iters

        result.final_result = current_result
        result.total_cost_usd = self._ledger.total_cost_usd

        self._audit.append_event(
            "run_completed",
            run_id=self._run_id,
            extra={
                "total_cost_usd": result.total_cost_usd,
                "iterations_completed": len(result.iterations),
                "meta_judges_flagged": len(meta_report.flagged_judges) if meta_report else 0,
                "meta_judges_corrected": sum(1 for m in meta_iters if m.adopted),
                "stop_reason": result.stop_reason,
            },
        )

        return result

    async def _run_eval(self, label: str) -> EvalRunResult:
        """Run a full evaluation using the current global registry."""
        logger.info("self_learn_eval  label=%s", label)
        harness = EvalHarness(self._config.eval_config, ledger=self._ledger)
        return await harness.run()

    async def _compare_ab(
        self,
        section_key: str,
        proposed: ProposedSection,
        baseline_result: EvalRunResult,
    ) -> ABResult:
        """Run the candidate eval and compare against baseline."""
        registry = get_prompt_registry()
        old_section = registry.get_section(section_key)
        old_content = old_section.content
        old_version = old_section.version

        registry.override_section(section_key, proposed.content, proposed.version)

        try:
            candidate_result = await self._run_eval("candidate")
        finally:
            registry.override_section(section_key, old_content, old_version)

        ab = ABHarness(
            config=self._config.eval_config,
            ledger=self._ledger,
            audit=self._audit,
        )
        return await ab.compare(baseline_result, candidate_result)

    async def _run_meta_phase(
        self,
        current_result: EvalRunResult,
    ) -> tuple[MetaReport | None, list[MetaIterationResult]]:
        """Darwin Godel Machine: evaluate judges and propose corrections."""
        logger.info("self_learn_meta_phase  starting")
        oracle = MetaOracle()
        meta_evaluator = MetaEvaluator(oracle)
        meta_report = meta_evaluator.evaluate(current_result)

        self._audit.append_event(
            "meta_eval_completed",
            run_id=self._run_id,
            extra=meta_report.to_dict(),
        )

        meta_iterations: list[MetaIterationResult] = []

        if not meta_report.any_flagged:
            logger.info("self_learn_meta_phase  no judges flagged")
            return meta_report, meta_iterations

        for jr in meta_report.flagged_judges:
            if self._ledger.remaining_budget < self._estimate_iteration_cost():
                logger.info("self_learn_meta_phase  budget exhausted, skipping judge %s", jr.judge_name)
                break

            mi = MetaIterationResult(
                judge_name=jr.judge_name,
                section_key=jr.section_key,
            )

            failures = self._build_judge_failure_digest(jr)

            meta_iter_num = self._config.max_iterations + len(meta_iterations) + 1
            try:
                proposed = await self._proposer.propose(
                    registry=get_prompt_registry(),
                    section_key=jr.section_key,
                    failures=failures,
                    iteration=meta_iter_num,
                )
            except (ProposalRejectedError, BudgetExceededError) as exc:
                mi.skipped = True
                mi.reason = str(exc)
                meta_iterations.append(mi)
                self._audit.append_event(
                    "meta_candidate_rejected",
                    section_key=jr.section_key,
                    run_id=self._run_id,
                    rationale=str(exc),
                )
                continue

            mi.proposed = proposed
            self._audit.append_event(
                "meta_candidate_proposed",
                section_key=jr.section_key,
                new_version=proposed.version,
                run_id=self._run_id,
                rationale=proposed.rationale,
            )

            registry = get_prompt_registry()
            old_section = registry.get_section(jr.section_key)
            old_content = old_section.content
            old_version = old_section.version

            registry.override_section(jr.section_key, proposed.content, proposed.version)
            try:
                candidate_result = await self._run_eval("meta_candidate")
            except BudgetExceededError:
                registry.override_section(jr.section_key, old_content, old_version)
                mi.skipped = True
                mi.reason = "budget_exceeded"
                meta_iterations.append(mi)
                break
            finally:
                registry.override_section(jr.section_key, old_content, old_version)

            meta_ab = MetaABHarness(oracle, ledger=self._ledger, audit=self._audit)
            ab_result = meta_ab.compare(
                jr.judge_name, jr.section_key,
                current_result, candidate_result,
            )
            mi.meta_ab_result = ab_result

            if ab_result.adopt:
                registry.override_section(jr.section_key, proposed.content, proposed.version)
                mi.adopted = True
                mi.reason = ab_result.reason
                self._audit.append_event(
                    "meta_correction_adopted",
                    section_key=jr.section_key,
                    new_version=proposed.version,
                    run_id=self._run_id,
                    metrics_before={"accuracy": ab_result.baseline_accuracy},
                    metrics_after={"accuracy": ab_result.candidate_accuracy},
                    decision="adopted",
                    rationale=ab_result.reason,
                )
            else:
                mi.adopted = False
                mi.reason = ab_result.reason
                self._audit.append_event(
                    "meta_correction_rejected",
                    section_key=jr.section_key,
                    new_version=proposed.version,
                    run_id=self._run_id,
                    metrics_before={"accuracy": ab_result.baseline_accuracy},
                    metrics_after={"accuracy": ab_result.candidate_accuracy},
                    decision="rejected",
                    rationale=ab_result.reason,
                )

            meta_iterations.append(mi)

        return meta_report, meta_iterations

    @staticmethod
    def _build_judge_failure_digest(jr: JudgeReport) -> SectionFailures:
        """Convert meta-eval flaws into a SectionFailures for the proposer."""
        from app.eval.failure_miner import FailureExcerpt

        excerpts = []
        for flaw in jr.flaws:
            excerpts.append(FailureExcerpt(
                scenario_id="meta_eval",
                rule_id=flaw.rule_id or "aggregate",
                rule_name=flaw.flaw_type,
                score=1.0 - jr.oracle_accuracy,
                explanation=flaw.description,
                source="meta_evaluation",
            ))

        return SectionFailures(
            section_key=jr.section_key,
            weighted_count=float(len(jr.flaws)) * 3.0,
            excerpts=excerpts,
        )

    def _estimate_iteration_cost(self) -> float:
        """Conservative estimate of cost for one full iteration."""
        avg = self._ledger.total_cost_usd / max(self._ledger.total_calls, 1)
        calls_per_eval = max(self._ledger.total_calls, 10)
        return avg * calls_per_eval * 1.5


def _composite_mean(result: EvalRunResult) -> float:
    vals = [s.composite_score for s in result.scores]
    return sum(vals) / len(vals) if vals else 0.0


def _config_dict(config: SelfLearnConfig) -> dict[str, Any]:
    return {
        "eval_config": config.eval_config.model_dump(mode="json"),
        "max_iterations": config.max_iterations,
        "budget_usd": config.budget_usd,
        "proposer_model": config.proposer_model,
    }
