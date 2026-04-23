"""CLI entry point for the self-learning loop.

Usage::

    python -m app.eval.run_self_learn --seed 42 --n-per-persona 2 --iterations 3

Writes artefacts under ``data/self_learn/<run_id>/``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

from app.eval.models import EvalConfig
from app.eval.self_learn import SelfLearnConfig, SelfLearningLoop
from app.logging_config import setup_logging
from app.services.prompt_registry import get_prompt_registry, reset_prompt_registry


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the self-learning prompt-improvement loop.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-persona", type=int, default=2)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--budget", type=float, default=20.0)
    p.add_argument("--sim-model", type=str, default=None)
    p.add_argument("--judge-model", type=str, default=None)
    p.add_argument("--proposer-model", type=str, default=None)
    p.add_argument("--output", type=str, default="data/self_learn")
    return p.parse_args()


async def _main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    args = _parse_args()

    eval_config = EvalConfig(
        seed=args.seed,
        n_per_persona=args.n_per_persona,
        sim_model=args.sim_model,
        judge_model=args.judge_model,
    )

    config = SelfLearnConfig(
        eval_config=eval_config,
        max_iterations=args.iterations,
        budget_usd=args.budget,
        proposer_model=args.proposer_model,
        output_dir=args.output,
    )

    reset_prompt_registry()
    from app.eval.audit_trail import AuditTrail
    audit = AuditTrail()
    get_prompt_registry(audit_trail=audit)

    loop = SelfLearningLoop(config)
    result = await loop.run()

    run_dir = Path(args.output) / result.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "config.json", {
        "eval_config": eval_config.model_dump(mode="json"),
        "max_iterations": config.max_iterations,
        "budget_usd": config.budget_usd,
        "proposer_model": config.proposer_model,
    })

    if result.baseline_result:
        _write_json(
            run_dir / "baseline_summary.json",
            _eval_summary(result.baseline_result),
        )

    timeline: list[dict] = []
    for i, ir in enumerate(result.iterations):
        iter_dir = run_dir / f"iteration_{ir.iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        iter_data: dict = {
            "iteration": ir.iteration,
            "section_key": ir.section_key,
            "adopted": ir.adopted,
            "reason": ir.reason,
            "skipped": ir.skipped,
        }
        if ir.proposed:
            _write_text(iter_dir / "candidate_prompt.txt", ir.proposed.content)
            iter_data["proposed_version"] = ir.proposed.version
            iter_data["change_summary"] = ir.proposed.change_summary
            iter_data["rationale"] = ir.proposed.rationale

        if ir.ab_result:
            ab_data = {
                "baseline_composite_mean": ir.ab_result.baseline_composite_mean,
                "candidate_composite_mean": ir.ab_result.candidate_composite_mean,
                "p_value": ir.ab_result.p_value,
                "cohen_d": ir.ab_result.cohen_d,
                "significant": ir.ab_result.significant,
                "adopt": ir.ab_result.adopt,
                "reason": ir.ab_result.reason,
                "compliance_regressions": [
                    {
                        "rule_id": r.rule_id,
                        "baseline_pass_rate": r.baseline_pass_rate,
                        "candidate_pass_rate": r.candidate_pass_rate,
                        "delta": r.delta,
                        "regressed": r.regressed,
                    }
                    for r in ir.ab_result.compliance_regressions
                ],
            }
            _write_json(iter_dir / "ab_result.json", ab_data)
            iter_data.update(ab_data)

        _write_json(iter_dir / "decision.json", iter_data)
        timeline.append(iter_data)

    _write_json(run_dir / "evolution_timeline.json", timeline)

    from app.services.prompt_registry import get_prompt_registry as gpr
    _write_json(run_dir / "final_registry_snapshot.json", gpr().snapshot())

    _write_json(run_dir / "cost_report.json", {
        "total_cost_usd": result.total_cost_usd,
        "ledger_calls": loop.ledger.total_calls,
        "by_role": loop.ledger.cost_by_role(),
    })

    logger.info("Self-learning run complete  run_id=%s  dir=%s", result.run_id, run_dir)
    logger.info(
        "  iterations=%d  adopted=%d  stop_reason=%s  cost=$%.4f",
        len(result.iterations),
        sum(1 for ir in result.iterations if ir.adopted),
        result.stop_reason,
        result.total_cost_usd,
    )


def _eval_summary(r) -> dict:
    vals = [s.composite_score for s in r.scores]
    mean = sum(vals) / len(vals) if vals else 0.0
    return {
        "n_conversations": len(r.conversations),
        "n_scored": len(r.scores),
        "composite_mean": mean,
        "cost_usd": r.cost.estimated_cost_usd,
    }


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
