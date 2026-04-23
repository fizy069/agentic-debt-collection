"""CLI entry point for the evaluation harness.

Usage::

    python -m app.eval.run_eval --seed 42 --n-per-persona 2 --output data/eval_results

Produces four JSON files in the output directory:
  - conversations.json  — raw per-conversation records
  - scores.json         — per-conversation judge scores
  - summary.json        — aggregate metrics + statistics
  - cost_report.json    — LLM API spend breakdown
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

from app.eval.harness import EvalHarness
from app.eval.models import EvalConfig
from app.eval.stats import compute_summary
from app.logging_config import setup_logging


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the self-learning evaluation harness.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility.")
    parser.add_argument(
        "--n-per-persona", type=int, default=2,
        help="Number of scenarios per borrower persona.",
    )
    parser.add_argument(
        "--sim-model", type=str, default=None,
        help=(
            "Model for the borrower simulator.  "
            "Default: EVAL_SIM_MODEL env, then the active provider's default "
            "(claude-haiku-4-5 for Anthropic, gpt-4o-mini for OpenAI)."
        ),
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
        help=(
            "Model for the judges.  "
            "Default: EVAL_JUDGE_MODEL env, then the active provider's default "
            "(claude-haiku-4-5 for Anthropic, gpt-4o-mini for OpenAI)."
        ),
    )
    parser.add_argument(
        "--output", type=str, default="data/eval_results",
        help="Output directory for result JSON files.",
    )
    return parser.parse_args()


async def _main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)

    args = _parse_args()
    config = EvalConfig(
        seed=args.seed,
        n_per_persona=args.n_per_persona,
        sim_model=args.sim_model,
        judge_model=args.judge_model,
        output_dir=args.output,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting evaluation harness  config=%s", config.model_dump())

    harness = EvalHarness(config)
    result = await harness.run()

    conversations_path = output_dir / "conversations.json"
    conversations_path.write_text(
        json.dumps(
            [c.model_dump(mode="json") for c in result.conversations],
            indent=2,
        ),
        encoding="utf-8",
    )

    scores_path = output_dir / "scores.json"
    scores_path.write_text(
        json.dumps(
            [s.model_dump(mode="json") for s in result.scores],
            indent=2,
        ),
        encoding="utf-8",
    )

    composite_values = [s.composite_score for s in result.scores]
    summary_data = {
        "metrics": [m.model_dump(mode="json") for m in result.metrics],
        "composite_distribution": compute_summary(composite_values),
        "n_conversations": len(result.conversations),
        "n_scored": len(result.scores),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")

    cost_path = output_dir / "cost_report.json"
    cost_path.write_text(
        json.dumps(result.cost.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    logger.info("Results written to %s", output_dir.resolve())
    logger.info(
        "Summary: %d conversations, composite mean=%.3f, estimated cost=$%.4f",
        len(result.conversations),
        summary_data["composite_distribution"].get("mean", 0),
        result.cost.estimated_cost_usd,
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
