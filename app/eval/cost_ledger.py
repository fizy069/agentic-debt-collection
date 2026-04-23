"""Token-accurate cost tracking with a hard budget cap.

Records every LLM call (role, model, input/output tokens, timestamp) and
computes spend from a per-model pricing table.  The ``$20`` default cap
is overridable via the ``MAX_EVAL_COST_USD`` environment variable.

Persists a JSONL call log at ``data/audit/cost_ledger/<run_id>.jsonl``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_BUDGET_USD = 20.0

# $/1M tokens — conservative upper bounds; update as pricing evolves.
_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_1M, output_per_1M)
    "claude-haiku-4-5":      (0.80,  4.00),
    "claude-3-5-haiku":      (0.80,  4.00),
    "claude-sonnet-4-5":     (3.00, 15.00),
    "claude-3-5-sonnet":     (3.00, 15.00),
    "gpt-4o-mini":           (0.15,  0.60),
    "gpt-4o":                (2.50, 10.00),
    "stub":                  (0.00,  0.00),
}

_FALLBACK_INPUT_PER_1M = 3.00
_FALLBACK_OUTPUT_PER_1M = 15.00


class BudgetExceededError(RuntimeError):
    """Raised when cumulative spend would exceed the hard budget cap."""

    def __init__(self, spent: float, budget: float) -> None:
        self.spent = spent
        self.budget = budget
        super().__init__(
            f"Budget exceeded: ${spent:.4f} spent against ${budget:.2f} cap"
        )


@dataclass
class LLMCallRecord:
    role: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


class CostLedger:
    """Accumulates per-call token counts and dollar spend."""

    def __init__(
        self,
        run_id: str,
        budget_usd: float | None = None,
        base_dir: str | Path = "data/audit/cost_ledger",
    ) -> None:
        self._run_id = run_id
        env_budget = os.getenv("MAX_EVAL_COST_USD")
        self._budget = budget_usd or (float(env_budget) if env_budget else _DEFAULT_BUDGET_USD)
        self._calls: list[LLMCallRecord] = []
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

        self._log_path = Path(base_dir) / f"{run_id}.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self._budget - self._total_cost)

    @property
    def total_input_tokens(self) -> int:
        return self._total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._total_output_tokens

    @property
    def total_calls(self) -> int:
        return len(self._calls)

    def record(
        self,
        role: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> LLMCallRecord:
        """Record a completed LLM call and accumulate cost."""
        cost = self._compute_cost(model, input_tokens, output_tokens)
        entry = LLMCallRecord(
            role=role,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._calls.append(entry)
        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens

        self._persist(entry)
        return entry

    def check_budget_or_raise(self) -> None:
        """Raise ``BudgetExceededError`` if spend has hit the cap."""
        if self._total_cost >= self._budget:
            raise BudgetExceededError(self._total_cost, self._budget)

    def cost_by_role(self) -> dict[str, float]:
        """Aggregate spend per role (sim, judge, proposer, agent)."""
        totals: dict[str, float] = {}
        for c in self._calls:
            totals[c.role] = totals.get(c.role, 0.0) + c.cost_usd
        return totals

    def calls_by_role(self) -> dict[str, int]:
        """Aggregate call count per role."""
        totals: dict[str, int] = {}
        for c in self._calls:
            totals[c.role] = totals.get(c.role, 0) + 1
        return totals

    def _compute_cost(
        self, model: str, input_tokens: int, output_tokens: int,
    ) -> float:
        for key, (inp_price, out_price) in _PRICING.items():
            if key in model:
                return (
                    input_tokens * inp_price / 1_000_000
                    + output_tokens * out_price / 1_000_000
                )
        return (
            input_tokens * _FALLBACK_INPUT_PER_1M / 1_000_000
            + output_tokens * _FALLBACK_OUTPUT_PER_1M / 1_000_000
        )

    def _persist(self, entry: LLMCallRecord) -> None:
        try:
            with open(self._log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(entry)) + "\n")
        except OSError:
            logger.warning("cost_ledger_persist_failed  path=%s", self._log_path)
