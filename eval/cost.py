"""Cost estimation and a running actual-spend accumulator (issue #47).

Conor's standing constraint on this benchmark is staged, predictable spend. The
harness had no cost visibility until after a run finished. This module gives a
run two things:

1. **A preflight ESTIMATE** (:func:`estimate_run_cost`) computed BEFORE the first
   call from the resolved roster (models x scenarios x reliability-runs) and the
   per-evaluation token priors in ``eval.config`` plus the judge panel. It is a
   conservative upper-ish figure so the printed number over-states rather than
   under-states the bill.

2. **A running ACTUAL-spend accumulator** (:class:`CostAccumulator`) fed from the
   token usage captured per call during the run (agent + simulators + judges).
   When a ``--max-cost`` cap is set, the run loop checks the accumulator after
   each completed evaluation and stops submitting new ones once the cap is
   crossed, letting in-flight work finish and still writing all artifacts.

Both paths price tokens with the same ``TOKEN_COSTS`` table, so the estimate and
the actual are computed on a like-for-like basis (the only difference is priors
vs. measured tokens).
"""

import logging
import threading

from eval.config import (
    JUDGES,
    PER_EVAL_TOKEN_PRIORS,
    SEPARATE_JUDGE_INPUT_MULTIPLIER,
    TOKEN_COSTS,
)

logger = logging.getLogger(__name__)


# Exit code used when a run stops early because the --max-cost budget was hit.
# Distinct from 0 (clean finish) and 1 (generic error / SystemExit) so a wrapper
# script or CI can tell "stopped on budget, completed work is on disk" apart from
# a crash.
BUDGET_EXCEEDED_EXIT_CODE = 7


def token_cost(model_id: str, input_tokens: float, output_tokens: float) -> float:
    """Price input/output tokens for a model id using TOKEN_COSTS (USD).

    A model id absent from TOKEN_COSTS contributes $0 (the same fallback the row
    cost uses) and is logged at DEBUG — the preflight already warns loudly about
    missing pricing, so this stays quiet to avoid per-call log spam.
    """
    costs = TOKEN_COSTS.get(model_id)
    if costs is None:
        logger.debug("No TOKEN_COSTS entry for %s; pricing it at $0", model_id)
        return 0.0
    return input_tokens * costs["input"] / 1_000_000 + output_tokens * costs["output"] / 1_000_000


def _judge_estimate_per_eval(judge_keys, separate_judge_calls: bool) -> float:
    """Estimated judge-side cost for ONE evaluation across the panel.

    Each requested judge is priced from the per-eval judge priors at the judge's
    own configured model id. The combined path sends the transcript once; the
    separate path sends it once per dimension, so judge INPUT is multiplied by
    ``SEPARATE_JUDGE_INPUT_MULTIPLIER`` in that mode (output is per-dimension
    either way and is left at the prior).
    """
    input_mult = SEPARATE_JUDGE_INPUT_MULTIPLIER if separate_judge_calls else 1.0
    total = 0.0
    for key in judge_keys:
        judge = JUDGES[key]
        total += token_cost(
            judge.model_id,
            PER_EVAL_TOKEN_PRIORS["judge_input"] * input_mult,
            PER_EVAL_TOKEN_PRIORS["judge_output"],
        )
    return total


def _sim_estimate_per_eval(user_sim_model_id: str, tool_sim_model_id: str) -> float:
    """Estimated simulator-side cost for ONE evaluation.

    The ``sim_input`` / ``sim_output`` priors are the SUMMED user+tool simulator
    tokens for one conversation. We split that budget evenly across the two
    simulator model ids so a sensitivity run that swaps one simulator to a
    different-priced family (issue #50) is reflected in the estimate.
    """
    half_in = PER_EVAL_TOKEN_PRIORS["sim_input"] / 2
    half_out = PER_EVAL_TOKEN_PRIORS["sim_output"] / 2
    return token_cost(user_sim_model_id, half_in, half_out) + token_cost(
        tool_sim_model_id, half_in, half_out
    )


def estimate_run_cost(
    *,
    models,
    n_scenarios: int,
    reliability_runs: int,
    judge_keys,
    user_sim_model_id: str,
    tool_sim_model_id: str,
    separate_judge_calls: bool,
) -> dict:
    """Estimate total run cost (USD) BEFORE any call, from priors (issue #47).

    ``models`` is the resolved roster (list of ``{"name", "model_id", ...}``
    dicts). The estimate is::

        per-model agent cost  = n_scenarios * reliability_runs * agent_priors
        sim + judge cost      = n_evals_total * (sim_per_eval + judge_per_eval)

    where ``n_evals_total = len(models) * n_scenarios * reliability_runs``. The
    simulators and judges are priced once per evaluation regardless of which model
    is under test, so they scale with the total evaluation count.

    Returns a dict with the total and a per-model agent breakdown so the run can
    log the estimate at startup and a reader can see where the money goes::

        {
            "total_usd": float,
            "n_evals_total": int,
            "agent_by_model": {model_name: usd, ...},
            "agent_total_usd": float,
            "sim_total_usd": float,
            "judge_total_usd": float,
        }
    """
    evals_per_model = n_scenarios * reliability_runs
    n_evals_total = len(models) * evals_per_model

    agent_by_model: dict[str, float] = {}
    for m in models:
        agent_by_model[m["name"]] = evals_per_model * token_cost(
            m["model_id"],
            PER_EVAL_TOKEN_PRIORS["agent_input"],
            PER_EVAL_TOKEN_PRIORS["agent_output"],
        )
    agent_total = sum(agent_by_model.values())

    sim_per_eval = _sim_estimate_per_eval(user_sim_model_id, tool_sim_model_id)
    judge_per_eval = _judge_estimate_per_eval(judge_keys, separate_judge_calls)
    sim_total = n_evals_total * sim_per_eval
    judge_total = n_evals_total * judge_per_eval

    return {
        "total_usd": agent_total + sim_total + judge_total,
        "n_evals_total": n_evals_total,
        "agent_by_model": agent_by_model,
        "agent_total_usd": agent_total,
        "sim_total_usd": sim_total,
        "judge_total_usd": judge_total,
    }


class CostAccumulator:
    """Thread-safe running tally of ACTUAL spend during a run (issue #47).

    Models are evaluated on a thread pool, so the accumulator must be safe to add
    to concurrently. Each completed evaluation adds its measured agent + simulator
    + judge cost; the run loop then reads :meth:`total` to decide whether the
    ``--max-cost`` cap has been crossed.

    ``max_cost`` of ``None`` means no cap (the default — a rehearsal sets one).
    :meth:`exceeded` is always ``False`` when there is no cap.
    """

    def __init__(self, max_cost: float | None = None):
        self._lock = threading.Lock()
        self._total = 0.0
        self._by_model: dict[str, float] = {}
        self.max_cost = max_cost

    def add(self, model_name: str, cost_usd: float) -> float:
        """Add one evaluation's cost; return the new running total (USD)."""
        with self._lock:
            self._total += cost_usd
            self._by_model[model_name] = self._by_model.get(model_name, 0.0) + cost_usd
            return self._total

    def total(self) -> float:
        with self._lock:
            return self._total

    def by_model(self) -> dict:
        with self._lock:
            return dict(self._by_model)

    def model_total(self, model_name: str) -> float:
        with self._lock:
            return self._by_model.get(model_name, 0.0)

    def exceeded(self) -> bool:
        """True iff a cap is set and the running total has reached/crossed it."""
        if self.max_cost is None:
            return False
        with self._lock:
            return self._total >= self.max_cost
