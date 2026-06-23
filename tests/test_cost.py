"""Tests for the cost estimator and the --max-cost budget guard (issue #47).

All offline: estimator math is pure, and the budget-abort path drives
run_eval.main with stubbed model evaluation (no API calls).
"""

import json

import pytest

from eval.config import PER_EVAL_TOKEN_PRIORS
from eval.cost import (
    BUDGET_EXCEEDED_EXIT_CODE,
    CostAccumulator,
    estimate_run_cost,
    token_cost,
)


class TestTokenCost:
    def test_prices_from_table(self):
        # gpt-5.5: $5 in / $30 out per million.
        assert token_cost("gpt-5.5-2026-04-23", 1_000_000, 0) == pytest.approx(5.0)
        assert token_cost("gpt-5.5-2026-04-23", 0, 1_000_000) == pytest.approx(30.0)
        assert token_cost("gpt-5.5-2026-04-23", 2_000_000, 1_000_000) == pytest.approx(40.0)

    def test_unknown_model_is_zero(self):
        assert token_cost("no-such-model", 1_000_000, 1_000_000) == 0.0


class TestEstimator:
    def test_single_model_single_eval_math(self):
        # One model, one scenario, one run, one judge — hand-computable.
        model = {"name": "GPT-5.5", "model_id": "gpt-5.5-2026-04-23", "provider": "openai"}
        est = estimate_run_cost(
            models=[model],
            n_scenarios=1,
            reliability_runs=1,
            judge_keys=["opus"],
            user_sim_model_id="gpt-4.1-mini-2025-04-14",
            tool_sim_model_id="gpt-4.1-mini-2025-04-14",
            separate_judge_calls=False,
        )
        # Agent: priors * gpt-5.5 price.
        agent = token_cost(
            "gpt-5.5-2026-04-23",
            PER_EVAL_TOKEN_PRIORS["agent_input"],
            PER_EVAL_TOKEN_PRIORS["agent_output"],
        )
        assert est["agent_by_model"]["GPT-5.5"] == pytest.approx(agent)
        assert est["agent_total_usd"] == pytest.approx(agent)
        assert est["n_evals_total"] == 1
        # Judge: opus priced from the priors (combined path, single judge).
        judge = token_cost(
            "claude-opus-4-6",
            PER_EVAL_TOKEN_PRIORS["judge_input"],
            PER_EVAL_TOKEN_PRIORS["judge_output"],
        )
        assert est["judge_total_usd"] == pytest.approx(judge)
        assert est["judge_total_usd"] > 0  # judges are now priced
        assert est["total_usd"] == pytest.approx(
            est["agent_total_usd"] + est["sim_total_usd"] + est["judge_total_usd"]
        )

    def test_scales_with_scenarios_and_runs(self):
        model = {"name": "GPT-5.5", "model_id": "gpt-5.5-2026-04-23", "provider": "openai"}
        base = estimate_run_cost(
            models=[model],
            n_scenarios=1,
            reliability_runs=1,
            judge_keys=["opus"],
            user_sim_model_id="gpt-4.1-mini-2025-04-14",
            tool_sim_model_id="gpt-4.1-mini-2025-04-14",
            separate_judge_calls=False,
        )
        scaled = estimate_run_cost(
            models=[model],
            n_scenarios=5,
            reliability_runs=3,
            judge_keys=["opus"],
            user_sim_model_id="gpt-4.1-mini-2025-04-14",
            tool_sim_model_id="gpt-4.1-mini-2025-04-14",
            separate_judge_calls=False,
        )
        # 5 scenarios * 3 runs = 15x the single-eval estimate.
        assert scaled["total_usd"] == pytest.approx(base["total_usd"] * 15)
        assert scaled["n_evals_total"] == 15

    def test_separate_judge_path_costs_more_input(self):
        model = {"name": "GPT-5.5", "model_id": "gpt-5.5-2026-04-23", "provider": "openai"}
        kwargs = dict(
            models=[model],
            n_scenarios=1,
            reliability_runs=1,
            judge_keys=["opus"],
            user_sim_model_id="gpt-4.1-mini-2025-04-14",
            tool_sim_model_id="gpt-4.1-mini-2025-04-14",
        )
        combined = estimate_run_cost(separate_judge_calls=False, **kwargs)
        separate = estimate_run_cost(separate_judge_calls=True, **kwargs)
        assert separate["judge_total_usd"] > combined["judge_total_usd"]


class TestAccumulator:
    def test_add_and_total(self):
        acc = CostAccumulator(max_cost=None)
        assert acc.total() == 0.0
        acc.add("GPT-5.5", 1.5)
        acc.add("GPT-5.5", 0.5)
        acc.add("Haiku", 2.0)
        assert acc.total() == pytest.approx(4.0)
        assert acc.model_total("GPT-5.5") == pytest.approx(2.0)
        assert acc.by_model() == {"GPT-5.5": pytest.approx(2.0), "Haiku": pytest.approx(2.0)}

    def test_no_cap_never_exceeds(self):
        acc = CostAccumulator(max_cost=None)
        acc.add("m", 1_000_000.0)
        assert acc.exceeded() is False

    def test_cap_exceeded_at_or_past_threshold(self):
        acc = CostAccumulator(max_cost=5.0)
        acc.add("m", 4.99)
        assert acc.exceeded() is False
        acc.add("m", 0.02)  # 5.01 >= 5.0
        assert acc.exceeded() is True


# --------------------------------------------------------------------------- #
# --max-cost integration: budget stop leaves completed work + distinct exit.
# --------------------------------------------------------------------------- #
def _stub_runner_loop(run_eval, monkeypatch, per_eval_cost):
    """Patch evaluate_scenario so each call returns a fixed (row, cost), no API.

    The row is the minimal shape build_result_row / downstream code reads.
    """

    def fake_evaluate(runner, scenario, agent_spec, tracer, judge_keys, **kw):
        row = {
            "model": agent_spec.name,
            "scenario_id": scenario.id,
            "domain": scenario.domain.value,
            "category": scenario.category,
            "efficacy": 0.5,
            "cost_usd": 0.0,
            "latency_ms": 1.0,
        }
        return row, per_eval_cost

    monkeypatch.setattr(run_eval, "evaluate_scenario", fake_evaluate)


def _scenarios(n):
    from eval.config import Domain
    from eval.simulation.runner import Scenario

    return [
        Scenario(
            id=f"banking_x_{i:04d}_aaaa1111",
            domain=Domain.BANKING,
            persona={"name": "T"},
            user_goals=["g"],
            tools=[{"name": "lookup", "description": "d"}],
            category="adaptive_tool_use",
            initial_message="hi",
            ground_truth={"accounts": {"a1": {"balance": 100}}},
            expected_state_changes=[{"path": "accounts.a1.balance", "expected": 50}],
        )
        for i in range(n)
    ]


def test_max_cost_aborts_with_distinct_exit_and_persists(tmp_path, monkeypatch):
    import scripts.run_eval as run_eval

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    output = results_dir / "results_20260610_999999.parquet"

    scenarios = _scenarios(10)
    monkeypatch.setattr(run_eval, "load_scenarios", lambda domain, seed: (scenarios, []))
    monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
    monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
    # Avoid building a real SimulationRunner (would create API clients).
    monkeypatch.setattr(run_eval, "SimulationRunner", lambda *a, **k: object())
    # Each evaluation "costs" $1; cap at $3 -> stop after ~3 evaluations.
    _stub_runner_loop(run_eval, monkeypatch, per_eval_cost=1.0)

    argv = [
        "run_eval",
        "--domains",
        "banking",
        "--models",
        "GPT-5.5",
        "--reliability-runs",
        "1",
        "--no-artifacts",
        "--parallel-models",
        "1",
        "--max-cost",
        "3",
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc:
        run_eval.main()
    assert exc.value.code == BUDGET_EXCEEDED_EXIT_CODE

    # Completed work is on disk: parquet + manifest written before the exit.
    assert output.exists()
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cost"]["budget_stopped"] is True
    assert manifest["cost"]["max_cost_usd"] == 3.0
    # Stopped early: fewer than the full 10 evaluations ran (3-ish before the cap).
    assert manifest["cost"]["actual_usd"] >= 3.0
    import pandas as pd

    df = pd.read_parquet(output)
    assert 0 < len(df) < 10


def test_no_cap_runs_to_completion_exit_zero(tmp_path, monkeypatch):
    import scripts.run_eval as run_eval

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    output = results_dir / "results_20260610_888888.parquet"

    scenarios = _scenarios(4)
    monkeypatch.setattr(run_eval, "load_scenarios", lambda domain, seed: (scenarios, []))
    monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
    monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
    monkeypatch.setattr(run_eval, "SimulationRunner", lambda *a, **k: object())
    _stub_runner_loop(run_eval, monkeypatch, per_eval_cost=1.0)

    argv = [
        "run_eval",
        "--domains",
        "banking",
        "--models",
        "GPT-5.5",
        "--reliability-runs",
        "1",
        "--no-artifacts",
        "--parallel-models",
        "1",
        "--output",
        str(output),
    ]
    monkeypatch.setattr("sys.argv", argv)

    # No cap -> no SystemExit raised; all 4 scenarios complete.
    run_eval.main()
    import pandas as pd

    df = pd.read_parquet(output)
    assert len(df) == 4
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cost"]["budget_stopped"] is False
    assert manifest["cost"]["actual_usd"] == pytest.approx(4.0)
