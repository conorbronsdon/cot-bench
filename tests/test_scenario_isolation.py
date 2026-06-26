"""Per-scenario harness-exception isolation (issue #88).

A single scenario's harness fault (a provider 5xx, a scoring bug, a malformed
coded transition) must NOT kill a whole model's run and discard its other
completed, already-paid-for rows. The run loop catches the exception, records the
failed run as an ``ungradable`` row, and continues. These tests pin that the
recorded row is schema-correct + classified ungradable, and that the loop
survives a raising evaluation and still returns the model's other rows.

Fully offline: evaluate_scenario is stubbed; no API calls, no SimulationRunner.
"""

import scripts.run_eval as run_eval
from eval.config import Domain
from eval.providers.registry import ModelSpec
from eval.simulation.runner import Scenario
from scripts.run_eval import OUTCOME_UNGRADABLE, ungradable_row


def _scenario(i=0):
    return Scenario(
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


_SPEC = ModelSpec(name="GPT-5.5", model_id="x", provider="openai")


class TestUngradableRow:
    def test_row_is_ungradable_with_zeroed_grade(self):
        row = ungradable_row(_scenario(0), _SPEC, "RuntimeError: boom")
        assert row["outcome"] == OUTCOME_UNGRADABLE
        assert row["efficacy"] == 0.0
        assert row["scenario_id"] == "banking_x_0000_aaaa1111"
        assert row["model"] == "GPT-5.5"
        assert row["domain"] == "banking"
        # Agent-only published Cost dimension is 0 for a run that never ran.
        assert row["cost_usd"] == 0.0

    def test_row_has_full_schema(self):
        # Built through the SAME build_result_row as the live path, so the row
        # carries every column a normal row does (no schema drift that would make
        # the parquet ragged / break aggregation).
        row = ungradable_row(_scenario(0), _SPEC, "RuntimeError: boom")
        for col in (
            "scenario_id",
            "domain",
            "model",
            "efficacy",
            "task_completion",
            "tool_selection",
            "state_score",
            "state_gradable",
            "cost_usd",
            "latency_ms",
            "total_turns",
            "tc_agreement",
            "ts_agreement",
            "outcome",
        ):
            assert col in row, f"ungradable row missing column {col!r}"


class TestRunLoopIsolatesAScenarioFault:
    def test_one_raising_scenario_does_not_kill_the_model(self, monkeypatch):
        scenarios = [_scenario(0), _scenario(1)]

        def fake_evaluate(runner, scenario, agent_spec, tracer, judge_keys, **kw):
            if scenario.id == scenarios[0].id:
                raise RuntimeError("boom in scoring")
            row = {
                "model": agent_spec.name,
                "scenario_id": scenario.id,
                "domain": scenario.domain.value,
                "efficacy": 0.8,
                "outcome": "pass",
                "cost_usd": 0.0,
                "latency_ms": 1.0,
            }
            return row, 0.001

        monkeypatch.setattr(run_eval, "evaluate_scenario", fake_evaluate)

        # runner=object() skips building a real SimulationRunner (no API client).
        results = run_eval._run_model_scenarios(
            {"name": "GPT-5.5", "model_id": "x", "provider": "openai"},
            [Domain.BANKING],
            {Domain.BANKING: scenarios},
            reliability_runs=1,
            judge_keys=["opus"],
            tracer=None,
            run_id="r",
            artifacts_root=None,
            runner=object(),
        )

        by_id = {r["scenario_id"]: r for r in results}
        # Both scenarios produced a row — the raising one did NOT abort the model.
        assert set(by_id) == {scenarios[0].id, scenarios[1].id}
        # The raising scenario is recorded ungradable; the other is its normal row.
        assert by_id[scenarios[0].id]["outcome"] == OUTCOME_UNGRADABLE
        assert by_id[scenarios[0].id]["efficacy"] == 0.0
        assert by_id[scenarios[1].id]["outcome"] == "pass"
        assert by_id[scenarios[1].id]["efficacy"] == 0.8

    def test_all_scenarios_raising_still_returns_ungradable_rows_not_crash(self, monkeypatch):
        scenarios = [_scenario(0), _scenario(1)]

        def always_raise(runner, scenario, agent_spec, tracer, judge_keys, **kw):
            raise RuntimeError("systemic failure")

        monkeypatch.setattr(run_eval, "evaluate_scenario", always_raise)

        results = run_eval._run_model_scenarios(
            {"name": "GPT-5.5", "model_id": "x", "provider": "openai"},
            [Domain.BANKING],
            {Domain.BANKING: scenarios},
            reliability_runs=1,
            judge_keys=["opus"],
            tracer=None,
            run_id="r",
            artifacts_root=None,
            runner=object(),
        )
        # The model returns ungradable rows (not an empty list / a raise); the
        # ungradable-rate publish gate is what blocks a systemic failure, not a
        # discarded run.
        assert len(results) == 2
        assert all(r["outcome"] == OUTCOME_UNGRADABLE for r in results)
