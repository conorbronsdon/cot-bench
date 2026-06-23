"""Tests for sim-model overrides (issue #50).

Offline: provider inference is pure; the end-to-end checks drive run_eval.main
with a stubbed evaluation and assert the override lands in pre_registration.json
and on result rows.
"""

import json

from eval.config import DEFAULT_SIMULATION, Domain, SimulationConfig
from eval.providers.registry import infer_provider
from eval.simulation.runner import Scenario


class TestInferProvider:
    def test_slug_form_is_openrouter(self):
        assert infer_provider("deepseek/deepseek-v4-pro") == "openrouter"

    def test_claude_is_anthropic(self):
        assert infer_provider("claude-sonnet-4-6") == "anthropic"
        assert infer_provider("Claude-Opus-4-6") == "anthropic"

    def test_gemini_is_google(self):
        assert infer_provider("gemini-3.5-flash") == "google"

    def test_gpt_is_openai(self):
        assert infer_provider("gpt-4.1-mini-2025-04-14") == "openai"
        assert infer_provider("o3-mini") == "openai"

    def test_unknown_defaults_openai(self):
        # Historical sim default — an unrecognized id behaves as before.
        assert infer_provider("some-mystery-model") == "openai"


class TestSimulationConfigDefaults:
    def test_defaults_are_openai_gpt_mini(self):
        # SimulationConfig remains the single source of the sim defaults.
        assert DEFAULT_SIMULATION.user_simulator_model == "gpt-4.1-mini-2025-04-14"
        assert DEFAULT_SIMULATION.tool_simulator_model == "gpt-4.1-mini-2025-04-14"
        assert DEFAULT_SIMULATION.user_simulator_provider == "openai"
        assert DEFAULT_SIMULATION.tool_simulator_provider == "openai"

    def test_override_carries_provider(self):
        cfg = SimulationConfig(
            user_simulator_model="claude-sonnet-4-6",
            user_simulator_provider="anthropic",
        )
        assert cfg.user_simulator_model == "claude-sonnet-4-6"
        assert cfg.user_simulator_provider == "anthropic"
        # Tool sim untouched -> still the default.
        assert cfg.tool_simulator_model == "gpt-4.1-mini-2025-04-14"


def _scenario(i=0):
    return Scenario(
        id=f"banking_x_{i:04d}_aaaa1111",
        domain=Domain.BANKING,
        persona={"name": "T"},
        user_goals=["check balance"],
        tools=[{"name": "lookup", "description": "look up account"}],
        category="adaptive_tool_use",
        initial_message="hi",
        ground_truth={"accounts": {"a1": {"balance": 100}}},
        expected_state_changes=[{"path": "accounts.a1.balance", "expected": 50}],
    )


def _run_with_overrides(tmp_path, monkeypatch, extra_argv):
    import scripts.run_eval as run_eval

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    output = results_dir / "results_20260610_777777.parquet"

    scenarios = [_scenario(0)]
    monkeypatch.setattr(run_eval, "load_scenarios", lambda domain, seed: (scenarios, []))
    monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
    monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
    monkeypatch.setattr(run_eval, "SimulationRunner", lambda *a, **k: object())

    def fake_evaluate(runner, scenario, agent_spec, tracer, judge_keys, **kw):
        # The sim model ids are threaded onto the row from sim_result; here we
        # simulate that by reading them off the runner's config is not available
        # (runner is a stub), so the production row-building path is exercised via
        # a SimpleNamespace sim_result fed through build_result_row instead.
        from types import SimpleNamespace

        from eval.scoring.judge import ConsensusResult

        sim = SimpleNamespace(
            total_latency_ms=1.0,
            total_turns=1,
            total_input_tokens=10,
            total_output_tokens=5,
            completed=True,
            ended_by="user_sim",
            state_progress_at_end=1.0,
            premature_end=False,
            resolved_model="gpt-5.5",
            user_sim_model=run_eval_sim_models["user"],
            tool_sim_model=run_eval_sim_models["tool"],
        )
        consensus = ConsensusResult(
            scenario_id=scenario.id,
            rubric_type="task_completion",
            judge_results=[],
            consensus_score=0.8,
            agreement_rate=None,
            max_disagreement=None,
            n_judges_requested=1,
            n_judges_valid=1,
        )
        ts = ConsensusResult(
            scenario_id=scenario.id,
            rubric_type="tool_selection",
            judge_results=[],
            consensus_score=0.8,
            agreement_rate=None,
            max_disagreement=None,
            n_judges_requested=1,
            n_judges_valid=1,
        )
        row = run_eval.build_result_row(
            scenario, agent_spec, sim, consensus, ts, 0.8, 0.0, state_result=None
        )
        return row, 0.0

    # Capture the resolved sim config the run builds, so the fake sim_result can
    # echo the override ids onto the row (the stub runner can't).
    run_eval_sim_models = {"user": None, "tool": None}
    orig_simconfig = run_eval.SimulationConfig

    def capturing_simconfig(**kwargs):
        cfg = orig_simconfig(**kwargs)
        run_eval_sim_models["user"] = cfg.user_simulator_model
        run_eval_sim_models["tool"] = cfg.tool_simulator_model
        return cfg

    monkeypatch.setattr(run_eval, "SimulationConfig", capturing_simconfig)
    monkeypatch.setattr(run_eval, "evaluate_scenario", fake_evaluate)

    argv = [
        "run_eval",
        "--domains",
        "banking",
        "--models",
        "GPT-5.5",
        "--judges",
        "opus",
        "--reliability-runs",
        "1",
        "--no-artifacts",
        "--parallel-models",
        "1",
        "--output",
        str(output),
        *extra_argv,
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.chdir(tmp_path)
    run_eval.main()
    return results_dir, output


def test_override_lands_in_pre_registration(tmp_path, monkeypatch):
    import pandas as pd

    results_dir, output = _run_with_overrides(
        tmp_path,
        monkeypatch,
        ["--user-sim-model", "claude-sonnet-4-6", "--tool-sim-model", "gpt-4.1-mini-2025-04-14"],
    )
    reg = json.loads((results_dir / "pre_registration.json").read_text(encoding="utf-8"))
    st = reg["seeds_and_temperatures"]
    assert st["user_simulator_model"] == "claude-sonnet-4-6"
    assert st["tool_simulator_model"] == "gpt-4.1-mini-2025-04-14"
    # Temperatures still recorded alongside.
    assert "user_simulator_temperature" in st

    # And the override is tagged on the result rows.
    df = pd.read_parquet(output)
    assert set(df["user_sim_model"]) == {"claude-sonnet-4-6"}
    assert set(df["tool_sim_model"]) == {"gpt-4.1-mini-2025-04-14"}


def test_default_when_no_override(tmp_path, monkeypatch):
    results_dir, _ = _run_with_overrides(tmp_path, monkeypatch, [])
    reg = json.loads((results_dir / "pre_registration.json").read_text(encoding="utf-8"))
    st = reg["seeds_and_temperatures"]
    # Defaults from SimulationConfig (the single source of truth).
    assert st["user_simulator_model"] == DEFAULT_SIMULATION.user_simulator_model
    assert st["tool_simulator_model"] == DEFAULT_SIMULATION.tool_simulator_model
