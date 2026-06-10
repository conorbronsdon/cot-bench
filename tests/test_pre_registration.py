"""Tests for run pre-registration (issue #38).

Pre-registration is only meaningful if the run's definition is committed to disk
BEFORE the results are known. These tests pin three properties:

1. **Hash determinism** — the same scenario set hashes to the same value;
   any change to scenario content (or adding/removing a scenario) changes it.
2. **Written before the run** — run_eval writes pre_registration.json before the
   evaluation loop makes any model/simulator/judge call.
3. **Post-run linkage** — the completion record (run_manifest.json) references
   the pre-registration by path and sha256, and carries the corpus hash.
"""

import json
from pathlib import Path

from eval.config import JUDGES, Domain
from eval.pre_registration import (
    PRE_REGISTRATION_FILENAME,
    build_pre_registration,
    canonical_scenario_bytes,
    file_sha256,
    scenario_set_hash,
    write_pre_registration,
)
from eval.simulation.runner import Scenario


def _scenario(scenario_id="banking_x_0000_aaaa1111", domain=Domain.BANKING, goals=None):
    return Scenario(
        id=scenario_id,
        domain=domain,
        persona={"name": "Test User", "traits": ["impatient"]},
        user_goals=goals or ["check balance", "transfer funds"],
        tools=[{"name": "lookup", "description": "look up account"}],
        category="adaptive_tool_use",
        initial_message="Hi, I need help.",
        ground_truth={"accounts": {"a1": {"balance": 100}}},
        expected_state_changes=[{"path": "accounts.a1.balance", "expected": 50}],
    )


class TestHashDeterminism:
    def test_same_set_same_hash(self):
        s = {Domain.BANKING: [_scenario()]}
        h1, idx1 = scenario_set_hash(s)
        h2, idx2 = scenario_set_hash(s)
        assert h1 == h2
        assert idx1 == idx2
        assert len(h1) == 64  # sha256 hex

    def test_independent_of_domain_iteration_order(self):
        a = _scenario("banking_x_0000_aaaa1111", Domain.BANKING)
        b = _scenario("cs_x_0000_bbbb2222", Domain.CUSTOMER_SUCCESS)
        forward = {Domain.BANKING: [a], Domain.CUSTOMER_SUCCESS: [b]}
        reverse = {Domain.CUSTOMER_SUCCESS: [b], Domain.BANKING: [a]}
        assert scenario_set_hash(forward)[0] == scenario_set_hash(reverse)[0]

    def test_independent_of_scenario_list_order(self):
        a = _scenario("banking_x_0000_aaaa1111")
        b = _scenario("banking_x_0001_cccc3333")
        assert (
            scenario_set_hash({Domain.BANKING: [a, b]})[0]
            == (scenario_set_hash({Domain.BANKING: [b, a]})[0])
        )

    def test_content_change_changes_hash(self):
        base = {Domain.BANKING: [_scenario(goals=["check balance"])]}
        changed = {Domain.BANKING: [_scenario(goals=["DRAIN the account"])]}
        assert scenario_set_hash(base)[0] != scenario_set_hash(changed)[0]

    def test_adding_a_scenario_changes_hash(self):
        a = _scenario("banking_x_0000_aaaa1111")
        b = _scenario("banking_x_0001_cccc3333")
        assert (
            scenario_set_hash({Domain.BANKING: [a]})[0]
            != (scenario_set_hash({Domain.BANKING: [a, b]})[0])
        )

    def test_canonical_bytes_ignore_key_order(self):
        # Same logical content, different key order -> identical canonical bytes.
        assert canonical_scenario_bytes({"a": 1, "b": 2}) == canonical_scenario_bytes(
            {"b": 2, "a": 1}
        )


class TestBuildPreRegistration:
    def _build(self, **overrides):
        kwargs = dict(
            run_id="results_20260610_000000",
            models=[{"name": "GPT-5.5", "model_id": "gpt-5.5-2026-04-23", "provider": "openai"}],
            scenarios_by_domain={Domain.BANKING: [_scenario()]},
            judges=JUDGES,
            judge_keys=list(JUDGES.keys()),
            reliability_runs=3,
            bootstrap_seed=42,
            agent_temperature=0.0,
            user_simulator_temperature=0.7,
            tool_simulator_temperature=0.0,
            separate_judge_calls=False,
        )
        kwargs.update(overrides)
        return build_pre_registration(**kwargs)

    def test_captures_required_fields(self):
        reg = self._build()
        assert reg["artifact_type"] == "pre_registration"
        assert reg["run_id"] == "results_20260610_000000"
        assert "timestamp" in reg
        assert reg["models_under_test"][0]["model_id"] == "gpt-5.5-2026-04-23"
        assert reg["domains"] == ["banking"]
        assert reg["scenario_set"]["n_scenarios"] == 1
        assert len(reg["scenario_set"]["sha256"]) == 64
        # Judge panel records configured model IDs, NOT resolved_model.
        names = {j["name"] for j in reg["judge_panel"]["judges"]}
        assert names == {JUDGES[k].name for k in JUDGES}
        flat = json.dumps(reg["judge_panel"])
        assert "resolved_model" not in {
            j_key for j in reg["judge_panel"]["judges"] for j_key in j.keys()
        }
        assert "resolved_model" in flat  # only inside the explanatory note
        assert reg["seeds_and_temperatures"]["bootstrap_seed"] == 42
        assert reg["seeds_and_temperatures"]["agent_temperature"] == 0.0
        assert reg["seeds_and_temperatures"]["user_simulator_temperature"] == 0.7
        assert "not" in reg["seeds_and_temperatures"]["reproducibility_note"].lower()

    def test_judge_prompt_mode_reflects_flag(self):
        assert self._build(separate_judge_calls=False)["judge_prompt_mode"] == "combined"
        assert self._build(separate_judge_calls=True)["judge_prompt_mode"] == "separate"

    def test_corpus_hash_matches_scenario_set_hash(self):
        s = {Domain.BANKING: [_scenario()]}
        reg = self._build(scenarios_by_domain=s)
        assert reg["scenario_set"]["sha256"] == scenario_set_hash(s)[0]

    def test_write_round_trips(self, tmp_path):
        reg = self._build()
        path = write_pre_registration(tmp_path, reg)
        assert path.name == PRE_REGISTRATION_FILENAME
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["scenario_set"]["sha256"] == reg["scenario_set"]["sha256"]


class TestWrittenBeforeRun:
    """The honesty property: the pre-registration file must exist on disk before
    the evaluation loop makes any model call. We assert this by patching the
    executor entry point (_run_model_scenarios) to fail if the file is missing
    when the first model is dispatched, then confirming run_eval.main wrote it.
    """

    def test_pre_registration_exists_before_first_model_call(self, tmp_path, monkeypatch):
        import scripts.run_eval as run_eval

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output = results_dir / "results_20260610_010101.parquet"

        scenarios = [_scenario()]
        monkeypatch.setattr(run_eval, "load_scenarios", lambda domain: scenarios)
        # Tracing is a no-op sink in tests; avoid touching the real exporter.
        monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
        monkeypatch.setattr(run_eval, "get_tracer", lambda: None)

        seen = {}

        def fake_run_model_scenarios(model_cfg, *a, **kw):
            # This stands in for the first model/simulator call. The pre-
            # registration MUST already be on disk at this point.
            prereg = results_dir / PRE_REGISTRATION_FILENAME
            seen["exists_at_dispatch"] = prereg.exists()
            return [
                {
                    "model": model_cfg["name"],
                    "scenario_id": scenarios[0].id,
                    "domain": "banking",
                    "category": "adaptive_tool_use",
                    "efficacy": 0.5,
                    "cost_usd": 0.0,
                    "latency_ms": 1.0,
                    "reliability_pass_rate": 1.0,
                }
            ]

        monkeypatch.setattr(run_eval, "_run_model_scenarios", fake_run_model_scenarios)

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
        run_eval.main()

        assert seen.get("exists_at_dispatch") is True, (
            "pre_registration.json must exist before the first model call"
        )

    def test_post_run_manifest_links_pre_registration(self, tmp_path, monkeypatch):
        import scripts.run_eval as run_eval

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output = results_dir / "results_20260610_020202.parquet"

        scenarios = [_scenario()]
        monkeypatch.setattr(run_eval, "load_scenarios", lambda domain: scenarios)
        monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
        monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
        monkeypatch.setattr(
            run_eval,
            "_run_model_scenarios",
            lambda model_cfg, *a, **kw: [
                {
                    "model": model_cfg["name"],
                    "scenario_id": scenarios[0].id,
                    "domain": "banking",
                    "category": "adaptive_tool_use",
                    "efficacy": 0.5,
                    "cost_usd": 0.0,
                    "latency_ms": 1.0,
                    "reliability_pass_rate": 1.0,
                }
            ],
        )

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
        run_eval.main()

        manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
        prereg_path = results_dir / PRE_REGISTRATION_FILENAME
        assert prereg_path.exists()

        link = manifest["pre_registration"]
        assert link["file"] == PRE_REGISTRATION_FILENAME
        assert Path(link["path"]).name == PRE_REGISTRATION_FILENAME
        # The recorded hash must match the actual file on disk.
        assert link["sha256"] == file_sha256(prereg_path)
        # And the corpus hash must match what's inside the pre-registration.
        prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
        assert link["corpus_sha256"] == prereg["scenario_set"]["sha256"]
