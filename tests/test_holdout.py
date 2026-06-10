"""Tests for the private-holdout scenario split (issue #31).

The holdout is an EXTERNAL, never-published scenario set run alongside the public
corpus so a public-vs-holdout efficacy gap acts as an overfitting tripwire. These
tests pin the harness behavior with SYNTHETIC dummy holdout scenarios in tmp
dirs — no real holdout content appears in this repo, its tests, or its fixtures.

Coverage:

1. **Loading** — run_eval.load_holdout_scenarios reads an external tree laid out
   like data/scenarios/ and tags every scenario holdout=True.
2. **Row tagging** — build_result_row stamps ``holdout`` from the scenario.
3. **Pre-registration** — the holdout gets its OWN corpus hash + count, with NO
   scenario IDs and NO per-scenario index (the privacy invariant).
4. **Aggregation** — compute_holdout_gap / compute_leaderboard split public vs
   holdout, publish a per-model gap, and never expose holdout scenario detail.
"""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from eval.config import Domain
from eval.pre_registration import (
    build_pre_registration,
    holdout_set_hash,
    scenario_set_hash,
)
from eval.simulation.runner import Scenario
from scripts.aggregate_results import compute_holdout_gap, compute_leaderboard
from scripts.run_eval import build_result_row, load_holdout_scenarios

# --- Synthetic dummy holdout scenarios (NOT real holdout content) ------------

DUMMY_HOLDOUT_SCENARIO = {
    "id": "banking_adaptive_tool_use_9999_dummy001",
    "category": "adaptive_tool_use",
    "schema_version": "0.2",
    "authorship": {"author_model": "human-handwritten"},
    "persona": {
        "name": "Dummy Holdout Persona",
        "age": 40,
        "occupation": "tester",
        "personality_traits": ["synthetic"],
        "tone": "neutral",
        "detail_level": "low",
        "background": "Fake persona used only to exercise the holdout loader.",
    },
    "user_goals": ["do a fake thing", "do another fake thing", "and a third"],
    "tools": [
        {"name": "noop", "description": "does nothing", "parameters": []},
        {"name": "noop2", "description": "also nothing", "parameters": []},
    ],
    "initial_message": "Hello, this is a synthetic holdout test message.",
    "difficulty": "medium",
    "ground_truth": {"accounts": {"X": {"balance": 1.0}}},
    "expected_state_changes": [],
}


def _write_dummy_holdout(tmp_path, domain="banking", n=1):
    """Write n synthetic holdout scenarios under a domain subdir. Returns root."""
    root = tmp_path / "holdout"
    dom_dir = root / domain
    dom_dir.mkdir(parents=True)
    for i in range(n):
        data = dict(DUMMY_HOLDOUT_SCENARIO)
        data["id"] = f"banking_adaptive_tool_use_9999_dummy{i:03d}"
        (dom_dir / f"{data['id']}.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def _holdout_scenario(scenario_id="banking_x_9999_dummyhh", holdout=True):
    return Scenario(
        id=scenario_id,
        domain=Domain.BANKING,
        persona={"name": "Dummy"},
        user_goals=["g1", "g2", "g3"],
        tools=[{"name": "noop", "description": "n"}],
        category="adaptive_tool_use",
        initial_message="synthetic holdout message",
        ground_truth={"accounts": {"X": {"balance": 1.0}}},
        expected_state_changes=[],
        holdout=holdout,
    )


# --- 1. Loading --------------------------------------------------------------


class TestLoadHoldout:
    def test_loads_and_tags_holdout(self, tmp_path):
        root = _write_dummy_holdout(tmp_path, n=2)
        scenarios = load_holdout_scenarios(root, Domain.BANKING)
        assert len(scenarios) == 2
        assert all(s.holdout is True for s in scenarios)
        assert all(s.domain is Domain.BANKING for s in scenarios)

    def test_missing_domain_returns_empty(self, tmp_path):
        root = _write_dummy_holdout(tmp_path, domain="banking", n=1)
        # No customer_success subdir -> empty list, not an error.
        assert load_holdout_scenarios(root, Domain.CUSTOMER_SUCCESS) == []


# --- 2. Row tagging ----------------------------------------------------------


class TestResultRowTagging:
    def _fake_consensus(self):
        return SimpleNamespace(
            consensus_score=0.5,
            agreement_rate=0.9,
            max_disagreement=0.1,
            n_judges_valid=3,
            parse_failures=[],
            api_failures=[],
            degraded=False,
            judge_results=[],
        )

    def _fake_sim(self):
        return SimpleNamespace(
            total_latency_ms=1.0,
            total_turns=3,
            total_input_tokens=10,
            total_output_tokens=10,
            completed=True,
        )

    def test_holdout_scenario_marks_row(self):
        agent = SimpleNamespace(name="GPT-5.5", model_id="gpt-5.5-2026-04-23")
        row = build_result_row(
            _holdout_scenario(holdout=True),
            agent,
            self._fake_sim(),
            self._fake_consensus(),
            self._fake_consensus(),
            0.5,
            0.0,
        )
        assert row["holdout"] is True

    def test_public_scenario_marks_row_false(self):
        agent = SimpleNamespace(name="GPT-5.5", model_id="gpt-5.5-2026-04-23")
        row = build_result_row(
            _holdout_scenario(holdout=False),
            agent,
            self._fake_sim(),
            self._fake_consensus(),
            self._fake_consensus(),
            0.5,
            0.0,
        )
        assert row["holdout"] is False


# --- 3. Pre-registration: hash + count only, no IDs --------------------------


class TestHoldoutPreRegistration:
    def test_hash_is_deterministic_and_count_only(self):
        s = {
            Domain.BANKING: [
                _holdout_scenario("banking_x_9999_a"),
                _holdout_scenario("banking_x_9999_b"),
            ]
        }
        h1, n1 = holdout_set_hash(s)
        h2, n2 = holdout_set_hash(s)
        assert h1 == h2
        assert n1 == n2 == 2
        assert len(h1) == 64

    def test_holdout_hash_matches_public_machinery(self):
        # Same content hashes identically whether treated as public or holdout —
        # the difference is only WHAT is recorded, not HOW it is hashed.
        s = {Domain.BANKING: [_holdout_scenario("banking_x_9999_a")]}
        public_hash, _index = scenario_set_hash(s)
        holdout_hash, _n = holdout_set_hash(s)
        assert public_hash == holdout_hash

    def test_change_changes_hash(self):
        a = {Domain.BANKING: [_holdout_scenario("banking_x_9999_a")]}
        b = {
            Domain.BANKING: [
                _holdout_scenario("banking_x_9999_a"),
                _holdout_scenario("banking_x_9999_b"),
            ]
        }
        assert holdout_set_hash(a)[0] != holdout_set_hash(b)[0]

    def _build(self, holdout):
        from eval.config import JUDGES

        return build_pre_registration(
            run_id="results_20260610_000000",
            models=[{"name": "GPT-5.5", "model_id": "gpt-5.5-2026-04-23", "provider": "openai"}],
            scenarios_by_domain={
                Domain.BANKING: [_holdout_scenario("banking_pub_0000_x", holdout=False)]
            },
            holdout_by_domain=holdout,
            judges=JUDGES,
            judge_keys=list(JUDGES.keys()),
            reliability_runs=3,
            bootstrap_seed=42,
            agent_temperature=0.0,
            user_simulator_temperature=0.7,
            tool_simulator_temperature=0.0,
            separate_judge_calls=False,
        )

    def test_holdout_block_has_hash_count_no_ids(self):
        holdout = {Domain.BANKING: [_holdout_scenario("banking_x_9999_secret")]}
        reg = self._build(holdout)
        block = reg["holdout_set"]
        assert block is not None
        assert len(block["sha256"]) == 64
        assert block["n_scenarios"] == 1
        # The privacy invariant: NO scenario IDs, NO index anywhere in the block.
        flat = json.dumps(block)
        assert "secret" not in flat
        assert "9999" not in flat
        assert "scenario_index" not in block
        assert "scenario_ids_by_domain" not in block

    def test_no_holdout_block_when_absent(self):
        reg = self._build(None)
        assert reg["holdout_set"] is None
        # And the public scenario_set still carries its index (unchanged).
        assert "scenario_index" in reg["scenario_set"]

    def test_full_prereg_never_contains_holdout_ids(self):
        holdout = {Domain.BANKING: [_holdout_scenario("banking_x_9999_topsecret")]}
        reg = self._build(holdout)
        assert "topsecret" not in json.dumps(reg)


# --- 4. Aggregation: gap + content exclusion ---------------------------------


def _agg_df(public_eff, holdout_eff, n_scen=4, n_runs=2, seed=3):
    """Build a results frame with both public and holdout rows per model.

    public_eff / holdout_eff: dict {model: base_efficacy} for each half.
    """
    rng = np.random.default_rng(seed)
    rows = []

    def _rows(eff_map, holdout, prefix):
        for model, base in eff_map.items():
            for s in range(n_scen):
                for r in range(n_runs):
                    eff = float(np.clip(base + rng.normal(0, 0.02), 0, 1))
                    rows.append(
                        {
                            "scenario_id": f"{prefix}_{s:02d}",
                            "domain": "banking",
                            "category": "adaptive_tool_use",
                            "model": model,
                            "holdout": holdout,
                            "efficacy": eff,
                            "task_completion": eff,
                            "tool_selection": eff,
                            "cost_usd": 0.01,
                            "latency_ms": 2000.0,
                            "total_turns": 5,
                            "reliability_pass_rate": base,
                            "reliability_consistency": 0.9,
                            "tc_agreement": 0.9,
                            "ts_agreement": 0.9,
                        }
                    )

    _rows(public_eff, False, "pub")
    _rows(holdout_eff, True, "hold")
    return pd.DataFrame(rows)


class TestComputeHoldoutGap:
    def test_gap_is_public_minus_holdout(self):
        df = _agg_df({"A": 0.9}, {"A": 0.6})
        gap = compute_holdout_gap(df)
        assert "A" in gap
        # public ~0.9, holdout ~0.6 -> gap ~ +0.3 (overfit signal)
        assert gap["A"]["holdout_gap"] > 0.2
        assert gap["A"]["public_score"] > gap["A"]["holdout_score"]

    def test_no_holdout_rows_returns_empty(self):
        df = _agg_df({"A": 0.8}, {})  # no holdout half
        assert compute_holdout_gap(df) == {}

    def test_missing_column_returns_empty(self):
        df = _agg_df({"A": 0.8}, {"A": 0.7}).drop(columns=["holdout"])
        assert compute_holdout_gap(df) == {}


class TestLeaderboardHoldout:
    def test_public_score_excludes_holdout_rows(self):
        # Public efficacy 0.9, holdout 0.3. The leaderboard efficacy must reflect
        # ONLY the public rows, not a blend.
        df = _agg_df({"A": 0.9, "B": 0.5}, {"A": 0.3, "B": 0.5})
        lb = compute_leaderboard(df)
        entry_a = next(m for m in lb["models"] if m["name"] == "A")
        assert abs(entry_a["efficacy"] - 0.9) < 0.05  # public, not blended down
        assert entry_a["holdout_score"] is not None
        assert entry_a["holdout_score"] < entry_a["efficacy"]
        assert entry_a["holdout_gap"] > 0.2

    def test_leaderboard_has_holdout_header(self):
        df = _agg_df({"A": 0.8}, {"A": 0.6})
        lb = compute_leaderboard(df)
        assert lb["holdout"]["present"] is True
        assert lb["holdout"]["models_with_gap"] == 1

    def test_no_holdout_header_when_absent(self):
        df = _agg_df({"A": 0.8}, {})
        lb = compute_leaderboard(df)
        assert lb["holdout"]["present"] is False
        for m in lb["models"]:
            assert m["holdout_score"] is None
            assert m["holdout_gap"] is None

    def test_no_holdout_scenario_ids_in_leaderboard(self):
        # The hard privacy invariant at the published-output boundary: no holdout
        # scenario_id ("hold_*") may appear anywhere in leaderboard.json.
        df = _agg_df({"A": 0.8}, {"A": 0.6})
        lb = compute_leaderboard(df)
        flat = json.dumps(lb)
        assert "hold_" not in flat

    def test_holdout_only_run_yields_empty_board(self):
        # A run with ONLY holdout rows has no public leaderboard to publish.
        df = _agg_df({}, {"A": 0.6})
        lb = compute_leaderboard(df)
        assert lb["models"] == []


# --- 5. End-to-end run_eval wiring (offline, faked model loop) ---------------


class TestRunEvalHoldoutWiring:
    """run_eval.main with --holdout-dir, with the model loop faked out (no API).

    Pins that the pre-registration written before any model call (a) records a
    holdout_set with hash + count and NO IDs, and (b) keeps the holdout out of
    the public scenario_set index. Mirrors test_pre_registration's offline style.
    """

    def test_holdout_prereg_written_with_hash_not_ids(self, tmp_path, monkeypatch):
        import scripts.run_eval as run_eval
        from eval.pre_registration import PRE_REGISTRATION_FILENAME

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output = results_dir / "results_20260610_030303.parquet"
        holdout_root = _write_dummy_holdout(tmp_path, domain="banking", n=2)

        public = [_holdout_scenario("banking_pub_0000_aaaa", holdout=False)]
        # Public loader returns the public scenario; holdout loader reads the dir.
        monkeypatch.setattr(run_eval, "load_scenarios", lambda domain: public)
        monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
        monkeypatch.setattr(run_eval, "get_tracer", lambda: None)

        captured = {}

        def fake_run_model_scenarios(model_cfg, domains, scenarios_by_domain, *a, **kw):
            # Record what the run loop was handed: holdout must be merged in here.
            captured["holdout_flags"] = [
                s.holdout for scs in scenarios_by_domain.values() for s in scs
            ]
            return [
                {
                    "model": model_cfg["name"],
                    "scenario_id": "banking_pub_0000_aaaa",
                    "domain": "banking",
                    "category": "adaptive_tool_use",
                    "holdout": False,
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
            "--holdout-dir",
            str(holdout_root),
            "--output",
            str(output),
        ]
        monkeypatch.setattr("sys.argv", argv)
        run_eval.main()

        # The run loop saw the 2 holdout scenarios merged with the 1 public one.
        assert captured["holdout_flags"].count(True) == 2
        assert captured["holdout_flags"].count(False) == 1

        prereg = json.loads((results_dir / PRE_REGISTRATION_FILENAME).read_text(encoding="utf-8"))
        # Public scenario_set carries ONLY the public scenario; holdout is excluded.
        assert prereg["scenario_set"]["n_scenarios"] == 1
        assert prereg["scenario_set"]["scenario_ids_by_domain"]["banking"] == [
            "banking_pub_0000_aaaa"
        ]
        # Holdout block: hash + count only, no IDs, and the dummy ID never leaks.
        assert prereg["holdout_set"]["n_scenarios"] == 2
        assert len(prereg["holdout_set"]["sha256"]) == 64
        assert "9999" not in json.dumps(prereg["holdout_set"])
        assert "dummy" not in json.dumps(prereg["holdout_set"])
        # Manifest carries the holdout hash + count, no IDs.
        manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["holdout"]["n_scenarios"] == 2
        assert manifest["scenario_counts"] == {"banking": 1}  # public only

    def test_missing_holdout_dir_errors(self, tmp_path, monkeypatch):
        import scripts.run_eval as run_eval

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output = results_dir / "results_20260610_040404.parquet"
        monkeypatch.setattr(
            run_eval, "load_scenarios", lambda domain: [_holdout_scenario(holdout=False)]
        )
        monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
        monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
        argv = [
            "run_eval",
            "--domains",
            "banking",
            "--models",
            "GPT-5.5",
            "--holdout-dir",
            str(tmp_path / "does_not_exist"),
            "--output",
            str(output),
        ]
        monkeypatch.setattr("sys.argv", argv)
        import pytest

        with pytest.raises(SystemExit):
            run_eval.main()
