"""Tests for checkpoint/resume (issue #48).

Offline: artifacts are written to disk directly (the schema eval/artifacts.py
emits), and run_eval.main is driven with stubbed evaluation so resume skips the
right tuples, merges reconstructed rows, and aborts on a corpus-hash mismatch.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from eval.artifacts import model_slug
from eval.config import Domain
from eval.pre_registration import build_pre_registration, scenario_set_hash
from eval.resume import (
    CorpusMismatchError,
    completed_tuples,
    verify_corpus_unchanged,
)
from eval.simulation.runner import Scenario


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


def _write_artifact(
    artifacts_root,
    run_id,
    model,
    scenario_id,
    run_index,
    *,
    efficacy_judges=0.8,
    tool_sim_parse_failures=0,
    llm_tool_sim_mutations=0,
    coded_transition_calls=0,
    llm_tool_sim_calls=0,
):
    """Write one per-evaluation artifact in the eval/artifacts.py schema."""
    out_dir = Path(artifacts_root) / run_id / model_slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario_id": scenario_id,
        "model": model,
        "run_index": run_index,
        "domain": "banking",
        "category": "adaptive_tool_use",
        "holdout": False,
        "evaluated_at": "2026-06-10T00:00:00+00:00",
        "transcript": [],
        "judges": {
            "task_completion": [
                {
                    "judge_name": "Claude Opus 4.6",
                    "rubric_type": "task_completion",
                    "overall_score": efficacy_judges,
                    "reasoning": "r",
                    "parse_failed": False,
                    "resolved_model": "claude-opus-4-6",
                    "raw_response": {"overall_score": efficacy_judges},
                }
            ],
            "tool_selection": [
                {
                    "judge_name": "Claude Opus 4.6",
                    "rubric_type": "tool_selection",
                    "overall_score": efficacy_judges,
                    "reasoning": "r",
                    "parse_failed": False,
                    "resolved_model": "claude-opus-4-6",
                    "raw_response": {"overall_score": efficacy_judges},
                }
            ],
        },
        "sim_meta": {
            "completed": True,
            "total_turns": 2,
            "input_tokens": 100,
            "output_tokens": 50,
            "latency_ms": 12.0,
            "error": None,
            "resolved_model": "gpt-5.5-2026-04-23",
            "ended_by": "user_sim",
            "state_progress_at_end": 1.0,
            "premature_end": False,
            "user_sim_model": "gpt-4.1-mini-2025-04-14",
            "tool_sim_model": "gpt-4.1-mini-2025-04-14",
            # Coded-vs-LLM authority split (#87 1b) + spine-trust guard (#87 ph3).
            "coded_transition_calls": coded_transition_calls,
            "llm_tool_sim_calls": llm_tool_sim_calls,
            "tool_sim_parse_failures": tool_sim_parse_failures,
            "llm_tool_sim_mutations": llm_tool_sim_mutations,
        },
        "state": {
            "score": 1.0,
            "checks": [{"passed": True, "detail": "ok"}],
            "final_world": {"accounts": {"a1": {"balance": 50}}},
        },
    }
    (out_dir / f"{scenario_id}_run{run_index}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _write_pre_registration(results_dir, public_by_domain):
    reg = build_pre_registration(
        run_id="results_resume_001",
        models=[{"name": "GPT-5.5", "model_id": "gpt-5.5-2026-04-23", "provider": "openai"}],
        scenarios_by_domain=public_by_domain,
        judges=__import__("eval.config", fromlist=["JUDGES"]).JUDGES,
        judge_keys=["opus"],
        reliability_runs=1,
        bootstrap_seed=42,
        agent_temperature=0.0,
        user_simulator_temperature=0.7,
        tool_simulator_temperature=0.0,
        separate_judge_calls=False,
    )
    path = Path(results_dir) / "pre_registration.json"
    path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg


class TestCompletedTuples:
    def test_scans_artifacts(self, tmp_path):
        root = tmp_path / "artifacts"
        run_id = "results_resume_001"
        _write_artifact(root, run_id, "GPT-5.5", "banking_x_0000_aaaa1111", 0)
        _write_artifact(root, run_id, "GPT-5.5", "banking_x_0001_aaaa1111", 0)
        done = completed_tuples(root, run_id, ["GPT-5.5"])
        assert done == {
            ("GPT-5.5", "banking_x_0000_aaaa1111", 0),
            ("GPT-5.5", "banking_x_0001_aaaa1111", 0),
        }

    def test_ignores_models_not_in_roster(self, tmp_path):
        root = tmp_path / "artifacts"
        run_id = "results_resume_001"
        _write_artifact(root, run_id, "GPT-5.5", "banking_x_0000_aaaa1111", 0)
        _write_artifact(root, run_id, "Other", "banking_x_0000_aaaa1111", 0)
        done = completed_tuples(root, run_id, ["GPT-5.5"])
        assert done == {("GPT-5.5", "banking_x_0000_aaaa1111", 0)}

    def test_missing_dir_is_empty(self, tmp_path):
        assert completed_tuples(tmp_path / "artifacts", "nope", ["GPT-5.5"]) == set()


class TestCorpusVerification:
    def test_match_passes(self, tmp_path):
        public = {Domain.BANKING: [_scenario(0)]}
        reg = _write_pre_registration(tmp_path, public)
        current_hash, _ = scenario_set_hash(public)
        # Should not raise.
        verify_corpus_unchanged(reg, current_public_hash=current_hash, current_holdout_hash=None)

    def test_mismatch_aborts(self, tmp_path):
        public = {Domain.BANKING: [_scenario(0)]}
        reg = _write_pre_registration(tmp_path, public)
        # Different corpus (a second scenario) -> different hash.
        changed = {Domain.BANKING: [_scenario(0), _scenario(1)]}
        changed_hash, _ = scenario_set_hash(changed)
        with pytest.raises(CorpusMismatchError):
            verify_corpus_unchanged(
                reg, current_public_hash=changed_hash, current_holdout_hash=None
            )

    def test_holdout_appearing_aborts(self, tmp_path):
        public = {Domain.BANKING: [_scenario(0)]}
        reg = _write_pre_registration(tmp_path, public)  # no holdout originally
        current_hash, _ = scenario_set_hash(public)
        with pytest.raises(CorpusMismatchError):
            verify_corpus_unchanged(
                reg, current_public_hash=current_hash, current_holdout_hash="deadbeef"
            )


# --------------------------------------------------------------------------- #
# End-to-end resume via run_eval.main (stubbed evaluation).
# --------------------------------------------------------------------------- #
def _setup_resume_run(tmp_path, monkeypatch, scenarios):
    import scripts.run_eval as run_eval

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    run_id = "results_resume_001"

    monkeypatch.setattr(run_eval, "load_scenarios", lambda domain, seed: (scenarios, []))
    monkeypatch.setattr(run_eval, "init_tracing", lambda **kw: None)
    monkeypatch.setattr(run_eval, "get_tracer", lambda: None)
    monkeypatch.setattr(run_eval, "SimulationRunner", lambda *a, **k: object())

    evaluated = []

    def fake_evaluate(runner, scenario, agent_spec, tracer, judge_keys, **kw):
        evaluated.append((agent_spec.name, scenario.id, kw["run_index"]))
        row = {
            "model": agent_spec.name,
            "scenario_id": scenario.id,
            "domain": scenario.domain.value,
            "category": scenario.category,
            "efficacy": 0.5,
            "cost_usd": 0.0,
            "latency_ms": 1.0,
        }
        return row, 0.5

    monkeypatch.setattr(run_eval, "evaluate_scenario", fake_evaluate)
    return run_eval, results_dir, run_id, evaluated


def test_resume_skips_completed_and_merges(tmp_path, monkeypatch):
    scenarios = [_scenario(0), _scenario(1), _scenario(2)]
    run_eval, results_dir, run_id, evaluated = _setup_resume_run(tmp_path, monkeypatch, scenarios)

    # Original pre-registration over the SAME corpus.
    _write_pre_registration(results_dir, {Domain.BANKING: scenarios})
    # Two of three scenarios already completed (artifacts on disk).
    artifacts_root = results_dir / "artifacts"
    _write_artifact(artifacts_root, run_id, "GPT-5.5", "banking_x_0000_aaaa1111", 0)
    _write_artifact(artifacts_root, run_id, "GPT-5.5", "banking_x_0001_aaaa1111", 0)

    argv = [
        "run_eval",
        "--domains",
        "banking",
        "--models",
        "GPT-5.5",
        "--reliability-runs",
        "1",
        "--parallel-models",
        "1",
        "--resume",
        run_id,
    ]
    # Place outputs under results_dir by setting the default output explicitly.
    argv += ["--output", str(results_dir / f"{run_id}.parquet")]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.chdir(tmp_path)
    run_eval.main()

    # Only the ONE not-yet-done scenario was actually evaluated.
    assert evaluated == [("GPT-5.5", "banking_x_0002_aaaa1111", 0)]

    # Final parquet merges 2 reconstructed + 1 new = 3 rows.
    df = pd.read_parquet(results_dir / f"{run_id}.parquet")
    assert len(df) == 3
    assert set(df["scenario_id"]) == {
        "banking_x_0000_aaaa1111",
        "banking_x_0001_aaaa1111",
        "banking_x_0002_aaaa1111",
    }

    # Manifest records the resume and does NOT write a second pre-registration.
    manifest = json.loads((results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["resumed"] is True
    assert manifest["resumed_rows"] == 2
    assert manifest["resumed_at"] is not None


def test_resume_aborts_on_corpus_mismatch(tmp_path, monkeypatch):
    # Pre-registration committed to ONE scenario; the live corpus now has THREE.
    scenarios = [_scenario(0), _scenario(1), _scenario(2)]
    run_eval, results_dir, run_id, evaluated = _setup_resume_run(tmp_path, monkeypatch, scenarios)
    _write_pre_registration(results_dir, {Domain.BANKING: [_scenario(0)]})
    artifacts_root = results_dir / "artifacts"
    _write_artifact(artifacts_root, run_id, "GPT-5.5", "banking_x_0000_aaaa1111", 0)

    argv = [
        "run_eval",
        "--domains",
        "banking",
        "--models",
        "GPT-5.5",
        "--reliability-runs",
        "1",
        "--parallel-models",
        "1",
        "--resume",
        run_id,
        "--output",
        str(results_dir / f"{run_id}.parquet"),
    ]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(CorpusMismatchError):
        run_eval.main()
    # Nothing evaluated — aborted before the run loop.
    assert evaluated == []


def test_reconstructed_rows_price_by_requested_model_id(tmp_path):
    """Review fix on PR #62: resumed rows must price cost_usd by the REQUESTED
    model id (mapped from the display name through the roster), not by
    resolved_model -- OpenRouter resolved ids are absent from TOKEN_COSTS and
    would silently zero the Cost dimension on every resumed row.
    """
    from eval.config import MODELS_UNDER_TEST, TOKEN_COSTS
    from eval.resume import rows_from_artifacts

    roster_model = MODELS_UNDER_TEST[0]
    costs = TOKEN_COSTS[roster_model["model_id"]]
    run_id = "results_test"
    _write_artifact(tmp_path, run_id, roster_model["name"], "banking_x_0000_aaaa1111", 0)

    rows = rows_from_artifacts(tmp_path, run_id, [roster_model["name"]])
    assert len(rows) == 1
    expected = 100 * costs["input"] / 1_000_000 + 50 * costs["output"] / 1_000_000
    assert expected > 0, "roster head model must have nonzero pricing for this test"
    assert abs(rows[0]["cost_usd"] - expected) < 1e-12


def test_reconstructed_rows_classify_failure_modes(tmp_path):
    """Resume-path parity for the failure taxonomy (issue #55): reconstructed
    rows go through the same build_result_row, so a failed artifact (judges low,
    reasoning text persisted) classifies exactly like the live path — and a
    passing artifact stays unclassified.
    """
    from eval.config import MODELS_UNDER_TEST
    from eval.resume import rows_from_artifacts

    name = MODELS_UNDER_TEST[0]["name"]
    run_id = "results_test"
    # Passing artifact (judges 0.8, state 1.0 -> efficacy >= 0.7).
    _write_artifact(tmp_path, run_id, name, "banking_x_0000_aaaa1111", 0, efficacy_judges=0.8)
    # Failing artifact: judges 0.2 with the default state 1.0 -> efficacy 0.44.
    # Its persisted reasoning ("r") has no keywords -> incomplete-task fallback.
    _write_artifact(tmp_path, run_id, name, "banking_x_0001_aaaa1111", 0, efficacy_judges=0.2)

    rows = {r["scenario_id"]: r for r in rows_from_artifacts(tmp_path, run_id, [name])}
    passed = rows["banking_x_0000_aaaa1111"]
    failed = rows["banking_x_0001_aaaa1111"]
    assert passed["failure_mode"] is None
    assert passed["failure_mode_source"] is None
    assert failed["failure_mode"] == "incomplete-task"
    assert failed["failure_mode_source"] == "fallback"


def test_resume_reconstructs_state_gradability_from_artifact(tmp_path):
    """Resume parity for the state-gradability gates (#87 ph3 + S3): a resumed row
    must reconstruct ``llm_tool_sim_mutations`` / ``tool_sim_parse_failures`` from
    the artifact's sim_meta and decide ``is_state_gradable`` IDENTICALLY to the live
    path. A missing/typo'd key or a wrong default here would silently regrade an
    excluded run and ship a corrupted leaderboard number — the exact failure phase 3
    exists to prevent. The default _write_artifact (both counters 0) is the gradable
    control; the two tainted artifacts must null state.
    """
    from eval.config import MODELS_UNDER_TEST
    from eval.resume import rows_from_artifacts

    name = MODELS_UNDER_TEST[0]["name"]
    run_id = "results_test"
    # Control: a fully-coded, clean run is gradable and keeps its state score.
    _write_artifact(tmp_path, run_id, name, "banking_x_0000_aaaa1111", 0)
    # Phase 3: an LLM-authored mutation taints the world -> non-gradable.
    _write_artifact(tmp_path, run_id, name, "banking_x_0001_aaaa1111", 0, llm_tool_sim_mutations=1)
    # S3: a tool-sim parse failure also taints the world -> non-gradable.
    _write_artifact(tmp_path, run_id, name, "banking_x_0002_aaaa1111", 0, tool_sim_parse_failures=1)

    rows = {r["scenario_id"]: r for r in rows_from_artifacts(tmp_path, run_id, [name])}
    clean = rows["banking_x_0000_aaaa1111"]
    llm_mut = rows["banking_x_0001_aaaa1111"]
    parse_fail = rows["banking_x_0002_aaaa1111"]

    # Control row: gradable, state score survives, both counters reconstruct to 0.
    assert clean["state_gradable"] is True
    assert clean["state_score"] == 1.0
    assert clean["llm_tool_sim_mutations"] == 0
    # LLM-mutation row: non-gradable, state nulled, count reconstructed.
    assert llm_mut["state_gradable"] is False
    assert llm_mut["state_score"] is None
    assert llm_mut["llm_tool_sim_mutations"] == 1
    # Parse-failure row: non-gradable for the pre-existing S3 reason (not an LLM mutation).
    assert parse_fail["state_gradable"] is False
    assert parse_fail["state_score"] is None
    assert parse_fail["llm_tool_sim_mutations"] == 0
