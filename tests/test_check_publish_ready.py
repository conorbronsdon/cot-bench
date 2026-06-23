"""Tests for the publish-completeness gate (scripts/check_publish_ready.py).

These pin the behavior that stops a scheduled run from silently shipping a
leaderboard missing models: a run_manifest.json with any models_failed must
block the publish (exit non-zero), unless --allow-partial is set, and a
missing manifest must fail with a helpful message rather than passing.

Pattern mirrors tests/test_empty_run_guards.py: call the check function
directly against tmp_path manifests.
"""

import json
from pathlib import Path

import pytest

from eval.config import JUDGES, MIN_SCENARIOS_FOR_PUBLISH, RELIABILITY_RUNS
from eval.simulation.profiles import DEFAULT_SIM_PROFILE
from eval.templating import DEFAULT_INSTANTIATION_SEED
from scripts.check_publish_ready import check_publish_ready


def _write_manifest(tmp_path: Path, **overrides) -> Path:
    manifest = {
        "timestamp": "2026-06-09T00:00:00+00:00",
        "models_requested": ["GPT-4.1", "Gemini 2.5 Pro", "Gemini 2.5 Flash"],
        "models_completed": ["GPT-4.1", "Gemini 2.5 Pro", "Gemini 2.5 Flash"],
        "models_failed": [],
        "domains": ["banking"],
        # Default to a count that clears the publish minimum so failed-model
        # tests exercise the failed-model path in isolation; scenario-count
        # tests override this explicitly.
        "scenario_counts": {"banking": MIN_SCENARIOS_FOR_PUBLISH},
        "reliability_runs": RELIABILITY_RUNS,
        # Defaults that clear the H2 gates so other tests exercise their target
        # condition in isolation; the H2 tests override these explicitly.
        "judges": {"requested": list(JUDGES.keys()), "resolved": [j.name for j in JUDGES.values()]},
        "sim_profile": DEFAULT_SIM_PROFILE,
    }
    manifest.update(overrides)
    path = tmp_path / "run_manifest.json"
    path.write_text(json.dumps(manifest))
    return path


class TestCheckPublishReady:
    def test_complete_manifest_returns_zero(self, tmp_path):
        path = _write_manifest(tmp_path)
        assert check_publish_ready(path) == 0

    def test_failed_models_returns_nonzero_and_names_them(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            models_completed=["GPT-4.1"],
            models_failed=["Gemini 2.5 Pro", "Gemini 2.5 Flash"],
        )
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "Gemini 2.5 Pro" in err
        assert "Gemini 2.5 Flash" in err
        assert "::error::" in err

    def test_allow_partial_returns_zero_with_warning(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            models_completed=["GPT-4.1"],
            models_failed=["Gemini 2.5 Pro"],
        )
        assert check_publish_ready(path, allow_partial=True) == 0
        err = capsys.readouterr().err
        assert "::warning::" in err
        assert "Gemini 2.5 Pro" in err

    def test_missing_manifest_returns_nonzero_with_helpful_message(self, tmp_path, capsys):
        path = tmp_path / "does_not_exist.json"
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert "manifest" in err.lower()

    def test_allow_partial_does_not_rescue_missing_manifest(self, tmp_path):
        # An absent manifest is a hard failure regardless of the escape hatch:
        # we can't know what (if anything) ran.
        path = tmp_path / "does_not_exist.json"
        assert check_publish_ready(path, allow_partial=True) != 0


class TestScenarioMinimum:
    def test_below_minimum_scenarios_blocks(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            scenario_counts={"banking": MIN_SCENARIOS_FOR_PUBLISH - 1},
        )
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert str(MIN_SCENARIOS_FOR_PUBLISH) in err
        assert "banking" in err

    def test_exactly_minimum_scenarios_passes(self, tmp_path):
        path = _write_manifest(
            tmp_path,
            scenario_counts={"banking": MIN_SCENARIOS_FOR_PUBLISH},
        )
        assert check_publish_ready(path) == 0

    def test_one_domain_below_minimum_blocks_even_if_other_ok(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            domains=["banking", "customer_success"],
            scenario_counts={
                "banking": MIN_SCENARIOS_FOR_PUBLISH + 10,
                "customer_success": 4,
            },
        )
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "customer_success" in err
        # The above-minimum domain should not be named as a blocker.
        assert "customer_success=4" in err

    def test_allow_partial_rescues_below_minimum(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            scenario_counts={"banking": 4},
        )
        assert check_publish_ready(path, allow_partial=True) == 0
        err = capsys.readouterr().err
        assert "::warning::" in err
        assert "banking" in err

    def test_failed_models_and_low_scenarios_both_reported(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            models_completed=["GPT-4.1"],
            models_failed=["Gemini 2.5 Pro"],
            scenario_counts={"banking": 4},
        )
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "Gemini 2.5 Pro" in err  # failed-model blocker
        assert str(MIN_SCENARIOS_FOR_PUBLISH) in err  # scenario blocker

    def test_missing_scenario_counts_does_not_crash(self, tmp_path):
        # Older manifests may lack scenario_counts; treat absent as "no domains
        # to check" rather than crashing (the failed-model gate still applies).
        path = _write_manifest(tmp_path, scenario_counts={})
        assert check_publish_ready(path) == 0


class TestJudgePanelGate:
    """H2: a non-default judge panel blocks the publish (distinct reason)."""

    def test_single_judge_blocks(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            judges={"requested": ["opus"], "resolved": ["Claude Opus 4.6"]},
        )
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert "Judge panel" in err
        assert "opus" in err

    def test_full_panel_passes(self, tmp_path):
        # Order-independent: a reordered full roster still passes.
        reordered = list(reversed(list(JUDGES.keys())))
        path = _write_manifest(
            tmp_path,
            judges={"requested": reordered, "resolved": [JUDGES[k].name for k in reordered]},
        )
        assert check_publish_ready(path) == 0

    def test_allow_partial_rescues_reduced_panel(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            judges={"requested": ["opus"], "resolved": ["Claude Opus 4.6"]},
        )
        assert check_publish_ready(path, allow_partial=True) == 0
        assert "::warning::" in capsys.readouterr().err

    def test_legacy_manifest_without_judges_not_gated(self, tmp_path):
        # A manifest predating the judges field is not blocked on this condition.
        path = _write_manifest(tmp_path)
        manifest = json.loads(path.read_text())
        del manifest["judges"]
        path.write_text(json.dumps(manifest))
        assert check_publish_ready(path) == 0


class TestReliabilityRunsGate:
    """H2: a non-default reliability_runs blocks the publish (distinct reason)."""

    def test_non_default_reliability_blocks(self, tmp_path, capsys):
        path = _write_manifest(tmp_path, reliability_runs=1)
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert "reliability_runs" in err
        assert str(RELIABILITY_RUNS) in err

    def test_default_reliability_passes(self, tmp_path):
        path = _write_manifest(tmp_path, reliability_runs=RELIABILITY_RUNS)
        assert check_publish_ready(path) == 0

    def test_allow_partial_rescues_non_default_reliability(self, tmp_path, capsys):
        path = _write_manifest(tmp_path, reliability_runs=1)
        assert check_publish_ready(path, allow_partial=True) == 0
        assert "::warning::" in capsys.readouterr().err

    def test_legacy_manifest_without_reliability_not_gated(self, tmp_path):
        path = _write_manifest(tmp_path)
        manifest = json.loads(path.read_text())
        del manifest["reliability_runs"]
        path.write_text(json.dumps(manifest))
        assert check_publish_ready(path) == 0


class TestSimProfileGate:
    """H2: a non-cooperative sim_profile blocks the publish (distinct reason)."""

    def test_noncooperative_profile_blocks(self, tmp_path, capsys):
        path = _write_manifest(tmp_path, sim_profile="adversarial")
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert "sim_profile" in err
        assert "adversarial" in err

    def test_cooperative_profile_passes(self, tmp_path):
        path = _write_manifest(tmp_path, sim_profile=DEFAULT_SIM_PROFILE)
        assert check_publish_ready(path) == 0

    def test_allow_partial_rescues_noncooperative(self, tmp_path, capsys):
        path = _write_manifest(tmp_path, sim_profile="adversarial")
        assert check_publish_ready(path, allow_partial=True) == 0
        assert "::warning::" in capsys.readouterr().err

    def test_legacy_manifest_without_sim_profile_not_gated(self, tmp_path):
        path = _write_manifest(tmp_path)
        manifest = json.loads(path.read_text())
        del manifest["sim_profile"]
        path.write_text(json.dumps(manifest))
        assert check_publish_ready(path) == 0


class TestScenarioLimitGate:
    """S1: a scenario-limited (prefix-subset) run blocks the publish.

    --scenario-limit slices a fixed lexicographic prefix, so a positive limit
    ships a non-representative subset. Absent / 0 means the full corpus and
    passes; --allow-partial rescues it like the other conditions.
    """

    def test_scenario_limited_run_blocks(self, tmp_path, capsys):
        path = _write_manifest(tmp_path, scenario_limit=30)
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert "scenario_limit" in err
        assert "30" in err

    def test_full_run_zero_limit_passes(self, tmp_path):
        path = _write_manifest(tmp_path, scenario_limit=0)
        assert check_publish_ready(path) == 0

    def test_allow_partial_rescues_scenario_limited(self, tmp_path, capsys):
        path = _write_manifest(tmp_path, scenario_limit=30)
        assert check_publish_ready(path, allow_partial=True) == 0
        err = capsys.readouterr().err
        assert "::warning::" in err
        assert "scenario_limit" in err

    def test_legacy_manifest_without_scenario_limit_not_gated(self, tmp_path):
        # Default _write_manifest omits scenario_limit -> treated as unlimited.
        path = _write_manifest(tmp_path)
        manifest = json.loads(path.read_text())
        assert "scenario_limit" not in manifest
        assert check_publish_ready(path) == 0


class TestTemplatingSeedGate:
    """Issue #60: a published TEMPLATED run must not use the default seed (0).

    The manifest's ``templating`` block is present only when the corpus actually
    contained templated scenarios, so it doubles as the "templating was used"
    signal. A non-templated run (no block) is never gated on this condition.
    """

    def _templating(self, seed):
        return {
            "instantiation_seed": seed,
            "n_templated_scenarios": 5,
            "template_corpus_sha256": "a" * 64,
            "instantiated_corpus_sha256": "b" * 64,
        }

    def test_templated_run_with_default_seed_blocks(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            templating=self._templating(DEFAULT_INSTANTIATION_SEED),
        )
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert "--random-instantiation-seed" in err
        assert "CI-only" in err

    def test_templated_run_with_fresh_seed_passes(self, tmp_path):
        path = _write_manifest(
            tmp_path,
            templating=self._templating(1234567),
        )
        assert check_publish_ready(path) == 0

    def test_templated_run_with_missing_seed_blocks(self, tmp_path, capsys):
        # Fail-closed: a templating block whose instantiation_seed is missing/None
        # (only reachable via a malformed/hand-edited manifest) must NOT pass — a
        # publish gate cannot certify a surface whose seed it can't confirm.
        path = _write_manifest(tmp_path, templating=self._templating(None))
        assert check_publish_ready(path) != 0
        assert "::error::" in capsys.readouterr().err

    def test_non_templated_run_with_default_seed_unaffected(self, tmp_path):
        # No templating block (non-templated run): seed 0 is fine, gate is silent.
        path = _write_manifest(tmp_path)
        manifest = json.loads(path.read_text())
        assert "templating" not in manifest
        assert check_publish_ready(path) == 0

    def test_allow_partial_rescues_templated_default_seed(self, tmp_path, capsys):
        path = _write_manifest(
            tmp_path,
            templating=self._templating(DEFAULT_INSTANTIATION_SEED),
        )
        assert check_publish_ready(path, allow_partial=True) == 0
        assert "::warning::" in capsys.readouterr().err


class TestSurfaceReuseGate:
    """Closes the gap left open by #82: the seed gate stops seed 0 but NOT reusing
    the same fresh seed across two published runs, which re-exposes a byte-identical
    instantiated surface. The collision key is ``instantiated_corpus_sha256``; a
    committed ledger of already-published surfaces makes the reuse detectable.
    """

    def _templating(self, surface_hash, seed=1234567):
        return {
            "instantiation_seed": seed,
            "n_templated_scenarios": 5,
            "template_corpus_sha256": "a" * 64,
            "instantiated_corpus_sha256": surface_hash,
        }

    def _ledger(self, tmp_path, *entries) -> Path:
        ledger = tmp_path / "published_surfaces.jsonl"
        ledger.write_text("".join(json.dumps(e) + "\n" for e in entries))
        return ledger

    def test_reused_surface_blocks(self, tmp_path, capsys):
        # The current run's surface hash is already in the ledger (a prior board
        # published it): block, naming the prior run + seed.
        surface = "c" * 64
        ledger = self._ledger(
            tmp_path,
            {
                "run_id": "results_20260101",
                "instantiation_seed": 1234567,
                "instantiated_corpus_sha256": surface,
                "published_at": "2026-01-01T00:00:00+00:00",
            },
        )
        path = _write_manifest(tmp_path, templating=self._templating(surface))
        assert check_publish_ready(path, ledger_path=ledger) != 0
        err = capsys.readouterr().err
        assert "::error::" in err
        assert "already published" in err
        assert "results_20260101" in err  # names the prior run
        assert "--random-instantiation-seed" in err

    def test_novel_surface_passes(self, tmp_path):
        # The ledger holds a different surface; this run's surface is new -> pass.
        ledger = self._ledger(
            tmp_path,
            {
                "run_id": "results_20260101",
                "instantiation_seed": 999,
                "instantiated_corpus_sha256": "d" * 64,
                "published_at": "2026-01-01T00:00:00+00:00",
            },
        )
        path = _write_manifest(tmp_path, templating=self._templating("e" * 64))
        assert check_publish_ready(path, ledger_path=ledger) == 0

    def test_non_templated_run_and_empty_or_missing_ledger_pass_silently(self, tmp_path, capsys):
        # A non-templated run (no templating block) is never gated on surface reuse,
        # even with a missing ledger.
        missing_ledger = tmp_path / "nope.jsonl"
        non_templated = _write_manifest(tmp_path)
        assert "templating" not in json.loads(non_templated.read_text())
        assert check_publish_ready(non_templated, ledger_path=missing_ledger) == 0
        assert "already published" not in capsys.readouterr().err

        # A templated run with no prior publishes (empty ledger) also passes — the
        # surface is novel by definition.
        empty_ledger = tmp_path / "empty.jsonl"
        empty_ledger.write_text("")
        templated = _write_manifest(tmp_path, templating=self._templating("f" * 64))
        assert check_publish_ready(templated, ledger_path=empty_ledger) == 0
        assert "already published" not in capsys.readouterr().err

    def test_allow_partial_rescues_reused_surface(self, tmp_path, capsys):
        surface = "1" * 64
        ledger = self._ledger(
            tmp_path,
            {
                "run_id": "results_20260101",
                "instantiation_seed": 1234567,
                "instantiated_corpus_sha256": surface,
                "published_at": "2026-01-01T00:00:00+00:00",
            },
        )
        path = _write_manifest(tmp_path, templating=self._templating(surface))
        assert check_publish_ready(path, allow_partial=True, ledger_path=ledger) == 0
        assert "::warning::" in capsys.readouterr().err


class TestAllH2BlockersReportedTogether:
    def test_every_distinct_reason_named_at_once(self, tmp_path, capsys):
        # All three H2 conditions plus a failed model and low scenarios should
        # each be named in one pass — distinct reasons, not first-fail-only.
        path = _write_manifest(
            tmp_path,
            models_completed=["GPT-4.1"],
            models_failed=["Gemini 2.5 Pro"],
            scenario_counts={"banking": 4},
            judges={"requested": ["opus"], "resolved": ["Claude Opus 4.6"]},
            reliability_runs=1,
            sim_profile="impatient",
        )
        assert check_publish_ready(path) != 0
        err = capsys.readouterr().err
        assert "Gemini 2.5 Pro" in err  # failed model
        assert str(MIN_SCENARIOS_FOR_PUBLISH) in err  # scenarios
        assert "Judge panel" in err  # judges
        assert "reliability_runs" in err  # reliability
        assert "sim_profile" in err  # profile


class TestMainExitCode:
    def test_main_exits_nonzero_on_failed_models(self, tmp_path, monkeypatch):
        path = _write_manifest(tmp_path, models_failed=["Gemini 2.5 Pro"])
        monkeypatch.setattr("sys.argv", ["check_publish_ready", "--manifest", str(path)])
        import scripts.check_publish_ready as mod

        with pytest.raises(SystemExit) as excinfo:
            mod.main()
        assert excinfo.value.code != 0
