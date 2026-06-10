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

from eval.config import MIN_SCENARIOS_FOR_PUBLISH
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
        "reliability_runs": 3,
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


class TestMainExitCode:
    def test_main_exits_nonzero_on_failed_models(self, tmp_path, monkeypatch):
        path = _write_manifest(tmp_path, models_failed=["Gemini 2.5 Pro"])
        monkeypatch.setattr("sys.argv", ["check_publish_ready", "--manifest", str(path)])
        import scripts.check_publish_ready as mod

        with pytest.raises(SystemExit) as excinfo:
            mod.main()
        assert excinfo.value.code != 0
