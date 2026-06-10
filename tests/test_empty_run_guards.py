"""Tests for the empty-run guards in run_eval and aggregate_results.

These pin the fail-loud behavior added after five consecutive weekly CI runs
silently produced nothing (no API keys configured): run_eval wrote an empty
parquet and exited 0, aggregate_results warned and exited 0, and the workflow
finally died at `git add` with a pathspec error that diagnosed nothing.
"""

import pandas as pd
import pytest

import scripts.aggregate_results as aggregate_results
from scripts.run_eval import assert_results_nonempty


class TestAssertResultsNonempty:
    def test_all_models_failed_exits_nonzero_and_names_them(self):
        with pytest.raises(SystemExit) as excinfo:
            assert_results_nonempty([], ["GPT-4.1", "Claude Sonnet 4.6"])
        msg = str(excinfo.value)
        assert "Claude Sonnet 4.6" in msg
        assert "GPT-4.1" in msg
        assert "preflight" in msg

    def test_zero_results_zero_failures_still_exits(self):
        # Degenerate case: nothing attempted (e.g. --models filter matched
        # no configured model). Still must not exit 0 with an empty parquet.
        with pytest.raises(SystemExit):
            assert_results_nonempty([], [])

    def test_partial_failure_does_not_exit(self, caplog):
        # One bad provider must not sink the run — but it must be visible.
        with caplog.at_level("WARNING"):
            assert_results_nonempty([{"model": "GPT-4.1"}], ["Mistral Large"])
        assert "Mistral Large" in caplog.text

    def test_full_success_is_silent(self):
        assert_results_nonempty([{"model": "GPT-4.1"}], [])


class TestAggregateEmptyGuard:
    def test_empty_results_exit_nonzero(self, monkeypatch):
        monkeypatch.setattr(aggregate_results, "load_all_results", lambda: pd.DataFrame())
        with pytest.raises(SystemExit) as excinfo:
            aggregate_results.main()
        assert "No results to aggregate" in str(excinfo.value)
