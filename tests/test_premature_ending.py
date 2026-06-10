"""Tests for the premature-ending rate in aggregate_results (#32, part 1).

The user-sim completion decoupling records, per run, whether the user sim ended
the conversation before the deterministic state check passed (``premature_end``).
aggregate_results must surface a per-model premature-ending rate on the
leaderboard so a miscalibrated simulator (ending early) is visible and
aggregable. These tests pin that wiring; no network — synthetic result rows only.
"""

import pandas as pd

from scripts.aggregate_results import compute_leaderboard


def _row(model, scenario_id, premature_end, **overrides):
    """A minimal result row with the columns compute_leaderboard touches."""
    row = {
        "model": model,
        "scenario_id": scenario_id,
        "domain": "banking",
        "category": "adaptive_tool_use",
        "efficacy": 0.8,
        "task_completion": 0.8,
        "tool_selection": 0.8,
        "state_score": 1.0,
        "cost_usd": 0.001,
        "latency_ms": 100.0,
        "total_turns": 3,
        "reliability_pass_rate": 1.0,
        "reliability_consistency": 1.0,
        "tc_agreement": 1.0,
        "ts_agreement": 1.0,
        "premature_end": premature_end,
    }
    row.update(overrides)
    return row


class TestPrematureEndingRate:
    def test_rate_is_fraction_of_premature_runs(self):
        # Model A: 4 runs, 1 premature -> rate 0.25.
        rows = [
            _row("A", "s0", True),
            _row("A", "s0", False),
            _row("A", "s1", False),
            _row("A", "s1", False),
        ]
        lb = compute_leaderboard(pd.DataFrame(rows))
        entry = next(m for m in lb["models"] if m["name"] == "A")
        assert entry["premature_end_rate"] == 0.25

    def test_zero_when_no_premature_endings(self):
        rows = [_row("A", "s0", False), _row("A", "s1", False)]
        lb = compute_leaderboard(pd.DataFrame(rows))
        entry = next(m for m in lb["models"] if m["name"] == "A")
        assert entry["premature_end_rate"] == 0.0

    def test_one_when_all_premature(self):
        rows = [_row("A", "s0", True), _row("A", "s1", True)]
        lb = compute_leaderboard(pd.DataFrame(rows))
        entry = next(m for m in lb["models"] if m["name"] == "A")
        assert entry["premature_end_rate"] == 1.0

    def test_per_model_independent(self):
        rows = [
            _row("A", "s0", True),
            _row("A", "s1", True),
            _row("B", "s0", False),
            _row("B", "s1", False),
        ]
        lb = compute_leaderboard(pd.DataFrame(rows))
        by_name = {m["name"]: m for m in lb["models"]}
        assert by_name["A"]["premature_end_rate"] == 1.0
        assert by_name["B"]["premature_end_rate"] == 0.0

    def test_missing_column_is_null_not_crash(self):
        # Legacy parquet without the premature_end column: rate is None.
        rows = [_row("A", "s0", False), _row("A", "s1", False)]
        df = pd.DataFrame(rows).drop(columns=["premature_end"])
        lb = compute_leaderboard(df)
        entry = next(m for m in lb["models"] if m["name"] == "A")
        assert entry["premature_end_rate"] is None
