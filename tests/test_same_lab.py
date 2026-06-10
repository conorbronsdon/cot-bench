"""Tests for the same-lab robustness check in aggregate_results."""

import pandas as pd

from scripts.aggregate_results import compute_same_lab_check

TC_COLS = ["tc_Kimi K2.6", "tc_GLM-4.6", "tc_Claude Opus 4.6"]
TS_COLS = ["ts_Kimi K2.6", "ts_GLM-4.6", "ts_Claude Opus 4.6"]
JUDGE_COLS = TC_COLS + TS_COLS


def _df(rows):
    return pd.DataFrame(rows)


def _row(model, tc_kimi, tc_glm, tc_opus, ts_kimi, ts_glm, ts_opus):
    tc_full = (tc_kimi + tc_glm + tc_opus) / 3
    ts_full = (ts_kimi + ts_glm + ts_opus) / 3
    return {
        "model": model,
        "task_completion": tc_full,
        "tool_selection": ts_full,
        "tc_Kimi K2.6": tc_kimi,
        "tc_GLM-4.6": tc_glm,
        "tc_Claude Opus 4.6": tc_opus,
        "ts_Kimi K2.6": ts_kimi,
        "ts_GLM-4.6": ts_glm,
        "ts_Claude Opus 4.6": ts_opus,
    }


class TestComputeSameLabCheck:
    def test_only_claude_contestants_get_a_check(self):
        df = _df(
            [
                _row("Claude Sonnet 4.6", 0.9, 0.8, 1.0, 0.7, 0.6, 0.9),
                _row("GPT-4.1", 0.8, 0.8, 0.8, 0.7, 0.7, 0.7),
            ]
        )
        checks = compute_same_lab_check(df, JUDGE_COLS)
        assert "Claude Sonnet 4.6" in checks
        assert "GPT-4.1" not in checks

    def test_delta_math(self):
        # Opus rates its sibling 1.0 while open judges average 0.85:
        # full = (0.9+0.8+1.0)/3 = 0.9, excl = 0.85, delta = +0.05.
        df = _df([_row("Claude Sonnet 4.6", 0.9, 0.8, 1.0, 0.7, 0.6, 0.9)])
        check = compute_same_lab_check(df, JUDGE_COLS)["Claude Sonnet 4.6"]
        assert check["task_completion_excl_same_lab"] == 0.85
        assert check["task_completion_delta"] == 0.05
        # ts: full = (0.7+0.6+0.9)/3 ~= 0.7333, excl = 0.65, delta ~= +0.0833
        assert check["tool_selection_excl_same_lab"] == 0.65
        assert abs(check["tool_selection_delta"] - 0.0833) < 1e-3

    def test_no_same_lab_judge_columns_returns_empty(self):
        cols = ["tc_Kimi K2.6", "ts_Kimi K2.6"]
        df = _df([_row("Claude Sonnet 4.6", 0.9, 0.8, 1.0, 0.7, 0.6, 0.9)])
        assert compute_same_lab_check(df, cols) == {}

    def test_excluded_columns_named(self):
        df = _df([_row("Claude Haiku 4.5", 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)])
        check = compute_same_lab_check(df, JUDGE_COLS)["Claude Haiku 4.5"]
        assert check["excluded_judge_columns"] == ["tc_Claude Opus 4.6", "ts_Claude Opus 4.6"]
        # Identical scores -> zero delta.
        assert check["task_completion_delta"] == 0.0
