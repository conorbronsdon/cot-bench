"""Tests for the generalized per-judge-vs-consensus deltas (issue #27)."""

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import compute_judge_deltas, compute_same_lab_check

TC_COLS = ["tc_Kimi K2.6", "tc_GLM-4.6", "tc_Claude Opus 4.6"]
TS_COLS = ["ts_Kimi K2.6", "ts_GLM-4.6", "ts_Claude Opus 4.6"]
JUDGE_COLS = TC_COLS + TS_COLS


def _row(model, tc, ts, tc_consensus=None, ts_consensus=None):
    """tc/ts are (kimi, glm, opus); consensus defaults to the median of the three."""
    tc_med = sorted(tc)[1] if tc_consensus is None else tc_consensus
    ts_med = sorted(ts)[1] if ts_consensus is None else ts_consensus
    return {
        "model": model,
        "task_completion": tc_med,
        "tool_selection": ts_med,
        "tc_Kimi K2.6": tc[0],
        "tc_GLM-4.6": tc[1],
        "tc_Claude Opus 4.6": tc[2],
        "ts_Kimi K2.6": ts[0],
        "ts_GLM-4.6": ts[1],
        "ts_Claude Opus 4.6": ts[2],
    }


class TestComputeJudgeDeltas:
    def test_every_judge_gets_a_delta(self):
        df = pd.DataFrame([_row("GPT-4.1", (0.8, 0.9, 0.7), (0.6, 0.7, 0.5))])
        out = compute_judge_deltas(df, JUDGE_COLS)
        # All contestants get an entry (not just same-lab ones).
        assert "GPT-4.1" in out
        tc = out["GPT-4.1"]["task_completion"]
        assert set(tc.keys()) == {"Kimi K2.6", "GLM-4.6", "Claude Opus 4.6"}
        for judge_block in tc.values():
            assert "mean" in judge_block
            assert "delta" in judge_block

    def test_delta_sign_and_value(self):
        # Single row: consensus (median) tc = 0.8. Kimi=0.8 -> delta 0;
        # GLM=0.9 (more generous) -> delta = 0.8-0.9 = -0.1;
        # Opus=0.7 (harsher) -> delta = 0.8-0.7 = +0.1.
        df = pd.DataFrame([_row("GPT-4.1", (0.8, 0.9, 0.7), (0.6, 0.7, 0.5))])
        tc = compute_judge_deltas(df, JUDGE_COLS)["GPT-4.1"]["task_completion"]
        assert tc["Kimi K2.6"]["delta"] == pytest.approx(0.0)
        assert tc["GLM-4.6"]["delta"] == pytest.approx(-0.1)
        assert tc["Claude Opus 4.6"]["delta"] == pytest.approx(0.1)
        assert tc["GLM-4.6"]["mean"] == pytest.approx(0.9)

    def test_means_over_multiple_rows(self):
        df = pd.DataFrame(
            [
                _row("M", (0.8, 1.0, 0.6), (0.5, 0.5, 0.5)),
                _row("M", (0.6, 0.8, 0.4), (0.5, 0.5, 0.5)),
            ]
        )
        out = compute_judge_deltas(df, JUDGE_COLS)["M"]["task_completion"]
        # Kimi mean = (0.8+0.6)/2 = 0.7; consensus mean = (0.8+0.6)/2 = 0.7 (medians).
        assert out["Kimi K2.6"]["mean"] == pytest.approx(0.7)
        assert out["Kimi K2.6"]["delta"] == pytest.approx(0.0)

    def test_nan_judge_mean_is_handled(self):
        # A judge that parse-failed everywhere for a model -> NaN mean -> None.
        df = pd.DataFrame([_row("M", (0.8, 0.9, np.nan), (0.6, 0.7, np.nan))])
        out = compute_judge_deltas(df, JUDGE_COLS)["M"]["task_completion"]
        assert out["Claude Opus 4.6"]["mean"] is None
        assert out["Claude Opus 4.6"]["delta"] is None

    def test_no_judge_columns_returns_empty(self):
        df = pd.DataFrame([{"model": "M", "task_completion": 0.5, "tool_selection": 0.5}])
        assert compute_judge_deltas(df, []) == {}

    def test_same_lab_check_still_works(self):
        # Regression: generalizing deltas must not break the same-lab check.
        df = pd.DataFrame([_row("Claude Sonnet 4.6", (0.9, 0.8, 1.0), (0.7, 0.6, 0.9))])
        check = compute_same_lab_check(df, JUDGE_COLS)
        assert "Claude Sonnet 4.6" in check
        assert "task_completion_delta" in check["Claude Sonnet 4.6"]
