"""Tests for inter-judge Krippendorff alpha wired into aggregate_results."""

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import _alpha_from_columns, compute_judge_alpha

TC_COLS = ["tc_Kimi K2.6", "tc_GLM-4.6", "tc_Claude Opus 4.6"]
TS_COLS = ["ts_Kimi K2.6", "ts_GLM-4.6", "ts_Claude Opus 4.6"]
JUDGE_COLS = TC_COLS + TS_COLS


def _row(model, tc, ts):
    """tc/ts are 3-tuples of (kimi, glm, opus) scores; NaN allowed."""
    return {
        "model": model,
        "tc_Kimi K2.6": tc[0],
        "tc_GLM-4.6": tc[1],
        "tc_Claude Opus 4.6": tc[2],
        "ts_Kimi K2.6": ts[0],
        "ts_GLM-4.6": ts[1],
        "ts_Claude Opus 4.6": ts[2],
    }


class TestAlphaFromColumns:
    def test_high_agreement_positive_alpha(self):
        df = pd.DataFrame(
            [
                _row("A", (0.8, 0.85, 0.82), (0.6, 0.6, 0.65)),
                _row("A", (0.4, 0.45, 0.42), (0.3, 0.35, 0.3)),
            ]
        )
        alpha = _alpha_from_columns(df, TC_COLS)
        assert alpha is not None
        assert alpha > 0.8

    def test_single_column_is_none(self):
        df = pd.DataFrame([_row("A", (0.8, 0.85, 0.82), (0.6, 0.6, 0.65))])
        assert _alpha_from_columns(df, ["tc_Kimi K2.6"]) is None

    def test_nan_cells_are_missing(self):
        # A parse-failed judge on a row leaves NaN; alpha still computes.
        df = pd.DataFrame(
            [
                _row("A", (0.8, 0.85, np.nan), (0.6, 0.6, 0.65)),
                _row("A", (0.4, 0.45, 0.42), (0.3, np.nan, 0.3)),
            ]
        )
        alpha = _alpha_from_columns(df, TC_COLS)
        assert alpha is not None


class TestComputeJudgeAlpha:
    def test_overall_and_per_model(self):
        df = pd.DataFrame(
            [
                _row("A", (0.8, 0.85, 0.82), (0.6, 0.6, 0.65)),
                _row("A", (0.4, 0.45, 0.42), (0.3, 0.35, 0.3)),
                _row("B", (0.9, 0.9, 0.9), (0.5, 0.55, 0.5)),
                _row("B", (0.2, 0.25, 0.2), (0.7, 0.7, 0.72)),
            ]
        )
        out = compute_judge_alpha(df, JUDGE_COLS)
        assert "task_completion" in out
        assert "tool_selection" in out
        assert set(out["per_model"].keys()) == {"A", "B"}
        # Each per-model dimension is a float or None.
        for model_block in out["per_model"].values():
            assert "task_completion" in model_block
            assert "tool_selection" in model_block

    def test_constant_dimension_is_none(self):
        # Every judge gives the same constant -> no variation -> alpha undefined.
        df = pd.DataFrame(
            [
                _row("A", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                _row("A", (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        out = compute_judge_alpha(df, JUDGE_COLS)
        assert out["task_completion"] is None
        assert out["tool_selection"] is None

    def test_values_are_rounded(self):
        df = pd.DataFrame(
            [
                _row("A", (0.8, 0.85, 0.82), (0.6, 0.61, 0.65)),
                _row("A", (0.4, 0.45, 0.42), (0.3, 0.35, 0.31)),
            ]
        )
        out = compute_judge_alpha(df, JUDGE_COLS)
        alpha = out["task_completion"]
        assert alpha == pytest.approx(round(alpha, 4))
