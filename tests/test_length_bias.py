"""Tests for the length-bias regression (issue #30)."""

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import _ols_slope, compute_length_bias


class TestOLSSlope:
    def test_perfect_linear_fit(self):
        # y = 2 + 3x exactly.
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 + 3.0 * x
        fit = _ols_slope(x, y)
        assert fit["slope"] == pytest.approx(3.0)
        assert fit["intercept"] == pytest.approx(2.0)
        assert fit["r_squared"] == pytest.approx(1.0)
        # Perfect fit -> zero residual -> infinite t -> t_stat None but significant.
        assert fit["t_stat"] is None
        assert fit["significant"] is True

    def test_known_regression_reference(self):
        # Classic worked example: x=[1,2,3,4,5], y=[2,4,5,4,5].
        # Least squares gives slope 0.6, intercept 2.2 (standard textbook result).
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 5.0, 4.0, 5.0])
        fit = _ols_slope(x, y)
        assert fit["slope"] == pytest.approx(0.6, abs=1e-6)
        assert fit["intercept"] == pytest.approx(2.2, abs=1e-6)
        # R^2 for this example is 0.6 (textbook).
        assert fit["r_squared"] == pytest.approx(0.6, abs=1e-3)

    def test_t_stat_significance(self):
        # Strong, noisy positive trend -> significant; flat noise -> not.
        rng = np.random.default_rng(0)
        x = np.arange(50, dtype=float)
        y_trend = 0.01 * x + rng.normal(0, 0.05, size=50)
        assert _ols_slope(x, y_trend)["significant"] is True

        y_flat = rng.normal(0.5, 0.05, size=50)
        fit_flat = _ols_slope(x, y_flat)
        assert fit_flat["significant"] is False

    def test_zero_variance_x_is_none(self):
        x = np.array([3.0, 3.0, 3.0, 3.0])
        y = np.array([0.1, 0.2, 0.3, 0.4])
        assert _ols_slope(x, y) is None

    def test_too_few_points_is_none(self):
        assert _ols_slope(np.array([1.0, 2.0]), np.array([0.1, 0.2])) is None


class TestComputeLengthBias:
    def _df(self, tokens, tc, ts):
        return pd.DataFrame({"output_tokens": tokens, "task_completion": tc, "tool_selection": ts})

    def test_positive_length_bias_detected(self):
        # task_completion rises with output_tokens -> positive significant slope.
        tokens = list(range(100, 100 + 30 * 10, 10))
        tc = [0.3 + 0.0005 * t for t in tokens]
        ts = [0.5] * len(tokens)  # flat -> no tool-selection length bias
        out = compute_length_bias(self._df(tokens, tc, ts))
        assert out["task_completion"]["slope"] > 0
        assert out["task_completion"]["significant"] is True
        # Flat tool_selection: slope ~ 0, not significant.
        assert out["tool_selection"]["significant"] is False

    def test_missing_output_tokens_returns_empty(self):
        df = pd.DataFrame({"task_completion": [0.5, 0.6, 0.7], "tool_selection": [0.4, 0.5, 0.6]})
        assert compute_length_bias(df) == {}

    def test_nan_rows_dropped(self):
        df = pd.DataFrame(
            {
                "output_tokens": [100.0, 200.0, np.nan, 300.0, 400.0],
                "task_completion": [0.3, 0.4, 0.5, np.nan, 0.6],
                "tool_selection": [0.5, 0.5, 0.5, 0.5, 0.5],
            }
        )
        out = compute_length_bias(df)
        # task_completion fit uses the 3 fully-present rows; should still produce a fit.
        assert "task_completion" in out
        assert out["task_completion"]["n"] == 3

    def test_no_length_variation_omits_dimension(self):
        df = self._df([100.0] * 5, [0.3, 0.4, 0.5, 0.6, 0.7], [0.5] * 5)
        assert compute_length_bias(df) == {}
