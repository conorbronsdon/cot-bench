"""Tests for pass^k aggregation in aggregate_results."""

import pandas as pd
import pytest

from scripts.aggregate_results import compute_pass_hat_k_by_model


def _rows(model, per_scenario_pass_hat):
    """Build run-rows for a model.

    per_scenario_pass_hat: list of (p1, p2, p3) tuples, one per scenario; each
    scenario contributes 3 identical run-rows (run_eval writes the scenario's
    pass^k onto every run-row of that scenario).
    """
    rows = []
    for s, (p1, p2, p3) in enumerate(per_scenario_pass_hat):
        for _ in range(3):
            rows.append(
                {
                    "model": model,
                    "scenario_id": f"sc_{s}",
                    "reliability_pass_hat_1": p1,
                    "reliability_pass_hat_2": p2,
                    "reliability_pass_hat_3": p3,
                }
            )
    return rows


class TestComputePassHatKByModel:
    def test_mean_over_scenarios(self):
        # Model A: scenario 1 all-pass (1,1,1), scenario 2 two-of-three (2/3,1/3,0).
        df = pd.DataFrame(_rows("A", [(1.0, 1.0, 1.0), (2 / 3, 1 / 3, 0.0)]))
        out = compute_pass_hat_k_by_model(df)
        assert out["A"]["1"] == pytest.approx((1.0 + 2 / 3) / 2, abs=1e-4)
        assert out["A"]["2"] == pytest.approx((1.0 + 1 / 3) / 2, abs=1e-4)
        assert out["A"]["3"] == pytest.approx((1.0 + 0.0) / 2, abs=1e-4)

    def test_multiple_models(self):
        df = pd.DataFrame(_rows("A", [(1.0, 1.0, 1.0)]) + _rows("B", [(1 / 3, 0.0, 0.0)]))
        out = compute_pass_hat_k_by_model(df)
        assert set(out.keys()) == {"A", "B"}
        assert out["A"]["3"] == pytest.approx(1.0)
        assert out["B"]["3"] == pytest.approx(0.0)

    def test_no_pass_hat_columns_returns_empty_blocks(self):
        df = pd.DataFrame([{"model": "A", "scenario_id": "s0"}])
        out = compute_pass_hat_k_by_model(df)
        assert out == {"A": {}}

    def test_k_keys_sorted_numerically(self):
        df = pd.DataFrame(_rows("A", [(0.5, 0.25, 0.1)]))
        out = compute_pass_hat_k_by_model(df)
        assert list(out["A"].keys()) == ["1", "2", "3"]
