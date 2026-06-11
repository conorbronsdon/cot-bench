"""Tests for bootstrap confidence intervals and rank bands.

Covers scripts/aggregate_results.py's statistical-honesty additions:

- bootstrap CIs contain the point estimate,
- constant scores collapse to a zero-width (degenerate) CI,
- rank-band clustering groups CI-overlapping models and separates a clearly
  better one,
- the single-model path does not crash,
- the leaderboard JSON gains the new keys (CIs, n_scenarios, n_rows,
  reliability_consistency, rank_band, statistical_note).

The synthetic frame mirrors the row schema from scripts/run_eval.build_result_row
(the columns compute_leaderboard / compute_bootstrap_cis actually read).
"""

import numpy as np
import pandas as pd
import pytest

from scripts.aggregate_results import (
    _intervals_overlap,
    assign_rank_bands,
    compute_bootstrap_cis,
    compute_leaderboard,
)

# Columns compute_leaderboard reads; build a minimal but schema-faithful frame.
_REQUIRED_COLS = {
    "scenario_id",
    "domain",
    "category",
    "model",
    "efficacy",
    "task_completion",
    "tool_selection",
    "cost_usd",
    "latency_ms",
    "total_turns",
    "reliability_pass_rate",
    "reliability_consistency",
    "tc_agreement",
    "ts_agreement",
}


def _make_df(model_scores, n_scenarios=6, n_runs=3, seed=7):
    """Build a results frame: models x scenarios x runs.

    model_scores: dict {model_name: base_efficacy}. Each row's efficacy is the
    base plus a small seeded jitter (constant base -> noisy but separable).
    Cost/latency/reliability are derived deterministically so CLEAR is defined.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for model, base in model_scores.items():
        for s in range(n_scenarios):
            for r in range(n_runs):
                eff = float(np.clip(base + rng.normal(0, 0.03), 0, 1))
                rows.append(
                    {
                        "scenario_id": f"sc_{s:02d}",
                        "domain": "banking",
                        "category": "adaptive_tool_use",
                        "model": model,
                        "efficacy": eff,
                        "task_completion": eff,
                        "tool_selection": eff,
                        "cost_usd": 0.01 * (1 + base),
                        "latency_ms": 2000.0 * (2 - base),
                        "total_turns": 5,
                        "reliability_pass_rate": base,
                        "reliability_consistency": 0.9,
                        "tc_agreement": 0.9,
                        "ts_agreement": 0.9,
                    }
                )
    return pd.DataFrame(rows)


def _make_constant_df(model_scores, n_scenarios=6, n_runs=3):
    """Like _make_df but every score is exactly the base (zero variance)."""
    rows = []
    for model, base in model_scores.items():
        for s in range(n_scenarios):
            for r in range(n_runs):
                rows.append(
                    {
                        "scenario_id": f"sc_{s:02d}",
                        "domain": "banking",
                        "category": "adaptive_tool_use",
                        "model": model,
                        "efficacy": base,
                        "task_completion": base,
                        "tool_selection": base,
                        "cost_usd": 0.01,
                        "latency_ms": 2000.0,
                        "total_turns": 5,
                        "reliability_pass_rate": base,
                        "reliability_consistency": 1.0,
                        "tc_agreement": 0.9,
                        "ts_agreement": 0.9,
                    }
                )
    return pd.DataFrame(rows)


class TestSyntheticFrameSchema:
    def test_frame_has_required_columns(self):
        df = _make_df({"A": 0.8, "B": 0.6, "C": 0.4})
        assert _REQUIRED_COLS.issubset(set(df.columns))
        assert df["scenario_id"].nunique() == 6
        assert len(df) == 3 * 6 * 3  # models x scenarios x runs


class TestBootstrapCIs:
    def test_ci_contains_point_estimate(self):
        df = _make_df({"A": 0.85, "B": 0.6, "C": 0.4})
        models = ["A", "B", "C"]
        cis = compute_bootstrap_cis(df, models)
        # Point estimate = mean efficacy over scenario means.
        for m in models:
            mdf = df[df["model"] == m]
            point = mdf.groupby("scenario_id")["efficacy"].mean().mean()
            lo, hi = cis[m]["efficacy_ci"]
            assert lo <= point <= hi, f"{m}: {lo} <= {point} <= {hi}"

    def test_constant_scores_give_zero_width_efficacy_ci(self):
        df = _make_constant_df({"A": 0.8, "B": 0.5})
        cis = compute_bootstrap_cis(df, ["A", "B"])
        for m in ["A", "B"]:
            lo, hi = cis[m]["efficacy_ci"]
            assert hi - lo == pytest.approx(0.0, abs=1e-9)

    def test_single_scenario_gives_degenerate_ci(self):
        df = _make_df({"A": 0.7, "B": 0.5}, n_scenarios=1)
        cis = compute_bootstrap_cis(df, ["A", "B"])
        # With one scenario every resample is identical -> zero-width efficacy CI.
        for m in ["A", "B"]:
            lo, hi = cis[m]["efficacy_ci"]
            assert hi - lo == pytest.approx(0.0, abs=1e-9)

    def test_single_model_does_not_crash(self):
        df = _make_df({"A": 0.7})
        cis = compute_bootstrap_cis(df, ["A"])
        assert "efficacy_ci" in cis["A"]
        assert "clear_score_ci" in cis["A"]
        # Single-model CLEAR mirrors efficacy.
        assert cis["A"]["clear_score_ci"] == cis["A"]["efficacy_ci"]

    def test_reproducible_with_fixed_seed(self):
        df = _make_df({"A": 0.8, "B": 0.6})
        a = compute_bootstrap_cis(df, ["A", "B"])
        b = compute_bootstrap_cis(df, ["A", "B"])
        assert a == b


class TestIntervalOverlap:
    def test_overlap_true(self):
        assert _intervals_overlap([0.1, 0.5], [0.4, 0.9])

    def test_overlap_false(self):
        assert not _intervals_overlap([0.1, 0.3], [0.5, 0.9])

    def test_touching_counts_as_overlap(self):
        assert _intervals_overlap([0.1, 0.5], [0.5, 0.9])

    def test_none_does_not_overlap(self):
        assert not _intervals_overlap(None, [0.1, 0.5])
        assert not _intervals_overlap([None, None], [0.1, 0.5])


class TestRankBands:
    def test_two_overlapping_one_separated(self):
        # Leader and second overlap -> band 1; third clearly lower -> band 2.
        models = [
            {"name": "A", "clear_score": 0.80, "clear_score_ci": [0.70, 0.90]},
            {"name": "B", "clear_score": 0.78, "clear_score_ci": [0.68, 0.88]},
            {"name": "C", "clear_score": 0.40, "clear_score_ci": [0.30, 0.50]},
        ]
        assign_rank_bands(models)
        assert [m["rank_band"] for m in models] == [1, 1, 2]

    def test_all_separated(self):
        models = [
            {"name": "A", "clear_score": 0.9, "clear_score_ci": [0.85, 0.95]},
            {"name": "B", "clear_score": 0.6, "clear_score_ci": [0.55, 0.65]},
            {"name": "C", "clear_score": 0.3, "clear_score_ci": [0.25, 0.35]},
        ]
        assign_rank_bands(models)
        assert [m["rank_band"] for m in models] == [1, 2, 3]

    def test_all_overlapping_single_band(self):
        models = [
            {"name": "A", "clear_score": 0.6, "clear_score_ci": [0.4, 0.8]},
            {"name": "B", "clear_score": 0.55, "clear_score_ci": [0.4, 0.7]},
            {"name": "C", "clear_score": 0.5, "clear_score_ci": [0.45, 0.75]},
        ]
        assign_rank_bands(models)
        assert all(m["rank_band"] == 1 for m in models)

    def test_missing_ci_falls_to_own_band(self):
        models = [
            {"name": "A", "clear_score": 0.8, "clear_score_ci": [0.7, 0.9]},
            {"name": "B", "clear_score": 0.7, "clear_score_ci": None},
        ]
        assign_rank_bands(models)
        assert models[0]["rank_band"] == 1
        assert models[1]["rank_band"] == 2


class TestComputeLeaderboardKeys:
    def test_leaderboard_has_new_keys(self):
        df = _make_df({"A": 0.85, "B": 0.6, "C": 0.4})
        lb = compute_leaderboard(df)
        assert "statistical_note" in lb
        assert "n_rank_bands" in lb
        for m in lb["models"]:
            assert "clear_score_ci" in m
            assert "efficacy_ci" in m
            assert "n_scenarios" in m
            assert "n_rows" in m
            assert "reliability_consistency" in m
            assert "rank_band" in m
            assert len(m["efficacy_ci"]) == 2
            assert len(m["clear_score_ci"]) == 2
        # n_rows accounts for runs (6 scenarios x 3 runs).
        assert lb["models"][0]["n_rows"] == 18
        assert lb["models"][0]["n_scenarios"] == 6

    def test_statistical_note_flags_below_minimum(self):
        df = _make_df({"A": 0.8, "B": 0.6}, n_scenarios=4)
        lb = compute_leaderboard(df)
        assert "publish minimum" in lb["statistical_note"]

    def test_single_model_leaderboard_does_not_crash(self):
        df = _make_df({"A": 0.7})
        lb = compute_leaderboard(df)
        assert len(lb["models"]) == 1
        m = lb["models"][0]
        assert m["rank_band"] == 1
        assert m["clear_score_ci"] == m["efficacy_ci"]

    def test_empty_df_returns_empty(self):
        lb = compute_leaderboard(pd.DataFrame())
        assert lb["models"] == []


class TestClearWeightsSingleSource:
    """H5: the point-estimate CLEAR must use CLEAR_WEIGHTS, not duplicated
    hardcoded weights. Editing one path used to silently desync the published
    score from its bootstrap CI; these pin that the two share one weight source.
    """

    def test_point_estimate_clear_equals_dict_weighted(self):
        from scripts.aggregate_results import CLEAR_WEIGHTS, _min_max_norm

        df = _make_df({"A": 0.85, "B": 0.6, "C": 0.4})
        lb = compute_leaderboard(df)

        # Re-derive the expected CLEAR independently, reading the weights from the
        # single source (CLEAR_WEIGHTS). If the point estimate ever reverts to a
        # hardcoded literal that disagrees with the dict, this fails.
        overall = (
            df.groupby("model")
            .agg(
                efficacy=("efficacy", "mean"),
                cost_per_task=("cost_usd", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                reliability=("reliability_pass_rate", "mean"),
            )
            .reset_index()
        )
        eff_n = _min_max_norm(overall["efficacy"].to_numpy(dtype=float))
        rel_n = _min_max_norm(overall["reliability"].to_numpy(dtype=float))
        cost_n = _min_max_norm(overall["cost_per_task"].to_numpy(dtype=float), invert=True)
        lat_n = _min_max_norm(overall["avg_latency_ms"].to_numpy(dtype=float), invert=True)
        expected = (
            eff_n * CLEAR_WEIGHTS["efficacy"]
            + rel_n * CLEAR_WEIGHTS["reliability"]
            + cost_n * CLEAR_WEIGHTS["cost_per_task"]
            + lat_n * CLEAR_WEIGHTS["avg_latency_ms"]
        )
        expected_by_model = dict(zip(overall["model"], expected))

        for m in lb["models"]:
            assert m["clear_score"] == pytest.approx(
                round(expected_by_model[m["name"]], 4), abs=1e-9
            )

    def test_clear_weights_sum_to_one(self):
        from scripts.aggregate_results import CLEAR_WEIGHTS

        assert sum(CLEAR_WEIGHTS.values()) == pytest.approx(1.0)

    def test_point_estimate_and_bootstrap_share_clear_helper(self):
        # Both the point estimate and the bootstrap go through _clear_from_means,
        # so a degenerate single-model field yields the documented efficacy-only
        # CLEAR on both paths.
        df = _make_df({"A": 0.7})
        lb = compute_leaderboard(df)
        m = lb["models"][0]
        assert m["clear_score"] == round(m["efficacy"], 4)
