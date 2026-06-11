"""Tests for macro-averaged scores and published failure profiles (issue #55).

Covers scripts/aggregate_results.py's #55 additions:

- compute_macro_efficacy: macro vs micro divergence under category imbalance,
  equality under balance, legacy frames without a category column,
- compute_macro_bootstrap_cis: determinism (fixed seed), point estimate inside
  the interval, single-scenario categories, degenerate zero-width CIs,
- compute_failure_profiles: counts/rates, the all-pass empty profile, legacy
  frames, out-of-vocabulary modes,
- compute_leaderboard wiring: per-model macro/profile fields, top-level
  categories/category_scores/failure_taxonomy, null-agent and holdout exclusion
  (the same public-only pinning the micro efficacy obeys), and that the
  pre-existing micro bootstrap CIs are unchanged by the new machinery.
"""

import numpy as np
import pandas as pd

from eval.providers.null_agent import NULL_AGENT_NAME
from eval.scoring.failure_modes import FAILURE_MODES
from scripts.aggregate_results import (
    compute_bootstrap_cis,
    compute_failure_profiles,
    compute_leaderboard,
    compute_macro_bootstrap_cis,
    compute_macro_efficacy,
)


def _rows(
    model,
    *,
    scenarios,
    category,
    efficacy,
    domain="banking",
    n_runs=2,
    holdout=False,
    failure_mode=None,
    jitter=0.0,
    seed=5,
):
    """Rows for one model over a list of scenario ids, one category."""
    rng = np.random.default_rng(seed)
    out = []
    for sid in scenarios:
        for _ in range(n_runs):
            eff = float(np.clip(efficacy + (rng.normal(0, jitter) if jitter else 0.0), 0, 1))
            out.append(
                {
                    "scenario_id": sid,
                    "domain": domain,
                    "category": category,
                    "model": model,
                    "holdout": holdout,
                    "efficacy": eff,
                    "task_completion": eff,
                    "tool_selection": eff,
                    "cost_usd": 0.01,
                    "latency_ms": 2000.0,
                    "total_turns": 5,
                    "reliability_pass_rate": 0.8,
                    "reliability_consistency": 0.9,
                    "tc_agreement": 0.9,
                    "ts_agreement": 0.9,
                    "failure_mode": failure_mode,
                }
            )
    return out


def _imbalanced_df(models=("A", "B"), jitter=0.0):
    """Frequent-easy vs rare-hard: 9 easy scenarios at high efficacy, 1 hard
    scenario at low efficacy. Micro is dominated by the easy mass; macro
    weights the two categories equally."""
    rows = []
    for i, model in enumerate(models):
        easy = 0.9 - 0.1 * i
        hard = 0.1
        rows += _rows(
            model,
            scenarios=[f"easy_{s:02d}" for s in range(9)],
            category="easy_cat",
            efficacy=easy,
            jitter=jitter,
            seed=11 + i,
        )
        rows += _rows(
            model,
            scenarios=["hard_00"],
            category="hard_cat",
            efficacy=hard,
            failure_mode="incomplete-task",
            jitter=jitter,
            seed=23 + i,
        )
    return pd.DataFrame(rows)


class TestComputeMacroEfficacy:
    def test_macro_weights_categories_equally(self):
        df = _imbalanced_df(models=("A",))
        micro = df["efficacy"].mean()  # 18 easy rows at 0.9, 2 hard at 0.1 -> 0.82
        macro = compute_macro_efficacy(df, "category")["A"]
        assert abs(micro - 0.82) < 1e-9
        assert abs(macro - 0.5) < 1e-9  # (0.9 + 0.1) / 2
        assert macro < micro  # the rare-hard category is no longer drowned

    def test_macro_equals_micro_when_balanced(self):
        rows = _rows("A", scenarios=["s1", "s2"], category="c1", efficacy=0.8) + _rows(
            "A", scenarios=["s3", "s4"], category="c2", efficacy=0.4
        )
        df = pd.DataFrame(rows)
        macro = compute_macro_efficacy(df, "category")["A"]
        assert abs(macro - df["efficacy"].mean()) < 1e-9

    def test_missing_column_returns_empty(self):
        df = _imbalanced_df().drop(columns=["category"])
        assert compute_macro_efficacy(df, "category") == {}

    def test_model_missing_a_category_averages_its_own_categories(self):
        rows = _rows("A", scenarios=["s1"], category="c1", efficacy=0.9) + _rows(
            "A", scenarios=["s2"], category="c2", efficacy=0.5
        )
        rows += _rows("B", scenarios=["s1"], category="c1", efficacy=0.6)
        df = pd.DataFrame(rows)
        macro = compute_macro_efficacy(df, "category")
        assert abs(macro["A"] - 0.7) < 1e-9
        assert abs(macro["B"] - 0.6) < 1e-9  # only its own category counts

    def test_works_for_domain_grouping(self):
        rows = _rows("A", scenarios=["s1"], category="c", efficacy=0.9, domain="banking") + _rows(
            "A", scenarios=["s2"], category="c", efficacy=0.3, domain="customer_success"
        )
        df = pd.DataFrame(rows)
        assert abs(compute_macro_efficacy(df, "domain")["A"] - 0.6) < 1e-9


class TestComputeMacroBootstrapCis:
    def test_deterministic_across_calls(self):
        df = _imbalanced_df(jitter=0.05)
        a = compute_macro_bootstrap_cis(df, ["A", "B"], "category")
        b = compute_macro_bootstrap_cis(df, ["A", "B"], "category")
        assert a == b

    def test_ci_contains_macro_point_estimate(self):
        df = _imbalanced_df(jitter=0.05)
        macro = compute_macro_efficacy(df, "category")
        cis = compute_macro_bootstrap_cis(df, ["A", "B"], "category")
        for model in ("A", "B"):
            lo, hi = cis[model]
            assert lo <= macro[model] <= hi

    def test_constant_scores_give_zero_width_ci(self):
        # No jitter: every resample reproduces the same category means.
        df = _imbalanced_df(jitter=0.0)
        cis = compute_macro_bootstrap_cis(df, ["A"], "category")
        lo, hi = cis["A"]
        assert lo == hi

    def test_single_scenario_category_contributes_without_spread(self):
        # hard_cat has ONE scenario: it always resamples to itself, so all the
        # interval width comes from the easy category's resampling.
        df = _imbalanced_df(jitter=0.05)
        cis = compute_macro_bootstrap_cis(df, ["A"], "category")
        lo, hi = cis["A"]
        assert lo is not None and hi is not None
        assert hi >= lo

    def test_missing_column_or_empty_returns_empty(self):
        df = _imbalanced_df().drop(columns=["category"])
        assert compute_macro_bootstrap_cis(df, ["A"], "category") == {}
        assert compute_macro_bootstrap_cis(pd.DataFrame(), ["A"], "category") == {}

    def test_micro_cis_unchanged_by_macro_machinery(self):
        # The macro bootstrap uses its OWN seeded generator; the pre-existing
        # micro CIs must be byte-identical whether or not macro CIs are computed.
        df = _imbalanced_df(jitter=0.05)
        before = compute_bootstrap_cis(df, ["A", "B"])
        compute_macro_bootstrap_cis(df, ["A", "B"], "category")
        after = compute_bootstrap_cis(df, ["A", "B"])
        assert before == after


class TestComputeFailureProfiles:
    def test_counts_and_rates(self):
        rows = _rows(
            "A",
            scenarios=["s1"],
            category="c",
            efficacy=0.2,
            failure_mode="premature-end",
        )  # 2 rows (2 runs)
        rows += _rows("A", scenarios=["s2"], category="c", efficacy=0.9, failure_mode=None)
        df = pd.DataFrame(rows)
        profile = compute_failure_profiles(df)["A"]
        assert profile["n_rows"] == 4
        assert profile["n_failures"] == 2
        assert profile["failure_rate"] == 0.5
        assert profile["modes"]["premature-end"] == {"count": 2, "rate": 0.5}
        # Full vocabulary always present, zeros included.
        assert set(FAILURE_MODES) <= set(profile["modes"])
        assert profile["modes"]["wrong-parameters"] == {"count": 0, "rate": 0.0}

    def test_all_pass_model_gets_explicit_empty_profile(self):
        df = pd.DataFrame(
            _rows("A", scenarios=["s1", "s2"], category="c", efficacy=0.95, failure_mode=None)
        )
        profile = compute_failure_profiles(df)["A"]
        assert profile["n_failures"] == 0
        assert profile["failure_rate"] == 0.0
        assert all(v["count"] == 0 for v in profile["modes"].values())

    def test_legacy_frame_without_column_returns_empty(self):
        df = pd.DataFrame(
            _rows("A", scenarios=["s1"], category="c", efficacy=0.5, failure_mode=None)
        ).drop(columns=["failure_mode"])
        assert compute_failure_profiles(df) == {}

    def test_unknown_mode_is_kept_not_dropped(self):
        # Future taxonomy growth must not silently vanish from the counts.
        df = pd.DataFrame(
            _rows("A", scenarios=["s1"], category="c", efficacy=0.2, failure_mode="new-mode")
        )
        profile = compute_failure_profiles(df)["A"]
        assert profile["modes"]["new-mode"]["count"] == 2
        assert profile["n_failures"] == 2

    def test_counts_sum_to_n_failures(self):
        rows = _rows(
            "A", scenarios=["s1"], category="c", efficacy=0.2, failure_mode="premature-end"
        )
        rows += _rows(
            "A", scenarios=["s2"], category="c", efficacy=0.3, failure_mode="wrong-parameters"
        )
        df = pd.DataFrame(rows)
        profile = compute_failure_profiles(df)["A"]
        assert sum(v["count"] for v in profile["modes"].values()) == profile["n_failures"]


class TestLeaderboardWiring:
    def _board_df(self):
        df = _imbalanced_df(jitter=0.02)
        # Holdout half: different efficacy + a failure mode that must NEVER
        # reach the published profiles (public-only pinning).
        hold = []
        for model in ("A", "B"):
            hold += _rows(
                model,
                scenarios=["hold_00"],
                category="easy_cat",
                efficacy=0.2,
                holdout=True,
                failure_mode="hallucinated-capability",
            )
        # Null-agent rows: excluded from every published aggregate.
        null = _rows(
            NULL_AGENT_NAME,
            scenarios=[f"easy_{s:02d}" for s in range(9)] + ["hard_00"],
            category="easy_cat",
            efficacy=0.0,
            failure_mode="incomplete-task",
        )
        return pd.concat([df, pd.DataFrame(hold), pd.DataFrame(null)], ignore_index=True)

    def test_model_entries_carry_macro_and_profile(self):
        lb = compute_leaderboard(self._board_df())
        for entry in lb["models"]:
            assert entry["efficacy_macro_category"] is not None
            assert entry["efficacy_macro_domain"] is not None
            lo, hi = entry["efficacy_macro_category_ci"]
            assert lo is not None and hi is not None and lo <= hi
            assert entry["failure_profile"]["n_rows"] > 0
        entry_a = next(m for m in lb["models"] if m["name"] == "A")
        # Macro pulled well below micro by the rare-hard category.
        assert entry_a["efficacy_macro_category"] < entry_a["efficacy"]

    def test_toplevel_categories_and_taxonomy(self):
        lb = compute_leaderboard(self._board_df())
        assert lb["categories"] == ["easy_cat", "hard_cat"]
        assert set(lb["category_scores"]) == {"easy_cat", "hard_cat"}
        for records in lb["category_scores"].values():
            for rec in records:
                assert {"model", "efficacy", "n_scenarios"} <= set(rec)
        assert lb["failure_taxonomy"]["modes"] == list(FAILURE_MODES)

    def test_profiles_and_macro_pinned_to_public_rows_only(self):
        df = self._board_df()
        lb = compute_leaderboard(df)
        public = df[~df["holdout"]]
        public = public[public["model"] != NULL_AGENT_NAME]
        for entry in lb["models"]:
            mdf = public[public["model"] == entry["name"]]
            # Profile row count == PUBLIC rows; the holdout failure mode
            # (hallucinated-capability) never reaches the published profile.
            assert entry["failure_profile"]["n_rows"] == len(mdf)
            assert entry["failure_profile"]["modes"]["hallucinated-capability"]["count"] == 0
            # Macro recomputed from public rows matches the published value.
            expected = mdf.groupby("category")["efficacy"].mean().mean()
            assert abs(entry["efficacy_macro_category"] - expected) < 1e-4

    def test_null_agent_absent_from_new_aggregates(self):
        lb = compute_leaderboard(self._board_df())
        names = {m["name"] for m in lb["models"]}
        assert NULL_AGENT_NAME not in names
        for records in lb["category_scores"].values():
            assert NULL_AGENT_NAME not in {r["model"] for r in records}

    def test_legacy_frame_degrades_to_none(self):
        df = _imbalanced_df(jitter=0.02).drop(columns=["category", "failure_mode"])
        lb = compute_leaderboard(df)
        assert lb["categories"] == []
        assert lb["category_scores"] == {}
        for entry in lb["models"]:
            assert entry["efficacy_macro_category"] is None
            assert entry["efficacy_macro_category_ci"] == [None, None]
            assert entry["failure_profile"] is None
            # Domain column still exists, so the domain macro is still computed.
            assert entry["efficacy_macro_domain"] is not None

    def test_statistical_note_mentions_macro(self):
        lb = compute_leaderboard(self._board_df())
        assert "Macro-averaged" in lb["statistical_note"]
