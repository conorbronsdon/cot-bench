"""Consistency bands + corpus-health stats (issue #71, WolfBench learnings).

Two published surfaces, both derived from data already in the parquet:

1. ``consistency_band`` per model entry — solid (passed every reliability run;
   mathematically pass^k at k = n), avg (mean per-scenario pass rate), and
   best-of (passed at least once) rates. Key ABSENT when the parquet has no
   repeated runs.
2. ``corpus_health`` top-level block — scenarios never passed by any model /
   passed by every model, as COUNTS plus the WolfBench-style headline string.
   Scenario ids are logged at INFO for maintainers, never published.

Plus the sacred exclusion tripwires: both are computed from PUBLIC,
cooperative, non-null-agent rows only — a holdout, null-agent, or
behavioral-profile row must never move either surface, and no holdout scenario
id may appear anywhere in the published JSON.
"""

import json
import logging

import pandas as pd
import pytest

from eval.providers.null_agent import NULL_AGENT_NAME
from eval.scoring.rubrics import PASS_THRESHOLD, compute_pass_hat_k
from eval.simulation.profiles import ADVERSARIAL_PROFILE
from scripts.aggregate_results import (
    compute_consistency_bands,
    compute_corpus_health,
    compute_leaderboard,
)

PASS = PASS_THRESHOLD + 0.1  # a passing efficacy
FAIL = PASS_THRESHOLD - 0.2  # a failing efficacy


def _row(model, scenario, efficacy, *, holdout=False, sim_profile=None):
    """One result row with every column compute_leaderboard's agg requires."""
    row = {
        "scenario_id": scenario,
        "domain": "banking",
        "category": "adaptive_tool_use",
        "model": model,
        "holdout": holdout,
        "efficacy": float(efficacy),
        "task_completion": float(efficacy),
        "tool_selection": float(efficacy),
        "cost_usd": 0.01,
        "latency_ms": 2000.0,
        "total_turns": 5,
        "reliability_pass_rate": 0.5,
        "reliability_consistency": 0.9,
        "tc_agreement": 0.9,
        "ts_agreement": 0.9,
    }
    if sim_profile is not None:
        row["sim_profile"] = sim_profile
    return row


def _df(run_effs_by_model_scenario, **kw):
    """Frame from {model: {scenario: [per-run efficacies]}}."""
    rows = []
    for model, scenarios in run_effs_by_model_scenario.items():
        for scenario, effs in scenarios.items():
            for eff in effs:
                rows.append(_row(model, scenario, eff, **kw))
    return pd.DataFrame(rows)


# --- 1. Consistency bands: solid / avg / best math ----------------------------


class TestComputeConsistencyBands:
    def test_solid_avg_best_math(self):
        # s0: 3/3 pass, s1: 2/3 pass, s2: 0/3 pass.
        df = _df(
            {
                "A": {
                    "s0": [PASS, PASS, PASS],
                    "s1": [PASS, PASS, FAIL],
                    "s2": [FAIL, FAIL, FAIL],
                }
            }
        )
        band = compute_consistency_bands(df)["A"]
        assert band["solid_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert band["avg_pass_rate"] == pytest.approx((1.0 + 2 / 3 + 0.0) / 3, abs=1e-4)
        assert band["best_of_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert band["n_scenarios"] == 3
        assert band["n_runs"] == 3

    def test_solid_is_pass_hat_k_at_k_equals_n(self):
        # The issue's derivation: solid == mean over scenarios of pass^k at k=n.
        per_scenario_passes = [(3, 3), (2, 3), (0, 3)]
        expected = sum(compute_pass_hat_k(c, n)[n] for c, n in per_scenario_passes) / 3
        df = _df(
            {
                "A": {
                    "s0": [PASS, PASS, PASS],
                    "s1": [PASS, PASS, FAIL],
                    "s2": [FAIL, FAIL, FAIL],
                }
            }
        )
        assert compute_consistency_bands(df)["A"]["solid_rate"] == pytest.approx(expected, abs=1e-4)

    def test_ordering_invariant(self):
        df = _df(
            {
                "A": {"s0": [PASS, FAIL, PASS], "s1": [FAIL, FAIL, PASS]},
                "B": {"s0": [PASS, PASS, PASS], "s1": [FAIL, FAIL, FAIL]},
            }
        )
        for band in compute_consistency_bands(df).values():
            assert band["solid_rate"] <= band["avg_pass_rate"] <= band["best_of_rate"]

    def test_keyed_per_model(self):
        df = _df(
            {
                "A": {"s0": [PASS, PASS]},
                "B": {"s0": [FAIL, FAIL]},
            }
        )
        bands = compute_consistency_bands(df)
        assert set(bands) == {"A", "B"}
        assert bands["A"]["solid_rate"] == 1.0
        assert bands["B"]["best_of_rate"] == 0.0

    def test_single_run_parquet_returns_empty(self):
        # No repeated runs anywhere -> no distribution -> {} (key then absent).
        df = _df({"A": {"s0": [PASS], "s1": [FAIL]}})
        assert compute_consistency_bands(df) == {}

    def test_missing_columns_returns_empty(self):
        df = _df({"A": {"s0": [PASS, FAIL]}}).drop(columns=["efficacy"])
        assert compute_consistency_bands(df) == {}
        assert compute_consistency_bands(pd.DataFrame()) == {}


# --- 2. Corpus health: counts + headline --------------------------------------


class TestComputeCorpusHealth:
    def _frame(self):
        # s_easy: passed by both models; s_mid: passed by A only (B fails);
        # s_dead: passed by nobody.
        return _df(
            {
                "A": {
                    "s_easy": [PASS, FAIL],
                    "s_mid": [FAIL, PASS],
                    "s_dead": [FAIL, FAIL],
                },
                "B": {
                    "s_easy": [PASS, PASS],
                    "s_mid": [FAIL, FAIL],
                    "s_dead": [FAIL, FAIL],
                },
            }
        )

    def test_counts_and_headline(self):
        health = compute_corpus_health(self._frame())
        assert health["total_scenarios"] == 3
        assert health["passed_at_least_once"] == 2
        assert health["never_passed"] == 1
        assert health["passed_by_every_model"] == 1
        assert health["n_models"] == 2
        assert health["pass_threshold"] == PASS_THRESHOLD
        assert health["headline"] == (
            "2 of 3 scenarios passed at least once; 1 passed by every model"
        )

    def test_never_passed_ids_logged_not_published(self, caplog):
        with caplog.at_level(logging.INFO, logger="scripts.aggregate_results"):
            health = compute_corpus_health(self._frame())
        # Maintainer eyes: the dead scenario id appears in the INFO log...
        assert any("s_dead" in rec.getMessage() for rec in caplog.records)
        # ...but never in the published block (counts + headline only).
        assert "s_dead" not in json.dumps(health)
        assert "s_mid" not in json.dumps(health)

    def test_unattempted_scenario_not_passed_by_every_model(self):
        # s_solo is only attempted (and passed) by A; B never ran it, so it must
        # NOT count as passed by every model.
        df = _df(
            {
                "A": {"s_both": [PASS], "s_solo": [PASS]},
                "B": {"s_both": [PASS]},
            }
        )
        health = compute_corpus_health(df)
        assert health["total_scenarios"] == 2
        assert health["passed_at_least_once"] == 2
        assert health["passed_by_every_model"] == 1

    def test_missing_columns_returns_none(self):
        assert compute_corpus_health(pd.DataFrame()) is None
        df = _df({"A": {"s0": [PASS]}}).drop(columns=["efficacy"])
        assert compute_corpus_health(df) is None


# --- 3. Leaderboard integration + exclusion tripwires --------------------------


def _public_frame():
    """Two models, two public scenarios, 2 runs: s0 solid for both, s1 dead."""
    return _df(
        {
            "A": {"s0": [PASS, PASS], "s1": [FAIL, FAIL]},
            "B": {"s0": [PASS, PASS], "s1": [FAIL, FAIL]},
        }
    )


_EXPECTED_HEADLINE = "1 of 2 scenarios passed at least once; 1 passed by every model"


class TestLeaderboardIntegration:
    def test_surfaces_published(self):
        lb = compute_leaderboard(_public_frame())
        assert lb["corpus_health"]["headline"] == _EXPECTED_HEADLINE
        assert lb["corpus_health"]["total_scenarios"] == 2
        for m in lb["models"]:
            band = m["consistency_band"]
            assert band["solid_rate"] == 0.5
            assert band["avg_pass_rate"] == 0.5
            assert band["best_of_rate"] == 0.5
            assert band["n_runs"] == 2

    def test_consistency_band_absent_on_single_run_parquet(self):
        df = _df({"A": {"s0": [PASS], "s1": [FAIL]}})
        lb = compute_leaderboard(df)
        for m in lb["models"]:
            assert "consistency_band" not in m
        # Corpus health needs no repeats; it is still published.
        assert lb["corpus_health"]["total_scenarios"] == 2

    def test_holdout_rows_cannot_move_either_surface(self):
        # Holdout half: a scenario every model SOLIDLY passes plus one nobody
        # passes. If holdout leaked, total_scenarios jumps to 4, never_passed
        # to 2, and every solid/avg/best rate moves by construction.
        public = _public_frame()
        holdout = _df(
            {
                "A": {"hold_pass": [PASS, PASS], "hold_dead": [FAIL, FAIL]},
                "B": {"hold_pass": [PASS, PASS], "hold_dead": [FAIL, FAIL]},
            },
            holdout=True,
        )
        lb = compute_leaderboard(pd.concat([public, holdout], ignore_index=True))
        assert lb["corpus_health"]["total_scenarios"] == 2
        assert lb["corpus_health"]["never_passed"] == 1
        assert lb["corpus_health"]["headline"] == _EXPECTED_HEADLINE
        for m in lb["models"]:
            assert m["consistency_band"]["solid_rate"] == 0.5
            assert m["consistency_band"]["n_scenarios"] == 2

    def test_no_holdout_scenario_id_anywhere_in_published_json(self):
        # The hard privacy invariant at the published-output boundary, now with
        # the corpus_health block present: no holdout scenario id ("hold_*")
        # may appear anywhere in leaderboard.json.
        public = _public_frame()
        holdout = _df({"A": {"hold_dead": [FAIL, FAIL]}}, holdout=True)
        lb = compute_leaderboard(pd.concat([public, holdout], ignore_index=True))
        assert "corpus_health" in lb
        assert "hold_" not in json.dumps(lb)

    def test_null_agent_rows_cannot_move_either_surface(self):
        # The do-nothing null agent fails everything; if it leaked it would
        # zero passed_by_every_model and add a third bands entry.
        public = _public_frame()
        null_rows = _df({NULL_AGENT_NAME: {"s0": [FAIL, FAIL], "s1": [FAIL, FAIL]}})
        lb = compute_leaderboard(pd.concat([public, null_rows], ignore_index=True))
        assert lb["corpus_health"]["passed_by_every_model"] == 1
        assert lb["corpus_health"]["n_models"] == 2
        assert lb["corpus_health"]["headline"] == _EXPECTED_HEADLINE
        assert {m["name"] for m in lb["models"]} == {"A", "B"}

    def test_noncooperative_rows_cannot_move_either_surface(self):
        # Adversarial-profile rows where everything fails: if they leaked,
        # solid rates drop and never_passed grows.
        public = _public_frame()
        adversarial = _df(
            {"A": {"s0": [FAIL, FAIL], "s1": [FAIL, FAIL]}},
            sim_profile=ADVERSARIAL_PROFILE,
        )
        public["sim_profile"] = None  # legacy-null counts as cooperative
        lb = compute_leaderboard(pd.concat([public, adversarial], ignore_index=True))
        assert lb["corpus_health"]["headline"] == _EXPECTED_HEADLINE
        entry_a = next(m for m in lb["models"] if m["name"] == "A")
        assert entry_a["consistency_band"]["solid_rate"] == 0.5
