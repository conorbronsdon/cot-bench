"""Schema-coupling guard between the leaderboard frontend and aggregate_results.

frontend/index.html is a dependency-free static page that reads
``data/results/leaderboard.json`` (the file aggregate_results.py writes). It has
no build step and no shared type with the Python side, so the only thing keeping
the two in sync is that every key the JS dereferences is actually emitted by
``compute_leaderboard``. These tests pin that contract:

1. Every top-level and per-model key the frontend reads is a subset of what
   ``compute_leaderboard`` emits (so a rename/removal on the Python side that
   would silently break the page fails CI here instead).
2. The new fields surfaced by the frontend (holdout gap, pass^k, premature-end)
   are present in the emitted entry with the expected types.

The "expected keys" lists below are the ground truth for what index.html touches;
update them in lockstep when the page starts reading a new field.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate_results import compute_leaderboard

FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "index.html"

# Keys index.html dereferences off the top-level leaderboard object. Kept as the
# authoritative list of the page's coupling to the aggregate output.
FRONTEND_TOPLEVEL_KEYS = {
    "updated",
    "models",
    "statistical_note",
    "holdout",
}

# Keys index.html dereferences off each entry in leaderboard.models.
FRONTEND_MODEL_KEYS = {
    "name",
    "clear_score",
    "clear_score_ci",
    "efficacy",
    "task_completion",
    "tool_selection",
    "cost_per_task_usd",
    "reliability",
    "avg_latency_ms",
    "judge_agreement",
    "rank_band",
    # New fields surfaced by the holdout column + reliability detail line.
    "holdout_gap",
    "holdout_score",
    "premature_end_rate",
    "reliability_pass_hat_k",
}


def _build_df(with_holdout=True, with_premature=True, with_pass_hat=True):
    """A realistic results frame exercising every column the leaderboard reads."""
    rng = np.random.default_rng(11)
    judges = ["Kimi K2.6", "GLM-4.6", "Claude Opus 4.6"]
    models = {"GPT-5.5": 0.88, "Claude Sonnet 4.6": 0.78, "GPT-4.1 (anchor)": 0.55}
    halves = [(False, 0.0, "pub")]
    if with_holdout:
        halves.append((True, -0.15, "hold"))
    rows = []
    for holdout, adj, prefix in halves:
        for model, base in models.items():
            for s in range(32):
                for r in range(3):
                    eff = float(np.clip(base + adj + rng.normal(0, 0.03), 0, 1))
                    row = {
                        "scenario_id": f"{prefix}_{s:02d}",
                        "domain": "banking" if s % 2 else "customer_success",
                        "category": "adaptive_tool_use",
                        "model": model,
                        "holdout": holdout,
                        "efficacy": eff,
                        "task_completion": eff,
                        "tool_selection": eff,
                        "state_score": eff,
                        "cost_usd": 0.01 * (1 + list(models).index(model)),
                        "latency_ms": 2000.0,
                        "total_turns": 5,
                        "output_tokens": 500 + r * 50,
                        "reliability_pass_rate": base,
                        "reliability_consistency": 0.9,
                        "tc_agreement": 0.9,
                        "ts_agreement": 0.9,
                    }
                    if with_premature:
                        row["premature_end"] = r == 0 and model == "GPT-4.1 (anchor)"
                    if with_pass_hat:
                        row["reliability_pass_hat_1"] = base
                        row["reliability_pass_hat_2"] = base * 0.9
                        row["reliability_pass_hat_3"] = base * 0.8
                    for j in judges:
                        row[f"tc_{j}"] = float(np.clip(eff + rng.normal(0, 0.05), 0, 1))
                        row[f"ts_{j}"] = float(np.clip(eff + rng.normal(0, 0.05), 0, 1))
                    rows.append(row)
    return pd.DataFrame(rows)


class TestFrontendSchemaCoupling:
    def test_toplevel_keys_are_subset_of_emitted(self):
        lb = compute_leaderboard(_build_df())
        missing = FRONTEND_TOPLEVEL_KEYS - set(lb.keys())
        assert not missing, f"frontend reads top-level keys aggregate no longer emits: {missing}"

    def test_model_keys_are_subset_of_emitted(self):
        lb = compute_leaderboard(_build_df())
        assert lb["models"], "fixture produced no models"
        emitted = set(lb["models"][0].keys())
        missing = FRONTEND_MODEL_KEYS - emitted
        assert not missing, f"frontend reads model keys aggregate no longer emits: {missing}"

    def test_new_surfaced_fields_present_and_typed(self):
        lb = compute_leaderboard(_build_df())
        assert lb["holdout"]["present"] is True
        entry = lb["models"][0]
        # Holdout gap/score are floats here (this model has both halves).
        assert isinstance(entry["holdout_gap"], float)
        assert isinstance(entry["holdout_score"], float)
        # premature_end_rate is a float (a rate), pass^k is a dict keyed by k.
        assert isinstance(entry["premature_end_rate"], float)
        assert isinstance(entry["reliability_pass_hat_k"], dict)
        assert entry["reliability_pass_hat_k"], "expected per-k pass^k values"

    def test_new_fields_degrade_to_none_on_legacy_data(self):
        # No holdout / premature / pass^k columns (legacy parquet). The frontend
        # treats null as "render nothing extra"; aggregate must emit the keys with
        # null/empty so the JS `?.`/`!= null` guards have something to read.
        lb = compute_leaderboard(
            _build_df(with_holdout=False, with_premature=False, with_pass_hat=False)
        )
        assert lb["holdout"]["present"] is False
        entry = lb["models"][0]
        assert entry["holdout_gap"] is None
        assert entry["holdout_score"] is None
        assert entry["premature_end_rate"] is None
        assert entry["reliability_pass_hat_k"] == {}


class TestFrontendReadsDocumentedKeys:
    """Cheap drift guard: the keys we claim the frontend reads actually appear in
    index.html. Catches the inverse mistake — pruning a field from the page (so it
    silently stops surfacing) without updating this test's key list."""

    def test_declared_model_keys_appear_in_html(self):
        html = FRONTEND.read_text(encoding="utf-8")
        # Strip the demo-data block: it references model fields too, but we want to
        # confirm the *render* path reads each declared key, not the demo literal.
        html_no_demo = re.sub(r"function renderDemo\(\).*?^\s{8}\}", "", html, flags=re.S | re.M)
        for key in FRONTEND_MODEL_KEYS:
            assert key in html_no_demo, f"declared frontend key '{key}' not found in index.html"
