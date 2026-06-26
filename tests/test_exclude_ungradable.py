"""Tests for excluding ungradable rows from published aggregates (issue #88).

An ``ungradable`` episode is a HARNESS fault (simulator/agent error, zero valid
judges, or an incomplete graded world), not a real failure. Such a row still
carries a placeholder ``efficacy`` near 0.0, and ``episode_outcome``'s contract
says folding that 0.0 into the board "would understate the model and poison the
board." These tests pin that the exclusion is actually enforced — at the
aggregation entry point (efficacy / CLEAR / pass^k) and at the run-time/resume
reliability computation — so a routine judge-provider hiccup cannot silently drag
a model's published score toward zero.

Fully offline: hand-built result frames, no API calls.
"""

import pandas as pd

from scripts.aggregate_results import compute_leaderboard, exclude_ungradable
from scripts.run_eval import OUTCOME_FAIL, OUTCOME_PASS, OUTCOME_UNGRADABLE, _recompute_reliability


def _row(model: str, **over) -> dict:
    """A minimal results row carrying the columns the aggregates read."""
    row = {
        "scenario_id": "banking_adaptive_tool_use_0001",
        "domain": "banking",
        "model": model,
        "efficacy": 0.9,
        "task_completion": 0.9,
        "tool_selection": 0.9,
        "state_score": 0.9,
        "cost_usd": 0.0,
        "latency_ms": 1.0,
        "total_turns": 1,
        "reliability_pass_rate": 1.0,
        "reliability_consistency": 1.0,
        "tc_agreement": None,
        "ts_agreement": None,
        "outcome": OUTCOME_PASS,
    }
    row.update(over)
    return row


class TestExcludeUngradable:
    def test_drops_only_ungradable_rows(self):
        df = pd.DataFrame(
            [
                _row("GPT-5.5", outcome=OUTCOME_PASS),
                _row("GPT-5.5", outcome=OUTCOME_FAIL),
                _row("GPT-5.5", outcome=OUTCOME_UNGRADABLE),
            ]
        )
        kept = exclude_ungradable(df)
        assert len(kept) == 2
        assert OUTCOME_UNGRADABLE not in set(kept["outcome"])

    def test_no_outcome_column_passes_through(self):
        # Legacy artifacts / hand-built frames predating the column: nothing to drop.
        df = pd.DataFrame([{"model": "GPT-5.5", "efficacy": 0.8}])
        kept = exclude_ungradable(df)
        assert len(kept) == 1

    def test_empty_frame_passes_through(self):
        assert exclude_ungradable(pd.DataFrame()).empty


class TestUngradableDoesNotMovePublishedEfficacy:
    def test_ungradable_row_excluded_from_model_efficacy(self):
        # One gradable pass (0.9) + one ungradable placeholder (0.0) for the same
        # model. The published efficacy must be 0.9 (the pass), NOT 0.45 (the mean
        # that includes the harness-fault 0.0). A second model anchors the board.
        df = pd.DataFrame(
            [
                _row("GPT-5.5", scenario_id="s1", efficacy=0.9, outcome=OUTCOME_PASS),
                _row(
                    "GPT-5.5",
                    scenario_id="s2",
                    efficacy=0.0,
                    task_completion=0.0,
                    tool_selection=0.0,
                    state_score=0.0,
                    outcome=OUTCOME_UNGRADABLE,
                ),
                _row("Claude Sonnet 4.6", scenario_id="s1", efficacy=0.7),
            ]
        )
        board = compute_leaderboard(df)
        by_name = {m["name"]: m for m in board["models"]}
        assert by_name["GPT-5.5"]["efficacy"] == 0.9

    def test_all_ungradable_model_drops_off_board(self):
        # A model whose every row was ungradable has no gradable evidence — it must
        # not appear on the published board at all (vs. appearing with a fake 0.0).
        df = pd.DataFrame(
            [
                _row("GPT-5.5", efficacy=0.8, outcome=OUTCOME_PASS),
                _row("BrokenModel", efficacy=0.0, outcome=OUTCOME_UNGRADABLE),
            ]
        )
        board = compute_leaderboard(df)
        assert {m["name"] for m in board["models"]} == {"GPT-5.5"}


class TestReliabilityExcludesUngradable:
    def test_recompute_reliability_ignores_ungradable_runs(self):
        # Same (model, scenario), 3 reliability runs: 2 gradable passes + 1
        # ungradable. Reliability must be computed over the 2 gradable runs (a
        # perfect pass_rate), NOT dragged by the ungradable 0.0.
        rows = [
            _row("GPT-5.5", efficacy=0.9, outcome=OUTCOME_PASS),
            _row("GPT-5.5", efficacy=0.9, outcome=OUTCOME_PASS),
            _row("GPT-5.5", efficacy=0.0, outcome=OUTCOME_UNGRADABLE),
        ]
        _recompute_reliability(rows, reliability_runs=3)
        # Both gradable runs passed -> pass_rate 1.0 over the gradable set.
        assert rows[0]["reliability_pass_rate"] == 1.0
        assert rows[1]["reliability_pass_rate"] == 1.0

    def test_recompute_reliability_all_ungradable_leaves_columns_untouched(self):
        # A scenario whose every run was ungradable has no honest reliability; the
        # recompute must not overwrite the columns with a fabricated value.
        sentinel = -123.0
        rows = [
            _row(
                "GPT-5.5",
                efficacy=0.0,
                outcome=OUTCOME_UNGRADABLE,
                reliability_pass_rate=sentinel,
            ),
        ]
        _recompute_reliability(rows, reliability_runs=1)
        assert rows[0]["reliability_pass_rate"] == sentinel
