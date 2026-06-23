"""First-class pass/fail/ungradable episode outcome (issue #88).

Pins the harness-fault classes that make an episode UNGRADABLE (excluded, not
folded in as a silent 0.0) and the normal pass/fail split otherwise.
"""

from types import SimpleNamespace

from eval.scoring.rubrics import PASS_THRESHOLD
from scripts.run_eval import (
    OUTCOME_FAIL,
    OUTCOME_PASS,
    OUTCOME_UNGRADABLE,
    episode_outcome,
)


def _sim(*, error=None, ended_by="user_sim"):
    return SimpleNamespace(error=error, ended_by=ended_by)


def _judges(n_valid):
    return SimpleNamespace(n_judges_valid=n_valid)


def _outcome(efficacy, *, sim=None, tc=2, ts=2, state_result=None, state_gradable=True):
    return episode_outcome(
        efficacy,
        sim_result=sim or _sim(),
        tc_result=_judges(tc),
        ts_result=_judges(ts),
        state_result=state_result,
        state_gradable=state_gradable,
    )


# --- normal pass/fail -------------------------------------------------------- #


def test_high_efficacy_passes():
    assert _outcome(0.9) == OUTCOME_PASS


def test_at_threshold_passes():
    assert _outcome(PASS_THRESHOLD) == OUTCOME_PASS


def test_below_threshold_fails():
    assert _outcome(PASS_THRESHOLD - 0.01) == OUTCOME_FAIL


def test_zero_efficacy_is_fail_not_ungradable():
    """A clean run the agent simply failed is FAIL, never ungradable."""
    assert _outcome(0.0) == OUTCOME_FAIL


# --- ungradable: harness faults --------------------------------------------- #


def test_simulator_error_is_ungradable():
    assert _outcome(0.9, sim=_sim(error="boom")) == OUTCOME_UNGRADABLE


def test_ended_by_error_is_ungradable():
    assert _outcome(0.9, sim=_sim(ended_by="error")) == OUTCOME_UNGRADABLE


def test_no_valid_task_completion_judge_is_ungradable():
    assert _outcome(0.9, tc=0, ts=2) == OUTCOME_UNGRADABLE


def test_no_valid_tool_selection_judge_is_ungradable():
    assert _outcome(0.9, tc=2, ts=0) == OUTCOME_UNGRADABLE


def test_incomplete_graded_world_is_ungradable():
    """A stateful scenario whose world is non-gradable -> ungradable."""
    state = {"score": 0.5, "n_passed": 1, "n_total": 2}
    assert _outcome(0.9, state_result=state, state_gradable=False) == OUTCOME_UNGRADABLE


def test_gradable_stateful_world_is_graded_normally():
    state = {"score": 1.0, "n_passed": 2, "n_total": 2}
    assert _outcome(0.9, state_result=state, state_gradable=True) == OUTCOME_PASS


def test_stateless_scenario_unaffected_by_state_gradable():
    """state_result None (no ground_truth) is never the ungradable trigger."""
    assert _outcome(0.9, state_result=None, state_gradable=True) == OUTCOME_PASS


def test_harness_error_beats_a_passing_efficacy():
    """Priority: a harness error is ungradable even if efficacy would pass."""
    assert _outcome(1.0, sim=_sim(error="x"), tc=3, ts=3) == OUTCOME_UNGRADABLE
