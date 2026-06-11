"""Tests for the failure-mode taxonomy (issue #55).

Covers eval/scoring/failure_modes.py and its wiring into build_result_row:

- pass/fail gating against the shared reliability pass threshold,
- the deterministic-first precedence order (state-grader policy violation >
  premature-end flag > state-grader partial progress > judge keywords >
  incomplete-task fallback),
- keyword classification per mode, keyword priority, and case-insensitivity,
- a tie between a deterministic signal and a keyword signal resolving
  deterministic-first,
- judge_reasoning_text excluding parse-failed judges,
- build_result_row emitting failure_mode / failure_mode_source columns.
"""

from dataclasses import dataclass, field

import pytest

from eval.scoring.failure_modes import (
    FAILURE_MODES,
    HALLUCINATED_CAPABILITY,
    INCOMPLETE_TASK,
    POLICY_VIOLATION,
    PREMATURE_END,
    SOURCE_FALLBACK,
    SOURCE_JUDGE_KEYWORD,
    SOURCE_PREMATURE_FLAG,
    SOURCE_STATE_GRADER,
    TOOL_SELECTION_ERROR,
    WRONG_PARAMETERS,
    classify_failure,
    judge_reasoning_text,
)
from eval.scoring.rubrics import PASS_THRESHOLD
from eval.scoring.state_check import score_state_changes


def _partial_state(n_passed=1, n_total=3):
    """A state result with partial verifiable progress (some pass, some fail)."""
    checks = [{"passed": i < n_passed, "detail": f"check {i}"} for i in range(n_total)]
    return {
        "score": n_passed / n_total,
        "checks": checks,
        "n_passed": n_passed,
        "n_total": n_total,
    }


def _mutation_state():
    """The REAL grader's no-unauthorized-mutation failure (not a hand-built
    detail string), so the classifier's prefix match is pinned to the grader."""
    initial = {"accounts": {"A": {"balance": 100.0}}}
    final = {"accounts": {"A": {"balance": 50.0}}}
    return score_state_changes(initial, final, [])


class TestPassFailGate:
    def test_pass_returns_none(self):
        assert classify_failure(0.9) is None

    def test_exactly_at_threshold_is_a_pass(self):
        assert classify_failure(PASS_THRESHOLD) is None

    def test_just_below_threshold_is_a_failure(self):
        result = classify_failure(PASS_THRESHOLD - 0.01)
        assert result is not None
        assert result["mode"] in FAILURE_MODES

    def test_none_efficacy_returns_none(self):
        # An unscored row is not a classified failure.
        assert classify_failure(None) is None

    def test_pass_with_premature_flag_still_none(self):
        # premature_end can be true on a run that still scores above threshold
        # (judges + partial state can clear 0.7); the taxonomy only covers
        # failures — premature_end_rate covers the rest.
        assert classify_failure(0.85, premature_end=True) is None


class TestDeterministicSignals:
    def test_unauthorized_mutation_is_policy_violation(self):
        result = classify_failure(0.3, state_result=_mutation_state())
        assert result == {"mode": POLICY_VIOLATION, "source": SOURCE_STATE_GRADER}

    def test_premature_flag_is_premature_end(self):
        result = classify_failure(0.3, premature_end=True)
        assert result == {"mode": PREMATURE_END, "source": SOURCE_PREMATURE_FLAG}

    def test_policy_violation_outranks_premature_flag(self):
        result = classify_failure(0.3, state_result=_mutation_state(), premature_end=True)
        assert result["mode"] == POLICY_VIOLATION

    def test_partial_state_progress_is_incomplete_task(self):
        result = classify_failure(0.3, state_result=_partial_state(1, 3))
        assert result == {"mode": INCOMPLETE_TASK, "source": SOURCE_STATE_GRADER}

    def test_premature_flag_outranks_partial_progress(self):
        result = classify_failure(0.3, state_result=_partial_state(2, 5), premature_end=True)
        assert result["mode"] == PREMATURE_END

    def test_zero_state_progress_falls_through_to_keywords(self):
        # 0/n passed says nothing about WHY — keyword assist decides.
        result = classify_failure(
            0.2,
            state_result=_partial_state(0, 3),
            judge_reasoning="The agent called the wrong tool at turn 2.",
        )
        assert result == {"mode": TOOL_SELECTION_ERROR, "source": SOURCE_JUDGE_KEYWORD}

    def test_full_state_pass_but_low_efficacy_falls_through(self):
        # All assertions passed yet judges scored low: no deterministic failure
        # evidence, so keywords (here: none) -> fallback.
        result = classify_failure(0.5, state_result=_partial_state(3, 3))
        assert result == {"mode": INCOMPLETE_TASK, "source": SOURCE_FALLBACK}


class TestKeywordAssist:
    @pytest.mark.parametrize(
        ("reasoning", "expected"),
        [
            ("The agent hallucinated a refund capability.", HALLUCINATED_CAPABILITY),
            ("Claimed to have escalated, but no such tool exists.", HALLUCINATED_CAPABILITY),
            ("Initiated the transfer without verifying the customer's identity.", POLICY_VIOLATION),
            ("This is a clear policy breach: unauthorized account access.", POLICY_VIOLATION),
            ("Right tool, but the wrong amount was passed for the transfer.", WRONG_PARAMETERS),
            ("The call used an incorrect parameter for account_id.", WRONG_PARAMETERS),
            ("The agent should have called report_suspicious_transaction.", TOOL_SELECTION_ERROR),
            ("It never called the escalation tool at all.", TOOL_SELECTION_ERROR),
        ],
    )
    def test_keyword_maps_to_mode(self, reasoning, expected):
        result = classify_failure(0.3, judge_reasoning=reasoning)
        assert result == {"mode": expected, "source": SOURCE_JUDGE_KEYWORD}

    def test_keywords_case_insensitive(self):
        result = classify_failure(0.3, judge_reasoning="THE AGENT HALLUCINATED A TOOL.")
        assert result["mode"] == HALLUCINATED_CAPABILITY

    def test_priority_hallucination_beats_tool_selection(self):
        # Both signals present: the more specific mode (earlier priority) wins.
        reasoning = "The agent used the wrong tool and hallucinated a capability."
        result = classify_failure(0.3, judge_reasoning=reasoning)
        assert result["mode"] == HALLUCINATED_CAPABILITY

    def test_priority_wrong_parameters_beats_tool_selection(self):
        reasoning = "Wrong parameter on the lookup; also a redundant second call."
        result = classify_failure(0.3, judge_reasoning=reasoning)
        assert result["mode"] == WRONG_PARAMETERS

    def test_deterministic_beats_keyword(self):
        # Tie between a deterministic signal (partial state progress) and a
        # keyword signal (wrong tool): deterministic-first wins.
        result = classify_failure(
            0.3,
            state_result=_partial_state(1, 3),
            judge_reasoning="The agent picked the wrong tool entirely.",
        )
        assert result == {"mode": INCOMPLETE_TASK, "source": SOURCE_STATE_GRADER}

    def test_premature_flag_beats_keyword(self):
        result = classify_failure(
            0.3,
            premature_end=True,
            judge_reasoning="The agent hallucinated a capability.",
        )
        assert result["mode"] == PREMATURE_END

    def test_no_signals_falls_back_to_incomplete_task(self):
        result = classify_failure(0.3, judge_reasoning="Mediocre performance overall.")
        assert result == {"mode": INCOMPLETE_TASK, "source": SOURCE_FALLBACK}

    def test_empty_reasoning_falls_back(self):
        result = classify_failure(0.3, judge_reasoning="")
        assert result == {"mode": INCOMPLETE_TASK, "source": SOURCE_FALLBACK}


# --- judge_reasoning_text ----------------------------------------------------


@dataclass
class _FakeJudgeResult:
    judge_name: str
    reasoning: str
    parse_failed: bool = False


@dataclass
class _FakeConsensus:
    judge_results: list = field(default_factory=list)


class TestJudgeReasoningText:
    def test_concatenates_valid_judges_across_dimensions(self):
        tc = _FakeConsensus([_FakeJudgeResult("Kimi", "tc says wrong tool")])
        ts = _FakeConsensus([_FakeJudgeResult("GLM", "ts says wrong parameter")])
        text = judge_reasoning_text(tc, ts)
        assert "tc says wrong tool" in text
        assert "ts says wrong parameter" in text

    def test_excludes_parse_failed_judges(self):
        tc = _FakeConsensus(
            [
                _FakeJudgeResult("Kimi", "real assessment"),
                _FakeJudgeResult("Opus", "Failed to parse judge response", parse_failed=True),
            ]
        )
        text = judge_reasoning_text(tc)
        assert "real assessment" in text
        assert "Failed to parse" not in text

    def test_empty_panels_yield_empty_string(self):
        assert judge_reasoning_text(_FakeConsensus(), _FakeConsensus()) == ""


# --- build_result_row wiring -------------------------------------------------


@dataclass
class _FakeDomain:
    value: str = "banking"


@dataclass
class _FakeScenario:
    id: str = "s1"
    domain: _FakeDomain = field(default_factory=_FakeDomain)
    category: str = "scope_management"


@dataclass
class _FakeSpec:
    name: str = "GPT-4.1"


@dataclass
class _FakeSim:
    total_latency_ms: float = 1234.5
    total_turns: int = 4
    total_input_tokens: int = 100
    total_output_tokens: int = 50
    completed: bool = True
    ended_by: str = "user_sim"
    state_progress_at_end: float | None = None
    premature_end: bool = False


def _consensus(judge_results=None, score=0.5):
    from eval.scoring.judge import ConsensusResult

    return ConsensusResult(
        scenario_id="s1",
        rubric_type="task_completion",
        judge_results=judge_results or [],
        consensus_score=score,
        agreement_rate=None,
        max_disagreement=None,
        n_judges_requested=3,
        n_judges_valid=len(judge_results or []),
        parse_failures=[],
        api_failures=[],
        degraded=False,
    )


def _judge(reasoning, parse_failed=False):
    from eval.scoring.judge import JudgeResult

    return JudgeResult(
        judge_name="Kimi",
        rubric_type="task_completion",
        overall_score=0.4,
        reasoning=reasoning,
        raw_response={},
        latency_ms=10.0,
        parse_failed=parse_failed,
    )


class TestBuildResultRowFailureColumns:
    def _row(self, efficacy, *, sim=None, tc=None, ts=None, state_result=None):
        from scripts.run_eval import build_result_row

        return build_result_row(
            _FakeScenario(),
            _FakeSpec(),
            sim or _FakeSim(),
            tc or _consensus(),
            ts or _consensus(),
            efficacy=efficacy,
            cost_usd=0.001,
            state_result=state_result,
        )

    def test_passing_row_has_null_failure_columns(self):
        row = self._row(0.9)
        assert row["failure_mode"] is None
        assert row["failure_mode_source"] is None

    def test_premature_failure_row_classified(self):
        row = self._row(0.4, sim=_FakeSim(premature_end=True))
        assert row["failure_mode"] == PREMATURE_END
        assert row["failure_mode_source"] == SOURCE_PREMATURE_FLAG

    def test_keyword_failure_row_uses_judge_reasoning(self):
        tc = _consensus([_judge("The agent used the wrong tool at turn 3.")])
        row = self._row(0.4, tc=tc)
        assert row["failure_mode"] == TOOL_SELECTION_ERROR
        assert row["failure_mode_source"] == SOURCE_JUDGE_KEYWORD

    def test_parse_failed_reasoning_not_used_for_keywords(self):
        # The parse-failure placeholder text must not feed keyword matching.
        tc = _consensus([_judge("wrong tool used here", parse_failed=True)])
        row = self._row(0.4, tc=tc)
        assert row["failure_mode"] == INCOMPLETE_TASK
        assert row["failure_mode_source"] == SOURCE_FALLBACK

    def test_state_grader_policy_violation_row(self):
        row = self._row(0.2, state_result=_mutation_state())
        assert row["failure_mode"] == POLICY_VIOLATION
        assert row["failure_mode_source"] == SOURCE_STATE_GRADER
