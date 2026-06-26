"""Tests for judge scoring utilities."""

from dataclasses import dataclass, field

import pytest

import eval.scoring.judge as judge_mod
from eval.scoring.judge import (
    ConsensusResult,
    JudgeResult,
    _parse_judge_response,
    score_with_all_judges,
    score_with_all_judges_combined,
)


class TestParseJudgeResponse:
    def test_valid_json(self):
        result = _parse_judge_response('{"overall_score": 0.85, "overall_reasoning": "Good"}')
        assert result["overall_score"] == 0.85

    def test_json_in_code_block(self):
        content = '```json\n{"overall_score": 0.7, "overall_reasoning": "OK"}\n```'
        result = _parse_judge_response(content)
        assert result["overall_score"] == 0.7

    def test_json_in_bare_code_block(self):
        content = 'Here is my evaluation:\n```\n{"overall_score": 0.6}\n```'
        result = _parse_judge_response(content)
        assert result["overall_score"] == 0.6

    def test_unparseable_returns_none(self):
        # Parse failure is now explicit (None), not a fabricated 0.0 result.
        assert _parse_judge_response("This is not JSON at all.") is None

    def test_nested_json(self):
        content = (
            '{"goal_scores": [{"goal": "a", "score": "COMPLETE"}], '
            '"overall_score": 0.9, "overall_reasoning": "All good"}'
        )
        result = _parse_judge_response(content)
        assert result["overall_score"] == 0.9
        assert "goal_scores" in result


# --- Consensus orchestration tests -----------------------------------------
#
# These monkeypatch score_with_judge so no real API calls are made. Each fake
# is keyed by judge config so we can drive different judges to different
# outcomes within one score_with_all_judges call.


def _valid(judge_name, score, rubric_type="task_completion"):
    return JudgeResult(
        judge_name=judge_name,
        rubric_type=rubric_type,
        overall_score=score,
        reasoning="ok",
        raw_response={"overall_score": score},
        latency_ms=1.0,
    )


def _parse_failed(judge_name, rubric_type="task_completion"):
    return JudgeResult(
        judge_name=judge_name,
        rubric_type=rubric_type,
        overall_score=0.0,
        reasoning="Failed to parse judge response",
        raw_response={},
        latency_ms=1.0,
        parse_failed=True,
    )


def _install_fake_scorer(monkeypatch, behavior):
    """Patch score_with_judge with a per-judge behavior map.

    ``behavior`` maps judge config name -> either a JudgeResult to return, or
    an Exception instance to raise (simulating an API failure).
    """

    def fake(judge, system_prompt, rubric_prompt, rubric_type):
        outcome = behavior[judge.name]
        if isinstance(outcome, Exception):
            raise outcome
        # Re-stamp rubric_type so helpers don't need to know it up front.
        return JudgeResult(
            judge_name=outcome.judge_name,
            rubric_type=rubric_type,
            overall_score=outcome.overall_score,
            reasoning=outcome.reasoning,
            raw_response=outcome.raw_response,
            latency_ms=outcome.latency_ms,
            parse_failed=outcome.parse_failed,
        )

    monkeypatch.setattr(judge_mod, "score_with_judge", fake)


@pytest.fixture
def three_judges(monkeypatch):
    """Replace JUDGES with three simple configs (kimi/glm/opus by name)."""

    @dataclass(frozen=True)
    class _Cfg:
        name: str

    fake_judges = {
        "kimi": _Cfg(name="Kimi"),
        "glm": _Cfg(name="GLM"),
        "opus": _Cfg(name="Opus"),
    }
    monkeypatch.setattr(judge_mod, "JUDGES", fake_judges)
    return fake_judges


class TestConsensus:
    def test_happy_path_unchanged(self, monkeypatch, three_judges):
        _install_fake_scorer(
            monkeypatch,
            {
                "Kimi": _valid("Kimi", 0.8),
                "GLM": _valid("GLM", 0.9),
                "Opus": _valid("Opus", 0.7),
            },
        )
        res = score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert res.n_judges_requested == 3
        assert res.n_judges_valid == 3
        # Consensus is the MEDIAN of valid judge scores: median(0.8,0.9,0.7)=0.8.
        assert res.consensus_score == pytest.approx(0.8)
        assert res.parse_failures == []
        assert res.api_failures == []
        assert res.degraded is False
        # 3 pairs: |0.8-0.9|=0.1, |0.8-0.7|=0.1 agree; |0.9-0.7| is 0.2 in
        # float terms (0.2000...07) so it falls just outside the <=0.2 band.
        # 2 of 3 pairs agree (preserving the existing agreement logic).
        assert res.agreement_rate == pytest.approx(2 / 3)
        assert res.max_disagreement == pytest.approx(0.2)

    def test_consensus_median_robust_to_outlier(self, monkeypatch, three_judges):
        # One rogue judge (0.1) cannot drag the consensus: median(0.8,0.9,0.1)=0.8,
        # whereas a mean would be 0.6. This is the whole point of the median switch.
        _install_fake_scorer(
            monkeypatch,
            {
                "Kimi": _valid("Kimi", 0.8),
                "GLM": _valid("GLM", 0.9),
                "Opus": _valid("Opus", 0.1),
            },
        )
        res = score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert res.consensus_score == pytest.approx(0.8)
        # max_disagreement still reflects the full spread for transparency.
        assert res.max_disagreement == pytest.approx(0.8)

    def test_consensus_median_two_judges_is_midpoint(self, monkeypatch, three_judges):
        # With two valid judges the median equals their mean (midpoint).
        _install_fake_scorer(
            monkeypatch,
            {
                "Kimi": _valid("Kimi", 0.4),
                "GLM": _valid("GLM", 0.8),
                "Opus": _parse_failed("Opus"),
            },
        )
        res = score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert res.n_judges_valid == 2
        assert res.consensus_score == pytest.approx(0.6)

    def test_parse_failure_excluded_from_consensus(self, monkeypatch, three_judges):
        # (a) one judge parse-fails twice -> excluded, consensus = mean of 2.
        _install_fake_scorer(
            monkeypatch,
            {
                "Kimi": _valid("Kimi", 0.8),
                "GLM": _valid("GLM", 0.9),
                "Opus": _parse_failed("Opus"),
            },
        )
        res = score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert res.n_judges_valid == 2
        assert res.parse_failures == ["Opus"]
        assert res.api_failures == []
        assert res.consensus_score == pytest.approx((0.8 + 0.9) / 2)
        # parse-failed kept in judge_results for transparency
        assert len(res.judge_results) == 3
        assert res.degraded is False

    def test_parse_failure_then_retry_success_included(self, monkeypatch):
        # (b) score_with_judge itself handles the retry. Drive the API layer:
        # first call returns garbage, second returns valid JSON -> included.
        @dataclass(frozen=True)
        class _Cfg:
            name: str = "Solo"
            provider: str = "anthropic"
            model_id: str = "x"
            temperature: float = 0.0
            max_tokens: int = 4096
            endpoint: str | None = None

        calls = {"n": 0}

        def fake_api(judge, system_prompt, rubric_prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "not json", "fake-model-v1", (10, 5)
            return (
                '{"overall_score": 0.75, "overall_reasoning": "second try"}',
                "fake-model-v1",
                (10, 5),
            )

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        result = judge_mod.score_with_judge(_Cfg(), "sys", "rub", "task_completion")
        assert calls["n"] == 2  # retried once
        assert result.parse_failed is False
        assert result.overall_score == pytest.approx(0.75)

    def test_parse_failure_twice_flags_parse_failed(self, monkeypatch):
        @dataclass(frozen=True)
        class _Cfg:
            name: str = "Solo"
            provider: str = "anthropic"
            model_id: str = "x"
            temperature: float = 0.0
            max_tokens: int = 4096
            endpoint: str | None = None

        calls = {"n": 0}

        def fake_api(judge, system_prompt, rubric_prompt):
            calls["n"] += 1
            return "still not json", "fake-model-v1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        result = judge_mod.score_with_judge(_Cfg(), "sys", "rub", "task_completion")
        assert calls["n"] == 2  # initial + one retry, then gives up
        assert result.parse_failed is True

    @pytest.mark.parametrize("bad_score", [5.0, -1.0])
    def test_out_of_range_score_is_parse_failure(self, monkeypatch, bad_score):
        # B1: an overall_score outside [0,1] is rejected as a parse failure (not
        # clamped), so it retries once then is excluded. Both calls return the
        # same out-of-range score -> parse_failed.
        @dataclass(frozen=True)
        class _Cfg:
            name: str = "Solo"
            provider: str = "anthropic"
            model_id: str = "x"
            temperature: float = 0.0
            max_tokens: int = 4096
            endpoint: str | None = None

        calls = {"n": 0}

        def fake_api(judge, system_prompt, rubric_prompt):
            calls["n"] += 1
            return (
                f'{{"overall_score": {bad_score}, "overall_reasoning": "off-scale"}}',
                "fake-model-v1",
                (10, 5),
            )

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        result = judge_mod.score_with_judge(_Cfg(), "sys", "rub", "task_completion")
        assert calls["n"] == 2  # initial + one retry, then gives up
        assert result.parse_failed is True

    def test_api_failure_recorded(self, monkeypatch, three_judges):
        # (c) one judge raises -> api_failures records it; consensus from 2.
        _install_fake_scorer(
            monkeypatch,
            {
                "Kimi": _valid("Kimi", 0.8),
                "GLM": _valid("GLM", 0.6),
                "Opus": RuntimeError("503 from provider"),
            },
        )
        res = score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert res.api_failures == ["Opus"]
        assert res.parse_failures == []
        assert res.n_judges_valid == 2
        assert res.consensus_score == pytest.approx((0.8 + 0.6) / 2)
        assert res.degraded is False

    def test_all_judges_fail(self, monkeypatch, three_judges):
        # (d) all judges fail -> n_judges_valid == 0, degraded, agreement None.
        _install_fake_scorer(
            monkeypatch,
            {
                "Kimi": RuntimeError("boom"),
                "GLM": _parse_failed("GLM"),
                "Opus": RuntimeError("boom"),
            },
        )
        res = score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert res.n_judges_valid == 0
        assert res.degraded is True
        assert res.consensus_score == 0.0
        assert res.agreement_rate is None
        assert res.max_disagreement is None
        assert res.api_failures == ["Kimi", "Opus"] or sorted(res.api_failures) == [
            "Kimi",
            "Opus",
        ]
        assert res.parse_failures == ["GLM"]

    def test_single_valid_judge_agreement_none_and_degraded(self, monkeypatch, three_judges):
        # (e) single valid judge -> agreement_rate None, degraded True.
        _install_fake_scorer(
            monkeypatch,
            {
                "Kimi": _valid("Kimi", 0.85),
                "GLM": _parse_failed("GLM"),
                "Opus": RuntimeError("down"),
            },
        )
        res = score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert res.n_judges_valid == 1
        assert res.consensus_score == pytest.approx(0.85)
        assert res.agreement_rate is None
        assert res.max_disagreement is None
        assert res.degraded is True

    def test_concurrency_preserved(self, monkeypatch, three_judges):
        # Sanity: all three requested judges are actually invoked.
        seen = set()

        def fake(judge, system_prompt, rubric_prompt, rubric_type):
            seen.add(judge.name)
            return _valid(judge.name, 0.5)

        monkeypatch.setattr(judge_mod, "score_with_judge", fake)
        score_with_all_judges("sys", "rub", "task_completion", "s1")
        assert seen == {"Kimi", "GLM", "Opus"}


# --- Combined judge (one call scores both rubrics) tests --------------------
#
# These monkeypatch score_with_judge_combined (for orchestration tests) or
# _call_judge_api (for the single-judge retry/parse tests) so no real API
# calls are made.


def _combined_pair(judge_name, tc_score, ts_score):
    """A (task_completion, tool_selection) JudgeResult pair from one judge."""
    return (
        JudgeResult(
            judge_name=judge_name,
            rubric_type="task_completion",
            overall_score=tc_score,
            reasoning="ok-tc",
            raw_response={"overall_score": tc_score, "overall_reasoning": "ok-tc"},
            latency_ms=1.0,
        ),
        JudgeResult(
            judge_name=judge_name,
            rubric_type="tool_selection",
            overall_score=ts_score,
            reasoning="ok-ts",
            raw_response={"overall_score": ts_score, "overall_reasoning": "ok-ts"},
            latency_ms=1.0,
        ),
    )


def _combined_failed_pair(judge_name):
    """A both-dimensions parse-failed pair (the coupling rule)."""
    return tuple(
        JudgeResult(
            judge_name=judge_name,
            rubric_type=rt,
            overall_score=0.0,
            reasoning="Failed to parse combined judge response",
            raw_response={},
            latency_ms=1.0,
            parse_failed=True,
        )
        for rt in ("task_completion", "tool_selection")
    )


def _install_fake_combined_scorer(monkeypatch, behavior):
    """Patch score_with_judge_combined with a per-judge behavior map.

    ``behavior`` maps judge config name -> either a (tc, ts) JudgeResult pair,
    or an Exception instance to raise (simulating a combined API failure).
    """

    def fake(judge, system_prompt, combined_prompt):
        outcome = behavior[judge.name]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(judge_mod, "score_with_judge_combined", fake)


@dataclass(frozen=True)
class _CombinedCfg:
    name: str = "Solo"
    provider: str = "anthropic"
    model_id: str = "x"
    temperature: float = 0.0
    max_tokens: int = 4096
    endpoint: str | None = None


class TestCombinedConsensus:
    def test_happy_path_two_consensus_results(self, monkeypatch, three_judges):
        # One combined call per judge produces a (tc, ts) consensus pair with
        # the SAME shape as two separate score_with_all_judges calls.
        _install_fake_combined_scorer(
            monkeypatch,
            {
                "Kimi": _combined_pair("Kimi", 0.8, 0.6),
                "GLM": _combined_pair("GLM", 0.9, 0.7),
                "Opus": _combined_pair("Opus", 0.7, 0.5),
            },
        )
        tc, ts = score_with_all_judges_combined("sys", "combined", "s1")

        # Task completion consensus matches the separate-path happy path exactly.
        assert tc.rubric_type == "task_completion"
        assert tc.n_judges_requested == 3
        assert tc.n_judges_valid == 3
        # Median consensus: median(0.8,0.9,0.7)=0.8.
        assert tc.consensus_score == pytest.approx(0.8)
        assert tc.parse_failures == []
        assert tc.api_failures == []
        assert tc.degraded is False
        assert tc.agreement_rate == pytest.approx(2 / 3)
        assert tc.max_disagreement == pytest.approx(0.2)

        # Tool selection consensus is its own panel.
        assert ts.rubric_type == "tool_selection"
        assert ts.n_judges_valid == 3
        # Median consensus: median(0.6,0.7,0.5)=0.6.
        assert ts.consensus_score == pytest.approx(0.6)
        assert ts.degraded is False

    def test_one_dimension_missing_fails_both(self, monkeypatch):
        # The combined response parses but tool_selection has no overall_score.
        # The whole judge must be parse-failed for BOTH dimensions.
        def fake_api(judge, system_prompt, rubric_prompt):
            body = (
                '{"task_completion": {"overall_score": 0.9, "overall_reasoning": "ok"}, '
                '"tool_selection": {"overall_reasoning": "no score field"}}'
            )
            return body, "fake-model-v1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = judge_mod.score_with_judge_combined(_CombinedCfg(), "sys", "combined")
        assert tc_jr.parse_failed is True
        assert ts_jr.parse_failed is True
        assert tc_jr.overall_score == 0.0
        assert ts_jr.overall_score == 0.0

    @pytest.mark.parametrize("bad_score", [5.0, -1.0])
    def test_out_of_range_score_fails_both(self, monkeypatch, bad_score):
        # B1: one dimension carries an out-of-range overall_score. It is rejected
        # (not clamped), so the whole judge is parse-failed for BOTH dimensions.
        def fake_api(judge, system_prompt, rubric_prompt):
            body = (
                f'{{"task_completion": {{"overall_score": {bad_score}, '
                '"overall_reasoning": "ok"}, '
                '"tool_selection": {"overall_score": 0.6, "overall_reasoning": "ok"}}'
            )
            return body, "fake-model-v1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = judge_mod.score_with_judge_combined(_CombinedCfg(), "sys", "combined")
        assert tc_jr.parse_failed is True
        assert ts_jr.parse_failed is True
        assert tc_jr.overall_score == 0.0
        assert ts_jr.overall_score == 0.0

    def test_retry_then_success_included(self, monkeypatch):
        # First call: tool_selection missing its score -> rejected. Second call:
        # both dimensions valid -> included. One retry, like the two-call path.
        calls = {"n": 0}

        def fake_api(judge, system_prompt, rubric_prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"task_completion": {"overall_score": 0.9}}', "m1", (10, 5)
            return (
                '{"task_completion": {"overall_score": 0.9, "overall_reasoning": "a"}, '
                '"tool_selection": {"overall_score": 0.6, "overall_reasoning": "b"}}',
                "m1",
                (10, 5),
            )

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = judge_mod.score_with_judge_combined(_CombinedCfg(), "sys", "combined")
        assert calls["n"] == 2  # retried once
        assert tc_jr.parse_failed is False
        assert ts_jr.parse_failed is False
        assert tc_jr.overall_score == pytest.approx(0.9)
        assert ts_jr.overall_score == pytest.approx(0.6)

    def test_parse_failure_twice_flags_both(self, monkeypatch):
        calls = {"n": 0}

        def fake_api(judge, system_prompt, rubric_prompt):
            calls["n"] += 1
            return "still not json", "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = judge_mod.score_with_judge_combined(_CombinedCfg(), "sys", "combined")
        assert calls["n"] == 2  # initial + one retry, then gives up
        assert tc_jr.parse_failed is True
        assert ts_jr.parse_failed is True

    def test_api_failure_recorded_on_both(self, monkeypatch, three_judges):
        # A raised exception in the combined call drops the judge from BOTH
        # panels — it is one call, so it fails for both dimensions identically.
        _install_fake_combined_scorer(
            monkeypatch,
            {
                "Kimi": _combined_pair("Kimi", 0.8, 0.6),
                "GLM": _combined_pair("GLM", 0.6, 0.4),
                "Opus": RuntimeError("503 from provider"),
            },
        )
        tc, ts = score_with_all_judges_combined("sys", "combined", "s1")
        assert tc.api_failures == ["Opus"]
        assert ts.api_failures == ["Opus"]
        assert tc.n_judges_valid == 2
        assert ts.n_judges_valid == 2
        assert tc.consensus_score == pytest.approx((0.8 + 0.6) / 2)
        assert ts.consensus_score == pytest.approx((0.6 + 0.4) / 2)
        assert tc.degraded is False
        assert ts.degraded is False

    def test_combined_parse_failure_excluded_from_both(self, monkeypatch, three_judges):
        # A both-dimension parse-failed judge is excluded from both consensuses
        # and recorded in parse_failures for both.
        _install_fake_combined_scorer(
            monkeypatch,
            {
                "Kimi": _combined_pair("Kimi", 0.8, 0.6),
                "GLM": _combined_pair("GLM", 0.9, 0.7),
                "Opus": _combined_failed_pair("Opus"),
            },
        )
        tc, ts = score_with_all_judges_combined("sys", "combined", "s1")
        assert tc.parse_failures == ["Opus"]
        assert ts.parse_failures == ["Opus"]
        assert tc.n_judges_valid == 2
        assert ts.n_judges_valid == 2
        assert tc.consensus_score == pytest.approx((0.8 + 0.9) / 2)
        # parse-failed judge kept in judge_results for transparency, both panels
        assert len(tc.judge_results) == 3
        assert len(ts.judge_results) == 3


# --- Row building tests -----------------------------------------------------
#
# build_result_row is pure; we feed it faked ConsensusResult/sim objects and
# assert the row shape, especially the None-agreement degraded paths.


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


def _consensus(**kwargs):
    defaults = dict(
        scenario_id="s1",
        rubric_type="task_completion",
        judge_results=[],
        consensus_score=0.0,
        agreement_rate=None,
        max_disagreement=None,
        n_judges_requested=3,
        n_judges_valid=0,
        parse_failures=[],
        api_failures=[],
        degraded=False,
    )
    defaults.update(kwargs)
    return ConsensusResult(**defaults)


class TestBuildResultRow:
    def _build(self, tc, ts):
        from scripts.run_eval import build_result_row

        return build_result_row(
            _FakeScenario(),
            _FakeSpec(),
            _FakeSim(),
            tc,
            ts,
            efficacy=0.5,
            cost_usd=0.001,
        )

    def test_full_panel_row(self):
        tc = _consensus(
            judge_results=[_valid("Kimi", 0.8), _valid("GLM", 0.9)],
            consensus_score=0.85,
            agreement_rate=1.0,
            max_disagreement=0.1,
            n_judges_valid=2,
        )
        ts = _consensus(
            rubric_type="tool_selection",
            judge_results=[_valid("Kimi", 0.7, "tool_selection")],
            consensus_score=0.7,
            agreement_rate=None,
            max_disagreement=None,
            n_judges_valid=1,
            degraded=True,
        )
        row = self._build(tc, ts)
        assert row["tc_n_judges"] == 2
        assert row["ts_n_judges"] == 1
        assert row["tc_agreement"] == 1.0
        assert row["ts_agreement"] is None  # single judge -> undefined
        assert row["ts_max_disagreement"] is None
        assert row["ts_degraded"] is True
        assert row["tc_Kimi"] == 0.8
        assert row["high_disagreement"] is False  # None disagreement treated as 0

    def test_none_agreement_does_not_crash(self):
        # All-failed panel: agreement/disagreement None, must round cleanly.
        tc = _consensus(parse_failures=["GLM"], api_failures=["Kimi", "Opus"], degraded=True)
        ts = _consensus(rubric_type="tool_selection", api_failures=["Kimi"], degraded=True)
        row = self._build(tc, ts)
        assert row["tc_agreement"] is None
        assert row["tc_max_disagreement"] is None
        assert row["tc_parse_failures"] == 1
        assert row["tc_api_failures"] == 2
        assert row["tc_n_judges"] == 0
        assert row["high_disagreement"] is False

    def test_parse_failed_judge_excluded_from_score_columns(self):
        tc = _consensus(
            judge_results=[_valid("Kimi", 0.8), _parse_failed("Opus")],
            consensus_score=0.8,
            n_judges_valid=1,
            parse_failures=["Opus"],
            agreement_rate=None,
            max_disagreement=None,
            degraded=True,
        )
        ts = _consensus(rubric_type="tool_selection", n_judges_valid=0, degraded=True)
        row = self._build(tc, ts)
        assert "tc_Kimi" in row
        # parse-failed judge must NOT emit a misleading 0.0 score column
        assert "tc_Opus" not in row

    def test_state_columns_none_without_state_result(self):
        # Legacy path: build_result_row called without a state_result.
        tc = _consensus(judge_results=[_valid("Kimi", 0.8)], consensus_score=0.8, n_judges_valid=1)
        ts = _consensus(rubric_type="tool_selection", n_judges_valid=0)
        row = self._build(tc, ts)
        assert row["state_score"] is None
        assert row["state_checks_passed"] is None
        assert row["state_checks_total"] is None

    def test_combined_and_separate_rows_identical(self, monkeypatch, three_judges):
        # Regression: build_result_row must produce byte-identical column output
        # whether the (tc, ts) consensus pair came from the combined one-call
        # path or the separate two-call path, given equivalent judge scores.
        from scripts.run_eval import build_result_row

        scores = {"Kimi": (0.8, 0.6), "GLM": (0.9, 0.7), "Opus": (0.7, 0.5)}

        # Combined path: one call per judge yields the (tc, ts) pair.
        _install_fake_combined_scorer(
            monkeypatch,
            {name: _combined_pair(name, tc, ts) for name, (tc, ts) in scores.items()},
        )
        c_tc, c_ts = score_with_all_judges_combined("sys", "combined", "s1")

        # Separate path: two independent calls, same per-judge scores.
        _install_fake_scorer(
            monkeypatch,
            {name: _valid(name, tc) for name, (tc, _ts) in scores.items()},
        )
        s_tc = score_with_all_judges("sys", "tc_prompt", "task_completion", "s1")
        _install_fake_scorer(
            monkeypatch,
            {name: _valid(name, ts, "tool_selection") for name, (_tc, ts) in scores.items()},
        )
        s_ts = score_with_all_judges("sys", "ts_prompt", "tool_selection", "s1")

        combined_row = build_result_row(
            _FakeScenario(), _FakeSpec(), _FakeSim(), c_tc, c_ts, efficacy=0.5, cost_usd=0.001
        )
        separate_row = build_result_row(
            _FakeScenario(), _FakeSpec(), _FakeSim(), s_tc, s_ts, efficacy=0.5, cost_usd=0.001
        )
        assert combined_row.keys() == separate_row.keys()
        assert combined_row == separate_row

    def test_state_columns_present_with_state_result(self):
        from scripts.run_eval import build_result_row

        tc = _consensus(judge_results=[_valid("Kimi", 0.8)], consensus_score=0.8, n_judges_valid=1)
        ts = _consensus(rubric_type="tool_selection", n_judges_valid=1)
        state_result = {
            "score": 0.5,
            "checks": [{"passed": True, "detail": "x"}, {"passed": False, "detail": "y"}],
            "n_passed": 1,
            "n_total": 2,
        }
        row = build_result_row(
            _FakeScenario(),
            _FakeSpec(),
            _FakeSim(),
            tc,
            ts,
            efficacy=0.5,
            cost_usd=0.001,
            state_result=state_result,
        )
        assert row["state_score"] == 0.5
        assert row["state_checks_passed"] == 1
        assert row["state_checks_total"] == 2
        # S3: a clean stateful run is gradable and carries the graded score.
        assert row["state_gradable"] is True
        assert row["tool_sim_parse_failures"] == 0

    def test_state_nongradable_when_tool_sim_parse_failure(self):
        # S3: a stateful scenario whose run had a tool-sim parse failure is NOT
        # gradable — state columns null, state_gradable False, count surfaced.
        from scripts.run_eval import build_result_row

        @dataclass
        class _SimWithParseFailure:
            total_latency_ms: float = 1234.5
            total_turns: int = 4
            total_input_tokens: int = 100
            total_output_tokens: int = 50
            completed: bool = True
            tool_sim_parse_failures: int = 1

        tc = _consensus(judge_results=[_valid("Kimi", 0.8)], consensus_score=0.8, n_judges_valid=1)
        ts = _consensus(rubric_type="tool_selection", n_judges_valid=1)
        state_result = {
            "score": 1.0,
            "checks": [{"passed": True, "detail": "x"}],
            "n_passed": 1,
            "n_total": 1,
        }
        row = build_result_row(
            _FakeScenario(),
            _FakeSpec(),
            _SimWithParseFailure(),
            tc,
            ts,
            efficacy=0.5,
            cost_usd=0.001,
            state_result=state_result,
        )
        assert row["state_gradable"] is False
        assert row["state_score"] is None
        assert row["state_checks_passed"] is None
        assert row["state_checks_total"] is None
        assert row["tool_sim_parse_failures"] == 1

    def test_state_nongradable_when_llm_authored_mutation(self):
        # #87 phase 3: a stateful scenario whose graded world was mutated by the
        # LLM tool-sim fallback (an unregistered tool) is NOT gradable — the spine
        # trusts only coded mutations. State columns null, count surfaced.
        from scripts.run_eval import build_result_row

        @dataclass
        class _SimWithLLMMutation:
            total_latency_ms: float = 1234.5
            total_turns: int = 4
            total_input_tokens: int = 100
            total_output_tokens: int = 50
            completed: bool = True
            tool_sim_parse_failures: int = 0
            llm_tool_sim_mutations: int = 1

        tc = _consensus(judge_results=[_valid("Kimi", 0.8)], consensus_score=0.8, n_judges_valid=1)
        ts = _consensus(rubric_type="tool_selection", n_judges_valid=1)
        state_result = {
            "score": 1.0,
            "checks": [{"passed": True, "detail": "x"}],
            "n_passed": 1,
            "n_total": 1,
        }
        row = build_result_row(
            _FakeScenario(),
            _FakeSpec(),
            _SimWithLLMMutation(),
            tc,
            ts,
            efficacy=0.5,
            cost_usd=0.001,
            state_result=state_result,
        )
        assert row["state_gradable"] is False
        assert row["state_score"] is None
        assert row["llm_tool_sim_mutations"] == 1
        # A clean parse-failure count must NOT be what nulled it.
        assert row["tool_sim_parse_failures"] == 0

    def test_state_gradable_when_llm_fallback_was_read_only(self):
        # #87 phase 3: a read-only LLM fallback (no mutation) does NOT taint the
        # world — llm_tool_sim_mutations stays 0 and the state grade is trusted.
        from scripts.run_eval import build_result_row

        @dataclass
        class _SimReadOnlyFallback:
            total_latency_ms: float = 1.0
            total_turns: int = 2
            total_input_tokens: int = 10
            total_output_tokens: int = 5
            completed: bool = True
            tool_sim_parse_failures: int = 0
            llm_tool_sim_calls: int = 1
            llm_tool_sim_mutations: int = 0

        tc = _consensus(judge_results=[_valid("Kimi", 0.8)], consensus_score=0.8, n_judges_valid=1)
        ts = _consensus(rubric_type="tool_selection", n_judges_valid=1)
        state_result = {
            "score": 1.0,
            "checks": [{"passed": True, "detail": "x"}],
            "n_passed": 1,
            "n_total": 1,
        }
        row = build_result_row(
            _FakeScenario(),
            _FakeSpec(),
            _SimReadOnlyFallback(),
            tc,
            ts,
            efficacy=0.5,
            cost_usd=0.001,
            state_result=state_result,
        )
        assert row["state_gradable"] is True
        assert row["state_score"] == 1.0
        assert row["llm_tool_sim_mutations"] == 0

    def test_stateless_scenario_gradable_with_parse_failure(self):
        # S3: a parse failure on a STATELESS scenario (no state_result) leaves
        # nothing to null — state_gradable stays True, state columns stay None.
        from scripts.run_eval import build_result_row

        @dataclass
        class _SimWithParseFailure:
            total_latency_ms: float = 1.0
            total_turns: int = 1
            total_input_tokens: int = 1
            total_output_tokens: int = 1
            completed: bool = True
            tool_sim_parse_failures: int = 2

        tc = _consensus(judge_results=[_valid("Kimi", 0.8)], consensus_score=0.8, n_judges_valid=1)
        ts = _consensus(rubric_type="tool_selection", n_judges_valid=1)
        row = build_result_row(
            _FakeScenario(),
            _FakeSpec(),
            _SimWithParseFailure(),
            tc,
            ts,
            efficacy=0.5,
            cost_usd=0.001,
            state_result=None,
        )
        assert row["state_gradable"] is True
        assert row["state_score"] is None
        assert row["tool_sim_parse_failures"] == 2
