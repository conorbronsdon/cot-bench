"""Tests for judge scoring utilities."""

from dataclasses import dataclass, field

import pytest

import eval.scoring.judge as judge_mod
from eval.scoring.judge import (
    ConsensusResult,
    JudgeResult,
    _parse_judge_response,
    score_with_all_judges,
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
        assert res.consensus_score == pytest.approx((0.8 + 0.9 + 0.7) / 3)
        assert res.parse_failures == []
        assert res.api_failures == []
        assert res.degraded is False
        # 3 pairs: |0.8-0.9|=0.1, |0.8-0.7|=0.1 agree; |0.9-0.7| is 0.2 in
        # float terms (0.2000...07) so it falls just outside the <=0.2 band.
        # 2 of 3 pairs agree (preserving the existing agreement logic).
        assert res.agreement_rate == pytest.approx(2 / 3)
        assert res.max_disagreement == pytest.approx(0.2)

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
                return "not json"
            return '{"overall_score": 0.75, "overall_reasoning": "second try"}'

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
            return "still not json"

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
