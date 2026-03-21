"""Tests for judge scoring utilities."""

from eval.scoring.judge import _parse_judge_response


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

    def test_unparseable_returns_zero(self):
        result = _parse_judge_response("This is not JSON at all.")
        assert result["overall_score"] == 0.0

    def test_nested_json(self):
        content = (
            '{"goal_scores": [{"goal": "a", "score": "COMPLETE"}], '
            '"overall_score": 0.9, "overall_reasoning": "All good"}'
        )
        result = _parse_judge_response(content)
        assert result["overall_score"] == 0.9
        assert "goal_scores" in result
