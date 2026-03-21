"""Tests for scoring logic — rubrics and reliability computation."""

import pytest

from eval.scoring.rubrics import EFFICACY_WEIGHTS, compute_reliability


class TestComputeReliability:
    def test_all_passing(self):
        result = compute_reliability([0.9, 0.85, 0.8])
        assert result["pass_rate"] == 1.0
        assert result["consistency"] > 0.8

    def test_all_failing(self):
        result = compute_reliability([0.3, 0.2, 0.4])
        assert result["pass_rate"] == 0.0

    def test_mixed_results(self):
        result = compute_reliability([0.9, 0.5, 0.8])
        assert result["pass_rate"] == pytest.approx(2 / 3)

    def test_perfect_consistency(self):
        result = compute_reliability([0.8, 0.8, 0.8])
        assert result["consistency"] == 1.0
        assert result["score_variance"] == pytest.approx(0.0)

    def test_empty_input(self):
        result = compute_reliability([])
        assert result["pass_rate"] == 0.0
        assert result["consistency"] == 0.0

    def test_single_run(self):
        result = compute_reliability([0.85])
        assert result["pass_rate"] == 1.0
        assert result["consistency"] == 1.0

    def test_custom_threshold(self):
        result = compute_reliability([0.6, 0.5, 0.4], threshold=0.5)
        assert result["pass_rate"] == pytest.approx(2 / 3)


class TestEfficacyWeights:
    def test_weights_sum_to_one(self):
        assert sum(EFFICACY_WEIGHTS.values()) == pytest.approx(1.0)

    def test_has_required_components(self):
        assert "task_completion" in EFFICACY_WEIGHTS
        assert "tool_selection" in EFFICACY_WEIGHTS
