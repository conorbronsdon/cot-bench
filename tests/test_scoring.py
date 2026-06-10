"""Tests for scoring logic — rubrics and reliability computation."""

import pytest

from eval.scoring.rubrics import (
    EFFICACY_WEIGHTS,
    compute_efficacy,
    compute_pass_hat_k,
    compute_reliability,
)


class TestComputeReliability:
    def test_all_passing(self):
        result = compute_reliability([0.9, 0.85, 0.8])
        assert result["pass_rate"] == 1.0
        # consistency = 1.0 - (0.9 - 0.8) = 0.9
        assert result["consistency"] == pytest.approx(0.9)

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

    def test_consistency_clamped_at_zero(self):
        # Scores outside [0, 1] shouldn't produce negative consistency
        result = compute_reliability([0.0, 1.0, 0.0])
        assert result["consistency"] == 0.0
        assert result["consistency"] >= 0.0


class TestComputePassHatK:
    def test_all_pass_is_one_at_every_k(self):
        # 3 of 3 pass -> pass^k = 1.0 for every k.
        out = compute_pass_hat_k(3, 3)
        assert out == {1: 1.0, 2: 1.0, 3: 1.0}

    def test_two_of_three_decays(self):
        # c=2, n=3: pass^1 = 2/3, pass^2 = C(2,2)/C(3,2) = 1/3, pass^3 = 0/1 = 0.
        out = compute_pass_hat_k(2, 3)
        assert out[1] == pytest.approx(2 / 3)
        assert out[2] == pytest.approx(1 / 3)
        assert out[3] == pytest.approx(0.0)

    def test_one_of_three(self):
        # c=1, n=3: pass^1 = 1/3, pass^2 = C(1,2)/C(3,2) = 0, pass^3 = 0.
        out = compute_pass_hat_k(1, 3)
        assert out[1] == pytest.approx(1 / 3)
        assert out[2] == pytest.approx(0.0)
        assert out[3] == pytest.approx(0.0)

    def test_pass_hat_1_equals_pass_rate(self):
        out = compute_pass_hat_k(2, 5)
        assert out[1] == pytest.approx(2 / 5)

    def test_zero_runs_empty(self):
        assert compute_pass_hat_k(0, 0) == {}

    def test_no_passes(self):
        out = compute_pass_hat_k(0, 3)
        assert out == {1: 0.0, 2: 0.0, 3: 0.0}


class TestReliabilityPassHatK:
    def test_all_passing_pass_hat_k(self):
        result = compute_reliability([0.9, 0.85, 0.8])
        assert result["pass_hat_k"] == {1: 1.0, 2: 1.0, 3: 1.0}

    def test_mixed_pass_hat_k_decays(self):
        # 2 of 3 pass at threshold 0.7.
        result = compute_reliability([0.9, 0.5, 0.8])
        assert result["pass_hat_k"][1] == pytest.approx(2 / 3)
        assert result["pass_hat_k"][3] == pytest.approx(0.0)

    def test_empty_pass_hat_k(self):
        assert compute_reliability([])["pass_hat_k"] == {}


class TestEfficacyWeights:
    def test_weights_sum_to_one(self):
        assert sum(EFFICACY_WEIGHTS.values()) == pytest.approx(1.0)

    def test_has_required_components(self):
        assert "task_completion" in EFFICACY_WEIGHTS
        assert "tool_selection" in EFFICACY_WEIGHTS
        assert "state_verification" in EFFICACY_WEIGHTS


class TestComputeEfficacy:
    def test_with_state_uses_hybrid_weights(self):
        # 0.4*tc + 0.3*ts + 0.3*state
        eff = compute_efficacy(1.0, 1.0, 1.0)
        assert eff == pytest.approx(1.0)
        eff = compute_efficacy(0.8, 0.6, 0.5)
        assert eff == pytest.approx(0.4 * 0.8 + 0.3 * 0.6 + 0.3 * 0.5)

    def test_state_zero_pulls_score_down(self):
        eff = compute_efficacy(1.0, 1.0, 0.0)
        assert eff == pytest.approx(0.7)  # 0.4 + 0.3 + 0.0

    def test_without_state_renormalizes_to_half_half(self):
        eff = compute_efficacy(0.8, 0.6, None)
        assert eff == pytest.approx(0.5 * 0.8 + 0.5 * 0.6)

    def test_without_state_full_scores(self):
        assert compute_efficacy(1.0, 1.0, None) == pytest.approx(1.0)
        assert compute_efficacy(0.0, 0.0, None) == pytest.approx(0.0)
