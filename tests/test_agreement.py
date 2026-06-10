"""Tests for Krippendorff's alpha (inter-judge reliability)."""

import math

import pytest

from eval.scoring.agreement import krippendorff_alpha


class TestKrippendorffAlphaReference:
    def test_interval_reference_example(self):
        """Validate against a worked interval example (coincidence-matrix method).

        The raters x units matrix below is the standard Krippendorff reliability-
        data example with missing values. Computing alpha at the interval level
        via the textbook coincidence-matrix method yields 0.8491071..., which our
        closed-form implementation must reproduce exactly.
        """
        nan = float("nan")
        raters = [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
            [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
        ]
        n_units = len(raters[0])
        units = [[r[u] for r in raters] for u in range(n_units)]
        alpha = krippendorff_alpha(units)
        assert alpha == pytest.approx(0.8491071428571429, abs=1e-9)

    def test_matches_independent_coincidence_matrix(self):
        """Cross-check the closed form against a direct coincidence-matrix build."""
        nan = float("nan")
        raters = [
            [1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
            [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
            [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
            [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan],
        ]
        n_units = len(raters[0])
        units_full = [[r[u] for r in raters] for u in range(n_units)]

        # Reference computation via coincidence matrix.
        units = [
            [v for v in col if not (isinstance(v, float) and math.isnan(v))] for col in units_full
        ]
        units = [u for u in units if len(u) >= 2]
        coincidence: dict[tuple, float] = {}
        for u in units:
            m = len(u)
            for i in range(m):
                for j in range(m):
                    if i != j:
                        coincidence[(u[i], u[j])] = coincidence.get((u[i], u[j]), 0.0) + 1.0 / (
                            m - 1
                        )
        n = sum(coincidence.values())
        nc: dict = {}
        for (c, _k), val in coincidence.items():
            nc[c] = nc.get(c, 0.0) + val
        vals = sorted(set(v for u in units for v in u))

        def delta(a, b):
            return (a - b) ** 2

        d_o = sum(coincidence.get((c, k), 0.0) * delta(c, k) for c in vals for k in vals) / n
        d_e = sum(nc[c] * nc[k] * delta(c, k) for c in vals for k in vals) / (n * (n - 1))
        expected = 1.0 - d_o / d_e

        assert krippendorff_alpha(units_full) == pytest.approx(expected, abs=1e-12)


class TestKrippendorffAlphaBehavior:
    def test_perfect_agreement_with_variation_is_one(self):
        # Judges agree exactly, and there is variation across units -> alpha 1.0.
        units = [[0.8, 0.8, 0.8], [0.5, 0.5, 0.5], [0.2, 0.2, 0.2]]
        assert krippendorff_alpha(units) == pytest.approx(1.0)

    def test_systematic_disagreement_is_negative(self):
        # Raters consistently flip -> worse than chance -> negative alpha.
        units = [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        assert krippendorff_alpha(units) < 0

    def test_three_judge_partial_agreement(self):
        # median-style panels: two units, tight clusters -> high but <1 alpha.
        units = [[0.8, 0.9, 0.7], [0.5, 0.6, 0.4]]
        assert krippendorff_alpha(units) == pytest.approx(0.7142857142857143, abs=1e-9)

    def test_constant_data_is_undefined(self):
        # No variation at all -> expected disagreement 0 -> alpha undefined.
        assert krippendorff_alpha([[0.5, 0.5, 0.5]]) is None

    def test_too_few_pairable_values_is_none(self):
        # A unit with a single present value is unpairable -> dropped; nothing left.
        assert krippendorff_alpha([[0.5, None, None]]) is None

    def test_missing_values_handled(self):
        # None and NaN are both treated as missing; units with 2+ values count.
        nan = float("nan")
        units = [[0.8, 0.9, None], [0.5, nan, 0.6], [0.2, 0.3, 0.25]]
        alpha = krippendorff_alpha(units)
        assert alpha is not None
        assert -1.0 <= alpha <= 1.0

    def test_none_when_fewer_than_two_raters(self):
        # A single rater column can never produce pairable values.
        assert krippendorff_alpha([[0.5], [0.6], [0.7]]) is None
