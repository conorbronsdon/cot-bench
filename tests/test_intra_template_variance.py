"""Intra-template score variance, the GSM-Symbolic memorization signal (issue #90).

Pins the pure measurement primitive: spread of efficacy across DISTINCT instances
(by instantiation_seed) of the same template (by scenario_id), per model. It is
empty until the frame carries >= 2 instances per template (the single-seed
reality today), and never leaks holdout detail.
"""

import math

import pandas as pd

from scripts.aggregate_results import compute_intra_template_variance


def _rows(records):
    return pd.DataFrame(records)


def test_empty_frame_returns_empty():
    assert compute_intra_template_variance(_rows([])) == {}


def test_missing_columns_returns_empty():
    df = _rows([{"model": "M", "efficacy": 0.5, "scenario_id": "t1"}])  # no instantiation_seed
    assert compute_intra_template_variance(df) == {}


def test_single_instance_per_template_is_no_signal():
    """One seed per template (today's reality) -> nothing to measure -> {}."""
    df = _rows(
        [
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 7, "efficacy": 0.9},
            {"model": "M", "scenario_id": "t2", "instantiation_seed": 7, "efficacy": 0.4},
        ]
    )
    assert compute_intra_template_variance(df) == {}


def test_two_identical_instances_have_zero_std():
    df = _rows(
        [
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 1, "efficacy": 0.8},
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 2, "efficacy": 0.8},
        ]
    )
    out = compute_intra_template_variance(df)
    assert out["M"]["intra_template_std"] == 0.0
    assert out["M"]["n_templates_measured"] == 1


def test_divergent_instances_produce_positive_std():
    """A model whose score swings as the slots change -> high intra-template std."""
    df = _rows(
        [
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 1, "efficacy": 1.0},
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 2, "efficacy": 0.0},
        ]
    )
    out = compute_intra_template_variance(df)
    # population std of {1.0, 0.0} = 0.5
    assert math.isclose(out["M"]["intra_template_std"], 0.5, abs_tol=1e-9)


def test_per_instance_means_collapse_reliability_repeats():
    """Multiple rows at the SAME seed (reliability repeats) average into one instance.

    Variance must be across DIFFERENT instances, not run-to-run noise within one.
    """
    df = _rows(
        [
            # instance seed=1 repeated twice -> mean 0.9
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 1, "efficacy": 1.0},
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 1, "efficacy": 0.8},
            # instance seed=2 repeated twice -> mean 0.3
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 2, "efficacy": 0.4},
            {"model": "M", "scenario_id": "t1", "instantiation_seed": 2, "efficacy": 0.2},
        ]
    )
    out = compute_intra_template_variance(df)
    # std of instance means {0.9, 0.3} = 0.3
    assert math.isclose(out["M"]["intra_template_std"], 0.3, abs_tol=1e-9)


def test_holdout_rows_excluded():
    df = _rows(
        [
            {
                "model": "M",
                "scenario_id": "t1",
                "instantiation_seed": 1,
                "efficacy": 1.0,
                "holdout": False,
            },
            {
                "model": "M",
                "scenario_id": "t1",
                "instantiation_seed": 2,
                "efficacy": 0.0,
                "holdout": False,
            },
            # holdout instances must not contribute
            {
                "model": "M",
                "scenario_id": "t9",
                "instantiation_seed": 1,
                "efficacy": 1.0,
                "holdout": True,
            },
            {
                "model": "M",
                "scenario_id": "t9",
                "instantiation_seed": 2,
                "efficacy": 1.0,
                "holdout": True,
            },
        ]
    )
    out = compute_intra_template_variance(df)
    assert out["M"]["n_templates_measured"] == 1  # only the public template t1


def test_per_model_isolation():
    df = _rows(
        [
            {"model": "A", "scenario_id": "t1", "instantiation_seed": 1, "efficacy": 1.0},
            {"model": "A", "scenario_id": "t1", "instantiation_seed": 2, "efficacy": 1.0},
            {"model": "B", "scenario_id": "t1", "instantiation_seed": 1, "efficacy": 1.0},
            {"model": "B", "scenario_id": "t1", "instantiation_seed": 2, "efficacy": 0.0},
        ]
    )
    out = compute_intra_template_variance(df)
    assert out["A"]["intra_template_std"] == 0.0
    assert math.isclose(out["B"]["intra_template_std"], 0.5, abs_tol=1e-9)
