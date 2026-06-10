"""Inter-judge reliability metrics for COT Bench.

Krippendorff's alpha is the field-standard, chance-corrected agreement metric
for 3+ raters on ordinal/interval labels (Cohen's kappa is only for 2 raters and
assumes nominal categories; a raw "within-0.2 rate" is neither chance-corrected
nor comparable across score distributions). See docs/methodology.md §3.

We implement alpha in-repo (rather than adding the ``krippendorff`` PyPI
dependency) because it is a small, well-specified function and the project keeps
its dependency surface deliberately thin. The implementation is validated in
tests against worked examples from Krippendorff's own reference material.

Alpha is computed from the reliability-data matrix: rows are *units* (here, one
graded scenario-run-dimension), columns are *raters* (the judges). Missing
values (a judge that parse-failed on a unit) are simply absent — alpha is
defined for incomplete data and only uses units with 2+ present values.

Definition (Krippendorff 2011, "Computing Krippendorff's Alpha-Reliability"):

    alpha = 1 - (D_o / D_e)

with observed and expected disagreement computed from the coincidence matrix.
The equivalent, numerically convenient form used here works directly from the
pairable values:

    D_o = (1 / n) * sum over units u of
              [ 1 / (m_u - 1) * sum over value pairs (c, k) in u of delta(c, k) ]
    D_e = (1 / (n * (n - 1))) * sum over all value pairs (c, k) in the
              whole dataset of delta(c, k)

where ``n`` is the total number of pairable values (values in units with m_u>=2),
``m_u`` is the count of present values in unit u, and ``delta`` is the squared
difference metric for the ordinal/interval level used here. This is algebraically
identical to the coincidence-matrix formulation and avoids materializing it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _interval_metric(a: float, b: float) -> float:
    """Squared-difference distance, the interval/ordinal metric for numeric scores.

    For the continuous 0-1 rubric scores cot-bench produces, the interval metric
    (squared difference) is the appropriate ordinal/interval distance: it treats
    the scores as a ratio scale where the *size* of a disagreement matters, not
    just whether two raters picked the same bucket.
    """
    d = a - b
    return d * d


def krippendorff_alpha(
    reliability_data: Sequence[Sequence[float | None]],
) -> float | None:
    """Krippendorff's alpha (interval/ordinal level) over a units x raters matrix.

    Args:
        reliability_data: One row per *unit* (scenario-run-dimension), one column
            per *rater* (judge). Use ``None`` (or ``float('nan')``) for a missing
            value (e.g. a judge that parse-failed on that unit). Rows may be
            ragged in effect — missing cells are ignored.

    Returns:
        Alpha in (-inf, 1]. Returns ``None`` when alpha is undefined: fewer than
        2 pairable values exist, or there is no variation in the data at all
        (expected disagreement 0). By convention, when observed disagreement is
        also 0 (perfect agreement on a constant) alpha is 1.0; we return ``None``
        only when expected disagreement is 0 *and* observed disagreement is 0,
        which carries no reliability information.
    """
    # Collect, per unit, the list of present (non-missing) values.
    units: list[list[float]] = []
    for row in reliability_data:
        present = [float(v) for v in row if v is not None and not _is_nan(v)]
        if len(present) >= 2:
            units.append(present)

    # Total pairable values across all units with m_u >= 2.
    n = sum(len(u) for u in units)
    if n < 2:
        return None

    # Observed disagreement.
    d_o = 0.0
    for u in units:
        m_u = len(u)
        pair_sum = 0.0
        for i in range(m_u):
            for j in range(m_u):
                if i != j:
                    pair_sum += _interval_metric(u[i], u[j])
        d_o += pair_sum / (m_u - 1)
    d_o /= n

    # Expected disagreement: all ordered value pairs across the whole dataset.
    all_values = [v for u in units for v in u]
    d_e = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                d_e += _interval_metric(all_values[i], all_values[j])
    d_e /= n * (n - 1)

    if d_e == 0:
        # No variation in the data. Perfect agreement on a constant carries no
        # reliability signal, so alpha is undefined.
        return None

    return 1.0 - (d_o / d_e)


def _is_nan(value: float | None) -> bool:
    """True for float NaN; False for None (handled by the caller) and real numbers."""
    return isinstance(value, float) and math.isnan(value)
