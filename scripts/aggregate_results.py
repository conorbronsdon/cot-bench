"""Aggregate evaluation results into leaderboard format.

Reads parquet result files and produces:
- leaderboard.json: structured leaderboard data for the frontend
- latest.csv: human-readable summary
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from eval.config import MIN_SCENARIOS_FOR_PUBLISH
from eval.providers.null_agent import NULL_AGENT_NAME
from eval.scoring.agreement import krippendorff_alpha
from eval.scoring.failure_modes import FAILURE_MODES
from eval.scoring.rubrics import PASS_THRESHOLD
from eval.simulation.profiles import DEFAULT_SIM_PROFILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")


def exclude_non_contestants(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that must never appear on the published leaderboard.

    The deterministic do-nothing ``null-agent`` is an anti-gaming *validation*
    contestant (it is run on request to confirm the bench scores a do-nothing
    agent near zero), not a ranked model. It is excluded here — at the single
    aggregation entry point — so it can never leak onto the leaderboard, history,
    CSV, or bootstrap/rank-band math regardless of how the run was invoked. The
    match is case-insensitive on the model name.
    """
    if df.empty or "model" not in df.columns:
        return df
    keep = df["model"].astype(str).str.lower() != NULL_AGENT_NAME.lower()
    dropped = int((~keep).sum())
    if dropped:
        logger.info(
            "Excluding %d row(s) for non-contestant '%s' from the leaderboard.",
            dropped,
            NULL_AGENT_NAME,
        )
    return df[keep]


def _cooperative_mask(df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows produced under the cooperative (default) sim profile.

    A missing ``sim_profile`` column or a null cell means the row predates the
    profile feature (issue #59) — those runs were all cooperative by
    construction, so they count as cooperative rather than being dropped.
    """
    if "sim_profile" not in df.columns:
        return pd.Series(True, index=df.index)
    return df["sim_profile"].fillna(DEFAULT_SIM_PROFILE).astype(str) == DEFAULT_SIM_PROFILE


def exclude_non_cooperative_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows produced under a non-cooperative user-sim profile (issue #59).

    The public leaderboard measures every model against the SAME simulated user:
    the cooperative default. Rows from the behavioral profiles (impatient /
    technically-confused / adversarial) are a different, deliberately harder
    condition — mixing them into the public aggregates would silently deflate
    whichever models happened to be run under them. Like the null-agent and
    holdout exclusions, this is enforced here at the single aggregation entry
    point so a stratified run can never leak into public efficacy regardless of
    how it was invoked. Non-cooperative rows are reported separately via
    :func:`compute_sim_profile_pass_rates` (the robustness table).
    """
    if df.empty:
        return df
    keep = _cooperative_mask(df)
    dropped = int((~keep).sum())
    if dropped:
        profiles = sorted(df.loc[~keep, "sim_profile"].astype(str).unique())
        logger.info(
            "Excluding %d row(s) from non-cooperative sim profile(s) %s from the "
            "public leaderboard (issue #59; see the persona-stratified table).",
            dropped,
            profiles,
        )
    return df[keep]


def compute_sim_profile_pass_rates(df: pd.DataFrame, threshold: float = PASS_THRESHOLD) -> dict:
    """Per-profile pass rates per model — the persona-stratified robustness table.

    Given result rows spanning one or more user-sim profiles (issue #59),
    returns::

        {model: {profile: {"pass_rate": .., "mean_efficacy": .., "n_rows": ..,
                           "n_scenarios": ..,
                           "delta_vs_cooperative": ..}}}

    A row "passes" when its efficacy reaches ``threshold`` — the same pass
    definition reliability uses (``eval.scoring.rubrics.PASS_THRESHOLD``), so
    "pass rate" means the same thing here as on the leaderboard.
    ``delta_vs_cooperative`` is ``cooperative_pass_rate - profile_pass_rate``
    (positive = the model does worse under that behavioral profile — the
    inflation the cooperative-only literature warns about); it is ``None`` on the
    cooperative entry itself and when the model has no cooperative rows. Rows
    with no ``sim_profile`` (legacy parquets) count as cooperative. Deterministic:
    plain groupby means over the input rows, no resampling.

    This is a reporting helper for the published robustness table; it is NOT part
    of ``compute_leaderboard`` — non-cooperative rows never reach the public
    aggregates (see :func:`exclude_non_cooperative_profiles`).
    """
    if df.empty or "model" not in df.columns or "efficacy" not in df.columns:
        return {}
    work = df.copy()
    if "sim_profile" in work.columns:
        work["sim_profile"] = work["sim_profile"].fillna(DEFAULT_SIM_PROFILE).astype(str)
    else:
        work["sim_profile"] = DEFAULT_SIM_PROFILE
    work["_passed"] = work["efficacy"].astype(float) >= threshold

    out: dict[str, dict] = {}
    for (model, profile), grp in sorted(
        work.groupby(["model", "sim_profile"]), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))
    ):
        out.setdefault(str(model), {})[str(profile)] = {
            "pass_rate": round(float(grp["_passed"].mean()), 4),
            "mean_efficacy": round(float(grp["efficacy"].mean()), 4),
            "n_rows": int(len(grp)),
            "n_scenarios": int(grp["scenario_id"].nunique()) if "scenario_id" in grp else None,
        }

    for profiles in out.values():
        coop = profiles.get(DEFAULT_SIM_PROFILE)
        for profile, entry in profiles.items():
            if profile == DEFAULT_SIM_PROFILE or coop is None:
                entry["delta_vs_cooperative"] = None
            else:
                entry["delta_vs_cooperative"] = round(coop["pass_rate"] - entry["pass_rate"], 4)
    return out


def compute_recovery_rates(df: pd.DataFrame) -> dict:
    """Per-model recovery rate over probe-carrying rows only (issue #57).

    A recovery probe injects a deterministic mid-conversation fault and grades —
    via state checking — whether the agent reached the correct end state DESPITE
    it. ``recovered`` is the boolean verdict on each probe row (None on non-probe
    rows). This computes, per model::

        {model: {"recovery_rate": .., "n_probe_rows": .., "n_probe_scenarios": ..,
                 "by_kind": {kind: {"recovery_rate": .., "n_rows": ..}}}}

    over ONLY the rows where a probe ran (``recovered`` is non-null). Returns
    ``{}`` when no probe rows exist — which is every run on the v1 corpus, since
    no published scenario carries a probe (demo probes live in test fixtures
    only). Deterministic: plain group means, no resampling. This is a reporting
    helper, NOT part of the public efficacy/CLEAR aggregates — probe rows are a
    separate dimension, exactly like the persona-stratified robustness table.
    """
    if df.empty or "recovered" not in df.columns or "model" not in df.columns:
        return {}
    # Probe rows are the ones with a non-null ``recovered`` verdict. A null/absent
    # column means no probes ran — return empty so nothing is emitted.
    work = df[df["recovered"].notna()].copy()
    if work.empty:
        return {}
    work["_recovered"] = work["recovered"].astype(bool)

    out: dict[str, dict] = {}
    for model, grp in sorted(work.groupby("model"), key=lambda kv: str(kv[0])):
        by_kind: dict[str, dict] = {}
        if "recovery_probe_kind" in grp.columns:
            for kind, kgrp in sorted(grp.groupby("recovery_probe_kind"), key=lambda kv: str(kv[0])):
                by_kind[str(kind)] = {
                    "recovery_rate": round(float(kgrp["_recovered"].mean()), 4),
                    "n_rows": int(len(kgrp)),
                }
        out[str(model)] = {
            "recovery_rate": round(float(grp["_recovered"].mean()), 4),
            "n_probe_rows": int(len(grp)),
            "n_probe_scenarios": (
                int(grp["scenario_id"].nunique()) if "scenario_id" in grp else None
            ),
            "by_kind": by_kind,
        }
    return out


# --- Bootstrap configuration ---
# Confidence intervals are estimated by resampling SCENARIOS (not individual
# rows) with replacement. The scenario is the exchangeable unit here: all runs
# of a scenario share the same task and persona, so their scores are correlated
# and must move together under resampling — resampling rows would understate
# uncertainty by treating correlated repeats as independent draws. See the
# "Uncertainty" subsection of docs/methodology.md.
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 42  # fixed for reproducible CIs across runs
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5

# CLEAR composite weights (kept in one place so the point estimate and each
# bootstrap replicate normalize + weight identically).
CLEAR_WEIGHTS = {
    "efficacy": 0.35,
    "reliability": 0.25,
    "cost_per_task": 0.20,  # inverted (lower is better)
    "avg_latency_ms": 0.20,  # inverted (lower is better)
}


def _min_max_norm(values: np.ndarray, invert: bool = False) -> np.ndarray:
    """Min-max normalize a 1-D array across models; degenerate range -> 0.5.

    Guards division by zero: when every model has the same value (or there is a
    single model) the range is 0 and we return 0.5 for all, matching the
    point-estimate path. ``invert`` flips the scale for lower-is-better dims.
    """
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    rng = vmax - vmin
    if rng <= 0:
        return np.full(values.shape, 0.5)
    norm = (values - vmin) / rng
    if invert:
        norm = 1.0 - norm
    return norm


def _clear_from_means(
    efficacy: np.ndarray,
    reliability: np.ndarray,
    cost: np.ndarray,
    latency: np.ndarray,
) -> np.ndarray:
    """Field-relative CLEAR composite from per-model mean dimensions.

    Normalization is across the supplied set of models, so the composite (and
    therefore its bootstrap uncertainty) reflects normalization variance, not
    just the variance of each raw dimension. Used for both the point estimate
    and every bootstrap replicate so they are computed by exactly one code path.
    """
    eff_n = _min_max_norm(efficacy)
    rel_n = _min_max_norm(reliability)
    cost_n = _min_max_norm(cost, invert=True)
    lat_n = _min_max_norm(latency, invert=True)
    return (
        eff_n * CLEAR_WEIGHTS["efficacy"]
        + rel_n * CLEAR_WEIGHTS["reliability"]
        + cost_n * CLEAR_WEIGHTS["cost_per_task"]
        + lat_n * CLEAR_WEIGHTS["avg_latency_ms"]
    )


def compute_bootstrap_cis(df: pd.DataFrame, models: list[str]) -> dict:
    """Paired scenario-bootstrap CIs for per-model efficacy and CLEAR score.

    Returns ``{model: {"efficacy_ci": [lo, hi], "clear_score_ci": [lo, hi]}}``.

    A single set of resampled scenario ids is drawn per replicate and applied to
    every model (a *paired* bootstrap), so the field — and thus the min-max
    normalization underlying CLEAR — is consistent within each replicate. Edge
    cases: a single scenario yields a degenerate (zero-width) CI equal to the
    point estimate rather than crashing; a single model skips CLEAR normalization
    (its CLEAR CI mirrors efficacy, matching compute_leaderboard's 1-model path).
    """
    scenario_ids = df["scenario_id"].unique()
    n_scenarios = len(scenario_ids)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Pre-split rows by (model, scenario) so each replicate is cheap index math.
    # mean_by[model] maps scenario_id -> per-dimension means over that scenario's
    # rows (all runs); resampling then just averages the selected scenarios.
    dims = ["efficacy", "reliability_pass_rate", "cost_usd", "latency_ms"]
    mean_by: dict[str, pd.DataFrame] = {}
    for model in models:
        mdf = df[df["model"] == model]
        # mean per dimension within each scenario (collapses runs)
        per_scenario = mdf.groupby("scenario_id")[dims].mean()
        # reindex to the full scenario universe so every model aligns by id;
        # missing scenarios for a model become NaN and are nan-averaged out.
        mean_by[model] = per_scenario.reindex(scenario_ids)

    single_model = len(models) < 2

    eff_samples: dict[str, list[float]] = {m: [] for m in models}
    clear_samples: dict[str, list[float]] = {m: [] for m in models}

    for _ in range(BOOTSTRAP_REPLICATES):
        # Resample scenario POSITIONS with replacement; same draw for all models.
        idx = rng.integers(0, n_scenarios, size=n_scenarios)

        eff_means = np.empty(len(models))
        rel_means = np.empty(len(models))
        cost_means = np.empty(len(models))
        lat_means = np.empty(len(models))

        for j, model in enumerate(models):
            sampled = mean_by[model].to_numpy()[idx]  # (n_scenarios, 4)
            # nanmean: a model missing some scenarios still aggregates cleanly.
            with np.errstate(invalid="ignore"):
                col_means = np.nanmean(sampled, axis=0)
            eff_means[j], rel_means[j], cost_means[j], lat_means[j] = col_means
            eff_samples[model].append(float(col_means[0]))

        if single_model:
            # No cross-model normalization possible; CLEAR == efficacy.
            clear_samples[models[0]].append(eff_means[0])
        else:
            clear_vals = _clear_from_means(eff_means, rel_means, cost_means, lat_means)
            for j, model in enumerate(models):
                clear_samples[model].append(float(clear_vals[j]))

    out = {}
    for model in models:
        eff_arr = np.asarray(eff_samples[model], dtype=float)
        clr_arr = np.asarray(clear_samples[model], dtype=float)
        out[model] = {
            "efficacy_ci": _percentile_ci(eff_arr),
            "clear_score_ci": _percentile_ci(clr_arr),
        }
    return out


def _percentile_ci(samples: np.ndarray) -> list:
    """2.5/97.5 percentile CI; nan-safe, returns [lo, hi] rounded to 4 dp.

    With one scenario every replicate is identical, so the percentiles collapse
    to the point estimate (zero-width CI) rather than producing spurious spread.
    """
    clean = samples[~np.isnan(samples)]
    if clean.size == 0:
        return [None, None]
    lo = float(np.percentile(clean, CI_LOW_PCT))
    hi = float(np.percentile(clean, CI_HIGH_PCT))
    return [round(lo, 4), round(hi, 4)]


def compute_macro_efficacy(df: pd.DataFrame, group_col: str) -> dict:
    """Per-model MACRO-averaged efficacy over ``group_col`` (issue #55).

    Macro = the unweighted mean over groups of each group's mean efficacy, so
    every category (or domain) counts equally regardless of how many scenarios
    it has — frequent-easy scenario clusters cannot drown rare-hard ones. The
    existing headline ``efficacy`` is the MICRO average (mean over all rows);
    both are published. Mirrors the micro convention: the group mean is the mean
    over that group's rows, and groups a model has no rows in are simply absent
    from its macro mean. Returns ``{model: macro}``; empty when the grouping
    column is missing (legacy parquets without a ``category`` column).
    """
    if df.empty or group_col not in df.columns:
        return {}
    per_group = df.groupby(["model", group_col])["efficacy"].mean()
    macro = per_group.groupby(level="model").mean()
    return {str(m): round(float(v), 4) for m, v in macro.items()}


def compute_macro_bootstrap_cis(df: pd.DataFrame, models: list[str], group_col: str) -> dict:
    """Stratified paired bootstrap CIs for macro-averaged efficacy (issue #55).

    The micro bootstrap (``compute_bootstrap_cis``) resamples scenarios
    UNIFORMLY, which is the right uncertainty for a mean over scenarios — but
    the macro statistic is a mean of per-GROUP means, so its resampling must
    preserve the group structure. Each replicate resamples scenarios WITHIN each
    group (with replacement, group size preserved), recomputes every group's
    mean, and averages the group means — a bootstrap of category means. As with
    the micro CIs, the same within-group draw is applied to every model (paired),
    runs are collapsed to scenario means first (runs of a scenario are
    correlated), B and the seed match the micro bootstrap (B=2000, seed 42), and
    a fresh seeded generator keeps the micro CIs byte-identical to before.

    Edge cases: a single-scenario group always resamples to itself (it
    contributes its own mean with zero spread — correct, not a crash); a model
    with no rows in a group nan-skips that group, matching the point estimate.
    Returns ``{model: [lo, hi]}``; empty when ``group_col`` is missing.
    """
    if df.empty or group_col not in df.columns or not models:
        return {}
    col = df[group_col].astype(str)
    groups = sorted(col.dropna().unique())
    if not groups:
        return {}

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    # Per group: (n_models, n_scenarios_in_group) matrix of per-scenario mean
    # efficacy (runs collapsed), NaN where a model lacks that scenario.
    group_mats: list[np.ndarray] = []
    for group in groups:
        gdf = df[col == group]
        scenario_ids = gdf["scenario_id"].unique()
        per_model = []
        for model in models:
            mdf = gdf[gdf["model"] == model]
            per_scenario = mdf.groupby("scenario_id")["efficacy"].mean().reindex(scenario_ids)
            per_model.append(per_scenario.to_numpy(dtype=float))
        group_mats.append(np.vstack(per_model))

    samples = np.empty((BOOTSTRAP_REPLICATES, len(models)))
    for b in range(BOOTSTRAP_REPLICATES):
        group_means = np.empty((len(groups), len(models)))
        for gi, mat in enumerate(group_mats):
            n = mat.shape[1]
            idx = rng.integers(0, n, size=n)  # same within-group draw for all models
            with np.errstate(invalid="ignore"):
                group_means[gi] = np.nanmean(mat[:, idx], axis=1)
        with np.errstate(invalid="ignore"):
            samples[b] = np.nanmean(group_means, axis=0)

    return {model: _percentile_ci(samples[:, j]) for j, model in enumerate(models)}


def compute_failure_profiles(df: pd.DataFrame) -> dict:
    """Per-model failure-mode profiles — counts and rates (issue #55).

    Aggregates the per-row ``failure_mode`` column (written by
    ``build_result_row``; null for passed runs) into the diagnostic
    practitioners actually use: how often, and in which of the six taxonomy
    modes, each model fails. Rates are per evaluated row, so an all-pass model
    publishes an explicit all-zeros profile rather than disappearing. The full
    mode vocabulary is always present (zero counts included) for a stable
    schema; an out-of-vocabulary mode string (future taxonomy growth) is
    appended rather than silently dropped, so counts always sum to n_failures.
    Returns ``{}`` for legacy parquets without the column — the caller publishes
    null per model. The caller passes the PUBLIC, null-agent-stripped frame, so
    holdout rows and non-contestants never reach a published profile.
    """
    if df.empty or "failure_mode" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        n_rows = int(len(mdf))
        failed = mdf["failure_mode"].dropna().astype(str)
        counts = failed.value_counts().to_dict()
        n_failures = int(failed.size)
        modes = {}
        for mode in [*FAILURE_MODES, *sorted(set(counts) - set(FAILURE_MODES))]:
            count = int(counts.get(mode, 0))
            modes[mode] = {"count": count, "rate": round(count / n_rows, 4)}
        out[str(model)] = {
            "n_rows": n_rows,
            "n_failures": n_failures,
            "failure_rate": round(n_failures / n_rows, 4),
            "modes": modes,
        }
    return out


def assign_rank_bands(models: list[dict]) -> None:
    """Cluster models into rank bands by clear_score CI overlap (in place).

    Greedy algorithm over models pre-sorted by clear_score descending:

      1. The current band leader is the highest-scoring model not yet banded.
      2. Every subsequent model whose clear_score_ci OVERLAPS the leader's
         clear_score_ci joins the leader's band (their ordering is not
         statistically distinguishable from the leader's).
      3. The first model that does NOT overlap the current leader's CI starts a
         new band and becomes its leader. Repeat.

    Two intervals [a_lo, a_hi] and [b_lo, b_hi] overlap when a_lo <= b_hi and
    b_lo <= a_hi. Models lacking a clear_score_ci (e.g. degenerate inputs) fall
    back to their own band. Writes ``rank_band`` (1-based) onto each entry.
    """
    band = 0
    leader_ci = None
    for m in models:
        ci = m.get("clear_score_ci")
        if leader_ci is None or ci is None or not _intervals_overlap(leader_ci, ci):
            band += 1
            leader_ci = ci
        m["rank_band"] = band


def _intervals_overlap(a: list, b: list) -> bool:
    """True when closed intervals a=[lo,hi] and b=[lo,hi] intersect."""
    if a is None or b is None or a[0] is None or b[0] is None:
        return False
    return a[0] <= b[1] and b[0] <= a[1]


def _round_or_none(value, ndigits):
    """round() that returns None on NaN/None.

    judge_agreement columns may be entirely NaN for a model (every row had
    fewer than 2 valid judges), in which case the grouped mean is NaN and a
    bare round() would emit NaN into JSON. Publish null instead.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return round(value, ndigits)


def _alpha_from_columns(df: pd.DataFrame, judge_cols: list[str]) -> float | None:
    """Krippendorff's alpha (interval) over the per-judge score columns.

    Each result row is a *unit* and each per-judge column (e.g. ``tc_Kimi K2.6``)
    is a *rater*. Missing cells (a judge that parse-failed on a row, so its score
    column is NaN there) are treated as missing values — alpha is defined for
    incomplete data. Returns ``None`` when alpha is undefined (fewer than two
    judge columns, or no usable variation), matching ``krippendorff_alpha``.
    """
    if len(judge_cols) < 2:
        return None
    sub = df[judge_cols]
    # units x raters: list-of-rows, NaN -> None so the metric skips it.
    reliability_data = [[None if pd.isna(v) else float(v) for v in row] for row in sub.to_numpy()]
    return krippendorff_alpha(reliability_data)


def compute_judge_alpha(df: pd.DataFrame, judge_columns: list[str]) -> dict:
    """Inter-judge Krippendorff alpha per rubric dimension (and per model).

    Returns ``{"task_completion": alpha|None, "tool_selection": alpha|None,
    "per_model": {model: {"task_completion": ..., "tool_selection": ...}}}``.

    Alpha is the primary, chance-corrected inter-judge reliability metric (the
    within-0.2 agreement rate is kept as a secondary human-readable readout in
    each model's ``judge_agreement`` block). It is computed over the published
    per-judge score columns so anyone can reproduce it from the released results.
    """
    tc_cols = [c for c in judge_columns if c.startswith("tc_")]
    ts_cols = [c for c in judge_columns if c.startswith("ts_")]

    per_model: dict[str, dict] = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        per_model[str(model)] = {
            "task_completion": _round_or_none(_alpha_from_columns(mdf, tc_cols), 4),
            "tool_selection": _round_or_none(_alpha_from_columns(mdf, ts_cols), 4),
        }

    return {
        "task_completion": _round_or_none(_alpha_from_columns(df, tc_cols), 4),
        "tool_selection": _round_or_none(_alpha_from_columns(df, ts_cols), 4),
        "per_model": per_model,
    }


def _ols_slope(x: np.ndarray, y: np.ndarray) -> dict | None:
    """Simple OLS of y on x (plain math, no deps). Returns slope diagnostics.

    Fits ``y = intercept + slope * x`` by least squares and returns the slope,
    intercept, R², the slope's standard error, its t-statistic, and a simple
    significance flag (``|t| > 1.96``, the ~5% two-sided normal threshold). All
    computed from closed-form formulas:

        slope = Cov(x, y) / Var(x)
        SE(slope) = sqrt( (RSS / (n - 2)) / Sxx )
        t = slope / SE(slope)

    Returns ``None`` when the fit is undefined: fewer than 3 points (no residual
    degrees of freedom for an SE), or x has zero variance (slope undefined).
    """
    n = x.size
    if n < 3:
        return None
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    sxx = float(((x - x_mean) ** 2).sum())
    if sxx <= 0:
        return None
    sxy = float(((x - x_mean) * (y - y_mean)).sum())
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    pred = intercept + slope * x
    residuals = y - pred
    rss = float((residuals**2).sum())
    syy = float(((y - y_mean) ** 2).sum())
    r_squared = 0.0 if syy <= 0 else 1.0 - rss / syy
    # Residual variance with n-2 degrees of freedom (two estimated params);
    # the SE of the slope follows from it and the spread of x (Sxx).
    resid_var = rss / (n - 2)
    se_slope = (resid_var / sxx) ** 0.5
    if se_slope == 0:
        # Perfect fit (or degenerate): slope is exact, t is infinite. Treat as
        # significant only if the slope is actually non-zero.
        t_stat = float("inf") if slope != 0 else 0.0
    else:
        t_stat = slope / se_slope
    return {
        "slope": round(slope, 8),
        "intercept": round(intercept, 6),
        "r_squared": round(r_squared, 4),
        "se_slope": round(se_slope, 8),
        "t_stat": round(t_stat, 4) if t_stat not in (float("inf"), float("-inf")) else None,
        "n": int(n),
        "significant": bool(abs(t_stat) > 1.96),
    }


def compute_length_bias(df: pd.DataFrame) -> dict:
    """OLS regression of judge scores on agent output length (length-bias check).

    Judges may favor verbose agents (the AlpacaEval length-bias lesson). The
    per-row agent ``output_tokens`` already exists, so we regress each judge-
    derived score on it and publish the slope so the bias is *measured*, not
    denied. A materially positive, significant slope means longer agent outputs
    get higher judge scores independent of correctness.

    We regress the two judge dimensions — ``task_completion`` and
    ``tool_selection`` consensus — on ``output_tokens`` across all rows (the bias
    is a property of the judge panel, not of any one model). The deterministic
    ``state_score`` is judge-independent so it is not a length-bias surface and is
    excluded.

    Returns ``{dimension: {slope, intercept, r_squared, se_slope, t_stat, n,
    significant}}`` for each dimension fit. A dimension is omitted when its fit is
    undefined (missing column, too few rows, or no length variation).
    """
    if "output_tokens" not in df.columns:
        return {}
    out: dict[str, dict] = {}
    for dim in ("task_completion", "tool_selection"):
        if dim not in df.columns:
            continue
        sub = df[["output_tokens", dim]].dropna()
        if sub.empty:
            continue
        x = sub["output_tokens"].to_numpy(dtype=float)
        y = sub[dim].to_numpy(dtype=float)
        fit = _ols_slope(x, y)
        if fit is not None:
            out[dim] = fit
    return out


def compute_consistency_bands(df: pd.DataFrame, threshold: float = PASS_THRESHOLD) -> dict:
    """Per-model solid / average / best-of consistency band (issue #71).

    WolfBench-style distribution-first presentation, derived entirely from the
    reliability repeats already in the parquet — no new data:

      * ``solid_rate`` — fraction of scenarios passed in EVERY run. This is
        exactly the tau-bench pass^k estimator at k = n (``C(c, n) / C(n, n)``
        is 1 iff c == n — see :func:`eval.scoring.rubrics.compute_pass_hat_k`),
        recomputed from the per-row efficacies so it stays correct even when
        run counts differ across scenarios (e.g. resume merges).
      * ``avg_pass_rate`` — mean over scenarios of the per-scenario pass
        fraction (pass^1 averaged over scenarios).
      * ``best_of_rate`` — fraction of scenarios passed in at least ONE run
        (pass@n).

    ``solid_rate <= avg_pass_rate <= best_of_rate`` by construction. A row
    "passes" at the same ``PASS_THRESHOLD`` every other published pass stat
    uses. Returns ``{}`` — and the leaderboard then OMITS the key entirely
    rather than publishing null/empty garbage — when the frame lacks the
    needed columns or contains no repeated runs (a single-run parquet has no
    consistency distribution: all three numbers would collapse to the same
    pass rate). The caller passes the PUBLIC, cooperative, non-null-agent
    frame, so holdout rows, behavioral-profile rows, and the null agent can
    never reach a published band.
    """
    needed = {"model", "scenario_id", "efficacy"}
    if df.empty or not needed <= set(df.columns):
        return {}
    runs_per = df.groupby(["model", "scenario_id"]).size()
    if int(runs_per.max()) < 2:
        # No reliability repeats anywhere -> no distribution to publish.
        return {}
    passed = df["efficacy"].astype(float) >= threshold
    per_scenario = (
        passed.groupby([df["model"], df["scenario_id"]]).agg(["all", "any", "mean"]).astype(float)
    )
    out: dict[str, dict] = {}
    for model, grp in per_scenario.groupby(level="model"):
        out[str(model)] = {
            "solid_rate": round(float(grp["all"].mean()), 4),
            "avg_pass_rate": round(float(grp["mean"].mean()), 4),
            "best_of_rate": round(float(grp["any"].mean()), 4),
            "n_scenarios": int(len(grp)),
            "n_runs": int(runs_per.loc[model].max()),
        }
    return out


def compute_corpus_health(df: pd.DataFrame, threshold: float = PASS_THRESHOLD) -> dict | None:
    """Corpus-level pass-distribution stats over the PUBLIC corpus (issue #71).

    WolfBench-style difficulty calibration / broken-task detection: scenarios
    never passed by ANY model in any run are possible defects (route to expert
    review); scenarios passed at least once by EVERY model carry no
    discriminative signal (hard-tier replacement candidates). "Passed by every
    model" means every model on the board passed the scenario in at least one
    of its runs — a scenario some model never attempted does not qualify.

    The published block carries COUNTS plus the headline string only — no
    scenario ids. Never-passed scenario ids are logged at INFO for maintainer
    eyes; the caller feeds the public-only frame (so even the log is id-safe),
    and keeping the JSON count-only means the holdout tripwire (no holdout
    scenario id ever in leaderboard.json) holds by construction.

    Returns ``None`` when the needed columns are missing — the leaderboard key
    is then absent, not null.
    """
    needed = {"model", "scenario_id", "efficacy"}
    if df.empty or not needed <= set(df.columns):
        return None
    passed = df["efficacy"].astype(float) >= threshold
    n_models = int(df["model"].nunique())

    # Scenario -> passed at least once by ANY model/run.
    by_scenario = passed.groupby(df["scenario_id"]).any()
    total = int(by_scenario.size)
    passed_at_least_once = int(by_scenario.sum())
    never_passed_ids = sorted(str(s) for s in by_scenario.index[~by_scenario])

    # Scenario -> how many distinct models passed it at least once. Comparing
    # against the FULL model count means a model that never attempted the
    # scenario correctly disqualifies it from "passed by every model".
    models_passing = (
        passed.groupby([df["scenario_id"], df["model"]]).any().groupby(level="scenario_id").sum()
    )
    passed_by_every_model = int((models_passing == n_models).sum())

    if never_passed_ids:
        logger.info(
            "Corpus health: %d scenario(s) never passed by any model "
            "(possible defects — route to expert review): %s",
            len(never_passed_ids),
            ", ".join(never_passed_ids),
        )

    headline = (
        f"{passed_at_least_once} of {total} scenarios passed at least once; "
        f"{passed_by_every_model} passed by every model"
    )
    return {
        "total_scenarios": total,
        "passed_at_least_once": passed_at_least_once,
        "never_passed": total - passed_at_least_once,
        "passed_by_every_model": passed_by_every_model,
        "n_models": n_models,
        "pass_threshold": threshold,
        "headline": headline,
        "note": (
            "Pass distribution over the public corpus (issue #71). A run passes "
            f"at efficacy >= {threshold}; a scenario counts as passed when any "
            "run of any model passes it. never_passed scenarios are possible "
            "defects (expert review); passed_by_every_model scenarios carry no "
            "discriminative signal (hard-tier replacement candidates). Counts "
            "only — scenario ids are logged for maintainers, never published. "
            "Holdout, null-agent, and non-cooperative rows are excluded."
        ),
    }


def compute_pass_hat_k_by_model(df: pd.DataFrame) -> dict:
    """Per-model pass^k (tau-bench) means, keyed by model name.

    Each row carries the per-scenario pass^k estimates in ``reliability_pass_hat_k``
    columns (one per k, written by run_eval). We mean those across rows per model;
    since every run-row of a scenario carries that scenario's pass^k, the row-mean
    is the mean over scenarios of the per-scenario pass^k — the model-level
    estimate. Returns ``{model: {"k": value, ...}}``; an empty per-model dict when
    no pass^k columns exist (legacy parquets).
    """
    k_cols = sorted(
        (c for c in df.columns if c.startswith("reliability_pass_hat_")),
        key=lambda c: int(c.rsplit("_", 1)[1]),
    )
    out: dict[str, dict] = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        block: dict[str, float] = {}
        for c in k_cols:
            k = c.rsplit("_", 1)[1]
            block[k] = _round_or_none(mdf[c].mean(), 4)
        out[str(model)] = block
    return out


def load_all_results() -> pd.DataFrame:
    """Load the most recent parquet result file.

    Each weekly run writes a complete snapshot of every model across every
    scenario, so the latest file alone is the full leaderboard — we intentionally
    use only that file rather than concatenating older runs (which would
    double-count models and mix stale results with fresh ones).
    """
    parquet_files = sorted(RESULTS_DIR.glob("results_*.parquet"))
    if not parquet_files:
        logger.warning("No result files found in %s", RESULTS_DIR)
        return pd.DataFrame()

    # Use the most recent file
    latest = parquet_files[-1]
    logger.info("Loading results from %s", latest)
    return pd.read_parquet(latest)


# --- Same-lab robustness check ---
# The Claude Opus judge shares a lab (Anthropic) with the Claude contestants.
# methodology.md discloses this pairing; the mitigation it promises is that
# per-judge scores are published so anyone can recompute consensus with the
# same-lab judge excluded. This check does that recomputation up front and
# publishes it per Anthropic contestant: judge-mean task-completion and
# tool-selection from the arm's-length (open) judges only, plus the delta vs
# the full panel. A materially positive delta (full > excluded) would mean the
# same-lab judge rates its siblings higher than the open judges do.
SAME_LAB_JUDGE_MARKER = "opus"  # matches the judge's display name in tc_/ts_ columns
SAME_LAB_CONTESTANT_MARKER = "claude"  # matches contestant display names


def _judge_name_from_column(col: str) -> str:
    """Strip the ``tc_``/``ts_`` rubric prefix to recover the judge display name."""
    return col[3:] if col.startswith(("tc_", "ts_")) else col


def compute_judge_deltas(df: pd.DataFrame, judge_columns: list[str]) -> dict:
    """Per-judge-vs-consensus delta stats for EVERY judge, keyed by model name.

    This generalizes the same-lab check (which only instrumented the Opus/Claude
    pairing) to all judges: for each contestant and each judge on the panel, we
    report that judge's mean task-completion / tool-selection for the model and
    its delta against the published full-panel consensus (``full - judge``).

    Sign convention matches ``compute_same_lab_check``: ``delta = consensus_mean -
    judge_mean``, so a **positive** delta means the panel consensus is higher than
    this judge (the judge rates the model *lower* than its peers), and a
    **negative** delta means the judge is more generous to this model than the
    panel. This is the raw material for the "does any judge systematically favor
    a model" launch finding — including, but not limited to, the same-lab case.

    Returns ``{model: {"task_completion": {judge: {"mean": .., "delta": ..}, ...},
    "tool_selection": {...}}}``. Like the same-lab check this is a diagnostic over
    the published per-judge columns (which already exclude parse failures), so
    small differences from row-level panel composition are expected.
    """
    tc_cols = [c for c in judge_columns if c.startswith("tc_")]
    ts_cols = [c for c in judge_columns if c.startswith("ts_")]
    if not tc_cols and not ts_cols:
        return {}

    deltas: dict[str, dict] = {}
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        entry: dict = {}
        for key, cols, full_col in (
            ("task_completion", tc_cols, "task_completion"),
            ("tool_selection", ts_cols, "tool_selection"),
        ):
            if not cols or full_col not in mdf.columns:
                continue
            full = float(mdf[full_col].mean())
            per_judge: dict[str, dict] = {}
            for col in cols:
                judge_mean = mdf[col].mean()
                per_judge[_judge_name_from_column(col)] = {
                    "mean": _round_or_none(judge_mean, 4),
                    "delta": _round_or_none(
                        None if pd.isna(judge_mean) else full - float(judge_mean),
                        4,
                    ),
                }
            entry[key] = per_judge
        deltas[str(model)] = entry
    return deltas


def compute_same_lab_check(df: pd.DataFrame, judge_columns: list[str]) -> dict:
    """Per-model same-lab robustness stats keyed by model name.

    For each contestant whose name matches ``SAME_LAB_CONTESTANT_MARKER``,
    recompute the mean task-completion / tool-selection over ONLY the per-judge
    score columns that do not match ``SAME_LAB_JUDGE_MARKER``, and report the
    delta against the published full-panel consensus columns. Models without a
    same-lab relationship are omitted (their entry is null in the leaderboard).

    Note this is a robustness diagnostic, not a re-scoring: the full-panel
    consensus excludes parse-failed judges row by row, while this check uses
    the published per-judge columns (which already exclude parse failures), so
    small differences from row-level panel composition are expected.
    """
    same_lab_cols = {c for c in judge_columns if SAME_LAB_JUDGE_MARKER in c.lower()}
    if not same_lab_cols:
        return {}

    open_tc = [c for c in judge_columns if c.startswith("tc_") and c not in same_lab_cols]
    open_ts = [c for c in judge_columns if c.startswith("ts_") and c not in same_lab_cols]
    if not open_tc and not open_ts:
        return {}

    checks: dict[str, dict] = {}
    for model in df["model"].unique():
        if SAME_LAB_CONTESTANT_MARKER not in str(model).lower():
            continue
        mdf = df[df["model"] == model]
        entry: dict = {
            "excluded_judge_columns": sorted(same_lab_cols),
        }
        for key, open_cols, full_col in (
            ("task_completion", open_tc, "task_completion"),
            ("tool_selection", open_ts, "tool_selection"),
        ):
            if not open_cols:
                continue
            # Per-row mean over available open-judge scores, then mean over rows.
            excl = float(mdf[open_cols].mean(axis=1).mean())
            full = float(mdf[full_col].mean())
            entry[f"{key}_excl_same_lab"] = _round_or_none(excl, 4)
            entry[f"{key}_delta"] = _round_or_none(full - excl, 4)
        checks[model] = entry
    return checks


def compute_holdout_gap(df: pd.DataFrame) -> dict:
    """Per-model public-vs-holdout efficacy split (issue #31).

    Returns ``{model: {"public_score": .., "holdout_score": .., "holdout_gap": ..}}``
    for models that have BOTH public and holdout rows. ``holdout_gap`` is
    ``public_score - holdout_score`` (mean efficacy): a positive gap means the
    model does better on the public corpus than on the private holdout — the
    overfitting tripwire. Models with no holdout rows are omitted (their gap is
    null in the leaderboard entry).

    Only per-MODEL aggregates are returned — never per-scenario holdout detail —
    so nothing about which holdout scenarios exist or how a model did on any one
    of them is exposed. ``df`` here is already null-agent-stripped by the caller.
    """
    if df.empty or "holdout" not in df.columns:
        return {}
    holdout_mask = df["holdout"].fillna(False).astype(bool)
    public_df = df[~holdout_mask]
    holdout_df = df[holdout_mask]
    if holdout_df.empty:
        return {}

    public_eff = public_df.groupby("model")["efficacy"].mean()
    holdout_eff = holdout_df.groupby("model")["efficacy"].mean()

    out: dict[str, dict] = {}
    for model in holdout_eff.index:
        pub = public_eff.get(model)
        hold = float(holdout_eff[model])
        entry = {
            "public_score": _round_or_none(pub, 4),
            "holdout_score": round(hold, 4),
            # Gap only when both halves exist; otherwise the comparison is
            # undefined and we publish null rather than a misleading number.
            "holdout_gap": (None if pub is None or pd.isna(pub) else round(float(pub) - hold, 4)),
        }
        out[str(model)] = entry
    return out


def compute_leaderboard(df: pd.DataFrame) -> dict:
    """Compute leaderboard rankings from raw results.

    The headline rankings (efficacy, CLEAR, per-domain, judge stats) are computed
    over the PUBLIC corpus only — the private holdout (issue #31) is split out
    first so it cannot move the public score, and is summarized per model as a
    public-vs-holdout gap. No per-scenario holdout detail ever reaches the
    leaderboard, history, or CSV.
    """
    # Strip non-contestants (e.g. the do-nothing null-agent) before any
    # aggregation so they never appear on the board or skew normalization.
    df = exclude_non_contestants(df)
    if df.empty:
        return {"models": [], "updated": "", "domains": []}

    # Persona-stratified robustness table (issue #59 / H4). This is the ONE
    # surface that legitimately reads non-cooperative rows — it reports how each
    # model holds up under the behavioral profiles vs cooperative — so it is
    # computed from the FULL frame here, BEFORE exclude_non_cooperative_profiles
    # below strips the non-coop rows for the public board. It must still respect
    # the OTHER two tripwires: null-agent rows are already gone (excluded just
    # above), and holdout rows must never appear here either (governance §4), so
    # they are dropped from this snapshot before computing the table. Only emitted
    # when non-cooperative rows actually exist, mirroring the holdout header
    # pattern (absent key on a cooperative-only run, so no empty surface ships).
    robustness_source = df
    if "holdout" in robustness_source.columns:
        robustness_source = robustness_source[
            ~robustness_source["holdout"].fillna(False).astype(bool)
        ]
    has_noncooperative = bool((~_cooperative_mask(robustness_source)).any())
    sim_profile_robustness = (
        compute_sim_profile_pass_rates(robustness_source) if has_noncooperative else None
    )

    # Recovery-probe table (issue #57 / H). Like the robustness table, this reads
    # from the full frame BEFORE the cooperative exclusion (a probe is orthogonal
    # to the sim profile — it can ride any profile), with null-agent rows already
    # gone and holdout rows dropped (governance §4: holdout detail never reaches a
    # published surface). recovery_rate is computed over probe-carrying rows ONLY
    # and emitted conditionally — absent on every v1 run, since no published
    # scenario carries a probe.
    recovery_rates = compute_recovery_rates(robustness_source)

    # Strip rows from non-cooperative user-sim profiles (issue #59) before ANY
    # aggregate — including the holdout gap below — so a behavioral-profile run
    # can never move public efficacy. They are published separately via
    # compute_sim_profile_pass_rates (the persona-stratified robustness table,
    # emitted under sim_profile_robustness below).
    df = exclude_non_cooperative_profiles(df)
    if df.empty:
        # A run with only non-cooperative rows has no public leaderboard.
        return {"models": [], "updated": "", "domains": []}

    # Private-holdout split (issue #31). Compute the per-model gap from the full
    # frame first, then drop holdout rows so every downstream aggregate (efficacy,
    # CLEAR, bootstrap CIs, per-domain, per-judge) is over the PUBLIC corpus only.
    # The leaderboard publishes the gap per model but never holdout scenario IDs.
    holdout_gap = compute_holdout_gap(df)
    if "holdout" in df.columns:
        df = df[~df["holdout"].fillna(False).astype(bool)]
        if df.empty:
            # A holdout-only run has no public leaderboard to publish.
            return {"models": [], "updated": "", "domains": []}

    # Overall scores per model (averaged across all scenarios and runs)
    overall = (
        df.groupby("model")
        .agg(
            efficacy=("efficacy", "mean"),
            task_completion=("task_completion", "mean"),
            tool_selection=("tool_selection", "mean"),
            cost_per_task=("cost_usd", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            reliability=("reliability_pass_rate", "mean"),
            reliability_consistency=("reliability_consistency", "mean"),
            avg_turns=("total_turns", "mean"),
            total_scenarios=("scenario_id", "nunique"),
            total_rows=("scenario_id", "size"),
            judge_agreement_tc=("tc_agreement", "mean"),
            judge_agreement_ts=("ts_agreement", "mean"),
        )
        .reset_index()
    )

    # Deterministic state-verification mean (v0.2). Computed separately and
    # merged so older parquets without a state_score column don't break the agg.
    # groupby.mean skips NaN, so a model with a mix of state-graded and legacy
    # scenarios averages only its graded rows; a model with no graded rows is NaN
    # and published as null (NaN-guarded in the entry below).
    if "state_score" in df.columns:
        state_means = df.groupby("model")["state_score"].mean().reset_index()
        overall = overall.merge(state_means, on="model", how="left")
    else:
        overall["state_score"] = np.nan

    # Premature-ending rate (#32): the fraction of a model's runs the user sim
    # ended while the deterministic state check was still below 1.0. This makes
    # the user-sim/goal-completion decoupling visible at the leaderboard level —
    # a high rate means the simulator is quitting before goals are verifiably met
    # (or the agent is leaving goals unmet but reassuring the sim). Merged
    # separately so legacy parquets without the column don't break the agg.
    if "premature_end" in df.columns:
        premature = (
            df.assign(_premature=df["premature_end"].fillna(False).astype(bool))
            .groupby("model")["_premature"]
            .mean()
            .reset_index()
            .rename(columns={"_premature": "premature_end_rate"})
        )
        overall = overall.merge(premature, on="model", how="left")
    else:
        overall["premature_end_rate"] = np.nan

    # Compute composite CLEAR score (field-relative min-max normalization).
    # Higher is better for efficacy and reliability; lower is better for cost and
    # latency (inverted). The point estimate routes through the SAME
    # _clear_from_means / CLEAR_WEIGHTS path the bootstrap replicates use, so the
    # published score and its CI cannot silently desynchronize (the weights live
    # in exactly one place — see CLEAR_WEIGHTS). test_clear_weights_single_source
    # pins the equality.
    if len(overall) > 1:
        overall["clear_score"] = _clear_from_means(
            overall["efficacy"].to_numpy(dtype=float),
            overall["reliability"].to_numpy(dtype=float),
            overall["cost_per_task"].to_numpy(dtype=float),
            overall["avg_latency_ms"].to_numpy(dtype=float),
        )
    else:
        overall["clear_score"] = overall["efficacy"]

    overall = overall.sort_values("clear_score", ascending=False)

    # Bootstrap confidence intervals (scenario-resampling, paired across models)
    model_order = overall["model"].tolist()
    bootstrap_cis = compute_bootstrap_cis(df, model_order)

    # Per-domain breakdown
    domain_scores = {}
    for domain in df["domain"].unique():
        domain_df = df[df["domain"] == domain]
        domain_agg = (
            domain_df.groupby("model")
            .agg(
                efficacy=("efficacy", "mean"),
                cost_per_task=("cost_usd", "mean"),
                avg_latency_ms=("latency_ms", "mean"),
                reliability=("reliability_pass_rate", "mean"),
            )
            .reset_index()
            .sort_values("efficacy", ascending=False)
        )
        domain_scores[domain] = domain_agg.to_dict("records")

    # Per-category breakdown (issue #55) — the micro view of the macro grouping,
    # so a reader can see WHICH category drags a model's macro score. Efficacy +
    # scenario depth only (cost/latency are not category-shaped questions).
    category_scores: dict[str, list] = {}
    if "category" in df.columns:
        for category in df["category"].dropna().unique():
            cat_df = df[df["category"] == category]
            cat_agg = (
                cat_df.groupby("model")
                .agg(
                    efficacy=("efficacy", "mean"),
                    n_scenarios=("scenario_id", "nunique"),
                )
                .reset_index()
                .sort_values("efficacy", ascending=False)
            )
            category_scores[str(category)] = cat_agg.to_dict("records")

    # Macro-averaged efficacy + stratified bootstrap CIs (issue #55): every
    # category (and domain) weighted equally, published ALONGSIDE the micro
    # headline. Empty dicts on legacy parquets without the grouping column.
    macro_category = compute_macro_efficacy(df, "category")
    macro_domain = compute_macro_efficacy(df, "domain")
    macro_category_cis = compute_macro_bootstrap_cis(df, model_order, "category")
    macro_domain_cis = compute_macro_bootstrap_cis(df, model_order, "domain")

    # Per-model failure-mode profiles (issue #55), over the PUBLIC corpus only
    # (df has holdout rows + non-contestants stripped above).
    failure_profiles = compute_failure_profiles(df)

    # Per-judge scores (transparency). Exclude the consensus accounting columns
    # (agreement, disagreement, validity counts, failure counts, degraded flags)
    # — those share the tc_/ts_ prefix but are not per-judge score columns.
    non_judge_suffixes = (
        "agreement",
        "max_disagreement",
        "n_judges",
        "parse_failures",
        "api_failures",
        "degraded",
    )
    judge_columns = [c for c in df.columns if c.startswith("tc_") or c.startswith("ts_")]
    judge_columns = [
        c for c in judge_columns if not any(c.endswith(suffix) for suffix in non_judge_suffixes)
    ]
    judge_scores = {}
    if judge_columns:
        for col in judge_columns:
            judge_agg = df.groupby("model")[col].mean().reset_index()
            judge_scores[col] = judge_agg.set_index("model")[col].to_dict()

    same_lab_checks = compute_same_lab_check(df, judge_columns)

    # Per-judge-vs-consensus deltas for EVERY judge (generalizes same_lab_check).
    judge_deltas = compute_judge_deltas(df, judge_columns)

    # Inter-judge reliability: Krippendorff's alpha (interval), the primary
    # chance-corrected metric. Per-model within-0.2 agreement stays as a
    # secondary readout in each model entry's judge_agreement block.
    judge_alpha = compute_judge_alpha(df, judge_columns)

    # pass^k (tau-bench): per-model mean of the per-scenario pass^k estimates,
    # one entry per k. Published alongside pass@3 (reliability) and consistency.
    pass_hat_k = compute_pass_hat_k_by_model(df)

    # Consistency bands + corpus health (issue #71, WolfBench learnings). Both
    # are computed HERE — after the null-agent, non-cooperative, and holdout
    # strips above — so they see PUBLIC, cooperative, contestant rows only.
    # Empty/None when the parquet lacks the data (e.g. a single-run parquet has
    # no consistency distribution); the keys are then omitted entirely.
    consistency_bands = compute_consistency_bands(df)
    corpus_health = compute_corpus_health(df)

    # Length-bias check: OLS of judge scores on agent output length, so verbosity
    # bias is a measured slope rather than an assumption.
    length_bias = compute_length_bias(df)

    # Build leaderboard JSON
    leaderboard = {
        "updated": pd.Timestamp.now().isoformat(),
        "version": "0.1.0",
        "metrics": ["efficacy", "cost", "reliability", "latency", "clear_score"],
        "models": [],
        "domains": list(df["domain"].unique()),
        "domain_scores": domain_scores,
        # Scenario categories present in this run (issue #55), with the per-
        # category micro breakdown backing the macro-averaged scores. Empty on
        # legacy parquets without a category column.
        "categories": sorted(category_scores),
        "category_scores": category_scores,
        # Failure-mode taxonomy header (issue #55): the fixed mode vocabulary +
        # how rows were classified. Per-model profiles live in each model entry.
        "failure_taxonomy": {
            "modes": list(FAILURE_MODES),
            "note": (
                "Failed evaluations (efficacy below the "
                f"{PASS_THRESHOLD} pass threshold) are classified "
                "deterministic-first: state-grader evidence and the premature-end "
                "instrumentation (#32) take precedence over judge-reasoning "
                "keyword matching; no additional LLM calls. Per-model counts and "
                "rates are in each entry's failure_profile."
            ),
        },
        "judge_scores": judge_scores,
        "judge_alpha": judge_alpha,
        "length_bias": length_bias,
        # Private-holdout summary (issue #31). Top-level flag + the count of
        # models that have a public-vs-holdout gap. Deliberately NO scenario IDs,
        # text, or counts of the holdout corpus itself — the per-model gaps live
        # in each model entry; this is only a "was a holdout run, for how many
        # models" header for the frontend.
        "holdout": {
            "present": bool(holdout_gap),
            "models_with_gap": len(holdout_gap),
            "note": (
                "Public-vs-holdout efficacy gap (gap = public - holdout). A "
                "positive gap flags a model that does better on the public corpus "
                "than on the private, never-published holdout — an overfitting "
                "signal. See docs/governance.md §4 and issue #31."
            ),
        }
        if holdout_gap
        else {"present": False},
    }

    # Persona-stratified robustness table (issue #59 / H4): per-model, per-profile
    # pass rates with each non-cooperative profile's delta vs the cooperative
    # condition — the cooperative-only inflation made into a published number.
    # Computed from the full frame above (before the cooperative exclusion) with
    # null-agent and holdout rows already removed. Key present ONLY when the run
    # included non-cooperative rows, so a normal cooperative leaderboard ships no
    # empty/misleading surface (mirrors the holdout "present" pattern). No profile
    # name ever appears unless a stratified run actually produced those rows.
    # Corpus-health block (issue #71): pass-distribution counts + the headline
    # line over the public corpus. Key omitted (not null) when uncomputable,
    # following the sim_profile_robustness / holdout conditional pattern.
    if corpus_health is not None:
        leaderboard["corpus_health"] = corpus_health

    if sim_profile_robustness:
        leaderboard["sim_profile_robustness"] = {
            "cooperative_profile": DEFAULT_SIM_PROFILE,
            "pass_threshold": PASS_THRESHOLD,
            "note": (
                "Per-model pass rate (efficacy >= the pass threshold) under each "
                "user-sim behavioral profile (issue #59). delta_vs_cooperative = "
                "cooperative_pass_rate - profile_pass_rate; positive means the "
                "model does worse under that profile. These rows are EXCLUDED from "
                "the public efficacy/CLEAR rankings above and reported only here. "
                "Holdout and null-agent rows are never included."
            ),
            "models": sim_profile_robustness,
        }

    # Recovery-probe robustness table (issue #57): per-model recovery rate over
    # probe-carrying rows, broken down by probe kind. Key present ONLY when probe
    # rows exist — so a normal run (the entire v1 corpus has no probes) ships no
    # empty surface, mirroring the holdout "present" and sim_profile_robustness
    # patterns. Probe rows feed THIS surface only; they never move public
    # efficacy/CLEAR (probe scenarios are a separate tier — see
    # docs/recovery-probes.md).
    if recovery_rates:
        leaderboard["recovery_probe_robustness"] = {
            "note": (
                "Per-model recovery rate over recovery-probe scenarios (issue "
                "#57): a deterministic mid-conversation fault is injected at turn "
                "4-5 and recovery is verified by state grading (correct end state "
                "reached DESPITE the fault, and the bad entity not acted on). "
                "recovery_rate is the fraction of probe rows the model recovered, "
                "with a by_kind breakdown. These rows are a SEPARATE dimension — "
                "they are excluded from the public efficacy/CLEAR rankings above. "
                "Holdout and null-agent rows are never included."
            ),
            "models": recovery_rates,
        }

    for _, row in overall.iterrows():
        cis = bootstrap_cis.get(row["model"], {})
        model_entry = {
            "name": row["model"],
            "clear_score": round(row["clear_score"], 4),
            "clear_score_ci": cis.get("clear_score_ci", [None, None]),
            "efficacy": round(row["efficacy"], 4),
            "efficacy_ci": cis.get("efficacy_ci", [None, None]),
            # Macro-averaged efficacy (issue #55): unweighted mean of per-
            # category (per-domain) means, published alongside the micro
            # ``efficacy`` above so frequent-easy scenario clusters cannot drown
            # rare-hard ones. CIs come from the category-stratified bootstrap
            # (same B/seed). None/[None, None] on legacy data without the column.
            "efficacy_macro_category": macro_category.get(row["model"]),
            "efficacy_macro_category_ci": macro_category_cis.get(row["model"], [None, None]),
            "efficacy_macro_domain": macro_domain.get(row["model"]),
            "efficacy_macro_domain_ci": macro_domain_cis.get(row["model"], [None, None]),
            "task_completion": round(row["task_completion"], 4),
            "tool_selection": round(row["tool_selection"], 4),
            "state_score": _round_or_none(row.get("state_score"), 4),
            # Private-holdout split (issue #31). ``efficacy`` above is the public
            # score; these expose the held-out score and the gap (public -
            # holdout) so an overfitting model — strong on the public corpus,
            # weaker on the never-published holdout — is visible. None when this
            # model has no holdout rows. Only per-model aggregates; no holdout
            # scenario detail is ever published.
            "holdout_score": holdout_gap.get(row["model"], {}).get("holdout_score"),
            "holdout_gap": holdout_gap.get(row["model"], {}).get("holdout_gap"),
            # Premature-ending rate (#32): share of this model's runs the user
            # sim ended before the deterministic state check passed. None for
            # legacy parquets that predate the column.
            "premature_end_rate": _round_or_none(row.get("premature_end_rate"), 4),
            # Failure-mode profile (issue #55): per-mode counts + rates over this
            # model's PUBLIC rows. An all-pass model gets an explicit all-zeros
            # profile; None only for legacy parquets without the column.
            "failure_profile": failure_profiles.get(row["model"]),
            "cost_per_task_usd": round(row["cost_per_task"], 6),
            "avg_latency_ms": round(row["avg_latency_ms"], 1),
            "reliability": round(row["reliability"], 4),
            "reliability_consistency": _round_or_none(row["reliability_consistency"], 4),
            # pass^k (tau-bench): all-k-trials-succeed probability per k, published
            # alongside pass@3 (``reliability``) and consistency, not replacing them.
            "reliability_pass_hat_k": pass_hat_k.get(row["model"], {}),
            "avg_turns": round(row["avg_turns"], 1),
            "scenarios_evaluated": int(row["total_scenarios"]),
            "n_scenarios": int(row["total_scenarios"]),
            "n_rows": int(row["total_rows"]),
            # Inter-judge agreement: alpha is the primary chance-corrected metric;
            # within_0_2 is the secondary human-readable readout (kept for
            # continuity). Both are reported per dimension.
            "judge_agreement": {
                "task_completion": _round_or_none(row["judge_agreement_tc"], 4),
                "tool_selection": _round_or_none(row["judge_agreement_ts"], 4),
            },
            "judge_alpha": judge_alpha["per_model"].get(row["model"]),
            # Per-judge-vs-consensus deltas for every judge on the panel, per
            # dimension — the generalized form of the same-lab check.
            "judge_deltas": judge_deltas.get(row["model"]),
            # Same-lab robustness: judge-mean scores with the same-lab judge
            # excluded, plus the delta vs the full panel. None for models with
            # no same-lab judge on the panel. Kept working alongside judge_deltas.
            "same_lab_check": same_lab_checks.get(row["model"]),
        }
        # Consistency band (issue #71): solid / avg / best-of over the
        # reliability runs. Key ABSENT (not null) when the parquet has no
        # repeated runs to derive a distribution from.
        band = consistency_bands.get(row["model"])
        if band is not None:
            model_entry["consistency_band"] = band
        leaderboard["models"].append(model_entry)

    # Rank bands: cluster models whose clear_score CIs overlap so the frontend
    # can show that orderings within a band are not statistically distinguishable.
    assign_rank_bands(leaderboard["models"])

    # Top-level note stating the scenario count and the band caveat. Uses the
    # max per-model scenario count as the field's scenario depth.
    max_scenarios = max((m["n_scenarios"] for m in leaderboard["models"]), default=0)
    n_bands = max((m["rank_band"] for m in leaderboard["models"]), default=0)
    note = (
        f"Scores are means over {max_scenarios} scenario(s) with 95% bootstrap "
        f"confidence intervals (B={BOOTSTRAP_REPLICATES} resamples over scenarios, "
        f"seed {BOOTSTRAP_SEED}). Models sharing a rank band have overlapping CLEAR "
        "intervals; their ordering within a band is not statistically distinguishable. "
        "Macro-averaged efficacy weights each scenario category (or domain) equally; "
        "its CI uses a category-stratified scenario bootstrap with the same B and seed."
    )
    if max_scenarios < MIN_SCENARIOS_FOR_PUBLISH:
        note += (
            f" NOTE: only {max_scenarios} scenario(s) evaluated — below the "
            f"{MIN_SCENARIOS_FOR_PUBLISH}-scenario publish minimum; treat all "
            "orderings as provisional."
        )
    leaderboard["statistical_note"] = note
    leaderboard["n_rank_bands"] = n_bands

    return leaderboard


def main():
    df = load_all_results()
    if df.empty:
        # Exit non-zero: in CI this runs right after run_eval, so an empty
        # results dir means the eval silently produced nothing. Returning 0
        # here used to let the workflow continue until `git add` died on the
        # missing leaderboard.json with no hint of the real cause.
        raise SystemExit(
            "No results to aggregate — data/results/ has no parquet output. "
            "Did the eval run produce anything?"
        )

    leaderboard = compute_leaderboard(df)

    # Save leaderboard JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = RESULTS_DIR / "leaderboard.json"
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    logger.info("Leaderboard saved to %s", leaderboard_path)

    # Append to history for trend tracking
    history_path = RESULTS_DIR / "history.jsonl"
    snapshot = {
        "timestamp": leaderboard["updated"],
        "models": {
            m["name"]: {
                "clear_score": m["clear_score"],
                "efficacy": m["efficacy"],
                "cost_per_task_usd": m["cost_per_task_usd"],
                "reliability": m["reliability"],
                "avg_latency_ms": m["avg_latency_ms"],
            }
            for m in leaderboard["models"]
        },
    }
    with open(history_path, "a") as f:
        f.write(json.dumps(snapshot) + "\n")
    logger.info("Appended snapshot to %s", history_path)

    # Save human-readable CSV
    csv_path = RESULTS_DIR / "latest.csv"
    models_df = pd.DataFrame(leaderboard["models"])
    models_df = models_df.sort_values("clear_score", ascending=False)
    models_df.to_csv(csv_path, index=False)
    logger.info("CSV saved to %s", csv_path)

    # Print summary
    print("\n=== COT Bench Leaderboard ===\n")
    print(f"Updated: {leaderboard['updated']}")
    print(f"Domains: {', '.join(leaderboard['domains'])}")
    print(f"Models evaluated: {len(leaderboard['models'])}")
    print()

    for i, model in enumerate(leaderboard["models"], 1):
        print(
            f"  #{i} {model['name']:25s} "
            f"CLEAR={model['clear_score']:.3f}  "
            f"Eff={model['efficacy']:.3f}  "
            f"Cost=${model['cost_per_task_usd']:.4f}  "
            f"Rel={model['reliability']:.3f}  "
            f"Lat={model['avg_latency_ms']:.0f}ms"
        )

    # Corpus-health headline (issue #71): the WolfBench-style distribution line.
    corpus_health = leaderboard.get("corpus_health")
    if corpus_health:
        print(f"\nCorpus health: {corpus_health['headline']}")


if __name__ == "__main__":
    main()
