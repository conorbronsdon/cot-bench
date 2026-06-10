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


def compute_leaderboard(df: pd.DataFrame) -> dict:
    """Compute leaderboard rankings from raw results."""
    # Strip non-contestants (e.g. the do-nothing null-agent) before any
    # aggregation so they never appear on the board or skew normalization.
    df = exclude_non_contestants(df)
    if df.empty:
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

    # Compute composite CLEAR score (normalized, equal weight)
    # Higher is better for efficacy and reliability
    # Lower is better for cost and latency — invert these
    if len(overall) > 1:
        for col in ["efficacy", "reliability"]:
            col_min, col_max = overall[col].min(), overall[col].max()
            rng = col_max - col_min
            overall[f"{col}_norm"] = (overall[col] - col_min) / rng if rng > 0 else 0.5

        for col in ["cost_per_task", "avg_latency_ms"]:
            col_min, col_max = overall[col].min(), overall[col].max()
            rng = col_max - col_min
            overall[f"{col}_norm"] = 1.0 - ((overall[col] - col_min) / rng) if rng > 0 else 0.5

        overall["clear_score"] = (
            overall["efficacy_norm"] * 0.35
            + overall["cost_per_task_norm"] * 0.20
            + overall["reliability_norm"] * 0.25
            + overall["avg_latency_ms_norm"] * 0.20
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
        "judge_scores": judge_scores,
        "judge_alpha": judge_alpha,
        "length_bias": length_bias,
    }

    for _, row in overall.iterrows():
        cis = bootstrap_cis.get(row["model"], {})
        model_entry = {
            "name": row["model"],
            "clear_score": round(row["clear_score"], 4),
            "clear_score_ci": cis.get("clear_score_ci", [None, None]),
            "efficacy": round(row["efficacy"], 4),
            "efficacy_ci": cis.get("efficacy_ci", [None, None]),
            "task_completion": round(row["task_completion"], 4),
            "tool_selection": round(row["tool_selection"], 4),
            "state_score": _round_or_none(row.get("state_score"), 4),
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
        "intervals; their ordering within a band is not statistically distinguishable."
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


if __name__ == "__main__":
    main()
