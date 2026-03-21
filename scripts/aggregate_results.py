"""Aggregate evaluation results into leaderboard format.

Reads parquet result files and produces:
- leaderboard.json: structured leaderboard data for the frontend
- latest.csv: human-readable summary
"""

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("data/results")


def load_all_results() -> pd.DataFrame:
    """Load and concatenate all parquet result files."""
    parquet_files = sorted(RESULTS_DIR.glob("results_*.parquet"))
    if not parquet_files:
        logger.warning("No result files found in %s", RESULTS_DIR)
        return pd.DataFrame()

    # Use the most recent file
    latest = parquet_files[-1]
    logger.info("Loading results from %s", latest)
    return pd.read_parquet(latest)


def compute_leaderboard(df: pd.DataFrame) -> dict:
    """Compute leaderboard rankings from raw results."""
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
            avg_turns=("total_turns", "mean"),
            total_scenarios=("scenario_id", "nunique"),
            judge_agreement_tc=("tc_agreement", "mean"),
            judge_agreement_ts=("ts_agreement", "mean"),
        )
        .reset_index()
    )

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

    # Per-judge scores (transparency)
    judge_columns = [c for c in df.columns if c.startswith("tc_") or c.startswith("ts_")]
    judge_columns = [c for c in judge_columns if c not in ("tc_agreement", "ts_agreement")]
    judge_scores = {}
    if judge_columns:
        for col in judge_columns:
            judge_agg = df.groupby("model")[col].mean().reset_index()
            judge_scores[col] = judge_agg.set_index("model")[col].to_dict()

    # Build leaderboard JSON
    leaderboard = {
        "updated": pd.Timestamp.now().isoformat(),
        "version": "0.1.0",
        "metrics": ["efficacy", "cost", "reliability", "latency", "clear_score"],
        "models": [],
        "domains": list(df["domain"].unique()),
        "domain_scores": domain_scores,
        "judge_scores": judge_scores,
    }

    for _, row in overall.iterrows():
        model_entry = {
            "name": row["model"],
            "clear_score": round(row["clear_score"], 4),
            "efficacy": round(row["efficacy"], 4),
            "task_completion": round(row["task_completion"], 4),
            "tool_selection": round(row["tool_selection"], 4),
            "cost_per_task_usd": round(row["cost_per_task"], 6),
            "avg_latency_ms": round(row["avg_latency_ms"], 1),
            "reliability": round(row["reliability"], 4),
            "avg_turns": round(row["avg_turns"], 1),
            "scenarios_evaluated": int(row["total_scenarios"]),
            "judge_agreement": {
                "task_completion": round(row["judge_agreement_tc"], 4),
                "tool_selection": round(row["judge_agreement_ts"], 4),
            },
        }
        leaderboard["models"].append(model_entry)

    return leaderboard


def main():
    df = load_all_results()
    if df.empty:
        logger.warning("No results to aggregate")
        return

    leaderboard = compute_leaderboard(df)

    # Save leaderboard JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard_path = RESULTS_DIR / "leaderboard.json"
    with open(leaderboard_path, "w") as f:
        json.dump(leaderboard, f, indent=2)
    logger.info("Leaderboard saved to %s", leaderboard_path)

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
