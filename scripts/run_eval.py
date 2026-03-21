"""CLI entry point for running COT Bench evaluations."""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from eval.config import (
    JUDGES,
    MODELS_UNDER_TEST,
    RELIABILITY_RUNS,
    TOKEN_COSTS,
    Domain,
)
from eval.providers.registry import ModelSpec
from eval.scoring.judge import score_with_all_judges
from eval.scoring.rubrics import (
    EFFICACY_WEIGHTS,
    JUDGE_SYSTEM_PROMPT,
    TASK_COMPLETION_RUBRIC,
    TOOL_SELECTION_RUBRIC,
    compute_reliability,
)
from eval.simulation.runner import Scenario, SimulationRunner
from eval.tracing import get_tracer, init_tracing, trace_judge_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_scenarios(domain: Domain) -> list[Scenario]:
    """Load scenarios from data directory."""
    scenario_dir = Path(f"data/scenarios/{domain.value}")
    if not scenario_dir.exists():
        logger.warning("Scenario directory not found: %s", scenario_dir)
        return []
    scenarios = []
    for path in sorted(scenario_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        scenarios.append(
            Scenario(
                id=data["id"],
                domain=domain,
                persona=data["persona"],
                user_goals=data["user_goals"],
                tools=data["tools"],
                category=data["category"],
                initial_message=data["initial_message"],
            )
        )
    return scenarios


def format_transcript(turns) -> str:
    """Format conversation turns into a readable transcript for judges."""
    lines = []
    for t in turns:
        prefix = {"user": "USER", "agent": "AGENT", "tool": "TOOL"}.get(
            t.role, t.role.upper()
        )
        lines.append(f"[Turn {t.turn_number} - {prefix}]: {t.content}")
        for tc in t.tool_calls:
            lines.append(
                f"  -> Tool Call: {tc.tool_name}({json.dumps(tc.arguments)})"
            )
            if tc.result:
                lines.append(f"  <- Tool Result: {tc.result[:500]}")
    return "\n".join(lines)


def evaluate_scenario(runner, scenario, agent_spec, tracer, judge_keys):
    """Run simulation + multi-judge scoring for one scenario, one model."""
    sim_result = runner.run(scenario, agent_spec)
    transcript = format_transcript(sim_result.turns)

    tools_desc = json.dumps([
        {"name": t.get("name"), "description": t.get("description")}
        for t in scenario.tools
    ])

    # Score: Task Completion
    tc_prompt = TASK_COMPLETION_RUBRIC.format(
        domain=scenario.domain.value,
        user_goals="\n".join(f"- {g}" for g in scenario.user_goals),
        available_tools=tools_desc,
        transcript=transcript,
    )
    tc_result = score_with_all_judges(
        JUDGE_SYSTEM_PROMPT, tc_prompt, "task_completion", scenario.id,
        judge_keys=judge_keys,
    )

    # Score: Tool Selection
    ts_prompt = TOOL_SELECTION_RUBRIC.format(
        domain=scenario.domain.value,
        available_tools=tools_desc,
        transcript=transcript,
    )
    ts_result = score_with_all_judges(
        JUDGE_SYSTEM_PROMPT, ts_prompt, "tool_selection", scenario.id,
        judge_keys=judge_keys,
    )

    # Trace judge evaluations
    for result in [tc_result, ts_result]:
        for jr in result.judge_results:
            trace_judge_evaluation(
                tracer, jr.judge_name, scenario.id, jr.rubric_type,
                jr.overall_score, jr.reasoning, jr.latency_ms,
            )

    # Compute efficacy (weighted combination)
    efficacy = (
        tc_result.consensus_score * EFFICACY_WEIGHTS["task_completion"]
        + ts_result.consensus_score * EFFICACY_WEIGHTS["tool_selection"]
    )

    # Compute cost
    costs = TOKEN_COSTS.get(agent_spec.model_id, {"input": 0, "output": 0})
    cost_usd = (
        sim_result.total_input_tokens * costs["input"] / 1_000_000
        + sim_result.total_output_tokens * costs["output"] / 1_000_000
    )

    return {
        "scenario_id": scenario.id,
        "domain": scenario.domain.value,
        "category": scenario.category,
        "model": agent_spec.name,
        "efficacy": round(efficacy, 4),
        "task_completion": round(tc_result.consensus_score, 4),
        "tool_selection": round(ts_result.consensus_score, 4),
        "cost_usd": round(cost_usd, 6),
        "latency_ms": round(sim_result.total_latency_ms, 1),
        "total_turns": sim_result.total_turns,
        "input_tokens": sim_result.total_input_tokens,
        "output_tokens": sim_result.total_output_tokens,
        "completed": sim_result.completed,
        "tc_agreement": round(tc_result.agreement_rate, 4),
        "ts_agreement": round(ts_result.agreement_rate, 4),
        # Per-judge scores for transparency
        **{
            f"tc_{jr.judge_name}": round(jr.overall_score, 4)
            for jr in tc_result.judge_results
        },
        **{
            f"ts_{jr.judge_name}": round(jr.overall_score, 4)
            for jr in ts_result.judge_results
        },
    }


def _run_model_scenarios(
    model_cfg, domains, scenarios_by_domain, reliability_runs, judge_keys, tracer
):
    """Evaluate a single model across all domains/scenarios. Runs in a thread."""
    agent_spec = ModelSpec(
        name=model_cfg["name"],
        model_id=model_cfg["model_id"],
        provider=model_cfg["provider"],
    )
    # Each model gets its own runner (owns its own simulator model instances)
    runner = SimulationRunner()
    results = []

    for domain, scenarios in scenarios_by_domain.items():
        logger.info("Evaluating %s on %s", agent_spec.name, domain.value)
        for scenario in scenarios:
            run_scores = []
            for run_idx in range(reliability_runs):
                logger.info(
                    "  %s / %s, run %d/%d",
                    agent_spec.name,
                    scenario.id,
                    run_idx + 1,
                    reliability_runs,
                )
                result = evaluate_scenario(
                    runner, scenario, agent_spec, tracer, judge_keys
                )
                result["run_index"] = run_idx
                results.append(result)
                run_scores.append(result["efficacy"])

            reliability = compute_reliability(run_scores)
            for r in results[-reliability_runs:]:
                r["reliability_pass_rate"] = reliability["pass_rate"]
                r["reliability_consistency"] = reliability["consistency"]

    return results


def main():
    parser = argparse.ArgumentParser(description="COT Bench — Agent Evaluation")
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=[d.value for d in Domain],
        default=[d.value for d in Domain],
        help="Domains to evaluate",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to evaluate (default: all)",
    )
    parser.add_argument(
        "--judges",
        nargs="+",
        choices=list(JUDGES.keys()),
        default=list(JUDGES.keys()),
        help="Judges to use",
    )
    parser.add_argument(
        "--reliability-runs",
        type=int,
        default=RELIABILITY_RUNS,
        help="Number of repeated runs for reliability scoring",
    )
    parser.add_argument(
        "--parallel-models",
        type=int,
        default=2,
        help="Number of models to evaluate concurrently",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/results/results.parquet",
        help="Output path for results",
    )
    args = parser.parse_args()

    # Init tracing
    init_tracing()
    tracer = get_tracer()

    # Load scenarios for all requested domains
    scenarios_by_domain: dict[Domain, list[Scenario]] = {}
    for domain_str in args.domains:
        domain = Domain(domain_str)
        scenarios = load_scenarios(domain)
        if scenarios:
            scenarios_by_domain[domain] = scenarios
            logger.info("Loaded %d scenarios for %s", len(scenarios), domain.value)
        else:
            logger.warning("No scenarios found for %s, skipping", domain.value)

    if not scenarios_by_domain:
        logger.error("No scenarios loaded. Run generate_data.py first.")
        return

    # Filter models
    models = MODELS_UNDER_TEST
    if args.models:
        models = [m for m in models if m["name"] in args.models]

    # Run models in parallel
    all_results = []
    with ThreadPoolExecutor(max_workers=args.parallel_models) as executor:
        futures = {
            executor.submit(
                _run_model_scenarios,
                model_cfg,
                args.domains,
                scenarios_by_domain,
                args.reliability_runs,
                args.judges,
                tracer,
            ): model_cfg["name"]
            for model_cfg in models
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(
                    "Completed %s: %d results", model_name, len(results)
                )
            except Exception:
                logger.exception("Failed evaluating %s", model_name)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_parquet(output_path, index=False)
    logger.info("Results saved to %s (%d rows)", output_path, len(df))

    # Also save CSV for human readability
    csv_path = output_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
    logger.info("CSV saved to %s", csv_path)

    # Print summary
    if len(df) > 0:
        summary = (
            df.groupby("model")
            .agg(
                efficacy=("efficacy", "mean"),
                cost=("cost_usd", "mean"),
                latency=("latency_ms", "mean"),
                reliability=("reliability_pass_rate", "mean"),
            )
            .sort_values("efficacy", ascending=False)
        )
        print("\n=== COT Bench Results ===\n")
        print(summary.to_string())


if __name__ == "__main__":
    main()
