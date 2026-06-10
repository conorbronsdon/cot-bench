"""CLI entry point for running COT Bench evaluations."""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eval.artifacts import write_run_artifact
from eval.config import (
    JUDGES,
    MODELS_UNDER_TEST,
    NULL_AGENT_MODEL,
    RELIABILITY_RUNS,
    TOKEN_COSTS,
    Domain,
)
from eval.providers.null_agent import NULL_AGENT_NAME
from eval.providers.registry import ModelSpec
from eval.scoring.judge import score_with_all_judges, score_with_all_judges_combined
from eval.scoring.rubrics import (
    COMBINED_RUBRIC,
    JUDGE_SYSTEM_PROMPT,
    TASK_COMPLETION_RUBRIC,
    TOOL_SELECTION_RUBRIC,
    compute_efficacy,
    compute_reliability,
)
from eval.scoring.state_check import score_state_changes
from eval.simulation.runner import Scenario, SimulationRunner
from eval.tracing import (
    get_tracer,
    init_tracing,
    trace_agent_turn,
    trace_judge_evaluation,
)

# Default directory (relative to the output parquet's parent) that holds
# per-run artifact subtrees: data/results/artifacts/{run_id}/...
ARTIFACTS_DIRNAME = "artifacts"

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
                ground_truth=data.get("ground_truth"),
                expected_state_changes=data.get("expected_state_changes"),
            )
        )
    return scenarios


def format_transcript(turns) -> str:
    """Format conversation turns into a readable transcript for judges.

    Turns are emitted in true conversational order:
    user -> agent (with tool calls) -> tool result(s) -> agent follow-up -> ...
    Tool-call requests are shown on the agent turn that issued them; tool results
    appear as their own TOOL turns immediately after, so judges read calls before
    their results.
    """
    lines = []
    for t in turns:
        prefix = {"user": "USER", "agent": "AGENT", "tool": "TOOL"}.get(t.role, t.role.upper())
        lines.append(f"[Turn {t.turn_number} - {prefix}]: {t.content}")
        for tc in t.tool_calls:
            lines.append(f"  -> Tool Call: {tc.tool_name}({json.dumps(tc.arguments)})")
    return "\n".join(lines)


def _round_or_none(value, ndigits):
    """round() that tolerates None/NaN — returns None instead of crashing.

    Agreement metrics are None when fewer than 2 valid judges scored a row;
    they must survive into the results frame as null rather than raising.
    """
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return round(value, ndigits)


def build_result_row(
    scenario,
    agent_spec,
    sim_result,
    tc_result,
    ts_result,
    efficacy,
    cost_usd,
    state_result=None,
):
    """Assemble the flat results row from simulation + consensus results.

    Pure (no I/O) so it can be unit-tested with faked ConsensusResult objects,
    including the None-agreement degraded paths. Parse-failed judges are kept
    in ``judge_results`` for accounting but excluded from the per-judge score
    columns (a parse failure is not a real 0.0 grade).

    ``state_result`` is the deterministic state-grading result (the dict from
    ``score_state_changes``) or ``None`` for legacy scenarios with no
    ground_truth. The ``state_score`` / ``state_checks_passed`` /
    ``state_checks_total`` columns are floats/ints when present and ``None``
    otherwise (NaN once in a DataFrame), so aggregation must NaN-guard them.
    """
    state_score = None if state_result is None else round(state_result["score"], 4)
    state_checks_passed = None if state_result is None else state_result["n_passed"]
    state_checks_total = None if state_result is None else state_result["n_total"]
    return {
        "scenario_id": scenario.id,
        "domain": scenario.domain.value,
        "category": scenario.category,
        "model": agent_spec.name,
        "efficacy": round(efficacy, 4),
        "task_completion": round(tc_result.consensus_score, 4),
        "tool_selection": round(ts_result.consensus_score, 4),
        "state_score": state_score,
        "state_checks_passed": state_checks_passed,
        "state_checks_total": state_checks_total,
        "cost_usd": round(cost_usd, 6),
        "latency_ms": round(sim_result.total_latency_ms, 1),
        "total_turns": sim_result.total_turns,
        "input_tokens": sim_result.total_input_tokens,
        "output_tokens": sim_result.total_output_tokens,
        "completed": sim_result.completed,
        # Provider-reported model actually served (vs the pinned request id)
        "resolved_model": getattr(sim_result, "resolved_model", None),
        "tc_agreement": _round_or_none(tc_result.agreement_rate, 4),
        "ts_agreement": _round_or_none(ts_result.agreement_rate, 4),
        "tc_max_disagreement": _round_or_none(tc_result.max_disagreement, 4),
        "ts_max_disagreement": _round_or_none(ts_result.max_disagreement, 4),
        "high_disagreement": (
            (tc_result.max_disagreement or 0.0) > 0.3 or (ts_result.max_disagreement or 0.0) > 0.3
        ),
        # Judge-panel accounting (per row, for transparency)
        "tc_n_judges": tc_result.n_judges_valid,
        "ts_n_judges": ts_result.n_judges_valid,
        "tc_parse_failures": len(tc_result.parse_failures),
        "ts_parse_failures": len(ts_result.parse_failures),
        "tc_api_failures": len(tc_result.api_failures),
        "ts_api_failures": len(ts_result.api_failures),
        "tc_degraded": tc_result.degraded,
        "ts_degraded": ts_result.degraded,
        # Per-judge scores for transparency (valid judges only)
        **{
            f"tc_{jr.judge_name}": round(jr.overall_score, 4)
            for jr in tc_result.judge_results
            if not jr.parse_failed
        },
        **{
            f"ts_{jr.judge_name}": round(jr.overall_score, 4)
            for jr in ts_result.judge_results
            if not jr.parse_failed
        },
    }


def _trace_agent_turns(tracer, agent_spec, sim_result):
    """Emit one OpenInference AGENT span per agent turn, post-hoc.

    Iterates the completed simulation's turns rather than instrumenting inside
    the runner — keeps eval/simulation/runner.py untouched. Tool calls on each
    agent turn are passed through so the span records what the agent invoked.
    """
    for turn in sim_result.turns:
        if turn.role != "agent":
            continue
        tool_calls = [{"name": tc.tool_name, "arguments": tc.arguments} for tc in turn.tool_calls]
        trace_agent_turn(
            tracer,
            model_name=agent_spec.name,
            input_text=f"<turn {turn.turn_number}>",
            output_text=turn.content,
            tool_calls=tool_calls or None,
            token_count_output=turn.token_count,
            latency_ms=turn.latency_ms,
        )


def evaluate_scenario(
    runner,
    scenario,
    agent_spec,
    tracer,
    judge_keys,
    run_id,
    run_index,
    artifacts_root=None,
    separate_judge_calls=False,
):
    """Run simulation + multi-judge scoring for one scenario, one model.

    Args:
        run_id: Stem of the output parquet — groups a run's artifacts/traces.
        run_index: Which reliability repeat this is (0-based).
        artifacts_root: Directory under which per-run artifact subtrees are
            written. ``None`` disables artifact persistence (``--no-artifacts``).
        separate_judge_calls: When True, score task completion and tool
            selection with TWO separate judge prompts (the legacy path, kept
            for A/B validation). Default False uses the combined single-prompt
            path — one judge call per judge instead of two, and the transcript
            is sent once instead of twice. Both paths produce identically-shaped
            ``tc_result`` / ``ts_result`` ConsensusResult objects, so everything
            downstream is unaffected.
    """
    sim_result = runner.run(scenario, agent_spec)
    transcript = format_transcript(sim_result.turns)

    # Trace agent turns post-hoc (one AGENT span per agent turn).
    _trace_agent_turns(tracer, agent_spec, sim_result)

    tools_desc = json.dumps(
        [{"name": t.get("name"), "description": t.get("description")} for t in scenario.tools]
    )
    user_goals = "\n".join(f"- {g}" for g in scenario.user_goals)

    if separate_judge_calls:
        # Legacy two-call path (A/B validation). Context + transcript are sent
        # twice — once per dimension.
        tc_prompt = TASK_COMPLETION_RUBRIC.format(
            domain=scenario.domain.value,
            user_goals=user_goals,
            available_tools=tools_desc,
            transcript=transcript,
        )
        tc_result = score_with_all_judges(
            JUDGE_SYSTEM_PROMPT,
            tc_prompt,
            "task_completion",
            scenario.id,
            judge_keys=judge_keys,
        )

        ts_prompt = TOOL_SELECTION_RUBRIC.format(
            domain=scenario.domain.value,
            available_tools=tools_desc,
            transcript=transcript,
        )
        ts_result = score_with_all_judges(
            JUDGE_SYSTEM_PROMPT,
            ts_prompt,
            "tool_selection",
            scenario.id,
            judge_keys=judge_keys,
        )
    else:
        # Combined path (default): one judge call scores both dimensions, with
        # the context + transcript sent once. Returns the same (tc, ts) pair.
        combined_prompt = COMBINED_RUBRIC.format(
            domain=scenario.domain.value,
            user_goals=user_goals,
            available_tools=tools_desc,
            transcript=transcript,
        )
        tc_result, ts_result = score_with_all_judges_combined(
            JUDGE_SYSTEM_PROMPT,
            combined_prompt,
            scenario.id,
            judge_keys=judge_keys,
        )

    # Trace judge evaluations
    for result in [tc_result, ts_result]:
        for jr in result.judge_results:
            trace_judge_evaluation(
                tracer,
                jr.judge_name,
                scenario.id,
                jr.rubric_type,
                jr.overall_score,
                jr.reasoning,
                jr.latency_ms,
            )

    # Deterministic state verification (v0.2). None for legacy scenarios with no
    # ground_truth, in which case Efficacy renormalizes to judge-only 0.5/0.5.
    state_result = score_state_changes(
        scenario.ground_truth,
        sim_result.final_world,
        scenario.expected_state_changes,
    )
    state_score = None if state_result is None else state_result["score"]

    # Compute efficacy (hybrid: judge dimensions + deterministic state, degrading
    # gracefully to 0.5/0.5 when there is no state score).
    efficacy = compute_efficacy(tc_result.consensus_score, ts_result.consensus_score, state_score)

    # Compute cost
    costs = TOKEN_COSTS.get(agent_spec.model_id)
    if costs is None:
        logger.warning(
            "No token costs for %s — cost will be $0. Add pricing to eval/config.py TOKEN_COSTS.",
            agent_spec.model_id,
        )
        costs = {"input": 0, "output": 0}
    cost_usd = (
        sim_result.total_input_tokens * costs["input"] / 1_000_000
        + sim_result.total_output_tokens * costs["output"] / 1_000_000
    )

    # Persist the per-evaluation artifact (transcript + raw judge outputs) so a
    # published score is auditable back to its evidence. Default on; disabled
    # when artifacts_root is None (--no-artifacts).
    if artifacts_root is not None:
        try:
            write_run_artifact(
                artifacts_root,
                run_id,
                scenario.id,
                agent_spec.name,
                run_index,
                sim_result,
                tc_result,
                ts_result,
                state=state_result,
            )
        except OSError:
            logger.exception(
                "Failed to persist artifact for %s / %s run %d",
                agent_spec.name,
                scenario.id,
                run_index,
            )

    return build_result_row(
        scenario,
        agent_spec,
        sim_result,
        tc_result,
        ts_result,
        efficacy,
        cost_usd,
        state_result=state_result,
    )


def _run_model_scenarios(
    model_cfg,
    domains,
    scenarios_by_domain,
    reliability_runs,
    judge_keys,
    tracer,
    run_id,
    artifacts_root=None,
    separate_judge_calls=False,
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
                    runner,
                    scenario,
                    agent_spec,
                    tracer,
                    judge_keys,
                    run_id=run_id,
                    run_index=run_idx,
                    artifacts_root=artifacts_root,
                    separate_judge_calls=separate_judge_calls,
                )
                result["run_index"] = run_idx
                result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
                results.append(result)
                run_scores.append(result["efficacy"])

            reliability = compute_reliability(run_scores)
            for r in results[-reliability_runs:]:
                r["reliability_pass_rate"] = reliability["pass_rate"]
                r["reliability_consistency"] = reliability["consistency"]
                # pass^k (tau-bench): one column per k. The headline is k = the
                # number of runs (all-k-succeed); intermediate k are published
                # too so the autonomy-horizon decay is visible. Aggregation means
                # these per-row, which averages the per-scenario pass^k estimates.
                for k, val in reliability["pass_hat_k"].items():
                    r[f"reliability_pass_hat_{k}"] = val

    return results


def assert_results_nonempty(all_results: list, failed_models: list[str]) -> None:
    """Exit non-zero when an eval run produced zero results.

    Per-model failures are logged and swallowed so one bad provider can't
    sink a whole run — but when EVERY model fails (the classic cause:
    missing API keys), the old behavior wrote an empty parquet and exited
    0, and the failure only surfaced steps later as a confusing
    `git add: pathspec did not match` in CI. Fail here, loudly, instead.
    """
    if all_results:
        if failed_models:
            logger.warning(
                "Partial run: %d model(s) failed and are missing from results: %s",
                len(failed_models),
                ", ".join(sorted(failed_models)),
            )
        return
    raise SystemExit(
        "All model evaluations failed — no results produced. "
        f"Failed models: {', '.join(sorted(failed_models)) or 'none attempted'}. "
        "Check API keys (python -m scripts.preflight) before re-running."
    )


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
        "--include-null-agent",
        action="store_true",
        help=(
            f"Also run the deterministic do-nothing '{NULL_AGENT_NAME}' agent "
            "(anti-gaming validation). It makes no tool calls and gives only a "
            "trivial reply, so the bench should score it near zero on both judges "
            "and the deterministic state checks. It is never on the leaderboard. "
            f"Equivalent to passing --models {NULL_AGENT_NAME}."
        ),
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
        "--scenario-limit",
        type=int,
        default=0,
        help="Limit scenarios per domain (0 = all, useful for quick tests)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output path for results "
            "(default: data/results/results_<UTC timestamp>.parquet, "
            "matching the glob aggregate_results.py expects)"
        ),
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help=(
            "Disable per-run artifact persistence (transcripts + raw judge "
            "outputs). Artifacts are written by default for auditability."
        ),
    )
    parser.add_argument(
        "--separate-judge-calls",
        action="store_true",
        help=(
            "Use the legacy two-call judge path (one call for task completion, "
            "one for tool selection) instead of the default combined single-call "
            "path. Kept for A/B validation of the combined prompt; the combined "
            "path halves judge calls and input tokens with no row-schema change."
        ),
    )
    args = parser.parse_args()

    # Resolve the default per-run so the filename matches the results_*.parquet
    # glob in aggregate_results.py; a static argparse default can't be timestamped.
    if args.output is None:
        timestamp = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        args.output = f"data/results/results_{timestamp}.parquet"

    # run_id groups a run's artifacts + traces; derive it from the output stem
    # so they sit alongside the results parquet they explain.
    output_path = Path(args.output)
    run_id = output_path.stem
    results_dir = output_path.parent
    artifacts_root = None if args.no_artifacts else results_dir / ARTIFACTS_DIRNAME

    # Init tracing. Default the trace dir to <results dir>/traces/{run_id}/ so
    # spans.jsonl is written to disk (real, durable traces) unless an explicit
    # COT_BENCH_TRACE_DIR override is set in the environment.
    trace_dir = os.environ.get("COT_BENCH_TRACE_DIR") or str(results_dir / "traces" / run_id)
    init_tracing(trace_dir=trace_dir)
    tracer = get_tracer()

    # Load scenarios for all requested domains
    scenarios_by_domain: dict[Domain, list[Scenario]] = {}
    for domain_str in args.domains:
        domain = Domain(domain_str)
        scenarios = load_scenarios(domain)
        if scenarios:
            if args.scenario_limit > 0:
                scenarios = scenarios[: args.scenario_limit]
            scenarios_by_domain[domain] = scenarios
            logger.info("Loaded %d scenarios for %s", len(scenarios), domain.value)
        else:
            logger.warning("No scenarios found for %s, skipping", domain.value)

    if not scenarios_by_domain:
        logger.error("No scenarios loaded. Run generate_data.py first.")
        return

    # Filter models. The null agent is NOT in MODELS_UNDER_TEST (so it never
    # runs as a real contestant); inject it only when explicitly requested via
    # --include-null-agent or --models null-agent.
    models = MODELS_UNDER_TEST
    if args.models:
        models = [m for m in models if m["name"] in args.models]
    if args.include_null_agent or (args.models and NULL_AGENT_NAME in args.models):
        if not any(m["name"] == NULL_AGENT_NAME for m in models):
            models = [*models, NULL_AGENT_MODEL]
        logger.info(
            "Including the deterministic '%s' agent (anti-gaming validation; "
            "expected to score near zero, excluded from the leaderboard).",
            NULL_AGENT_NAME,
        )

    # Run models in parallel
    all_results = []
    failed_models: list[str] = []
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
                run_id,
                artifacts_root,
                args.separate_judge_calls,
            ): model_cfg["name"]
            for model_cfg in models
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.info("Completed %s: %d results", model_name, len(results))
            except Exception:
                logger.exception("Failed evaluating %s", model_name)
                failed_models.append(model_name)

    assert_results_nonempty(all_results, failed_models)

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_parquet(output_path, index=False)
    logger.info("Results saved to %s (%d rows)", output_path, len(df))
    if artifacts_root is not None:
        logger.info("Per-run artifacts written under %s", Path(artifacts_root) / run_id)
    logger.info("Traces written to %s", Path(trace_dir) / "spans.jsonl")

    # Write a run manifest next to the parquet output. The downstream publish
    # gate (scripts/check_publish_ready.py) reads models_failed to block a
    # scheduled leaderboard commit that would silently ship missing models.
    # Overwritten each run; uses only data main() already has in scope.
    models_requested = [m["name"] for m in models]
    models_completed = sorted({r["model"] for r in all_results})
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "artifacts_dir": (str(Path(artifacts_root) / run_id) if artifacts_root else None),
        "trace_dir": str(trace_dir),
        "models_requested": models_requested,
        "models_completed": models_completed,
        "models_failed": sorted(failed_models),
        "domains": [d.value for d in scenarios_by_domain],
        "scenario_counts": {
            d.value: len(scenarios) for d, scenarios in scenarios_by_domain.items()
        },
        "reliability_runs": args.reliability_runs,
    }
    manifest_path = output_path.parent / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Run manifest saved to %s", manifest_path)

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
