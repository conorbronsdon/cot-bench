"""CLI entry point for running COT Bench evaluations."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from eval.artifacts import write_run_artifact
from eval.config import (
    DEFAULT_SIMULATION,
    JUDGES,
    MODELS_UNDER_TEST,
    NULL_AGENT_MODEL,
    RELIABILITY_RUNS,
    TOKEN_COSTS,
    Domain,
    SimulationConfig,
)
from eval.cost import (
    BUDGET_EXCEEDED_EXIT_CODE,
    CostAccumulator,
    estimate_run_cost,
    token_cost,
)
from eval.pre_registration import (
    PRE_REGISTRATION_FILENAME,
    build_pre_registration,
    file_sha256,
    holdout_set_hash,
    scenario_set_hash,
    write_pre_registration,
)
from eval.providers.null_agent import NULL_AGENT_NAME
from eval.providers.registry import ModelSpec, infer_provider
from eval.resume import (
    completed_tuples,
    load_pre_registration,
    rows_from_artifacts,
    verify_corpus_unchanged,
)
from eval.scoring.failure_modes import classify_failure, judge_reasoning_text
from eval.scoring.judge import score_with_all_judges, score_with_all_judges_combined
from eval.scoring.rubrics import (
    JUDGE_SYSTEM_PROMPT,
    build_combined_prompt,
    build_task_completion_prompt,
    build_tool_selection_prompt,
    compute_efficacy,
    compute_reliability,
    criteria_for_dimension,
)
from eval.scoring.state_check import score_state_changes
from eval.simulation.probes import RecoveryProbe
from eval.simulation.profiles import DEFAULT_SIM_PROFILE, SIM_PROFILES
from eval.simulation.runner import Scenario, SimulationRunner
from eval.tracing import (
    get_tracer,
    init_tracing,
    trace_agent_turn,
    trace_judge_evaluation,
)
from scripts.aggregate_results import BOOTSTRAP_SEED

# Default directory (relative to the output parquet's parent) that holds
# per-run artifact subtrees: data/results/artifacts/{run_id}/...
ARTIFACTS_DIRNAME = "artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# Env var that points at an EXTERNAL private-holdout scenario tree (issue #31).
# The holdout corpus is authored and stored OUTSIDE this repo and is never
# committed here; this var lets a run pull it in at evaluation time. The flag
# (--holdout-dir) takes precedence over the env var when both are set.
HOLDOUT_DIR_ENV = "COT_BENCH_HOLDOUT_DIR"

# Full installed-package list written alongside the manifest each run (H3). The
# repo floor-pins only (pyproject `>=`) and CI installs with a bare
# `pip install -e .`, so two "identical" runs can resolve different library
# versions; capturing the resolved environment makes which versions a run used an
# auditable fact rather than an assumption. Written next to run_manifest.json and
# uploaded as a CI artifact; the manifest records its sha256 + package count so
# the (committed) trend record can point at the exact environment.
ENV_FREEZE_FILENAME = "env_freeze.txt"


def capture_environment(freeze_path: Path) -> dict:
    """Write a sorted `pip freeze`-equivalent to ``freeze_path``; return a summary.

    Uses :mod:`importlib.metadata` rather than shelling out to ``pip`` so the
    list is the in-process resolved environment, deterministically sorted, and
    independent of whether ``pip`` is on PATH. The returned dict (python version
    + platform + freeze-file sha256 + package count) is embedded in the manifest;
    the full list lives in the file so the manifest stays compact.
    """
    dists = sorted(
        (
            f"{d.metadata['Name']}=={d.version}"
            for d in importlib.metadata.distributions()
            if d.metadata["Name"]
        ),
        key=str.lower,
    )
    body = "\n".join(dists) + ("\n" if dists else "")
    # Write bytes (not write_text) so the on-disk file is byte-for-byte the bytes
    # we hash — write_text would translate "\n" to the platform newline and the
    # recorded sha256 would no longer match the file (it must, so a drifted
    # environment is detectable by comparing the file's hash to the manifest).
    encoded = body.encode("utf-8")
    freeze_path.write_bytes(encoded)
    return {
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "freeze_file": freeze_path.name,
        "freeze_sha256": hashlib.sha256(encoded).hexdigest(),
        "n_packages": len(dists),
    }


def _scenario_from_dict(data: dict, domain: Domain, *, holdout: bool) -> Scenario:
    """Build a Scenario from a loaded JSON dict (shared by both loaders)."""
    return Scenario(
        id=data["id"],
        domain=domain,
        persona=data["persona"],
        user_goals=data["user_goals"],
        tools=data["tools"],
        category=data["category"],
        initial_message=data["initial_message"],
        ground_truth=data.get("ground_truth"),
        expected_state_changes=data.get("expected_state_changes"),
        # Atomic rubric criteria (issue #54); None for scenarios without them.
        rubric_criteria=data.get("rubric_criteria"),
        holdout=holdout,
        # Recovery probe (issue #57); None for scenarios without one (the entire
        # public corpus today). RecoveryProbe.from_dict validates kind/turn/
        # injection at load time, so a malformed probe fails before any run.
        recovery_probe=RecoveryProbe.from_dict(data.get("recovery_probe")),
    )


def load_scenarios(domain: Domain) -> list[Scenario]:
    """Load public scenarios from the in-repo data directory."""
    scenario_dir = Path(f"data/scenarios/{domain.value}")
    if not scenario_dir.exists():
        logger.warning("Scenario directory not found: %s", scenario_dir)
        return []
    scenarios = []
    for path in sorted(scenario_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        scenarios.append(_scenario_from_dict(data, domain, holdout=False))
    return scenarios


def load_holdout_scenarios(holdout_root: Path, domain: Domain) -> list[Scenario]:
    """Load private-holdout scenarios for a domain from an EXTERNAL directory.

    ``holdout_root`` is a directory laid out exactly like ``data/scenarios/``:
    one subdirectory per domain (``{holdout_root}/{domain}/*.json``), each file in
    the same v0.2 scenario schema. Every scenario loaded here is tagged
    ``holdout=True`` so its result rows carry ``holdout: true`` and the
    aggregation can split public vs holdout efficacy.

    Returning ``[]`` for a domain with no holdout subdir is intentional — a
    holdout need not cover every domain; the run simply has no holdout rows for
    the uncovered ones.
    """
    domain_dir = holdout_root / domain.value
    if not domain_dir.exists():
        # Path at DEBUG: CI logs on a public repo are public; don't reveal
        # where the private holdout lives.
        logger.debug("No holdout scenarios for domain %s under %s", domain.value, holdout_root)
        return []
    scenarios = []
    for path in sorted(domain_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        scenarios.append(_scenario_from_dict(data, domain, holdout=True))
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


def _judge_model_id(judge_name: str) -> str | None:
    """Map a judge's display name back to its configured model_id for pricing.

    JudgeResult carries the human-facing judge name (e.g. "Claude Opus 4.6"), but
    TOKEN_COSTS is keyed by model_id. This reverse lookup lets the cost guard price
    judge tokens (issue #47). Returns None for an unknown name (priced at $0).
    """
    for cfg in JUDGES.values():
        if cfg.name == judge_name:
            return cfg.model_id
    return None


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
    # Failure-mode taxonomy (issue #55): classify failed evaluations from signals
    # already on this row — deterministic state-grader / premature-end (#32)
    # evidence first, judge-reasoning keywords as assist, never a new LLM call.
    # Computed HERE (not at aggregation) so the resume path, which replays
    # artifacts through this same builder, classifies identically — the judge
    # reasoning lives in the consensus objects, not in the parquet.
    failure = classify_failure(
        efficacy,
        state_result=state_result,
        premature_end=bool(getattr(sim_result, "premature_end", False)),
        judge_reasoning=judge_reasoning_text(tc_result, ts_result),
    )
    return {
        "scenario_id": scenario.id,
        "domain": scenario.domain.value,
        "category": scenario.category,
        "model": agent_spec.name,
        # Private-holdout flag (issue #31). Every row from an external holdout
        # scenario is marked here so aggregation can compute a public-vs-holdout
        # gap and keep holdout scenario detail off the published per-scenario
        # surfaces. False for the public corpus.
        "holdout": bool(getattr(scenario, "holdout", False)),
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
        # User-sim completion decoupling (#32). ``completed`` no longer implies
        # the goals were met — the user sim only signals it is done talking.
        # ``ended_by`` records the stop cause (user_sim / max_turns / error),
        # ``state_progress_at_end`` the deterministic state-check fraction when
        # the conversation ended, and ``premature_end`` flags a sim that quit
        # before the state check passed. Aggregation means premature_end into a
        # per-model premature-ending rate.
        "ended_by": getattr(sim_result, "ended_by", "max_turns"),
        "state_progress_at_end": getattr(sim_result, "state_progress_at_end", None),
        "premature_end": bool(getattr(sim_result, "premature_end", False)),
        # Failure-mode taxonomy (issue #55). None when the run PASSED (efficacy
        # at/above the reliability pass threshold). ``failure_mode_source``
        # records the classification provenance (state-grader / premature-flag /
        # judge-keyword / fallback) so published failure counts are auditable.
        "failure_mode": None if failure is None else failure["mode"],
        "failure_mode_source": None if failure is None else failure["source"],
        # Provider-reported model actually served (vs the pinned request id)
        "resolved_model": getattr(sim_result, "resolved_model", None),
        # Simulator model ids actually used this run (issue #50). Recorded on every
        # row so a sensitivity-test delta — same agents, different user/tool sim —
        # can be computed by aggregating two runs. resolved_model above covers the
        # AGENT; these cover the simulators.
        "user_sim_model": getattr(sim_result, "user_sim_model", None),
        "tool_sim_model": getattr(sim_result, "tool_sim_model", None),
        # Behavioral user-sim profile (issue #59 part 1). Stamped per row so
        # persona-stratified pass rates are computable and so aggregation can
        # exclude non-cooperative rows from the public leaderboard (mirroring the
        # holdout and null-agent exclusions). Defaults to cooperative for sim
        # results that predate the field.
        "sim_profile": getattr(sim_result, "sim_profile", DEFAULT_SIM_PROFILE),
        # Recovery probe (issue #57). ``recovery_probe_kind`` is the DECLARED
        # probe kind (None for the non-probe majority — the entire public corpus
        # today); ``probe_fired`` records whether the injection was actually
        # delivered as a user turn; ``recovered`` is the deterministic recovery
        # verdict, None unless the probe fired (a conversation that ended before
        # probe.turn saw no fault, so it must not be graded). Aggregation
        # computes a per-model recovery_rate over fired-probe rows ONLY
        # (recovered non-null) and emits it conditionally, so a normal
        # cooperative run with no probe rows is byte-identical downstream;
        # probe_fired makes the rate's denominator auditable on the rows.
        "recovery_probe_kind": getattr(sim_result, "recovery_probe_kind", None),
        "recovered": getattr(sim_result, "recovered", None),
        "probe_fired": bool(getattr(sim_result, "probe_fired", False)),
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

    Returns ``(result_row, eval_cost_usd)`` — the flat results row and the TOTAL
    actual spend for this evaluation (agent + simulators + judges), the latter fed
    to the run's --max-cost accumulator (issue #47). The row's own ``cost_usd``
    column stays agent-only (the published CLEAR Cost dimension).

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

    # Atomic rubric criteria (issue #54). None for scenarios without them, in
    # which case every prompt below is byte-identical to the bare template
    # (asserted by tests) and the judges run exactly as before.
    rubric_criteria = getattr(scenario, "rubric_criteria", None)

    if separate_judge_calls:
        # Legacy two-call path (A/B validation). Context + transcript are sent
        # twice — once per dimension. Each prompt carries only ITS dimension's
        # criteria, and the same filtered list is passed to the judges so the
        # verdict block is validated against exactly what the prompt showed.
        tc_prompt = build_task_completion_prompt(
            domain=scenario.domain.value,
            user_goals=user_goals,
            available_tools=tools_desc,
            transcript=transcript,
            rubric_criteria=rubric_criteria,
        )
        tc_result = score_with_all_judges(
            JUDGE_SYSTEM_PROMPT,
            tc_prompt,
            "task_completion",
            scenario.id,
            judge_keys=judge_keys,
            rubric_criteria=criteria_for_dimension(rubric_criteria or [], "task_completion")
            or None,
        )

        ts_prompt = build_tool_selection_prompt(
            domain=scenario.domain.value,
            available_tools=tools_desc,
            transcript=transcript,
            rubric_criteria=rubric_criteria,
        )
        ts_result = score_with_all_judges(
            JUDGE_SYSTEM_PROMPT,
            ts_prompt,
            "tool_selection",
            scenario.id,
            judge_keys=judge_keys,
            rubric_criteria=criteria_for_dimension(rubric_criteria or [], "tool_selection") or None,
        )
    else:
        # Combined path (default): one judge call scores both dimensions, with
        # the context + transcript sent once. Returns the same (tc, ts) pair.
        # The combined prompt carries ALL criteria; the verdicts are split by
        # dimension inside score_with_judge_combined.
        combined_prompt = build_combined_prompt(
            domain=scenario.domain.value,
            user_goals=user_goals,
            available_tools=tools_desc,
            transcript=transcript,
            rubric_criteria=rubric_criteria,
        )
        tc_result, ts_result = score_with_all_judges_combined(
            JUDGE_SYSTEM_PROMPT,
            combined_prompt,
            scenario.id,
            judge_keys=judge_keys,
            rubric_criteria=rubric_criteria,
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

    # Compute cost. The PUBLISHED row cost (the CLEAR Cost dimension) is the AGENT
    # cost only — what it costs to run the model under test — so the leaderboard
    # number is unchanged by the cost guard. The simulator + judge costs are
    # harness overhead, priced separately below and fed only to the run's
    # actual-spend accumulator (issue #47), never into the row's cost_usd.
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

    # Total actual spend for THIS evaluation = agent + simulators + judges, each
    # priced at its own model id. This is what the --max-cost budget guard tracks.
    # Simulators are priced at the (possibly overridden, issue #50) sim model ids
    # recorded on the result; judges at each judge's configured model id. The
    # combined judge path attributes each call's tokens to the first dimension
    # only, so summing input/output_tokens across all JudgeResults counts each
    # call once.
    sim_cost = token_cost(
        sim_result.user_sim_model, sim_result.sim_input_tokens / 2, sim_result.sim_output_tokens / 2
    ) + token_cost(
        sim_result.tool_sim_model, sim_result.sim_input_tokens / 2, sim_result.sim_output_tokens / 2
    )
    judge_cost = 0.0
    for consensus in (tc_result, ts_result):
        for jr in consensus.judge_results:
            jr_costs = TOKEN_COSTS.get(_judge_model_id(jr.judge_name))
            if jr_costs is None:
                continue
            judge_cost += (
                jr.input_tokens * jr_costs["input"] / 1_000_000
                + jr.output_tokens * jr_costs["output"] / 1_000_000
            )
    eval_cost_usd = cost_usd + sim_cost + judge_cost

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
                # Persist the authoritative taxonomy + holdout flag (issues #46,
                # #31) so calibration stratifies and excludes correctly.
                domain=scenario.domain.value,
                category=scenario.category,
                holdout=bool(getattr(scenario, "holdout", False)),
            )
        except OSError:
            logger.exception(
                "Failed to persist artifact for %s / %s run %d",
                agent_spec.name,
                scenario.id,
                run_index,
            )

    row = build_result_row(
        scenario,
        agent_spec,
        sim_result,
        tc_result,
        ts_result,
        efficacy,
        cost_usd,
        state_result=state_result,
    )
    # Return the row plus the TOTAL actual spend for this evaluation (agent + sims
    # + judges) so the run loop can feed the --max-cost accumulator (issue #47).
    # The row's own cost_usd stays agent-only (the published Cost dimension).
    return row, eval_cost_usd


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
    runner=None,
    sim_config=None,
    cost_acc=None,
    completed_keys=None,
):
    """Evaluate a single model across all domains/scenarios. Runs in a thread.

    ``sim_config`` (issue #50) is the run's SimulationConfig, including any
    user/tool simulator-model overrides; the runner is built from it HERE, in the
    worker thread, so each model owns its own simulator instances and no API client
    is created on the main thread. ``runner`` lets a test inject a pre-built runner
    directly (skipping construction); when both are None a default runner is used.
    ``cost_acc`` is the run's
    shared :class:`CostAccumulator` (issue #47): each completed evaluation adds its
    actual spend, and once the cap is crossed this model stops submitting new
    evaluations (in-flight work already finished, its artifacts persisted). The
    per-evaluation artifact is the checkpoint — the parquet is only written at the
    end — so a budget stop leaves completed work auditable on disk.
    ``completed_keys`` is the set of ``(model, scenario_id, run_index)`` tuples
    already done in a resumed run (issue #48); matching tuples are skipped.
    """
    agent_spec = ModelSpec(
        name=model_cfg["name"],
        model_id=model_cfg["model_id"],
        provider=model_cfg["provider"],
    )
    # Each model gets its own runner (owns its own simulator model instances),
    # built from the run's sim config (sim-model overrides, issue #50) unless a
    # runner was injected directly (tests).
    if runner is None:
        runner = SimulationRunner(config=sim_config)
    completed_keys = completed_keys or set()
    results = []

    for domain, scenarios in scenarios_by_domain.items():
        logger.info("Evaluating %s on %s", agent_spec.name, domain.value)
        for scenario in scenarios:
            # Stop submitting new evaluations once the budget cap is crossed
            # (issue #47). In-flight evaluations for OTHER models keep running in
            # their own threads and finish; this model just stops here.
            if cost_acc is not None and cost_acc.exceeded():
                logger.warning(
                    "Budget cap reached ($%.4f >= $%.2f) — %s stops before %s "
                    "(completed work is persisted to artifacts)",
                    cost_acc.total(),
                    cost_acc.max_cost,
                    agent_spec.name,
                    scenario.id,
                )
                return results
            run_scores = []
            scored_indices = []
            for run_idx in range(reliability_runs):
                # Resume: skip a (model, scenario, run) tuple already completed in
                # the original run (issue #48). Its artifact is reloaded and merged
                # by the caller; we must not re-run (and re-pay for) it.
                if (agent_spec.name, scenario.id, run_idx) in completed_keys:
                    logger.info(
                        "  RESUME skip: %s / %s run %d/%d already completed",
                        agent_spec.name,
                        scenario.id,
                        run_idx + 1,
                        reliability_runs,
                    )
                    continue
                logger.info(
                    "  %s / %s, run %d/%d",
                    agent_spec.name,
                    scenario.id,
                    run_idx + 1,
                    reliability_runs,
                )
                result, eval_cost = evaluate_scenario(
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
                scored_indices.append(len(results) - 1)

                if cost_acc is not None:
                    running = cost_acc.add(agent_spec.name, eval_cost)
                    logger.info(
                        "  running cost: %s $%.4f / total $%.4f",
                        agent_spec.name,
                        cost_acc.model_total(agent_spec.name),
                        running,
                    )

            # Reliability is only computable over the runs scored THIS session;
            # a resumed run reattaches reliability from artifacts on merge, and a
            # fully-skipped scenario has no fresh rows to annotate here.
            if not scored_indices:
                continue
            reliability = compute_reliability(run_scores)
            for idx in scored_indices:
                r = results[idx]
                r["reliability_pass_rate"] = reliability["pass_rate"]
                r["reliability_consistency"] = reliability["consistency"]
                # pass^k (tau-bench): one column per k. The headline is k = the
                # number of runs (all-k-succeed); intermediate k are published
                # too so the autonomy-horizon decay is visible. Aggregation means
                # these per-row, which averages the per-scenario pass^k estimates.
                for k, val in reliability["pass_hat_k"].items():
                    r[f"reliability_pass_hat_{k}"] = val

    return results


def _recompute_reliability(rows: list[dict], reliability_runs: int) -> None:
    """Recompute reliability columns across a MERGED row set, in place (issue #48).

    On resume, a scenario's reliability runs can be split between the original
    session (reconstructed from artifacts) and the resumed session. Per-session
    reliability would then be computed over a partial set of runs. This regroups
    ALL rows by (model, scenario_id) and recomputes the reliability columns over
    every run of that scenario, so the merged parquet carries one honest
    reliability figure per (model, scenario) rather than two partial ones.

    Grouping is by (model, scenario_id) only — domain/category are scenario
    properties, so they are constant within a scenario id.
    """
    from collections import defaultdict

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["scenario_id"])].append(r)

    for group_rows in groups.values():
        run_scores = [r["efficacy"] for r in group_rows]
        reliability = compute_reliability(run_scores)
        for r in group_rows:
            r["reliability_pass_rate"] = reliability["pass_rate"]
            r["reliability_consistency"] = reliability["consistency"]
            for k, val in reliability["pass_hat_k"].items():
                r[f"reliability_pass_hat_{k}"] = val


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
        "--holdout-dir",
        type=str,
        default=None,
        help=(
            "Path to an EXTERNAL private-holdout scenario tree (issue #31), laid "
            "out like data/scenarios/ (one subdir per domain). Scenarios loaded "
            "from here are run alongside the public corpus, marked holdout=true in "
            "results, and produce a public-vs-holdout gap on the leaderboard. The "
            "holdout content is never stored in this repo; only its corpus hash "
            f"and count are pre-registered. Falls back to the {HOLDOUT_DIR_ENV} "
            "environment variable when the flag is omitted."
        ),
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
        "--resume",
        type=str,
        default=None,
        metavar="RUN_ID",
        help=(
            "Resume an interrupted run by its run_id (the results parquet stem, "
            "e.g. results_20260610_120000). Completed (model, scenario, run) "
            "evaluations are read from that run's artifact dir and skipped — never "
            "paid for twice — and their rows are merged with the new ones into the "
            "final parquet. The run continues under the ORIGINAL pre_registration "
            "(no new one is written); if the current scenario set no longer matches "
            "the corpus hash recorded there, the resume ABORTS (governance §3). "
            "Issue #48."
        ),
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help=(
            "Abort gracefully once actual accumulated spend (agent + simulators + "
            "judges, in USD) reaches this cap. In-flight evaluations finish and all "
            "artifacts/parquet/manifest for completed work are written; the process "
            f"then exits with code {BUDGET_EXCEEDED_EXIT_CODE}. Default: no cap "
            "(a rehearsal sets one). Issue #47."
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
        "--user-sim-model",
        type=str,
        default=None,
        help=(
            "Override the USER simulator model id for this run (default: "
            f"SimulationConfig.user_simulator_model = {DEFAULT_SIMULATION.user_simulator_model}). "
            "Provider is inferred from the id and routed through the registry (e.g. "
            "a claude-* id -> anthropic). Recorded in pre_registration.json and on "
            "result rows so a sensitivity-test delta is attributable. Issue #50."
        ),
    )
    parser.add_argument(
        "--tool-sim-model",
        type=str,
        default=None,
        help=(
            "Override the TOOL simulator model id for this run (default: "
            f"SimulationConfig.tool_simulator_model = {DEFAULT_SIMULATION.tool_simulator_model}). "
            "Provider inferred + registry-routed like --user-sim-model. Issue #50."
        ),
    )
    parser.add_argument(
        "--sim-profile",
        type=str,
        choices=sorted(SIM_PROFILES),
        default=DEFAULT_SIM_PROFILE,
        help=(
            "Behavioral profile for the USER simulator (issue #59 part 1). The "
            f"default ('{DEFAULT_SIM_PROFILE}') appends nothing to the sim prompt "
            "— byte-identical to pre-profile behavior. Non-cooperative profiles "
            "(impatient / technically-confused / adversarial) layer behavioral "
            "exemplars onto the same persona and goals. The profile is recorded "
            "in pre_registration.json, the run manifest, and per result row "
            "(sim_profile column); rows from non-cooperative profiles are "
            "EXCLUDED from the public leaderboard aggregates and feed the "
            "persona-stratified robustness table instead."
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

    # Resolve the output path / run_id. On --resume the run_id is FIXED to the
    # original run so artifacts, pre_registration.json, and the final parquet line
    # up with the run being continued; the default output path is then that run's
    # parquet (overwritten with the merged old+new rows). Otherwise the default is
    # a fresh timestamped name matching the results_*.parquet glob in
    # aggregate_results.py (a static argparse default can't be timestamped).
    if args.resume:
        if args.no_artifacts:
            raise SystemExit(
                "--resume needs the original run's per-evaluation artifacts to know "
                "what is already done, but --no-artifacts disables them. Re-run "
                "without --no-artifacts."
            )
        if args.output is None:
            args.output = f"data/results/{args.resume}.parquet"
    elif args.output is None:
        timestamp = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
        args.output = f"data/results/results_{timestamp}.parquet"

    # run_id groups a run's artifacts + traces; derive it from the output stem
    # so they sit alongside the results parquet they explain. On resume it is the
    # original run_id by construction.
    output_path = Path(args.output)
    run_id = args.resume if args.resume else output_path.stem
    results_dir = output_path.parent
    artifacts_root = None if args.no_artifacts else results_dir / ARTIFACTS_DIRNAME

    # Init tracing. Default the trace dir to <results dir>/traces/{run_id}/ so
    # spans.jsonl is written to disk (real, durable traces) unless an explicit
    # COT_BENCH_TRACE_DIR override is set in the environment.
    trace_dir = os.environ.get("COT_BENCH_TRACE_DIR") or str(results_dir / "traces" / run_id)
    init_tracing(trace_dir=trace_dir)
    tracer = get_tracer()

    # Load PUBLIC scenarios for all requested domains. Kept separate from the
    # holdout set (below) so the pre-registration hashes the public corpus on its
    # own — its scenario index is published, the holdout's is not.
    public_by_domain: dict[Domain, list[Scenario]] = {}
    for domain_str in args.domains:
        domain = Domain(domain_str)
        scenarios = load_scenarios(domain)
        if scenarios:
            if args.scenario_limit > 0:
                scenarios = scenarios[: args.scenario_limit]
            public_by_domain[domain] = scenarios
            logger.info("Loaded %d scenarios for %s", len(scenarios), domain.value)
        else:
            logger.warning("No scenarios found for %s, skipping", domain.value)

    if not public_by_domain:
        logger.error("No scenarios loaded. Run generate_data.py first.")
        return

    # Private holdout (issue #31): an EXTERNAL scenario tree, never stored in this
    # repo, run alongside the public corpus so a public-vs-holdout efficacy gap
    # acts as an overfitting tripwire. Resolved from --holdout-dir or the
    # COT_BENCH_HOLDOUT_DIR env var. Loaded into a SEPARATE structure so the
    # pre-registration can hash it as its own set (hash + count only — never the
    # IDs or content).
    holdout_dir = args.holdout_dir or os.environ.get(HOLDOUT_DIR_ENV)
    holdout_by_domain: dict[Domain, list[Scenario]] = {}
    if holdout_dir:
        holdout_root = Path(holdout_dir)
        if not holdout_root.exists():
            raise SystemExit(
                f"Holdout directory not found: {holdout_root}. Unset {HOLDOUT_DIR_ENV} / "
                "--holdout-dir or point it at the external holdout scenario tree."
            )
        for domain_str in args.domains:
            domain = Domain(domain_str)
            held = load_holdout_scenarios(holdout_root, domain)
            if not held:
                continue
            if args.scenario_limit > 0:
                held = held[: args.scenario_limit]
            holdout_by_domain[domain] = held
            logger.info("Loaded %d HOLDOUT scenarios for %s", len(held), domain.value)
        if not holdout_by_domain:
            # No path in the message: public CI logs must not reveal where the
            # private holdout lives.
            logger.warning(
                "Holdout dir set but no holdout scenarios matched the requested "
                "domains; running public corpus only."
            )

    # The run loop evaluates the PUBLIC corpus and the holdout together. Each
    # holdout scenario already carries holdout=True, so its result rows are tagged
    # for the public-vs-holdout split downstream; the two sets are hashed
    # separately in the pre-registration.
    scenarios_by_domain: dict[Domain, list[Scenario]] = {}
    for domain in (*public_by_domain, *holdout_by_domain):
        if domain in scenarios_by_domain:
            continue
        scenarios_by_domain[domain] = [
            *public_by_domain.get(domain, []),
            *holdout_by_domain.get(domain, []),
        ]

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

    # Resolve the simulation config for this run (issue #50). The simulator models
    # default to SimulationConfig (the single source of those defaults); the CLI
    # overrides swap one or both for the sensitivity test. The provider for an
    # overridden sim is inferred from its model id and routed through the existing
    # registry (a default id keeps provider "openai"). The resolved ids/providers
    # flow into the pre-registration and onto result rows.
    sim_config = SimulationConfig(
        max_turns=DEFAULT_SIMULATION.max_turns,
        user_simulator_model=args.user_sim_model or DEFAULT_SIMULATION.user_simulator_model,
        tool_simulator_model=args.tool_sim_model or DEFAULT_SIMULATION.tool_simulator_model,
        user_simulator_temperature=DEFAULT_SIMULATION.user_simulator_temperature,
        tool_simulator_temperature=DEFAULT_SIMULATION.tool_simulator_temperature,
        user_simulator_provider=(
            infer_provider(args.user_sim_model)
            if args.user_sim_model
            else DEFAULT_SIMULATION.user_simulator_provider
        ),
        tool_simulator_provider=(
            infer_provider(args.tool_sim_model)
            if args.tool_sim_model
            else DEFAULT_SIMULATION.tool_simulator_provider
        ),
        user_sim_profile=args.sim_profile,
    )
    if args.user_sim_model or args.tool_sim_model:
        logger.info(
            "Sim-model overrides (issue #50): user=%s (%s), tool=%s (%s)",
            sim_config.user_simulator_model,
            sim_config.user_simulator_provider,
            sim_config.tool_simulator_model,
            sim_config.tool_simulator_provider,
        )
    if sim_config.user_sim_profile != DEFAULT_SIM_PROFILE:
        logger.info(
            "Behavioral sim profile (issue #59): %s — result rows will be tagged "
            "sim_profile=%s and EXCLUDED from public leaderboard aggregates.",
            sim_config.user_sim_profile,
            sim_config.user_sim_profile,
        )

    # Pre-registration (issue #38): commit the run's definition to disk BEFORE
    # the first agent/simulator/judge call. This is what makes it a real
    # pre-registration rather than the post-hoc run_manifest.json below — the
    # run's models, exact scenario set (corpus sha256), judge panel, and
    # seeds/temps are fixed on disk before any number is known, so the maintainer
    # cannot retroactively choose which run "counts". The post-run manifest links
    # back to this file by path + hash. Agent under test runs at ModelSpec's
    # default temperature (0.0); simulator temps come from DEFAULT_SIMULATION.
    # The public set is pre-registered WITH its scenario index (IDs + per-scenario
    # digests). The holdout set (issue #31) is pre-registered with its corpus hash
    # and count ONLY — no IDs, no index — so the holdout is pinned (tamper-evident)
    # without revealing its content. ``holdout_by_domain`` is None when no holdout
    # was requested, which omits the holdout_set block entirely.
    pre_registration_path = results_dir / PRE_REGISTRATION_FILENAME
    if args.resume:
        # RESUME (issue #48): do NOT write a new pre-registration — the run
        # continues under the one written before the ORIGINAL run (governance §3).
        # Verify the current scenario set still matches the corpus hash recorded
        # there; abort on any drift so a resume cannot silently mix two run
        # definitions.
        pre_registration = load_pre_registration(results_dir, PRE_REGISTRATION_FILENAME)
        current_public_hash, _ = scenario_set_hash(public_by_domain)
        current_holdout_hash = holdout_set_hash(holdout_by_domain)[0] if holdout_by_domain else None
        verify_corpus_unchanged(
            pre_registration,
            current_public_hash=current_public_hash,
            current_holdout_hash=current_holdout_hash,
        )
        logger.info(
            "RESUME %s: corpus hash matches the original pre-registration; "
            "continuing under it (no new pre-registration written).",
            run_id,
        )
    else:
        pre_registration = build_pre_registration(
            run_id=run_id,
            models=models,
            scenarios_by_domain=public_by_domain,
            holdout_by_domain=holdout_by_domain or None,
            judges=JUDGES,
            judge_keys=args.judges,
            reliability_runs=args.reliability_runs,
            bootstrap_seed=BOOTSTRAP_SEED,
            agent_temperature=ModelSpec.temperature,
            user_simulator_temperature=DEFAULT_SIMULATION.user_simulator_temperature,
            tool_simulator_temperature=DEFAULT_SIMULATION.tool_simulator_temperature,
            user_simulator_model=sim_config.user_simulator_model,
            tool_simulator_model=sim_config.tool_simulator_model,
            user_sim_profile=sim_config.user_sim_profile,
            separate_judge_calls=args.separate_judge_calls,
            artifacts_dir=(str(Path(artifacts_root) / run_id) if artifacts_root else None),
            trace_dir=str(trace_dir),
        )
        pre_registration_path = write_pre_registration(results_dir, pre_registration)
        logger.info("Pre-registration written to %s (before any model call)", pre_registration_path)

    # Preflight cost ESTIMATE (issue #47): print the expected spend from the
    # resolved roster + per-eval token priors BEFORE the first call, so a run with
    # a tight budget can be sized up front. Total scenarios = public + holdout
    # across all evaluated domains (one evaluation per model x scenario x
    # reliability run). Conservative priors -> an over-stated, safe estimate.
    n_scenarios = sum(len(s) for s in scenarios_by_domain.values())
    estimate = estimate_run_cost(
        models=models,
        n_scenarios=n_scenarios,
        reliability_runs=args.reliability_runs,
        judge_keys=args.judges,
        user_sim_model_id=sim_config.user_simulator_model,
        tool_sim_model_id=sim_config.tool_simulator_model,
        separate_judge_calls=args.separate_judge_calls,
    )
    logger.info(
        "Cost ESTIMATE (priors, before any call): $%.2f total over %d evaluations "
        "[agent $%.2f / sims $%.2f / judges $%.2f]",
        estimate["total_usd"],
        estimate["n_evals_total"],
        estimate["agent_total_usd"],
        estimate["sim_total_usd"],
        estimate["judge_total_usd"],
    )
    if args.max_cost is not None:
        logger.info("Budget cap (--max-cost): $%.2f", args.max_cost)
        if estimate["total_usd"] > args.max_cost:
            logger.warning(
                "Estimated cost $%.2f EXCEEDS the cap $%.2f — the run will stop "
                "early once actual spend crosses the cap.",
                estimate["total_usd"],
                args.max_cost,
            )

    # Shared, thread-safe actual-spend accumulator for the run (issue #47).
    cost_acc = CostAccumulator(max_cost=args.max_cost)

    # Resume (issue #48): the set of (model, scenario_id, run_index) already
    # completed in the original run, read from its artifact dir. Empty for a fresh
    # run. These tuples are skipped (never re-paid) and their rows are merged back
    # in after the loop.
    completed_keys: set = set()
    model_names = [m["name"] for m in models]
    if args.resume:
        completed_keys = completed_tuples(artifacts_root, run_id, model_names)
        logger.info(
            "RESUME %s: %d completed evaluation(s) found in artifacts; they will be "
            "skipped and merged.",
            run_id,
            len(completed_keys),
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
                # Sim config (issue #50) is passed, not a built runner: each model
                # builds its OWN runner inside the worker thread from this config
                # (owning its simulator instances + per-run sim token counters), so
                # the overrides apply per model and no API client is created on the
                # main thread (keeps the stubbed tests offline).
                sim_config=sim_config,
                cost_acc=cost_acc,
                completed_keys=completed_keys,
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

    budget_stopped = cost_acc.exceeded()
    if budget_stopped:
        logger.warning(
            "BUDGET STOP: actual spend $%.4f reached cap $%.2f. Writing artifacts/"
            "parquet/manifest for completed work, then exiting %d.",
            cost_acc.total(),
            args.max_cost,
            BUDGET_EXCEEDED_EXIT_CODE,
        )

    # Resume merge (issue #48): reconstruct rows for the evaluations completed in
    # the original run from their artifacts and combine with the freshly-run rows.
    # Reliability is then recomputed across the MERGED set per (model, scenario),
    # so a scenario whose runs span the original + resumed sessions gets one
    # correct reliability figure instead of two partial ones.
    n_resumed_rows = 0
    if args.resume and completed_keys:
        resumed_rows = rows_from_artifacts(artifacts_root, run_id, model_names)
        n_resumed_rows = len(resumed_rows)
        all_results.extend(resumed_rows)
        _recompute_reliability(all_results, args.reliability_runs)
        logger.info(
            "RESUME %s: merged %d reconstructed row(s) from artifacts with %d new "
            "row(s); reliability recomputed across the merged set.",
            run_id,
            n_resumed_rows,
            len(all_results) - n_resumed_rows,
        )

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
    # Capture the resolved environment (H3): full package list to a sibling file,
    # summary (python version + sha256 + count) into the manifest below. Capture
    # failure must never lose the manifest of a completed (paid) run — degrade
    # to an honest marker instead.
    try:
        environment = capture_environment(output_path.parent / ENV_FREEZE_FILENAME)
    except Exception as exc:
        logger.warning("Environment capture failed: %s", exc)
        environment = {"capture_failed": str(exc)}
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        # Resume accounting (issue #48): True when this run continued an earlier
        # one under its original pre-registration. ``resumed_at`` records when the
        # resume happened (distinct from the original run's pre-registration
        # timestamp), and ``resumed_rows`` is how many completed evaluations were
        # merged back from artifacts rather than re-run.
        "resumed": bool(args.resume),
        "resumed_at": datetime.now(timezone.utc).isoformat() if args.resume else None,
        "resumed_rows": n_resumed_rows,
        "artifacts_dir": (str(Path(artifacts_root) / run_id) if artifacts_root else None),
        "trace_dir": str(trace_dir),
        "models_requested": models_requested,
        "models_completed": models_completed,
        "models_failed": sorted(failed_models),
        "domains": [d.value for d in public_by_domain],
        # PUBLIC scenario counts only — the holdout count is reported separately
        # below (count is publishable; the holdout's per-domain breakdown stays
        # coarse to avoid hinting at its composition).
        "scenario_counts": {d.value: len(scenarios) for d, scenarios in public_by_domain.items()},
        "reliability_runs": args.reliability_runs,
        # Judge panel used for this run (H2). The publish gate
        # (scripts/check_publish_ready.py) blocks a scheduled commit when the
        # panel is not the full default roster, because a single-judge board has
        # a different (and uncomparable) consensus than a leaderboard run.
        # ``requested`` is the panel keys the run was invoked with; ``resolved``
        # is the configured model name per key. The model a provider ACTUALLY
        # served (resolved_model) is knowable only at call time and is recorded
        # per call in the artifacts (governance §2) — the manifest records the
        # configured panel so the gate has something to check without the
        # artifacts.
        "judges": {
            "requested": list(args.judges),
            "resolved": [JUDGES[key].name for key in args.judges],
        },
        # Resolved environment (H3). The repo floor-pins dependencies and CI does
        # a bare `pip install -e .`, so the exact library versions a run used were
        # previously unrecorded. The full `pip freeze`-equivalent is written to
        # ``environment.freeze_file`` next to this manifest; here we keep the
        # python version/platform plus the freeze file's sha256 and package count
        # so the environment is at least recorded (and tamper-evident) per run.
        "environment": environment,
        # Behavioral user-sim profile for this run (issue #59 part 1). Mirrors the
        # value pre-registered in seeds_and_temperatures and stamped per row, so
        # the completion record states which behavioral condition the run was —
        # a non-cooperative run must never be mistaken for a leaderboard run.
        "sim_profile": sim_config.user_sim_profile,
        # Cost guard accounting (issue #47): the preflight estimate, the measured
        # actual spend (agent + simulators + judges) and its per-model breakdown,
        # the cap, and whether the run stopped because the cap was hit. Lets a
        # rehearsal reconcile estimate vs. actual and confirm a budget stop.
        "cost": {
            "estimate_usd": round(estimate["total_usd"], 6),
            "actual_usd": round(cost_acc.total(), 6),
            "actual_by_model": {k: round(v, 6) for k, v in cost_acc.by_model().items()},
            "max_cost_usd": args.max_cost,
            "budget_stopped": budget_stopped,
        },
        # Link the completion record back to the pre-registration written before
        # any model call (issue #38), so the pair is verifiable: the path and a
        # sha256 of the exact pre-registration file. The corpus_sha256 is lifted
        # from the pre-registration so the scenario set a run committed to is
        # visible in the manifest too.
        "pre_registration": {
            "file": PRE_REGISTRATION_FILENAME,
            "path": str(pre_registration_path),
            "sha256": file_sha256(pre_registration_path),
            "corpus_sha256": pre_registration["scenario_set"]["sha256"],
        },
        # Private holdout (issue #31): hash + count ONLY (no IDs), mirroring the
        # pre-registration. None when no holdout ran. The run_manifest.json is a
        # local/CI artifact (not committed to the repo), but it still records no
        # holdout content so even the artifact never carries the held-out set.
        "holdout": (
            {
                "corpus_sha256": pre_registration["holdout_set"]["sha256"],
                "n_scenarios": pre_registration["holdout_set"]["n_scenarios"],
            }
            if pre_registration.get("holdout_set")
            else None
        ),
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

    # Distinct exit ONLY after all artifacts/parquet/manifest/CSV are on disk, so
    # a budget stop still leaves completed work fully persisted and auditable
    # (issue #47). A wrapper/CI can tell this apart from a clean finish (0) or a
    # crash (non-zero from an exception/SystemExit elsewhere).
    if budget_stopped:
        raise SystemExit(BUDGET_EXCEEDED_EXIT_CODE)


if __name__ == "__main__":
    main()
