"""Per-run artifact persistence for COT Bench.

The leaderboard's pitch is "publish every score so you can verify our work."
A flat results row carries scores, token counts, and per-judge numbers — but
not the *conversation* those scores were derived from, nor the raw judge
outputs. Without those, a published score is unauditable: nobody can see WHY a
model got the grade it did.

This module serializes, per (scenario, model, run_index), the full conversation
transcript and every judge's raw output (including reasoning and the raw parsed
response) to a JSON file, so any published number can be traced back to its
evidence.

Layout::

    data/results/artifacts/{run_id}/{model-slug}/{scenario_id}_run{run_index}.json

One JSON file per evaluation. ``run_id`` is the stem of the output parquet so a
run's artifacts sit alongside the results they explain.
"""

import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from eval.simulation.profiles import DEFAULT_SIM_PROFILE


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for a model name.

    Lowercase, with every run of non-alphanumeric characters collapsed to a
    single hyphen and leading/trailing hyphens stripped::

        "GPT-4.1"            -> "gpt-4-1"
        "Claude Sonnet 4.6"  -> "claude-sonnet-4-6"
    """
    slug = re.sub(r"[^a-z0-9]+", "-", model_name.lower())
    return slug.strip("-")


def _serialize_transcript(turns) -> list[dict]:
    """Serialize ConversationTurn objects (with nested ToolCall) to dicts.

    ``dataclasses.asdict`` recurses into the nested ``tool_calls`` list, so the
    full set of fields — content, role, turn numbers, tool call/result/ids,
    latency, token counts — is preserved.
    """
    return [asdict(turn) for turn in turns]


def _serialize_judges(consensus_result) -> list[dict]:
    """Serialize every JudgeResult in a ConsensusResult for audit.

    Includes parse-failed judges (kept for transparency) and the raw parsed
    response, so a reader can see exactly what each judge returned.
    """
    return [
        {
            "judge_name": jr.judge_name,
            "rubric_type": jr.rubric_type,
            "overall_score": jr.overall_score,
            "reasoning": jr.reasoning,
            "parse_failed": jr.parse_failed,
            "resolved_model": getattr(jr, "resolved_model", ""),
            # Atomic rubric criteria (issue #54). When the scenario carried
            # criteria for this dimension: overall_score above is the
            # criterion-informed score, holistic_score is the judge's template
            # dimension score, and criteria_verdicts holds the per-criterion
            # {id, met, evidence} verdicts — persisted so the criterion-based vs
            # holistic (halo-effect) comparison is analyzable from artifacts.
            # Defaults (None/False/None) for criteria-less scenarios.
            "holistic_score": getattr(jr, "holistic_score", None),
            "criterion_informed": getattr(jr, "criterion_informed", False),
            "criteria_verdicts": getattr(jr, "criteria_verdicts", None),
            "raw_response": jr.raw_response,
        }
        for jr in consensus_result.judge_results
    ]


def build_artifact(
    scenario_id: str,
    model: str,
    run_index: int,
    sim_result,
    tc_result,
    ts_result,
    state=None,
    domain: str | None = None,
    category: str | None = None,
    holdout: bool = False,
) -> dict:
    """Assemble the artifact payload (pure — no I/O, unit-testable).

    ``state`` is the deterministic state-grading result for this run — the dict
    returned by ``score_state_changes`` (``score`` + ``checks``). It is optional
    (default ``None``) so the signature stays backward-compatible with callers
    that have no state grade; when supplied, a ``state`` block carrying the
    score, per-assertion checks, and the final world is added to the payload so a
    published state score is auditable back to the world it was computed from.

    ``domain`` and ``category`` are persisted at the top level (issue #46) so the
    calibration tooling stratifies on the authoritative scenario taxonomy
    (``customer_success`` / ``adaptive_tool_use``) rather than re-deriving it from
    the scenario-id prefix — real ids are ``cs_adaptive_tool_use_0001``, whose
    prefix is neither the domain nor a clean category. They are the exact strings
    on the ``Scenario`` (``domain.value`` and ``category``). ``holdout`` mirrors
    the private-holdout flag carried on the result row (issue #31) so a workbook
    sampler can exclude held-out transcripts before a sheet is ever shared with an
    external labeler.
    """
    payload = {
        "scenario_id": scenario_id,
        "model": model,
        "run_index": run_index,
        # Authoritative scenario taxonomy, persisted so calibration does not have
        # to re-derive it from the id (issue #46).
        "domain": domain,
        "category": category,
        # Private-holdout flag (issue #31): keep held-out transcripts out of
        # shared calibration workbooks by default.
        "holdout": bool(holdout),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "transcript": _serialize_transcript(sim_result.turns),
        "judges": {
            "task_completion": _serialize_judges(tc_result),
            "tool_selection": _serialize_judges(ts_result),
        },
        "sim_meta": {
            "completed": sim_result.completed,
            "total_turns": sim_result.total_turns,
            "input_tokens": sim_result.total_input_tokens,
            "output_tokens": sim_result.total_output_tokens,
            "latency_ms": sim_result.total_latency_ms,
            "error": sim_result.error,
            "resolved_model": getattr(sim_result, "resolved_model", None),
            # Simulator model ids used this run (issue #50), so a transcript is
            # auditable to which user/tool simulator produced it and a resume can
            # reconstruct the row faithfully.
            "user_sim_model": getattr(sim_result, "user_sim_model", None),
            "tool_sim_model": getattr(sim_result, "tool_sim_model", None),
            # Behavioral user-sim profile (issue #59 part 1): which user the
            # agent actually faced in this transcript. Defaults to cooperative
            # for sim results that predate the field, matching the row builder.
            "sim_profile": getattr(sim_result, "sim_profile", DEFAULT_SIM_PROFILE),
            # Recovery probe (issue #57): which probe kind (if any) perturbed this
            # transcript and whether the agent recovered. Both None for the
            # non-probe majority and for artifacts that predate the field, so a
            # resume reconstructs the row identically.
            "recovery_probe_kind": getattr(sim_result, "recovery_probe_kind", None),
            "recovered": getattr(sim_result, "recovered", None),
            # User-sim completion decoupling (#32). ``completed`` only says the
            # conversation stopped; these three say HOW it stopped and whether
            # the goals were verifiably met when it did, so a premature ending
            # (sim quit before the deterministic state check passed) is auditable
            # per run and aggregable into a premature-ending rate.
            "ended_by": getattr(sim_result, "ended_by", "max_turns"),
            "state_progress_at_end": getattr(sim_result, "state_progress_at_end", None),
            "premature_end": getattr(sim_result, "premature_end", False),
        },
    }
    if state is not None:
        payload["state"] = {
            "score": state.get("score"),
            "checks": state.get("checks"),
            "final_world": getattr(sim_result, "final_world", None),
        }
    return payload


def write_run_artifact(
    artifacts_root,
    run_id: str,
    scenario_id: str,
    model: str,
    run_index: int,
    sim_result,
    tc_result,
    ts_result,
    state=None,
    domain: str | None = None,
    category: str | None = None,
    holdout: bool = False,
) -> Path:
    """Write one evaluation's artifact to disk and return its path.

    ``artifacts_root`` is the directory that holds per-run subdirectories
    (typically ``data/results/artifacts``). ``state`` is the optional
    deterministic state-grading result (see :func:`build_artifact`).
    ``domain`` / ``category`` / ``holdout`` are passed through to the payload (see
    :func:`build_artifact`).
    """
    payload = build_artifact(
        scenario_id,
        model,
        run_index,
        sim_result,
        tc_result,
        ts_result,
        state,
        domain=domain,
        category=category,
        holdout=holdout,
    )
    out_dir = Path(artifacts_root) / run_id / model_slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_id}_run{run_index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
