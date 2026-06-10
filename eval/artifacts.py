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
) -> dict:
    """Assemble the artifact payload (pure — no I/O, unit-testable)."""
    return {
        "scenario_id": scenario_id,
        "model": model,
        "run_index": run_index,
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
        },
    }


def write_run_artifact(
    artifacts_root,
    run_id: str,
    scenario_id: str,
    model: str,
    run_index: int,
    sim_result,
    tc_result,
    ts_result,
) -> Path:
    """Write one evaluation's artifact to disk and return its path.

    ``artifacts_root`` is the directory that holds per-run subdirectories
    (typically ``data/results/artifacts``).
    """
    payload = build_artifact(scenario_id, model, run_index, sim_result, tc_result, ts_result)
    out_dir = Path(artifacts_root) / run_id / model_slug(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_id}_run{run_index}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path
