"""Checkpoint/resume for COT Bench runs (issue #48).

A crash or rate-limit death partway through a paid run used to lose everything not
yet in the final parquet — the parquet is written only at the very end. But each
evaluation already persists a per-evaluation artifact (transcript + raw judge
outputs + state + sim_meta) the moment it finishes (``eval/artifacts.py``). Those
artifacts ARE the checkpoint.

``--resume <run_id>`` reuses them:

1. :func:`completed_tuples` scans ``{artifacts_root}/{run_id}/`` and returns the
   set of ``(model_name, scenario_id, run_index)`` already done, so the run loop
   skips them (never paying twice).
2. :func:`rows_from_artifacts` reconstructs the flat result rows for those
   completed evaluations by feeding the persisted artifact back through the SAME
   ``build_result_row`` the live path uses — so reconstructed rows are
   column-for-column identical to freshly-computed ones (per-judge scores,
   agreement, efficacy, cost), with no parallel row-builder to drift.

Governance honesty constraint (docs/governance.md §3): a resumed run continues
under its ORIGINAL ``pre_registration.json``. The run definition is fixed before
results, so a resume must NOT write a new pre-registration and must ABORT if the
current scenario set no longer matches the one the original run committed to.
:func:`verify_corpus_unchanged` enforces that against the public (and holdout,
if any) corpus hashes.
"""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

from eval.scoring.judge import JudgeResult
from eval.simulation.profiles import DEFAULT_SIM_PROFILE

logger = logging.getLogger(__name__)


class CorpusMismatchError(RuntimeError):
    """Raised when a resumed run's scenario set no longer matches the original.

    Per governance §3 the run definition is fixed before results; resuming under a
    changed corpus would silently mix two run definitions, so we abort instead.
    """


def _artifact_paths(artifacts_root, run_id: str):
    """Yield every per-evaluation artifact JSON for a run, if the dir exists."""
    run_dir = Path(artifacts_root) / run_id
    if not run_dir.exists():
        return
    yield from run_dir.glob("*/*.json")


def completed_tuples(artifacts_root, run_id: str, model_names) -> set:
    """Set of ``(model_name, scenario_id, run_index)`` already completed.

    Reads each artifact's own ``model`` / ``scenario_id`` / ``run_index`` fields
    (authoritative — the filename slug is lossy), so the returned model names match
    the live roster exactly and the run loop's skip check lines up. ``model_names``
    is the resolved roster; artifacts for a model not in it are ignored (a resume
    that narrowed --models should not claim other models' work).
    """
    roster = set(model_names)
    done: set = set()
    for path in _artifact_paths(artifacts_root, run_id):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable artifact during resume scan: %s", path)
            continue
        model = payload.get("model")
        scenario_id = payload.get("scenario_id")
        run_index = payload.get("run_index")
        if model in roster and scenario_id is not None and run_index is not None:
            done.add((model, scenario_id, int(run_index)))
    return done


def _consensus_from_artifact(judges_block: list, scenario_id: str, rubric_type: str):
    """Rebuild a ConsensusResult from a serialized judge list for one rubric.

    The artifact stores each judge's ``overall_score`` / ``reasoning`` /
    ``parse_failed`` / ``resolved_model`` (see ``artifacts._serialize_judges``).
    Token counts were not serialized per judge (they are not needed for scoring),
    so reconstructed JudgeResults carry zero tokens — resume does not re-pay, so
    no cost needs re-attributing. The shared consensus math is reused so the
    rebuilt result is identical in shape and value to the original.
    """
    from eval.scoring.judge import _build_consensus

    results = [
        JudgeResult(
            judge_name=j.get("judge_name", ""),
            rubric_type=rubric_type,
            overall_score=float(j.get("overall_score", 0.0) or 0.0),
            reasoning=j.get("reasoning", ""),
            raw_response=j.get("raw_response", {}) or {},
            latency_ms=0.0,
            parse_failed=bool(j.get("parse_failed", False)),
            resolved_model=j.get("resolved_model", "") or "",
            # Atomic-rubric fields (issue #54); .get defaults cover artifacts
            # written before these fields existed.
            holistic_score=j.get("holistic_score"),
            criterion_informed=bool(j.get("criterion_informed", False)),
            criteria_verdicts=j.get("criteria_verdicts"),
        )
        for j in judges_block
    ]
    return _build_consensus(scenario_id, rubric_type, results, len(results), api_failures=[])


def _sim_namespace(payload: dict):
    """Build a sim_result-like object from the artifact's sim_meta block.

    ``build_result_row`` reads only attributes (token counts, latency, completed,
    ended_by, resolved_model, sim model ids, premature_end) — never methods — so a
    SimpleNamespace with those fields is a faithful stand-in. Fields absent in
    older artifacts fall back to the same defaults the live getattr() paths use.
    """
    meta = payload.get("sim_meta", {})
    return SimpleNamespace(
        total_latency_ms=meta.get("latency_ms", 0.0) or 0.0,
        total_turns=meta.get("total_turns", 0) or 0,
        total_input_tokens=meta.get("input_tokens", 0) or 0,
        total_output_tokens=meta.get("output_tokens", 0) or 0,
        completed=bool(meta.get("completed", False)),
        ended_by=meta.get("ended_by", "max_turns"),
        state_progress_at_end=meta.get("state_progress_at_end"),
        premature_end=bool(meta.get("premature_end", False)),
        resolved_model=meta.get("resolved_model"),
        user_sim_model=meta.get("user_sim_model"),
        tool_sim_model=meta.get("tool_sim_model"),
        # Behavioral user-sim profile (issue #59). Older artifacts predate the
        # field and were all cooperative by construction.
        sim_profile=meta.get("sim_profile", DEFAULT_SIM_PROFILE),
    )


def rows_from_artifacts(artifacts_root, run_id: str, model_names) -> list[dict]:
    """Reconstruct result rows for all completed evaluations of a run.

    Each row is rebuilt by feeding the persisted artifact back through the live
    ``build_result_row`` (via reconstructed ConsensusResult + sim stand-in +
    state), so columns match the fresh path exactly. Reliability columns are NOT
    set here — they are recomputed across reconstructed + freshly-run rows in
    run_eval after the merge, because reliability is a per-(model, scenario)
    aggregate over all of a scenario's runs, some of which may be new.
    """
    from scripts.run_eval import build_result_row

    roster = set(model_names)
    rows: list[dict] = []
    for path in _artifact_paths(artifacts_root, run_id):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Skipping unreadable artifact during resume merge: %s", path)
            continue
        if payload.get("model") not in roster:
            continue

        scenario_id = payload["scenario_id"]
        judges = payload.get("judges", {})
        tc_result = _consensus_from_artifact(
            judges.get("task_completion", []), scenario_id, "task_completion"
        )
        ts_result = _consensus_from_artifact(
            judges.get("tool_selection", []), scenario_id, "tool_selection"
        )
        sim_ns = _sim_namespace(payload)

        state = payload.get("state")
        state_result = None
        if state is not None:
            checks = state.get("checks") or []
            n_total = len(checks)
            n_passed = sum(1 for c in checks if c.get("passed"))
            state_result = {
                "score": state.get("score", 0.0) or 0.0,
                "checks": checks,
                "n_passed": n_passed,
                "n_total": n_total,
            }

        # Efficacy is recomputed from the rebuilt consensus + state, matching the
        # live evaluate_scenario composition.
        from eval.scoring.rubrics import compute_efficacy

        state_score = None if state_result is None else state_result["score"]
        efficacy = compute_efficacy(
            tc_result.consensus_score, ts_result.consensus_score, state_score
        )

        # Cost row column is agent-only (CLEAR Cost dimension), recomputed from the
        # persisted agent token counts — the same formula evaluate_scenario uses.
        # PRICE BY THE REQUESTED MODEL ID, exactly like the live path: the artifact
        # stores the display name, so map it back through the roster. Pricing by
        # resolved_model would zero the Cost dimension for OpenRouter models
        # (resolved ids are upstream slugs absent from TOKEN_COSTS).
        from eval.config import MODELS_UNDER_TEST, NULL_AGENT_MODEL, TOKEN_COSTS

        requested_id = next(
            (
                m["model_id"]
                for m in [*MODELS_UNDER_TEST, NULL_AGENT_MODEL]
                if m["name"] == payload["model"]
            ),
            None,
        )
        agent_costs = TOKEN_COSTS.get(requested_id) or TOKEN_COSTS.get(
            payload.get("sim_meta", {}).get("resolved_model")
        )
        if agent_costs is None:
            # Matches the live path's $0 fallback for an unpriced model.
            cost_usd = 0.0
        else:
            cost_usd = (
                sim_ns.total_input_tokens * agent_costs["input"] / 1_000_000
                + sim_ns.total_output_tokens * agent_costs["output"] / 1_000_000
            )

        agent_spec = SimpleNamespace(name=payload["model"])
        scenario = SimpleNamespace(
            id=scenario_id,
            domain=SimpleNamespace(value=payload.get("domain")),
            category=payload.get("category"),
            holdout=bool(payload.get("holdout", False)),
        )
        row = build_result_row(
            scenario,
            agent_spec,
            sim_ns,
            tc_result,
            ts_result,
            efficacy,
            cost_usd,
            state_result=state_result,
        )
        row["run_index"] = int(payload["run_index"])
        row["evaluated_at"] = payload.get("evaluated_at")
        rows.append(row)
    return rows


def load_pre_registration(results_dir, filename: str) -> dict:
    """Load the original pre_registration.json for a resumed run."""
    path = Path(results_dir) / filename
    if not path.exists():
        raise CorpusMismatchError(
            f"--resume requires the original {filename} in {results_dir}; not found. "
            "A resumed run must continue under the pre-registration written before "
            "the original run (governance §3)."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_corpus_unchanged(
    original_pre_registration: dict,
    *,
    current_public_hash: str,
    current_holdout_hash: str | None,
) -> None:
    """Abort a resume if the scenario set changed since the original run.

    Compares the current public corpus hash (and the holdout hash, if the original
    run had a holdout) against the values recorded in the original
    pre-registration. Any mismatch — a scenario edited, added, or removed, or a
    holdout that appeared/vanished/changed — raises :class:`CorpusMismatchError`.
    This is the honest behavior per governance §3: the run definition is fixed
    before results, so a resume cannot quietly adopt a different corpus.
    """
    original_public = original_pre_registration.get("scenario_set", {}).get("sha256")
    if original_public != current_public_hash:
        raise CorpusMismatchError(
            "Public scenario corpus changed since the original run "
            f"(original {original_public!r} != current {current_public_hash!r}). "
            "Resuming would mix two run definitions; aborting (governance §3). "
            "Start a new run instead."
        )

    original_holdout_block = original_pre_registration.get("holdout_set")
    original_holdout = original_holdout_block.get("sha256") if original_holdout_block else None
    if original_holdout != current_holdout_hash:
        raise CorpusMismatchError(
            "Holdout corpus changed since the original run "
            f"(original {original_holdout!r} != current {current_holdout_hash!r}). "
            "Resuming would mix two run definitions; aborting (governance §3)."
        )
