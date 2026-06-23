"""Publish-readiness gate for the weekly leaderboard.

Reads data/results/run_manifest.json (written by scripts.run_eval) and blocks
a scheduled publish when:

  1. Any model failed to evaluate (a partial board would silently ship missing
     models — e.g. a missing GOOGLE_API_KEY would drop both Gemini models, with
     only a warning that scrolls past in CI logs).
  2. Any evaluated domain has fewer than MIN_SCENARIOS_FOR_PUBLISH scenarios. At
     low scenario counts, leaderboard orderings are dominated by sampling noise
     (bootstrap CIs overlap almost completely), so the ranking is not meaningful.
     The methodological review set this hard minimum so the board stays honest
     while the corpus stays at its shipped 92-scenario size.
  3. The judge panel is not the full default roster (H2). A single-judge or
     reduced-panel board has a different, uncomparable consensus (and a null
     inter-judge alpha) — publishing it as "the leaderboard" would be misleading.
  4. reliability_runs differs from the default (H2). pass@3 / pass^k mean
     different things at a different repeat count; a board run at reliability=1
     is not the same measurement.
  5. The sim_profile is not the cooperative default (H2). The public board is
     defined against the cooperative simulated user; a behavioral-profile run is
     a different, deliberately harder condition. (The aggregation already strips
     non-cooperative rows, so a stratified run would yield an empty/partial
     board — this gate names the cause loudly rather than letting it surface as
     a confusing empty publish.)

  6. Templating was used but the instantiation seed was the default (0). The
     anti-memorization benefit of templating (issue #60) only exists when a
     published run instantiates the corpus with a FRESH seed; at seed 0 the
     surface is identical every run, so a memorized surface defeats the
     benchmark. Seed 0 is CI-only — a published templated run must pass
     --random-instantiation-seed. Non-templated runs have no templating block in
     the manifest and are never gated on this condition.

  7. scenario_limit was set (> 0). run_eval's --scenario-limit slices a fixed
     lexicographic prefix of each domain, so a positive limit ships a
     deterministic subset, not a representative sample — its orderings are not
     comparable to a full-corpus board. The per-domain scenario minimum does not
     catch this (a limit can still clear the minimum), so this is a distinct
     gate. Absent / 0 => unlimited (the full corpus), which passes.

Conditions 3-5 only trip via an explicit ``workflow_dispatch`` with non-default
inputs; the scheduled path always uses the defaults, so it passes them silently.
Condition 6 only applies once the corpus actually contains templated scenarios.

Exit codes:
  0  manifest is complete (no failed models, all domains >= minimum, full judge
     panel, default reliability_runs, cooperative profile), or --allow-partial set
  1  any blocking condition above, or the manifest is missing/unreadable

Escape hatch: --allow-partial (or ALLOW_PARTIAL_PUBLISH=true) downgrades every
blocking condition above (each reported distinctly) to a loud warning and exits
0, for deliberate partial / non-default publishes and previews.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from eval.config import JUDGES, MIN_SCENARIOS_FOR_PUBLISH, RELIABILITY_RUNS
from eval.simulation.profiles import DEFAULT_SIM_PROFILE
from eval.templating import DEFAULT_INSTANTIATION_SEED

MANIFEST_PATH = Path("data/results/run_manifest.json")


def check_publish_ready(manifest_path: Path = MANIFEST_PATH, allow_partial: bool = False) -> int:
    """Return an exit code for the publish gate.

    Returns 0 when the run is complete (or partial is explicitly allowed),
    1 when models failed or the manifest can't be read. Emits GitHub Actions
    workflow annotations (::error::/::warning::) so failures surface in the
    Actions UI, not just the raw log.
    """
    if not manifest_path.exists():
        print(
            f"::error::Run manifest not found at {manifest_path} — cannot verify run "
            "completeness. Did scripts.run_eval complete and write the manifest?",
            file=sys.stderr,
        )
        return 1

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"::error::Could not read run manifest {manifest_path}: {e}",
            file=sys.stderr,
        )
        return 1

    # Collect every blocking reason so the operator sees all problems at once,
    # rather than fixing failed-models only to be re-blocked on scenario counts.
    blockers: list[str] = []

    failed = manifest.get("models_failed") or []
    if failed:
        failed_list = ", ".join(failed)
        blockers.append(
            f"{len(failed)} model(s) failed to evaluate and would be silently missing: "
            f"{failed_list}. Fix the cause (often a missing API key — run "
            "python -m scripts.preflight) and re-run."
        )

    # Per-domain scenario-count minimum. Any evaluated domain below the bar makes
    # the whole board's orderings noise-dominated.
    scenario_counts = manifest.get("scenario_counts") or {}
    below_min = {
        domain: count
        for domain, count in scenario_counts.items()
        if count < MIN_SCENARIOS_FOR_PUBLISH
    }
    if below_min:
        detail = ", ".join(f"{d}={n}" for d, n in sorted(below_min.items()))
        blockers.append(
            f"{len(below_min)} domain(s) below the {MIN_SCENARIOS_FOR_PUBLISH}-scenario "
            f"publish minimum ({detail}). Orderings at this scenario count are dominated "
            "by sampling noise; add scenarios before publishing."
        )

    # Judge panel (H2). The board is only a leaderboard when scored by the full
    # default panel — a reduced panel changes the consensus and nulls the
    # inter-judge alpha. ``judges.requested`` is recorded by run_eval; legacy
    # manifests without it are not gated on this condition (treated as default).
    judges_block = manifest.get("judges") or {}
    requested_judges = judges_block.get("requested")
    if requested_judges is not None:
        full_panel = sorted(JUDGES.keys())
        if sorted(requested_judges) != full_panel:
            blockers.append(
                f"Judge panel was {sorted(requested_judges)}, not the full default "
                f"roster {full_panel}. A reduced/single-judge board has a different "
                "consensus (and a null inter-judge alpha) than a leaderboard run; "
                "re-run with the full panel before publishing."
            )

    # Reliability runs (H2). pass@3 / pass^k change meaning at a different repeat
    # count, so a board must use the default. Absent => legacy manifest, skip.
    reliability_runs = manifest.get("reliability_runs")
    if reliability_runs is not None and reliability_runs != RELIABILITY_RUNS:
        blockers.append(
            f"reliability_runs was {reliability_runs}, not the default "
            f"{RELIABILITY_RUNS}. pass@3 / pass^k mean different things at a "
            "different repeat count; re-run at the default before publishing."
        )

    # Sim profile (H2). The public board is defined against the cooperative
    # simulated user. Absent => legacy manifest (all cooperative), skip.
    sim_profile = manifest.get("sim_profile")
    if sim_profile is not None and sim_profile != DEFAULT_SIM_PROFILE:
        blockers.append(
            f"sim_profile was '{sim_profile}', not the cooperative default "
            f"'{DEFAULT_SIM_PROFILE}'. The public board measures every model "
            "against the same cooperative user; a behavioral-profile run is a "
            "different condition (and its rows are stripped from the public "
            "aggregates). Re-run cooperative before publishing."
        )

    # Template-instantiation seed (issue #60). The anti-memorization benefit of
    # templating only exists if a PUBLISHED run instantiates the corpus with a FRESH
    # seed: at the default seed (0) the surface is identical every run, so memorizing
    # a published surface defeats the benchmark and templating gains nothing. The
    # manifest's ``templating`` block is present ONLY when at least one scenario was
    # actually a template (n_templated_scenarios > 0), so it doubles as the
    # "templating was used" signal — a non-templated run has no block and is never
    # gated here. Block a templated publish that used seed 0; seed 0 is CI-only.
    templating = manifest.get("templating")
    if templating:
        instantiation_seed = templating.get("instantiation_seed")
        # Fail closed: block when the seed is the default (0) OR cannot be
        # confirmed fresh (missing/None). run_eval always records an int seed, so
        # None only arises from a malformed/hand-edited manifest — but a publish
        # gate on the live board must not pass a surface it can't prove is fresh.
        if instantiation_seed is None or instantiation_seed == DEFAULT_INSTANTIATION_SEED:
            blockers.append(
                f"Templating was used ({templating.get('n_templated_scenarios')} "
                f"templated scenario(s)) but the instantiation seed was "
                f"{instantiation_seed!r} (the default {DEFAULT_INSTANTIATION_SEED}, "
                "or unconfirmable). At the default seed the instantiated surface is "
                "identical every run, so a memorized surface defeats the benchmark "
                "and templating gains nothing. Re-run with --random-instantiation-seed "
                f"(which draws a fresh non-zero seed); seed "
                f"{DEFAULT_INSTANTIATION_SEED} is CI-only, not for a published board."
            )

    # --- S1: scenario-limited (non-representative) run -----------------------
    # Kept as its own self-contained block (separate from the conditions above)
    # to minimize merge conflicts with other in-flight publish-gate work.
    #
    # run_eval's --scenario-limit slices a fixed lexicographic prefix of each
    # domain (scenarios[:N]). A positive limit therefore ships a deterministic
    # subset, NOT a representative sample of the corpus, so its leaderboard
    # orderings are not comparable to a full-corpus board. The scenario-count
    # minimum above does not catch this: a limit of 30 can still clear the
    # minimum while quietly publishing a prefix-subset. Absent => legacy manifest
    # (treated as the unlimited default), skip.
    scenario_limit = manifest.get("scenario_limit")
    if scenario_limit is not None and scenario_limit > 0:
        blockers.append(
            f"scenario_limit was {scenario_limit} (> 0), so the run evaluated only a "
            "fixed lexicographic-prefix subset of each domain, not the full corpus. "
            "A prefix-subset board is non-representative and not comparable to a "
            "full-corpus leaderboard; re-run with --scenario-limit 0 (all scenarios) "
            "before publishing."
        )

    if not blockers:
        completed = manifest.get("models_completed") or []
        print(
            f"Run complete: {len(completed)} model(s) evaluated, none failed; "
            f"all domains meet the {MIN_SCENARIOS_FOR_PUBLISH}-scenario minimum. "
            "OK to publish."
        )
        return 0

    joined = " ".join(blockers)
    if allow_partial:
        print(
            f"::warning::Partial publish: {joined} Publishing anyway because "
            "--allow-partial / ALLOW_PARTIAL_PUBLISH=true was set.",
            file=sys.stderr,
        )
        return 0

    print(
        f"::error::Refusing to publish: {joined} Pass --allow-partial / set "
        "ALLOW_PARTIAL_PUBLISH=true to publish deliberately.",
        file=sys.stderr,
    )
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Gate the scheduled leaderboard publish on run completeness."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to run_manifest.json",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        default=os.environ.get("ALLOW_PARTIAL_PUBLISH", "").lower() == "true",
        help="Downgrade a partial run to a warning and exit 0 (or set ALLOW_PARTIAL_PUBLISH=true).",
    )
    args = parser.parse_args()
    sys.exit(check_publish_ready(args.manifest, args.allow_partial))


if __name__ == "__main__":
    main()
