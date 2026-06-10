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
     while the scenario set scales toward its ~80-scenario target.

Exit codes:
  0  manifest is complete (no failed models, all domains >= minimum), or
     --allow-partial set
  1  one or more models failed, a domain is below the scenario minimum, or the
     manifest is missing/unreadable

Escape hatch: --allow-partial (or ALLOW_PARTIAL_PUBLISH=true) downgrades a
partial run (failed models OR below-minimum scenario counts) to a loud warning
and exits 0, for deliberate partial publishes / previews.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from eval.config import MIN_SCENARIOS_FOR_PUBLISH

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
