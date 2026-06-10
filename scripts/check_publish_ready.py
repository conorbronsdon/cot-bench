"""Publish-readiness gate for the weekly leaderboard.

Reads data/results/run_manifest.json (written by scripts.run_eval) and blocks
a scheduled publish when any model failed to evaluate. Without this gate, a
week where e.g. GOOGLE_API_KEY is missing would publish a leaderboard silently
missing both Gemini models — the run_eval warning scrolls past in CI logs and
the partial board ships to the live site anyway.

Exit codes:
  0  manifest is complete (no failed models), or --allow-partial set
  1  one or more models failed, or the manifest is missing/unreadable

Escape hatch: --allow-partial (or ALLOW_PARTIAL_PUBLISH=true) downgrades a
partial run to a loud warning and exits 0, for deliberate partial publishes.
"""

import argparse
import json
import os
import sys
from pathlib import Path

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

    failed = manifest.get("models_failed") or []
    if not failed:
        completed = manifest.get("models_completed") or []
        print(f"Run complete: {len(completed)} model(s) evaluated, none failed. OK to publish.")
        return 0

    failed_list = ", ".join(failed)
    if allow_partial:
        print(
            f"::warning::Partial publish: {len(failed)} model(s) FAILED and are missing "
            f"from the leaderboard: {failed_list}. Publishing anyway because "
            "--allow-partial / ALLOW_PARTIAL_PUBLISH=true was set.",
            file=sys.stderr,
        )
        return 0

    print(
        f"::error::Refusing to publish an incomplete leaderboard: {len(failed)} model(s) "
        f"failed to evaluate and would be silently missing: {failed_list}. "
        "Fix the cause (often a missing API key — run python -m scripts.preflight) and "
        "re-run, or pass --allow-partial / set ALLOW_PARTIAL_PUBLISH=true to publish "
        "deliberately.",
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
