"""Append-only ledger of already-published instantiated surfaces (anti-memorization).

Background: ``scripts/check_publish_ready.py`` already blocks a published TEMPLATED
run that used the default seed 0 (PR #82), because at seed 0 the instantiated
surface is identical every run. But that gate does NOT stop *reusing the same
fresh seed* across two published runs — which re-exposes a byte-identical surface
and defeats the anti-memorization goal (issue #60) just like seed 0 does. The
collision key that actually matters is the **instantiated surface**, recorded in
the manifest's ``templating`` block as ``instantiated_corpus_sha256`` (the sha256
of the exact scenarios a published board exposed).

This module maintains the committed ledger at ``data/results/published_surfaces.jsonl``
(append-only JSONL, one line per PUBLISHED run)::

    {"run_id": ..., "instantiation_seed": ..., "instantiated_corpus_sha256": ...,
     "published_at": ...}

The publish gate reads this ledger and blocks a templated publish whose surface
hash already appears in it. This module owns the WRITE side: a successful
published run appends its surface here, wired into the publish step of
``.github/workflows/weekly-eval.yml`` so the next published run can be checked
against it.

Design notes:

* **No clock call here.** ``published_at`` is taken from the manifest's own
  ``timestamp`` (the run's UTC timestamp), so the ledger records the surface's
  own provenance and stays deterministic/testable — this module never calls a
  time function itself.
* **Idempotent-safe.** ``append_published_surface`` does not append a second line
  for a ``run_id`` already in the ledger (a re-run of the publish step, or a
  resumed publish, must not double-count). It returns whether it appended.
* **Templated runs only.** A run with no ``templating`` block (no templated
  scenarios) has no instantiated surface to track; nothing is appended. The
  caller (publish step) only runs on a published run, never a smoke run.
"""

import argparse
import json
import sys
from pathlib import Path

LEDGER_PATH = Path("data/results/published_surfaces.jsonl")


def read_published_hashes(ledger_path: Path = LEDGER_PATH) -> set[str]:
    """Return the set of ``instantiated_corpus_sha256`` values already published.

    A missing or empty ledger means nothing has been published yet -> empty set.
    Malformed lines are skipped rather than crashing the gate (the ledger is an
    append-only audit file; a single bad line must not block every future
    publish). The gate uses this set as its collision check.
    """
    if not ledger_path.exists():
        return set()
    hashes: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        h = entry.get("instantiated_corpus_sha256")
        if h:
            hashes.add(h)
    return hashes


def find_prior_publish(instantiated_hash: str, ledger_path: Path = LEDGER_PATH) -> dict | None:
    """Return the first ledger entry whose surface hash matches, or None.

    Used by the publish gate to name the prior published run (its ``run_id`` and
    ``instantiation_seed``) in the blocker message, so an operator can see which
    earlier board already exposed this surface.
    """
    if not ledger_path.exists():
        return None
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("instantiated_corpus_sha256") == instantiated_hash:
            return entry
    return None


def _recorded_run_ids(ledger_path: Path) -> set[str]:
    """Run ids already present in the ledger (for idempotent append)."""
    if not ledger_path.exists():
        return set()
    ids: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = entry.get("run_id")
        if rid is not None:
            ids.add(rid)
    return ids


def append_published_surface(ledger_path: Path, manifest: dict) -> bool:
    """Append this published run's instantiated surface to the ledger.

    Pulls ``instantiated_corpus_sha256`` and ``instantiation_seed`` from the
    manifest's ``templating`` block, ``run_id`` and ``published_at`` (the run's
    own ``timestamp``) from the manifest top level. Returns True if a line was
    appended, False if it was skipped.

    Skipped (returns False) when:
      * the manifest has no ``templating`` block (a non-templated run has no
        instantiated surface to track), or
      * the run_id is already in the ledger (idempotent: a re-run of the publish
        step must not double-append the same surface).

    Never calls a clock: ``published_at`` is the manifest's recorded timestamp.
    """
    templating = manifest.get("templating")
    if not templating:
        # Non-templated run: no instantiated surface exists. Nothing to record.
        return False

    instantiated_hash = templating.get("instantiated_corpus_sha256")
    if not instantiated_hash:
        # A templating block without the surface hash is malformed; do not write a
        # ledger entry that can't serve as a collision key.
        return False

    run_id = manifest.get("run_id")
    if run_id is not None and run_id in _recorded_run_ids(ledger_path):
        # Already recorded (idempotent re-run of the publish step).
        return False

    entry = {
        "run_id": run_id,
        "instantiation_seed": templating.get("instantiation_seed"),
        "instantiated_corpus_sha256": instantiated_hash,
        "published_at": manifest.get("timestamp"),
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Append a published run's instantiated surface to the anti-memorization "
            "ledger. No-op (exit 0) for a non-templated run."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/results/run_manifest.json"),
        help="Path to run_manifest.json for the published run.",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="Path to the published-surfaces ledger (JSONL).",
    )
    args = parser.parse_args()

    if not args.manifest.exists():
        print(
            f"::error::Run manifest not found at {args.manifest} — cannot record the "
            "published surface.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"::error::Could not read run manifest {args.manifest}: {e}", file=sys.stderr)
        sys.exit(1)

    appended = append_published_surface(args.ledger, manifest)
    if appended:
        print(f"Recorded published surface for run {manifest.get('run_id')!r} in {args.ledger}.")
    else:
        print(
            "No surface recorded (non-templated run, or run already in the ledger). "
            "This is expected and not an error."
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
