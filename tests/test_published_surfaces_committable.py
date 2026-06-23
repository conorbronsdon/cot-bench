"""Guard: every path the weekly publish step commits must be git-add-able.

This pins the H1/H6 bug. The publish step in
``.github/workflows/weekly-eval.yml`` runs::

    git add data/results/leaderboard.json data/results/latest.csv data/results/history.jsonl

If any of those paths is matched by a ``.gitignore`` rule, ``git add`` of an
explicitly-named ignored file exits non-zero ("Use -f if you really want to add
them") and the whole publish step dies *before* commit/push — so the first
scheduled leaderboard publish never ships. ``latest.csv`` regressed exactly this
way (``data/results/*.csv`` swallowed it); ``history.jsonl`` was never added at
all, discarding the append-only audit trail.

These tests are deterministic and offline: they read the real workflow file and
shell out to ``git check-ignore`` (no network). The publish paths are parsed
from the workflow rather than hardcoded, so adding a published surface to the
``git add`` line without un-ignoring it fails here instead of in CI.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "weekly-eval.yml"

# The surfaces we expect the publish step to commit. Pinned independently of the
# parse so a published surface silently dropped from the workflow `git add` line
# is also caught (test_publish_step_adds_expected_surfaces below).
EXPECTED_PUBLISHED_SURFACES = {
    "data/results/leaderboard.json",
    "data/results/latest.csv",
    "data/results/history.jsonl",
    "data/results/published_surfaces.jsonl",
}


def _publish_add_paths() -> list[str]:
    """Paths from the publish step's ``git add data/results/...`` line.

    Scans the workflow for the ``git add`` invocation that stages the
    ``data/results`` publish surfaces and returns the path arguments.
    """
    text = WORKFLOW.read_text(encoding="utf-8")
    matches = [
        line.strip() for line in text.splitlines() if re.search(r"\bgit add\b.*data/results/", line)
    ]
    assert matches, (
        "No `git add data/results/...` line found in weekly-eval.yml — the publish "
        "step changed shape; update this guard so it still checks the committed paths."
    )
    assert len(matches) == 1, f"Expected exactly one publish `git add` line, found: {matches}"
    tokens = matches[0].split()
    add_idx = tokens.index("add")
    return [t for t in tokens[add_idx + 1 :] if t.startswith("data/results/")]


def _check_ignored(path: str) -> bool:
    """True if ``path`` is matched by a .gitignore rule (i.e. would block add)."""
    result = subprocess.run(
        ["git", "check-ignore", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    # exit 0 => ignored; exit 1 => not ignored; >1 => error
    assert result.returncode in (0, 1), f"git check-ignore errored on {path}: {result.stderr}"
    return result.returncode == 0


@pytest.fixture(scope="module", autouse=True)
def _require_git():
    if shutil.which("git") is None:
        pytest.skip("git not available")
    # Confirm we're inside a work tree (check-ignore needs one).
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        pytest.skip("not inside a git work tree")


def test_publish_step_adds_expected_surfaces():
    """The workflow stages exactly the published surfaces we expect."""
    assert set(_publish_add_paths()) == EXPECTED_PUBLISHED_SURFACES


def test_no_published_surface_is_gitignored():
    """Every path the publish step adds must NOT be gitignored (the H1/H6 bug)."""
    ignored = [p for p in _publish_add_paths() if _check_ignored(p)]
    assert not ignored, (
        f"These published paths are gitignored, so the weekly publish `git add` "
        f"will die before commit/push: {ignored}. Add `!<path>` negations in "
        ".gitignore (see the 'Published leaderboard surfaces' block)."
    )


def test_generated_results_files_stay_ignored():
    """The negations must be exact — generic generated outputs stay ignored."""
    # Per-run parquet/csv and the transcript artifacts dir must remain ignored,
    # so a too-broad negation doesn't start committing transcripts or raw runs.
    for path in (
        "data/results/results_20260101.parquet",
        "data/results/results_20260101.csv",
        "data/results/artifacts/run/x.json",
    ):
        assert _check_ignored(path), f"{path} should be gitignored but is not — negation too broad"
