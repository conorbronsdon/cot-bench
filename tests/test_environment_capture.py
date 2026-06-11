"""H3: every run records its resolved environment.

The repo floor-pins dependencies (pyproject `>=`) and CI installs with a bare
`pip install -e .`, so two "identical" runs can resolve different library
versions and nothing used to record which. ``capture_environment`` writes the
full installed-package list to a sibling file and returns a summary (python
version + platform + freeze-file sha256 + package count) that ``run_eval`` embeds
in ``run_manifest.json``. These tests pin that the file is written and that the
recorded sha256 matches the file's contents.
"""

import hashlib
import sys

from scripts.run_eval import ENV_FREEZE_FILENAME, capture_environment


def test_writes_freeze_file_and_returns_summary(tmp_path):
    freeze_path = tmp_path / ENV_FREEZE_FILENAME
    summary = capture_environment(freeze_path)

    assert freeze_path.exists()
    body = freeze_path.read_text(encoding="utf-8")
    # Non-empty: pytest itself is installed, so there is always at least one dist.
    assert body.strip()
    # Each line is a name==version pin.
    for line in body.splitlines():
        assert "==" in line

    assert summary["python_version"] == sys.version.split()[0]
    assert summary["freeze_file"] == ENV_FREEZE_FILENAME
    assert summary["n_packages"] == len(body.splitlines())
    assert summary["n_packages"] > 0


def test_recorded_sha256_matches_file(tmp_path):
    freeze_path = tmp_path / ENV_FREEZE_FILENAME
    summary = capture_environment(freeze_path)
    expected = hashlib.sha256(freeze_path.read_bytes()).hexdigest()
    assert summary["freeze_sha256"] == expected


def test_deterministic_and_sorted(tmp_path):
    # Two captures of the same environment produce the same content + hash, and
    # the list is sorted case-insensitively so the digest is order-stable.
    a = capture_environment(tmp_path / "a.txt")
    b = capture_environment(tmp_path / "b.txt")
    assert a["freeze_sha256"] == b["freeze_sha256"]

    lines = (tmp_path / "a.txt").read_text(encoding="utf-8").splitlines()
    assert lines == sorted(lines, key=str.lower)
