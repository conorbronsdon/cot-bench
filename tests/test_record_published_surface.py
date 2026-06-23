"""Tests for the published-surfaces ledger writer (scripts/record_published_surface.py).

These pin the WRITE side of the anti-memorization ledger that closes the
same-surface-reuse gap left open by #82:

* a templated published run appends a correct line (run_id, seed, surface hash,
  published_at lifted from the manifest timestamp — never a clock call here);
* the append is idempotent: re-recording the same run_id does NOT double-append;
* a non-templated run records nothing (no instantiated surface exists);
* read_published_hashes / find_prior_publish read the ledger back correctly and
  tolerate a missing/empty ledger and malformed lines.
"""

import json

from scripts.record_published_surface import (
    append_published_surface,
    find_prior_publish,
    read_published_hashes,
)


def _manifest(run_id="results_20260101", seed=1234567, surface="a" * 64, templated=True) -> dict:
    manifest = {
        "run_id": run_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "models_completed": ["GPT-4.1"],
    }
    if templated:
        manifest["templating"] = {
            "instantiation_seed": seed,
            "n_templated_scenarios": 5,
            "template_corpus_sha256": "b" * 64,
            "instantiated_corpus_sha256": surface,
        }
    return manifest


class TestAppendPublishedSurface:
    def test_appends_correct_line(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        assert append_published_surface(ledger, _manifest()) is True

        lines = ledger.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry == {
            "run_id": "results_20260101",
            "instantiation_seed": 1234567,
            "instantiated_corpus_sha256": "a" * 64,
            # published_at is the manifest's own timestamp, not a fresh clock read.
            "published_at": "2026-01-01T00:00:00+00:00",
        }

    def test_does_not_double_append_same_run_id(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        assert append_published_surface(ledger, _manifest()) is True
        # Recording the same run_id again (re-run of the publish step) is a no-op.
        assert append_published_surface(ledger, _manifest()) is False
        assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1

    def test_distinct_run_ids_both_appended(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        append_published_surface(ledger, _manifest(run_id="run_a", surface="a" * 64))
        append_published_surface(ledger, _manifest(run_id="run_b", surface="c" * 64))
        assert len(ledger.read_text(encoding="utf-8").splitlines()) == 2

    def test_non_templated_run_records_nothing(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        assert append_published_surface(ledger, _manifest(templated=False)) is False
        assert not ledger.exists()

    def test_creates_parent_dir(self, tmp_path):
        ledger = tmp_path / "nested" / "dir" / "published_surfaces.jsonl"
        assert append_published_surface(ledger, _manifest()) is True
        assert ledger.exists()


class TestReadPublishedHashes:
    def test_missing_ledger_is_empty(self, tmp_path):
        assert read_published_hashes(tmp_path / "nope.jsonl") == set()

    def test_empty_ledger_is_empty(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        ledger.write_text("")
        assert read_published_hashes(ledger) == set()

    def test_reads_back_appended_hashes(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        append_published_surface(ledger, _manifest(run_id="run_a", surface="a" * 64))
        append_published_surface(ledger, _manifest(run_id="run_b", surface="c" * 64))
        assert read_published_hashes(ledger) == {"a" * 64, "c" * 64}

    def test_skips_malformed_lines(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        ledger.write_text(
            'not json\n{"instantiated_corpus_sha256": "a"}\n\n'
            '{"instantiated_corpus_sha256": "' + "b" * 64 + '"}\n'
        )
        assert read_published_hashes(ledger) == {"a", "b" * 64}


class TestFindPriorPublish:
    def test_returns_matching_entry(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        append_published_surface(ledger, _manifest(run_id="run_a", seed=42, surface="a" * 64))
        prior = find_prior_publish("a" * 64, ledger)
        assert prior is not None
        assert prior["run_id"] == "run_a"
        assert prior["instantiation_seed"] == 42

    def test_returns_none_when_no_match(self, tmp_path):
        ledger = tmp_path / "published_surfaces.jsonl"
        append_published_surface(ledger, _manifest(surface="a" * 64))
        assert find_prior_publish("z" * 64, ledger) is None

    def test_returns_none_for_missing_ledger(self, tmp_path):
        assert find_prior_publish("a" * 64, tmp_path / "nope.jsonl") is None
