"""Tests for the human judge-calibration tooling (issue #33).

The study runs after the rehearsal run, but the code path must be correct now.
These tests build small synthetic artifact JSONs matching the real schema
written by eval/artifacts.py and exercise:

- stratification determinism (same seed -> same sample; coverage spans bands)
- blindness (no judge scores anywhere in the workbook output)
- agreement math on known values (alpha reuse, MAD, Pearson)
- the full round trip: sample -> (fill labels) -> score

No network, no API spend.
"""

import json

import pytest

from scripts.calibration import (
    DIMENSIONS,
    Evaluation,
    _agreement,
    _band_for,
    build_key,
    compute_calibration,
    load_evaluations,
    read_labels,
    render_report,
    render_sheet,
    stratified_sample,
    write_workbook,
)

# --------------------------------------------------------------------------- #
# Synthetic artifact builders (match eval/artifacts.py build_artifact shape)
# --------------------------------------------------------------------------- #


def _artifact(
    scenario_id,
    model,
    run_index,
    domain,
    category,
    tc_judges,
    ts_judges,
    state_score=None,
):
    """Build one artifact dict matching the real per-evaluation schema.

    ``tc_judges`` / ``ts_judges`` are lists of (judge_name, score, parse_failed).
    """

    def judges(rubric_type, specs):
        return [
            {
                "judge_name": name,
                "rubric_type": rubric_type,
                "overall_score": score,
                "reasoning": f"{name} reasoning",
                "parse_failed": failed,
                "resolved_model": "",
                "raw_response": {"overall_score": score},
            }
            for name, score, failed in specs
        ]

    art = {
        "scenario_id": scenario_id,
        "model": model,
        "run_index": run_index,
        "domain": domain,
        "category": category,
        "evaluated_at": "2026-06-10T00:00:00+00:00",
        "transcript": [
            {"turn_number": 0, "role": "user", "content": "Move $500 to savings", "tool_calls": []},
            {
                "turn_number": 0,
                "role": "agent",
                "content": "Transferring now.",
                "tool_calls": [
                    {
                        "turn": 0,
                        "tool_name": "transfer_funds",
                        "arguments": {"amount": 500, "to": "savings"},
                        "result": "",
                        "tool_call_id": "call_0_1",
                    }
                ],
            },
            {"turn_number": 0, "role": "tool", "content": '{"status":"ok"}', "tool_calls": []},
            {"turn_number": 0, "role": "agent", "content": "Done.", "tool_calls": []},
        ],
        "judges": {
            "task_completion": judges("task_completion", tc_judges),
            "tool_selection": judges("tool_selection", ts_judges),
        },
        "sim_meta": {"completed": True, "total_turns": 2},
    }
    if state_score is not None:
        art["state"] = {"score": state_score, "checks": [], "final_world": {}}
    return art


def _write_artifacts_run(root, specs):
    """Write a run's artifact tree: root/{model-slug}/{scenario}_run{idx}.json."""
    from eval.artifacts import model_slug

    for art in specs:
        d = root / model_slug(art["model"])
        d.mkdir(parents=True, exist_ok=True)
        name = f"{art['scenario_id']}_run{art['run_index']}.json"
        (d / name).write_text(json.dumps(art), encoding="utf-8")


def _make_population(root, n_per_band=4):
    """A population spanning two domains, two categories, all three bands."""
    specs = []
    bands = {
        "low": [("Kimi", 0.0, False), ("GLM", 0.1, False), ("Opus", 0.0, False)],
        "mid": [("Kimi", 0.5, False), ("GLM", 0.5, False), ("Opus", 0.6, False)],
        "high": [("Kimi", 1.0, False), ("GLM", 0.9, False), ("Opus", 1.0, False)],
    }
    idx = 0
    for domain in ("banking", "customer_success"):
        for category in ("adaptive_tool_use", "scope_management"):
            for band, jspec in bands.items():
                for _ in range(n_per_band):
                    idx += 1
                    # state_score chosen so reconstructed efficacy lands in band:
                    # low all 0, mid all 0.5, high all ~1.
                    sval = {"low": 0.0, "mid": 0.5, "high": 1.0}[band]
                    specs.append(
                        _artifact(
                            f"{domain}_{idx:03d}",
                            "GPT-4.1",
                            0,
                            domain,
                            category,
                            jspec,
                            jspec,
                            state_score=sval,
                        )
                    )
    _write_artifacts_run(root, specs)
    return specs


# --------------------------------------------------------------------------- #
# Banding + loading
# --------------------------------------------------------------------------- #


class TestBanding:
    def test_band_edges(self):
        assert _band_for(0.0) == "low"
        assert _band_for(0.33) == "low"
        assert _band_for(0.5) == "mid"
        assert _band_for(0.66) == "mid"
        assert _band_for(0.8) == "high"
        assert _band_for(1.0) == "high"

    def test_load_reconstructs_consensus_and_band(self, tmp_path):
        _write_artifacts_run(
            tmp_path,
            [
                _artifact(
                    "banking_001",
                    "GPT-4.1",
                    0,
                    "banking",
                    "adaptive_tool_use",
                    [("Kimi", 0.8, False), ("GLM", 0.6, False), ("Opus", 1.0, False)],
                    [("Kimi", 0.7, False), ("GLM", 0.7, False), ("Opus", 0.5, False)],
                    state_score=1.0,
                )
            ],
        )
        evals = load_evaluations(tmp_path)
        assert len(evals) == 1
        ev = evals[0]
        # median of (0.8, 0.6, 1.0) = 0.8 ; (0.7, 0.7, 0.5) = 0.7
        assert ev.consensus["task_completion"] == 0.8
        assert ev.consensus["tool_selection"] == 0.7
        # efficacy = 0.4*0.8 + 0.3*0.7 + 0.3*1.0 = 0.83 -> high
        assert ev.efficacy == pytest.approx(0.83)
        assert ev.band == "high"

    def test_parse_failed_judge_excluded_from_consensus(self, tmp_path):
        _write_artifacts_run(
            tmp_path,
            [
                _artifact(
                    "banking_002",
                    "GPT-4.1",
                    0,
                    "banking",
                    "scope_management",
                    [("Kimi", 0.8, False), ("Opus", 0.0, True)],
                    [("Kimi", 0.6, False)],
                )
            ],
        )
        ev = load_evaluations(tmp_path)[0]
        # Opus parse-failed -> excluded; consensus is just Kimi's 0.8.
        assert ev.consensus["task_completion"] == 0.8
        assert "Opus" not in ev.per_judge["task_completion"]

    def test_domain_derived_from_scenario_id_when_absent(self, tmp_path):
        art = _artifact(
            "customer_success_009",
            "GPT-4.1",
            0,
            "banking",  # will be removed below
            "adaptive_tool_use",
            [("Kimi", 0.5, False)],
            [("Kimi", 0.5, False)],
        )
        del art["domain"]
        _write_artifacts_run(tmp_path, [art])
        ev = load_evaluations(tmp_path)[0]
        assert ev.domain == "customer_success"


# --------------------------------------------------------------------------- #
# Stratification determinism + coverage
# --------------------------------------------------------------------------- #


class TestStratification:
    def test_deterministic_same_seed(self, tmp_path):
        _make_population(tmp_path)
        evals = load_evaluations(tmp_path)
        a = stratified_sample(evals, 24, seed=33)
        b = stratified_sample(evals, 24, seed=33)
        assert [e.artifact_id for e in a] == [e.artifact_id for e in b]

    def test_different_seed_changes_sample(self, tmp_path):
        _make_population(tmp_path)
        evals = load_evaluations(tmp_path)
        a = stratified_sample(evals, 24, seed=33)
        b = stratified_sample(evals, 24, seed=99)
        assert [e.artifact_id for e in a] != [e.artifact_id for e in b]

    def test_order_independent(self, tmp_path):
        """Shuffling the input must not change the sampled set (determinism)."""
        import random as _r

        _make_population(tmp_path)
        evals = load_evaluations(tmp_path)
        shuffled = list(evals)
        _r.Random(7).shuffle(shuffled)
        a = stratified_sample(evals, 24, seed=33)
        b = stratified_sample(shuffled, 24, seed=33)
        assert sorted(e.artifact_id for e in a) == sorted(e.artifact_id for e in b)

    def test_spans_all_bands(self, tmp_path):
        _make_population(tmp_path)
        evals = load_evaluations(tmp_path)
        sample = stratified_sample(evals, 24, seed=33)
        bands = {e.band for e in sample}
        assert bands == {"low", "mid", "high"}

    def test_proportional_allocation_sums_to_n(self, tmp_path):
        _make_population(tmp_path)
        evals = load_evaluations(tmp_path)
        sample = stratified_sample(evals, 30, seed=33)
        assert len(sample) == 30

    def test_caps_at_population(self, tmp_path):
        _make_population(tmp_path, n_per_band=1)
        evals = load_evaluations(tmp_path)  # 2*2*3 = 12 total
        sample = stratified_sample(evals, 80, seed=33)
        assert len(sample) == 12


# --------------------------------------------------------------------------- #
# Blindness: judge scores must NOT appear in the workbook
# --------------------------------------------------------------------------- #


class TestBlindness:
    def test_sheet_has_no_judge_scores(self):
        ev = Evaluation(
            artifact_id="gpt-4-1/banking_001_run0",
            artifact_path=None,
            scenario_id="banking_001",
            model="GPT-4.1",
            run_index=0,
            domain="banking",
            category="adaptive_tool_use",
            transcript=[{"turn_number": 0, "role": "user", "content": "hi", "tool_calls": []}],
            consensus={"task_completion": 0.8, "tool_selection": 0.7},
            per_judge={
                "task_completion": {"Kimi": 0.8, "Opus": 0.9},
                "tool_selection": {"Kimi": 0.7},
            },
            state_score=1.0,
            efficacy=0.83,
            band="high",
        )
        sheet = render_sheet(ev, 1, 1)
        # The judge names and their scores must be absent.
        assert "Kimi" not in sheet
        assert "Opus" not in sheet
        assert "0.8" not in sheet
        assert "0.9" not in sheet
        assert "consensus" not in sheet.lower()
        # But the blank score block and rubric anchors ARE present.
        assert "task_completion: _" in sheet
        assert "tool_selection: _" in sheet
        assert "Task Completion (0.0-1.0)" in sheet

    def test_workbook_files_have_no_judge_scores(self, tmp_path):
        _make_population(tmp_path, n_per_band=2)
        evals = load_evaluations(tmp_path)
        sample = stratified_sample(evals, 12, seed=33)
        wb = tmp_path / "workbook"
        key_path = write_workbook(sample, wb, seed=33, source="src")

        # Key file carries the judge scores...
        key = json.loads(key_path.read_text(encoding="utf-8"))
        assert any(e["per_judge"]["task_completion"] for e in key["evaluations"].values())
        # ...the key file lives OUTSIDE the workbook dir.
        assert key_path.parent == wb.parent
        assert key_path not in list(wb.glob("*"))

        # No workbook sheet mentions a judge name.
        for sheet in wb.glob("*.md"):
            text = sheet.read_text(encoding="utf-8")
            assert "Kimi" not in text
            assert "GLM" not in text
            assert "Opus" not in text
            assert "per_judge" not in text


# --------------------------------------------------------------------------- #
# Agreement math on known values
# --------------------------------------------------------------------------- #


class TestAgreementMath:
    def test_perfect_agreement(self):
        a = _agreement([0.8, 0.5, 0.2], [0.8, 0.5, 0.2])
        assert a.mean_abs_diff == 0.0
        assert a.alpha == pytest.approx(1.0)
        assert a.pearson == pytest.approx(1.0)

    def test_mean_abs_diff_known(self):
        a = _agreement([1.0, 0.0, 0.5], [0.0, 1.0, 0.5])
        # |1-0| + |0-1| + |0.5-0.5| = 2.0 over 3 = 0.6667
        assert a.mean_abs_diff == pytest.approx(2.0 / 3.0)

    def test_systematic_flip_negative_alpha(self):
        a = _agreement([0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0])
        assert a.alpha < 0

    def test_negative_pearson(self):
        a = _agreement([0.0, 0.5, 1.0], [1.0, 0.5, 0.0])
        assert a.pearson == pytest.approx(-1.0)

    def test_alpha_matches_krippendorff_reference(self):
        """Our two-rater alpha equals krippendorff_alpha on the same matrix."""
        from eval.scoring.agreement import krippendorff_alpha

        human = [0.8, 0.6, 0.2, 0.9]
        judge = [0.7, 0.6, 0.3, 1.0]
        a = _agreement(human, judge)
        expected = krippendorff_alpha([[h, j] for h, j in zip(human, judge)])
        assert a.alpha == pytest.approx(expected)

    def test_empty_is_none(self):
        a = _agreement([], [])
        assert a.n == 0
        assert a.alpha is None
        assert a.mean_abs_diff is None


# --------------------------------------------------------------------------- #
# Label reading + round trip
# --------------------------------------------------------------------------- #


class TestLabelReading:
    def test_reads_filled_scores(self, tmp_path):
        _make_population(tmp_path, n_per_band=1)
        evals = load_evaluations(tmp_path)
        sample = stratified_sample(evals, 12, seed=33)
        wb = tmp_path / "wb"
        write_workbook(sample, wb, seed=33, source="src")

        # Fill the first sheet's scores.
        sheet = sorted(wb.glob("sheet_*.md"))[0]
        text = sheet.read_text(encoding="utf-8")
        text = text.replace("task_completion: _", "task_completion: 0.75")
        text = text.replace("tool_selection: _", "tool_selection: 0.5")
        sheet.write_text(text, encoding="utf-8")

        labels = read_labels(wb)
        # The filled sheet has scores; the rest are None.
        filled = [v for v in labels.values() if v["task_completion"] is not None]
        assert len(filled) == 1
        assert filled[0]["task_completion"] == 0.75
        assert filled[0]["tool_selection"] == 0.5

    def test_skipped_field_is_none(self, tmp_path):
        ev = Evaluation(
            artifact_id="gpt-4-1/banking_001_run0",
            artifact_path=None,
            scenario_id="banking_001",
            model="GPT-4.1",
            run_index=0,
            domain="banking",
            category="adaptive_tool_use",
            transcript=[],
            consensus={"task_completion": 0.8, "tool_selection": 0.7},
            per_judge={"task_completion": {"Kimi": 0.8}, "tool_selection": {"Kimi": 0.7}},
            state_score=None,
            efficacy=0.5,
            band="mid",
        )
        wb = tmp_path / "wb"
        write_workbook([ev], wb, seed=33, source="src")
        labels = read_labels(wb)
        assert labels["gpt-4-1/banking_001_run0"] == {
            "task_completion": None,
            "tool_selection": None,
        }


class TestRoundTrip:
    def test_sample_label_score(self, tmp_path):
        _make_population(tmp_path, n_per_band=2)
        evals = load_evaluations(tmp_path)
        sample = stratified_sample(evals, 12, seed=33)
        wb = tmp_path / "wb"
        key_path = write_workbook(sample, wb, seed=33, source="rehearsal")

        # Human labels every sheet EXACTLY equal to the judge consensus, so
        # alpha must be perfect and MAD zero (a known-value end-to-end check).
        key = json.loads(key_path.read_text(encoding="utf-8"))
        for sheet in sorted(wb.glob("sheet_*.md")):
            text = sheet.read_text(encoding="utf-8")
            import re as _re

            aid = _re.search(r"\*\*Artifact:\*\*\s*`([^`]+)`", text).group(1)
            cons = key["evaluations"][aid]["consensus"]
            text = text.replace("task_completion: _", f"task_completion: {cons['task_completion']}")
            text = text.replace("tool_selection: _", f"tool_selection: {cons['tool_selection']}")
            sheet.write_text(text, encoding="utf-8")

        labels = read_labels(wb)
        calib = compute_calibration(labels, key)
        assert calib["n_matched"] == 12
        for dim in DIMENSIONS:
            cons = calib["dimensions"][dim]["consensus"]
            assert cons["mean_abs_diff"] == pytest.approx(0.0)
            assert cons["alpha"] == pytest.approx(1.0)

        report = render_report(calib, key)
        assert "judge calibration report" in report
        assert "human vs **consensus**" in report

    def test_unmatched_label_reported(self, tmp_path):
        key = {
            "seed": 33,
            "source": "src",
            "evaluations": {
                "gpt-4-1/banking_001_run0": {
                    "consensus": {"task_completion": 0.8, "tool_selection": 0.7},
                    "per_judge": {
                        "task_completion": {"Kimi": 0.8},
                        "tool_selection": {"Kimi": 0.7},
                    },
                }
            },
        }
        labels = {
            "gpt-4-1/banking_001_run0": {"task_completion": 0.8, "tool_selection": 0.7},
            "gpt-4-1/ghost_run0": {"task_completion": 0.5, "tool_selection": 0.5},
        }
        calib = compute_calibration(labels, key)
        assert "gpt-4-1/ghost_run0" in calib["unmatched_artifacts"]
        assert calib["n_matched"] == 1


class TestKeyFile:
    def test_key_has_judge_scores_and_consensus(self, tmp_path):
        _make_population(tmp_path, n_per_band=1)
        evals = load_evaluations(tmp_path)
        sample = stratified_sample(evals, 6, seed=33)
        key = build_key(sample, seed=33, source="src")
        assert key["issue"] == 33
        for entry in key["evaluations"].values():
            assert set(entry["consensus"]) == set(DIMENSIONS)
            assert "per_judge" in entry
