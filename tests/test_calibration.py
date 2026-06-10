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
from types import SimpleNamespace

import pytest

from eval.artifacts import build_artifact, model_slug
from eval.simulation.runner import ConversationTurn, ToolCall
from scripts.calibration import (
    DIMENSIONS,
    Evaluation,
    _agreement,
    _band_for,
    build_key,
    compute_calibration,
    dedup_reliability_runs,
    exclude_holdout,
    load_evaluations,
    read_labels,
    render_report,
    render_sheet,
    sheet_id_for,
    stratified_sample,
    write_workbook,
)

# --------------------------------------------------------------------------- #
# Synthetic artifact builders — PRODUCTION-SHAPED via eval.artifacts.build_artifact
# --------------------------------------------------------------------------- #
#
# Tests build artifacts through the SAME build_artifact the eval run uses, so the
# synthetic JSON can never silently drift from the real schema (that drift is
# exactly what hid the issue #46 stratification bug). The fakes below carry only
# the attributes build_artifact reads off sim_result / consensus results.


def _sim_result():
    """A minimal sim result with real ConversationTurn/ToolCall dataclasses.

    build_artifact calls dataclasses.asdict on each turn, so these must be the
    real dataclasses, not dicts.
    """
    turns = [
        ConversationTurn(turn_number=0, role="user", content="Move $500 to savings"),
        ConversationTurn(
            turn_number=0,
            role="agent",
            content="Transferring now.",
            tool_calls=[
                ToolCall(
                    turn=0,
                    tool_name="transfer_funds",
                    arguments={"amount": 500, "to": "savings"},
                    result="",
                )
            ],
        ),
        ConversationTurn(turn_number=0, role="tool", content='{"status":"ok"}'),
        ConversationTurn(turn_number=0, role="agent", content="Done."),
    ]
    return SimpleNamespace(
        turns=turns,
        completed=True,
        total_turns=2,
        total_input_tokens=100,
        total_output_tokens=50,
        total_latency_ms=1234.0,
        error=None,
        resolved_model="fake/model",
        ended_by="user_sim",
        state_progress_at_end=1.0,
        premature_end=False,
        final_world={},
    )


def _consensus(rubric_type, specs):
    """Fake ConsensusResult exposing only .judge_results (what build_artifact reads).

    ``specs`` are (judge_name, score, parse_failed) tuples.
    """
    judge_results = [
        SimpleNamespace(
            judge_name=name,
            rubric_type=rubric_type,
            overall_score=score,
            reasoning=f"{name} reasoning",
            parse_failed=failed,
            resolved_model="",
            raw_response={"overall_score": score},
        )
        for name, score, failed in specs
    ]
    return SimpleNamespace(judge_results=judge_results)


def _artifact(
    scenario_id,
    model,
    run_index,
    domain,
    category,
    tc_judges,
    ts_judges,
    state_score=None,
    holdout=False,
):
    """Build one PRODUCTION-SHAPED artifact dict via eval.artifacts.build_artifact.

    ``tc_judges`` / ``ts_judges`` are lists of (judge_name, score, parse_failed).
    """
    state = None if state_score is None else {"score": state_score, "checks": []}
    return build_artifact(
        scenario_id=scenario_id,
        model=model,
        run_index=run_index,
        sim_result=_sim_result(),
        tc_result=_consensus("task_completion", tc_judges),
        ts_result=_consensus("tool_selection", ts_judges),
        state=state,
        domain=domain,
        category=category,
        holdout=holdout,
    )


def _write_artifacts_run(root, specs):
    """Write a run's artifact tree: root/{model-slug}/{scenario}_run{idx}.json."""
    for art in specs:
        d = root / model_slug(art["model"])
        d.mkdir(parents=True, exist_ok=True)
        name = f"{art['scenario_id']}_run{art['run_index']}.json"
        (d / name).write_text(json.dumps(art), encoding="utf-8")


# Real corpus uses "banking_" and "cs_" id prefixes; domain VALUES are
# "banking" / "customer_success". The mismatch (cs_ vs customer_success) is the
# crux of issue #46 — keep the test population faithful to it.
_DOMAIN_ID_PREFIX = {"banking": "banking", "customer_success": "cs"}


def _make_population(root, n_per_band=4):
    """A population spanning two domains, two categories, all three bands.

    Scenario ids follow the REAL corpus form (e.g. banking_adaptive_tool_use_0001
    / cs_scope_management_0007) so the loader is exercised against production-
    shaped ids, not fictional ones.
    """
    specs = []
    # Per-judge scores are chosen to be distinctive (not 0.0/0.1/0.5/1.0, which
    # appear verbatim in the rubric anchor prose) so the blindness scan tests a
    # real leak signal, not a coincidental rubric-text match. Medians still land
    # each stratum in its intended band.
    bands = {
        "low": [("kimi", 0.12, False), ("glm", 0.15, False), ("opus", 0.08, False)],
        "mid": [("kimi", 0.45, False), ("glm", 0.55, False), ("opus", 0.62, False)],
        "high": [("kimi", 0.92, False), ("glm", 0.88, False), ("opus", 0.95, False)],
    }
    idx = 0
    for domain in ("banking", "customer_success"):
        prefix = _DOMAIN_ID_PREFIX[domain]
        for category in ("adaptive_tool_use", "scope_management"):
            for band, jspec in bands.items():
                for _ in range(n_per_band):
                    idx += 1
                    # state_score chosen so reconstructed efficacy lands in band:
                    # low all 0, mid all 0.5, high all ~1.
                    sval = {"low": 0.0, "mid": 0.5, "high": 1.0}[band]
                    specs.append(
                        _artifact(
                            f"{prefix}_{category}_{idx:04d}",
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
                    "banking_adaptive_tool_use_0001",
                    "GPT-4.1",
                    0,
                    "banking",
                    "adaptive_tool_use",
                    [("kimi", 0.8, False), ("glm", 0.6, False), ("opus", 1.0, False)],
                    [("kimi", 0.7, False), ("glm", 0.7, False), ("opus", 0.5, False)],
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
                    "banking_scope_management_0002",
                    "GPT-4.1",
                    0,
                    "banking",
                    "scope_management",
                    [("kimi", 0.8, False), ("opus", 0.0, True)],
                    [("kimi", 0.6, False)],
                )
            ],
        )
        ev = load_evaluations(tmp_path)[0]
        # opus parse-failed -> excluded; consensus is just kimi's 0.8.
        assert ev.consensus["task_completion"] == 0.8
        assert "opus" not in ev.per_judge["task_completion"]

    def test_explicit_domain_and_category_used(self, tmp_path):
        """Post-#46 artifacts carry top-level domain/category — use them directly.

        Real id is cs_adaptive_tool_use_0001 but domain is customer_success: the
        explicit field must win, never the id prefix.
        """
        _write_artifacts_run(
            tmp_path,
            [
                _artifact(
                    "cs_adaptive_tool_use_0001",
                    "GPT-4.1",
                    0,
                    "customer_success",
                    "adaptive_tool_use",
                    [("kimi", 0.5, False)],
                    [("kimi", 0.5, False)],
                )
            ],
        )
        ev = load_evaluations(tmp_path)[0]
        assert ev.domain == "customer_success"
        assert ev.category == "adaptive_tool_use"

    @pytest.mark.parametrize(
        "scenario_id,expected_domain",
        [
            ("banking_adaptive_tool_use_0001", "banking"),
            ("cs_adaptive_tool_use_0001", "customer_success"),
            ("cs_churn_intervention_0002", "customer_success"),
            # 8-char-hash variant from a generated batch.
            ("banking_adaptive_tool_use_0000_28d29485", "banking"),
            ("cs_adaptive_tool_use_0000_59bb2918", "customer_success"),
        ],
    )
    def test_legacy_domain_from_real_id_prefix(self, tmp_path, scenario_id, expected_domain):
        """Legacy artifacts (no top-level domain) derive domain from the REAL id
        prefix: banking_ -> banking, cs_ -> customer_success. Category is unknown.
        """
        art = _artifact(
            scenario_id,
            "GPT-4.1",
            0,
            "banking",  # removed below to simulate a pre-#46 artifact
            "adaptive_tool_use",
            [("kimi", 0.5, False)],
            [("kimi", 0.5, False)],
        )
        del art["domain"]
        del art["category"]
        _write_artifacts_run(tmp_path, [art])
        ev = load_evaluations(tmp_path)[0]
        assert ev.domain == expected_domain
        assert ev.category == "unknown"


class TestArtifactSchema:
    """Guards against the issue #46 drift: the synthetic artifacts the calibration
    tests run on MUST match what eval.artifacts.build_artifact emits.
    """

    def test_synthetic_artifact_matches_build_artifact_keys(self):
        # Built by the test helper (which calls build_artifact) — its top-level
        # keys must equal a freshly-built artifact's. If build_artifact gains or
        # loses a field, this fails instead of letting the tests run on a stale
        # shape.
        synthetic = _artifact(
            "banking_adaptive_tool_use_0001",
            "GPT-4.1",
            0,
            "banking",
            "adaptive_tool_use",
            [("kimi", 0.5, False)],
            [("kimi", 0.5, False)],
            state_score=1.0,
        )
        direct = build_artifact(
            scenario_id="banking_adaptive_tool_use_0001",
            model="GPT-4.1",
            run_index=0,
            sim_result=_sim_result(),
            tc_result=_consensus("task_completion", [("kimi", 0.5, False)]),
            ts_result=_consensus("tool_selection", [("kimi", 0.5, False)]),
            state={"score": 1.0, "checks": []},
            domain="banking",
            category="adaptive_tool_use",
        )
        assert set(synthetic) == set(direct)

    def test_build_artifact_persists_domain_category_holdout(self):
        art = build_artifact(
            scenario_id="cs_adaptive_tool_use_0001",
            model="GPT-4.1",
            run_index=0,
            sim_result=_sim_result(),
            tc_result=_consensus("task_completion", [("kimi", 0.5, False)]),
            ts_result=_consensus("tool_selection", [("kimi", 0.5, False)]),
            domain="customer_success",
            category="adaptive_tool_use",
            holdout=True,
        )
        assert art["domain"] == "customer_success"
        assert art["category"] == "adaptive_tool_use"
        assert art["holdout"] is True


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
            artifact_id="gpt-4-1/banking_adaptive_tool_use_0001_run0",
            artifact_path=None,
            scenario_id="banking_adaptive_tool_use_0001",
            model="GPT-4.1",
            run_index=0,
            domain="banking",
            category="adaptive_tool_use",
            holdout=False,
            transcript=[{"turn_number": 0, "role": "user", "content": "hi", "tool_calls": []}],
            consensus={"task_completion": 0.8, "tool_selection": 0.7},
            per_judge={
                "task_completion": {"kimi": 0.8, "opus": 0.9},
                "tool_selection": {"kimi": 0.7},
            },
            state_score=1.0,
            efficacy=0.83,
            band="high",
        )
        sheet = render_sheet(ev, 1, 1, seed=33)
        # The judge names and their scores must be absent.
        assert "kimi" not in sheet.lower()
        assert "opus" not in sheet.lower()
        assert "0.8" not in sheet
        assert "0.9" not in sheet
        assert "consensus" not in sheet.lower()
        # Model identity must NOT appear (nit #5): no real artifact id / slug.
        assert "gpt-4-1" not in sheet
        assert "GPT-4.1" not in sheet
        assert ev.artifact_id not in sheet
        # But the blank score block and rubric anchors ARE present.
        assert "task_completion: _" in sheet
        assert "tool_selection: _" in sheet
        assert "Task Completion (0.0-1.0)" in sheet

    def test_workbook_files_have_no_judge_scores(self, tmp_path):
        """General blindness scan: no JUDGES name and no key file numeric value
        may appear in ANY emitted sheet — derived from config + the key file, not
        a hardcoded allowlist (finding #4).
        """
        from eval.config import JUDGES

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

        # Collect EVERY judge name from config and EVERY numeric score the key
        # holds (per-judge, consensus, efficacy, state_score), as the strings a
        # leak would print. This catches a future regression that printed, say, a
        # rounded consensus — not just the three names a hardcoded list knows.
        # The leak set is the per-judge and consensus scores from the key — the
        # numbers blindness must never surface. efficacy/state_score are excluded:
        # they coincide with the difficulty-band concept (intentionally surfaced)
        # and with rubric-anchor prose (which prints fixed 0.0/0.5/1.0 examples),
        # so they'd be coincidental matches, not regressions of the kind this
        # guards against (a printed judge/consensus score). The population's
        # judge scores are picked to be distinct from any rubric number.
        judge_names = set(JUDGES.keys())
        leak_numbers: set[str] = set()
        for e in key["evaluations"].values():
            for d in DIMENSIONS:
                for v in e["per_judge"][d].values():
                    leak_numbers.add(str(v))
                if e["consensus"][d] is not None:
                    leak_numbers.add(str(e["consensus"][d]))
        assert leak_numbers, "expected non-empty set of key score values to scan for"

        for sheet in wb.glob("*.md"):
            text = sheet.read_text(encoding="utf-8")
            lower = text.lower()
            for name in judge_names:
                assert name.lower() not in lower, f"judge name {name!r} leaked into {sheet.name}"
            assert "per_judge" not in text
            assert "consensus" not in lower
            for num in leak_numbers:
                assert num not in text, f"key value {num!r} leaked into {sheet.name}"


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
            artifact_id="gpt-4-1/banking_adaptive_tool_use_0001_run0",
            artifact_path=None,
            scenario_id="banking_adaptive_tool_use_0001",
            model="GPT-4.1",
            run_index=0,
            domain="banking",
            category="adaptive_tool_use",
            holdout=False,
            transcript=[],
            consensus={"task_completion": 0.8, "tool_selection": 0.7},
            per_judge={"task_completion": {"kimi": 0.8}, "tool_selection": {"kimi": 0.7}},
            state_score=None,
            efficacy=0.5,
            band="mid",
        )
        wb = tmp_path / "wb"
        write_workbook([ev], wb, seed=33, source="src")
        labels = read_labels(wb)
        # Labels are keyed by the opaque sheet token, not the model-revealing id.
        token = sheet_id_for(ev.artifact_id, 33)
        assert labels[token] == {
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

            aid = _re.search(r"\*\*Sheet:\*\*\s*`([^`]+)`", text).group(1)
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

    def test_key_carries_real_artifact_id_and_model(self, tmp_path):
        """The key (kept out of the workbook) DOES record the real id + model, so
        a result is auditable; the model identity is just absent from the sheets.
        """
        _make_population(tmp_path, n_per_band=1)
        evals = load_evaluations(tmp_path)
        sample = stratified_sample(evals, 6, seed=33)
        key = build_key(sample, seed=33, source="src")
        # Keyed by opaque token, with the real id + model inside each entry.
        for token, entry in key["evaluations"].items():
            assert token.startswith("tx-")
            assert "artifact_id" in entry
            assert entry["model"] == "GPT-4.1"


# --------------------------------------------------------------------------- #
# Private-holdout exclusion (issue #31 interaction) + reliability-run dedup
# --------------------------------------------------------------------------- #


class TestHoldoutExclusion:
    def test_holdout_loaded_from_artifact_flag(self, tmp_path):
        _write_artifacts_run(
            tmp_path,
            [
                _artifact(
                    "banking_adaptive_tool_use_0001",
                    "GPT-4.1",
                    0,
                    "banking",
                    "adaptive_tool_use",
                    [("kimi", 0.5, False)],
                    [("kimi", 0.5, False)],
                    holdout=True,
                )
            ],
        )
        ev = load_evaluations(tmp_path)[0]
        assert ev.holdout is True

    def test_excluded_by_default(self, tmp_path):
        specs = [
            _artifact(
                "banking_adaptive_tool_use_0001",
                "GPT-4.1",
                0,
                "banking",
                "adaptive_tool_use",
                [("kimi", 0.5, False)],
                [("kimi", 0.5, False)],
                holdout=False,
            ),
            _artifact(
                "banking_adaptive_tool_use_0002",
                "GPT-4.1",
                0,
                "banking",
                "adaptive_tool_use",
                [("kimi", 0.5, False)],
                [("kimi", 0.5, False)],
                holdout=True,
            ),
        ]
        _write_artifacts_run(tmp_path, specs)
        evals = load_evaluations(tmp_path)
        kept, dropped = exclude_holdout(evals)
        assert dropped == 1
        assert {e.scenario_id for e in kept} == {"banking_adaptive_tool_use_0001"}
        assert all(not e.holdout for e in kept)


class TestReliabilityDedup:
    def _runs(self, n_runs):
        return [
            _artifact(
                "banking_adaptive_tool_use_0001",
                "GPT-4.1",
                idx,
                "banking",
                "adaptive_tool_use",
                [("kimi", 0.5, False)],
                [("kimi", 0.5, False)],
            )
            for idx in range(n_runs)
        ]

    def test_collapses_to_one_per_scenario_model(self, tmp_path):
        _write_artifacts_run(tmp_path, self._runs(4))
        evals = load_evaluations(tmp_path)
        assert len(evals) == 4
        deduped = dedup_reliability_runs(evals, seed=33)
        assert len(deduped) == 1
        assert deduped[0].scenario_id == "banking_adaptive_tool_use_0001"

    def test_keeps_distinct_scenario_model_pairs(self, tmp_path):
        specs = self._runs(3) + [
            _artifact(
                "banking_adaptive_tool_use_0002",
                "GPT-4.1",
                0,
                "banking",
                "adaptive_tool_use",
                [("kimi", 0.5, False)],
                [("kimi", 0.5, False)],
            )
        ]
        _write_artifacts_run(tmp_path, specs)
        evals = load_evaluations(tmp_path)
        deduped = dedup_reliability_runs(evals, seed=33)
        assert {e.scenario_id for e in deduped} == {
            "banking_adaptive_tool_use_0001",
            "banking_adaptive_tool_use_0002",
        }

    def test_deterministic_choice(self, tmp_path):
        _write_artifacts_run(tmp_path, self._runs(5))
        evals = load_evaluations(tmp_path)
        a = dedup_reliability_runs(evals, seed=33)
        b = dedup_reliability_runs(evals, seed=33)
        assert [e.artifact_id for e in a] == [e.artifact_id for e in b]

    def test_seed_changes_choice(self, tmp_path):
        # With 5 runs, two different seeds should be able to pick different runs.
        _write_artifacts_run(tmp_path, self._runs(5))
        evals = load_evaluations(tmp_path)
        picks = {dedup_reliability_runs(evals, seed=s)[0].artifact_id for s in range(20)}
        assert len(picks) > 1
