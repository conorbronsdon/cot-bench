"""Tests for atomic rubric criteria (issue #54).

Covers the four load-bearing properties of the staged design:

1. **Byte-identical fallback** — a scenario WITHOUT rubric_criteria produces
   judge prompts (combined and separate) byte-identical to today's bare
   templates, and the judge call path is unchanged.
2. **Criterion-informed scoring** — when criteria exist for a dimension, the
   JudgeResult's overall_score is the weighted fraction of met criteria, with
   the judge's holistic template score preserved on ``holistic_score`` and the
   per-criterion verdicts attached (for the halo-effect comparison).
3. **Strict verdict parsing** — a judge asked for per-criterion verdicts that
   returns a missing/partial/duplicated/mistyped block is a parse failure
   (one retry, then excluded), same as the existing whole-judge rule.
4. **Validation + provenance** — criteria are 3-6 atomic items with unique ids,
   valid judge dimensions, sane weights, and an honest ``criteria_authorship``
   stamp whose author is not a contestant; the corpus hash covers criteria
   when present and is unchanged for criteria-less scenarios.

All tests are deterministic and offline (judge API calls are monkeypatched).
"""

import json
from dataclasses import dataclass

import pytest

import eval.scoring.judge as judge_mod
from eval.config import Domain
from eval.pre_registration import (
    _scenario_to_canonical_dict,
    scenario_set_hash,
)
from eval.scoring.judge import (
    _extract_criteria_verdicts,
    score_with_all_judges_combined,
    score_with_judge,
    score_with_judge_combined,
)
from eval.scoring.rubrics import (
    COMBINED_RUBRIC,
    TASK_COMPLETION_RUBRIC,
    TOOL_SELECTION_RUBRIC,
    aggregate_criterion_score,
    build_combined_prompt,
    build_task_completion_prompt,
    build_tool_selection_prompt,
    criteria_for_dimension,
)
from eval.simulation.runner import Scenario
from scripts.generate_data import stamp_criteria
from scripts.validate_scenarios import validate_scenario_dict

# --- Shared fixtures ---------------------------------------------------------

PROMPT_KWARGS = {
    "domain": "banking",
    "user_goals": "- check balance\n- transfer funds",
    "available_tools": '[{"name": "get_balance"}]',
    "transcript": "[Turn 1 - USER]: hi",
}

CRITERIA = [
    {
        "id": "verify_before_balance",
        "text": "Agent verified the customer's identity before disclosing any balance.",
        "dimension": "task_completion",
        "weight": 2.0,
    },
    {
        "id": "correct_transfer_amount",
        "text": "Agent initiated the transfer with the exact amount the user requested.",
        "dimension": "task_completion",
        "weight": 1.0,
    },
    {
        "id": "no_redundant_lookups",
        "text": "Agent did not repeat a lookup tool call whose result it already had.",
        "dimension": "tool_selection",
        "weight": 1.0,
    },
]


def _verdict_json(met_by_id, tc=0.9, ts=0.6):
    """A valid combined judge response carrying per-criterion verdicts."""
    return json.dumps(
        {
            "task_completion": {"overall_score": tc, "overall_reasoning": "ok-tc"},
            "tool_selection": {"overall_score": ts, "overall_reasoning": "ok-ts"},
            "rubric_criteria": [
                {"id": cid, "met": met, "evidence": f"turn ref for {cid}"}
                for cid, met in met_by_id.items()
            ],
        }
    )


@dataclass(frozen=True)
class _Cfg:
    name: str = "Solo"
    provider: str = "anthropic"
    model_id: str = "x"
    temperature: float = 0.0
    max_tokens: int = 4096
    endpoint: str | None = None


def _base_scenario_dict() -> dict:
    """A minimal valid (pre-v0.2) scenario dict for validator tests."""
    return {
        "id": "banking_adaptive_tool_use_9999",
        "category": "adaptive_tool_use",
        "persona": {
            "name": "Test Person",
            "age": 40,
            "occupation": "Engineer",
            "personality_traits": ["calm"],
            "tone": "neutral",
            "detail_level": "moderate",
            "background": "A test persona used only in unit tests.",
        },
        "user_goals": ["check balance", "transfer funds", "ask about rates"],
        "tools": [
            {"name": "get_balance", "description": "Get balance", "parameters": []},
            {"name": "transfer", "description": "Transfer funds", "parameters": []},
        ],
        "initial_message": "Hello, I need help with my account today.",
    }


def _criteria_scenario_dict() -> dict:
    data = _base_scenario_dict()
    data["rubric_criteria"] = [dict(c) for c in CRITERIA]
    data["criteria_authorship"] = {"criteria_author_model": "anthropic/claude-opus-4.8"}
    return data


# --- 1. Byte-identical fallback ----------------------------------------------


class TestPromptFallback:
    def test_combined_prompt_without_criteria_is_byte_identical(self):
        # THE backwards-compatibility guarantee: a criteria-less scenario's
        # combined judge prompt is unchanged from today's template, byte for byte.
        assert build_combined_prompt(**PROMPT_KWARGS) == COMBINED_RUBRIC.format(**PROMPT_KWARGS)
        assert build_combined_prompt(
            **PROMPT_KWARGS, rubric_criteria=None
        ) == COMBINED_RUBRIC.format(**PROMPT_KWARGS)
        # Empty list is treated the same as None.
        assert build_combined_prompt(**PROMPT_KWARGS, rubric_criteria=[]) == COMBINED_RUBRIC.format(
            **PROMPT_KWARGS
        )

    def test_task_completion_prompt_without_criteria_is_byte_identical(self):
        assert build_task_completion_prompt(**PROMPT_KWARGS) == TASK_COMPLETION_RUBRIC.format(
            **PROMPT_KWARGS
        )

    def test_tool_selection_prompt_without_criteria_is_byte_identical(self):
        kwargs = {k: v for k, v in PROMPT_KWARGS.items() if k != "user_goals"}
        assert build_tool_selection_prompt(**kwargs) == TOOL_SELECTION_RUBRIC.format(**kwargs)

    def test_separate_prompt_with_no_criteria_for_its_dimension_is_byte_identical(self):
        # Criteria exist, but all map to tool_selection — the task-completion
        # prompt must still be the bare template.
        ts_only = [c for c in CRITERIA if c["dimension"] == "tool_selection"]
        assert build_task_completion_prompt(
            **PROMPT_KWARGS, rubric_criteria=ts_only
        ) == TASK_COMPLETION_RUBRIC.format(**PROMPT_KWARGS)


class TestPromptWithCriteria:
    def test_combined_prompt_appends_section_after_base(self):
        base = COMBINED_RUBRIC.format(**PROMPT_KWARGS)
        prompt = build_combined_prompt(**PROMPT_KWARGS, rubric_criteria=CRITERIA)
        assert prompt.startswith(base)  # appended, never spliced into the template
        for c in CRITERIA:
            assert f"[{c['id']}]" in prompt
            assert c["text"] in prompt
        assert '"rubric_criteria"' in prompt  # the additional response field

    def test_criteria_listed_without_dimension_or_weight(self):
        # The judge sees only id + text: dimension mapping and weights are
        # aggregation details that could bias a pure met/unmet verdict.
        prompt = build_combined_prompt(**PROMPT_KWARGS, rubric_criteria=CRITERIA)
        section = prompt[len(COMBINED_RUBRIC.format(**PROMPT_KWARGS)) :]
        assert "task_completion" not in section
        assert "tool_selection" not in section
        assert "weight" not in section

    def test_separate_prompts_show_only_their_dimensions_criteria(self):
        tc_prompt = build_task_completion_prompt(**PROMPT_KWARGS, rubric_criteria=CRITERIA)
        assert "[verify_before_balance]" in tc_prompt
        assert "[no_redundant_lookups]" not in tc_prompt

        ts_kwargs = {k: v for k, v in PROMPT_KWARGS.items() if k != "user_goals"}
        ts_prompt = build_tool_selection_prompt(**ts_kwargs, rubric_criteria=CRITERIA)
        assert "[no_redundant_lookups]" in ts_prompt
        assert "[verify_before_balance]" not in ts_prompt


# --- 2. Aggregation math ------------------------------------------------------


class TestAggregateCriterionScore:
    def test_weighted_fraction(self):
        # task_completion: weights 2 (met) + 1 (unmet) -> 2/3.
        met = {"verify_before_balance": True, "correct_transfer_amount": False}
        score = aggregate_criterion_score(CRITERIA, met, "task_completion")
        assert score == pytest.approx(2 / 3)

    def test_all_met_is_one_and_none_met_is_zero(self):
        all_met = {c["id"]: True for c in CRITERIA}
        none_met = {c["id"]: False for c in CRITERIA}
        assert aggregate_criterion_score(CRITERIA, all_met, "task_completion") == 1.0
        assert aggregate_criterion_score(CRITERIA, none_met, "task_completion") == 0.0

    def test_dimension_without_criteria_returns_none(self):
        tc_only = criteria_for_dimension(CRITERIA, "task_completion")
        assert aggregate_criterion_score(tc_only, {}, "tool_selection") is None

    def test_weight_defaults_to_one(self):
        crits = [
            {"id": "a", "text": "x" * 20, "dimension": "task_completion"},
            {"id": "b", "text": "y" * 20, "dimension": "task_completion"},
        ]
        score = aggregate_criterion_score(crits, {"a": True, "b": False}, "task_completion")
        assert score == pytest.approx(0.5)

    def test_nonpositive_total_weight_falls_back_to_none(self):
        # Unvalidated input guard: never divide by zero mid-run.
        crits = [{"id": "a", "text": "x" * 20, "dimension": "task_completion", "weight": 0.0}]
        assert aggregate_criterion_score(crits, {"a": True}, "task_completion") is None


# --- 3. Verdict extraction (strict) -------------------------------------------


class TestExtractCriteriaVerdicts:
    def test_valid_block_normalized_in_criterion_order(self):
        parsed = json.loads(_verdict_json({c["id"]: True for c in reversed(CRITERIA)}))
        verdicts = _extract_criteria_verdicts(parsed, CRITERIA)
        assert [v["id"] for v in verdicts] == [c["id"] for c in CRITERIA]
        assert all(v["met"] is True for v in verdicts)
        assert all(v["evidence"] for v in verdicts)

    def test_missing_block_is_invalid(self):
        assert _extract_criteria_verdicts({"task_completion": {}}, CRITERIA) is None

    def test_partial_ids_are_invalid(self):
        parsed = json.loads(_verdict_json({"verify_before_balance": True}))
        assert _extract_criteria_verdicts(parsed, CRITERIA) is None

    def test_extra_unknown_id_is_invalid(self):
        met = {c["id"]: True for c in CRITERIA}
        met["invented_by_judge"] = True
        assert _extract_criteria_verdicts(json.loads(_verdict_json(met)), CRITERIA) is None

    def test_duplicate_id_is_invalid(self):
        parsed = json.loads(_verdict_json({c["id"]: True for c in CRITERIA}))
        parsed["rubric_criteria"].append(dict(parsed["rubric_criteria"][0]))
        assert _extract_criteria_verdicts(parsed, CRITERIA) is None

    def test_non_bool_met_is_invalid(self):
        parsed = json.loads(_verdict_json({c["id"]: True for c in CRITERIA}))
        parsed["rubric_criteria"][0]["met"] = "yes"
        assert _extract_criteria_verdicts(parsed, CRITERIA) is None

    def test_missing_evidence_defaults_to_empty_string(self):
        parsed = json.loads(_verdict_json({c["id"]: True for c in CRITERIA}))
        for v in parsed["rubric_criteria"]:
            v.pop("evidence")
        verdicts = _extract_criteria_verdicts(parsed, CRITERIA)
        assert verdicts is not None
        assert all(v["evidence"] == "" for v in verdicts)


# --- 4. Judge scoring with criteria -------------------------------------------


class TestCombinedJudgeWithCriteria:
    def test_criterion_informed_scores_and_holistic_preserved(self, monkeypatch):
        # tc criteria: verify (w2, met) + transfer (w1, unmet) -> 2/3, holistic 0.9.
        # ts criteria: no_redundant_lookups (met) -> 1.0, holistic 0.6.
        met = {
            "verify_before_balance": True,
            "correct_transfer_amount": False,
            "no_redundant_lookups": True,
        }

        def fake_api(judge, system_prompt, rubric_prompt):
            return _verdict_json(met), "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = score_with_judge_combined(_Cfg(), "sys", "combined", CRITERIA)

        assert tc_jr.criterion_informed is True
        assert tc_jr.overall_score == pytest.approx(2 / 3)
        assert tc_jr.holistic_score == pytest.approx(0.9)
        assert [v["id"] for v in tc_jr.criteria_verdicts] == [
            "verify_before_balance",
            "correct_transfer_amount",
        ]

        assert ts_jr.criterion_informed is True
        assert ts_jr.overall_score == pytest.approx(1.0)
        assert ts_jr.holistic_score == pytest.approx(0.6)
        assert [v["id"] for v in ts_jr.criteria_verdicts] == ["no_redundant_lookups"]

    def test_dimension_without_criteria_keeps_holistic(self, monkeypatch):
        # Only task_completion criteria exist: tool_selection keeps the judge's
        # holistic score and stays flagged as not criterion-informed.
        tc_only = criteria_for_dimension(CRITERIA, "task_completion")
        met = {c["id"]: True for c in tc_only}

        def fake_api(judge, system_prompt, rubric_prompt):
            return _verdict_json(met), "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = score_with_judge_combined(_Cfg(), "sys", "combined", tc_only)

        assert tc_jr.criterion_informed is True
        assert tc_jr.overall_score == pytest.approx(1.0)
        assert ts_jr.criterion_informed is False
        assert ts_jr.overall_score == pytest.approx(0.6)
        assert ts_jr.holistic_score is None
        assert ts_jr.criteria_verdicts is None

    def test_missing_verdict_block_is_parse_failure_for_both(self, monkeypatch):
        # Valid dimensions but NO rubric_criteria block when criteria were
        # requested -> whole-judge parse failure (after one retry), both dims.
        calls = {"n": 0}

        def fake_api(judge, system_prompt, rubric_prompt):
            calls["n"] += 1
            body = (
                '{"task_completion": {"overall_score": 0.9, "overall_reasoning": "a"}, '
                '"tool_selection": {"overall_score": 0.6, "overall_reasoning": "b"}}'
            )
            return body, "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = score_with_judge_combined(_Cfg(), "sys", "combined", CRITERIA)
        assert calls["n"] == 2  # retried once
        assert tc_jr.parse_failed is True
        assert ts_jr.parse_failed is True

    def test_invalid_verdicts_then_valid_on_retry_succeeds(self, monkeypatch):
        calls = {"n": 0}
        met = {c["id"]: True for c in CRITERIA}

        def fake_api(judge, system_prompt, rubric_prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                # Partial verdict block -> rejected.
                return _verdict_json({"verify_before_balance": True}), "m1", (10, 5)
            return _verdict_json(met), "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = score_with_judge_combined(_Cfg(), "sys", "combined", CRITERIA)
        assert calls["n"] == 2
        assert tc_jr.parse_failed is False
        assert tc_jr.overall_score == pytest.approx(1.0)

    def test_no_criteria_behaves_exactly_as_before(self, monkeypatch):
        # Legacy call (no criteria): holistic score is overall_score and all
        # criteria fields keep their defaults.
        def fake_api(judge, system_prompt, rubric_prompt):
            body = (
                '{"task_completion": {"overall_score": 0.9, "overall_reasoning": "a"}, '
                '"tool_selection": {"overall_score": 0.6, "overall_reasoning": "b"}}'
            )
            return body, "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        tc_jr, ts_jr = score_with_judge_combined(_Cfg(), "sys", "combined")
        assert tc_jr.overall_score == pytest.approx(0.9)
        assert tc_jr.criterion_informed is False
        assert tc_jr.holistic_score is None
        assert tc_jr.criteria_verdicts is None


class TestSeparateJudgeWithCriteria:
    def test_criterion_informed_score_on_separate_path(self, monkeypatch):
        # The separate path receives this dimension's criteria pre-filtered.
        tc_only = criteria_for_dimension(CRITERIA, "task_completion")

        def fake_api(judge, system_prompt, rubric_prompt):
            body = json.dumps(
                {
                    "overall_score": 0.95,
                    "overall_reasoning": "ok",
                    "rubric_criteria": [
                        {"id": "verify_before_balance", "met": True, "evidence": "t2"},
                        {"id": "correct_transfer_amount", "met": False, "evidence": "missing"},
                    ],
                }
            )
            return body, "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        jr = score_with_judge(_Cfg(), "sys", "prompt", "task_completion", tc_only)
        assert jr.criterion_informed is True
        assert jr.overall_score == pytest.approx(2 / 3)  # weights 2 met / 3 total
        assert jr.holistic_score == pytest.approx(0.95)

    def test_missing_verdicts_parse_fail_on_separate_path(self, monkeypatch):
        tc_only = criteria_for_dimension(CRITERIA, "task_completion")

        def fake_api(judge, system_prompt, rubric_prompt):
            return '{"overall_score": 0.95, "overall_reasoning": "ok"}', "m1", (10, 5)

        monkeypatch.setattr(judge_mod, "_call_judge_api", fake_api)
        jr = score_with_judge(_Cfg(), "sys", "prompt", "task_completion", tc_only)
        assert jr.parse_failed is True


class TestOrchestrationPassThrough:
    def test_combined_orchestrator_passes_criteria_to_each_judge(self, monkeypatch):
        seen = {}

        def fake(judge, system_prompt, combined_prompt, rubric_criteria=None):
            seen[judge.name] = rubric_criteria
            return (
                judge_mod.JudgeResult(
                    judge_name=judge.name,
                    rubric_type="task_completion",
                    overall_score=0.5,
                    reasoning="ok",
                    raw_response={},
                    latency_ms=1.0,
                ),
                judge_mod.JudgeResult(
                    judge_name=judge.name,
                    rubric_type="tool_selection",
                    overall_score=0.5,
                    reasoning="ok",
                    raw_response={},
                    latency_ms=1.0,
                ),
            )

        monkeypatch.setattr(judge_mod, "score_with_judge_combined", fake)
        monkeypatch.setattr(judge_mod, "JUDGES", {"solo": _Cfg(name="Solo")})
        score_with_all_judges_combined("sys", "combined", "s1", rubric_criteria=CRITERIA)
        assert seen["Solo"] == CRITERIA

    def test_combined_orchestrator_without_criteria_uses_legacy_call_shape(self, monkeypatch):
        # A strict 3-arg fake (no rubric_criteria parameter at all) must keep
        # working when no criteria are passed — the legacy call shape is part of
        # the backwards-compatibility contract.
        def fake(judge, system_prompt, combined_prompt):
            return (
                judge_mod.JudgeResult(
                    judge_name=judge.name,
                    rubric_type="task_completion",
                    overall_score=0.5,
                    reasoning="ok",
                    raw_response={},
                    latency_ms=1.0,
                ),
                judge_mod.JudgeResult(
                    judge_name=judge.name,
                    rubric_type="tool_selection",
                    overall_score=0.5,
                    reasoning="ok",
                    raw_response={},
                    latency_ms=1.0,
                ),
            )

        monkeypatch.setattr(judge_mod, "score_with_judge_combined", fake)
        monkeypatch.setattr(judge_mod, "JUDGES", {"solo": _Cfg(name="Solo")})
        tc, ts = score_with_all_judges_combined("sys", "combined", "s1")
        assert tc.n_judges_valid == 1


# --- 5. Validator -------------------------------------------------------------


class TestValidatorCriteria:
    def test_valid_criteria_scenario_passes(self):
        assert validate_scenario_dict(_criteria_scenario_dict()) == []

    def test_scenario_without_criteria_is_untouched(self):
        assert validate_scenario_dict(_base_scenario_dict()) == []

    @pytest.mark.parametrize("count", [2, 7])
    def test_count_out_of_bounds_fails(self, count):
        data = _criteria_scenario_dict()
        template = data["rubric_criteria"][0]
        data["rubric_criteria"] = [
            {**template, "id": f"crit_{i}", "text": f"Checkable criterion number {i} text."}
            for i in range(count)
        ]
        errors = validate_scenario_dict(data)
        assert any("expected 3-6" in e for e in errors)

    def test_duplicate_ids_fail(self):
        data = _criteria_scenario_dict()
        data["rubric_criteria"][1]["id"] = data["rubric_criteria"][0]["id"]
        assert any("duplicate id" in e for e in validate_scenario_dict(data))

    def test_short_text_fails(self):
        data = _criteria_scenario_dict()
        data["rubric_criteria"][0]["text"] = "good"
        assert any("text too short" in e for e in validate_scenario_dict(data))

    def test_invalid_dimension_fails(self):
        data = _criteria_scenario_dict()
        data["rubric_criteria"][0]["dimension"] = "cost"
        assert any("dimension 'cost'" in e for e in validate_scenario_dict(data))

    @pytest.mark.parametrize("weight", [0, -1.0, 11.0])
    def test_insane_weight_fails(self, weight):
        data = _criteria_scenario_dict()
        data["rubric_criteria"][0]["weight"] = weight
        assert any("out of range" in e for e in validate_scenario_dict(data))

    def test_missing_criteria_authorship_fails(self):
        data = _criteria_scenario_dict()
        del data["criteria_authorship"]
        errors = validate_scenario_dict(data)
        assert any("requires 'criteria_authorship'" in e for e in errors)

    def test_contestant_criteria_author_fails_family_aware(self):
        # "gpt-5.5" is a prefix of the pinned contestant snapshot — still blocked.
        data = _criteria_scenario_dict()
        data["criteria_authorship"]["criteria_author_model"] = "gpt-5.5"
        errors = validate_scenario_dict(data)
        assert any("must not write the grading criteria" in e for e in errors)

    def test_human_handwritten_criteria_author_allowed(self):
        data = _criteria_scenario_dict()
        data["criteria_authorship"]["criteria_author_model"] = "human-handwritten"
        assert validate_scenario_dict(data) == []

    def test_criteria_authorship_without_criteria_fails(self):
        data = _base_scenario_dict()
        data["criteria_authorship"] = {"criteria_author_model": "anthropic/claude-opus-4.8"}
        errors = validate_scenario_dict(data)
        assert any("without rubric_criteria" in e for e in errors)


# --- 6. Corpus hash coverage (pre-registration) --------------------------------


def _make_scenario(**overrides) -> Scenario:
    fields = dict(
        id="banking_adaptive_tool_use_9999",
        domain=Domain.BANKING,
        persona={"name": "Test"},
        user_goals=["g1", "g2", "g3"],
        tools=[{"name": "t1"}],
        category="adaptive_tool_use",
        initial_message="Hello there, I need help.",
    )
    fields.update(overrides)
    return Scenario(**fields)


class TestCorpusHashCoversCriteria:
    def test_criteria_less_scenario_canonical_dict_has_no_criteria_key(self):
        # Conditional inclusion: criteria-less scenarios keep EXACTLY the
        # pre-#54 canonical key set, so their digests (and any corpus hash over
        # them) are unchanged by this feature shipping.
        data = _scenario_to_canonical_dict(_make_scenario())
        assert set(data.keys()) == {
            "id",
            "domain",
            "persona",
            "user_goals",
            "tools",
            "category",
            "initial_message",
            "ground_truth",
            "expected_state_changes",
        }

    def test_adding_criteria_changes_the_corpus_hash(self):
        plain = _make_scenario()
        with_criteria = _make_scenario(rubric_criteria=[dict(c) for c in CRITERIA])
        hash_plain, _ = scenario_set_hash({Domain.BANKING: [plain]})
        hash_crit, _ = scenario_set_hash({Domain.BANKING: [with_criteria]})
        assert hash_plain != hash_crit

    def test_editing_one_criterion_changes_the_corpus_hash(self):
        a = _make_scenario(rubric_criteria=[dict(c) for c in CRITERIA])
        edited = [dict(c) for c in CRITERIA]
        edited[0]["text"] = "Agent verified identity before ANY account data was shared."
        b = _make_scenario(rubric_criteria=edited)
        assert (
            scenario_set_hash({Domain.BANKING: [a]})[0]
            != scenario_set_hash({Domain.BANKING: [b]})[0]
        )


# --- 7. Criteria stamping (provenance) -----------------------------------------


class TestStampCriteria:
    def test_stamps_criteria_and_provenance(self):
        data = _base_scenario_dict()
        stamped = stamp_criteria(
            data,
            [dict(c) for c in CRITERIA],
            criteria_author_model="anthropic/claude-opus-4.8",
            criteria_author_run="2026-06-11-claude-opus-batch",
        )
        assert stamped["rubric_criteria"] == CRITERIA
        assert stamped["criteria_authorship"]["criteria_author_model"] == (
            "anthropic/claude-opus-4.8"
        )
        assert stamped["criteria_authorship"]["criteria_author_run"] == (
            "2026-06-11-claude-opus-batch"
        )
        assert validate_scenario_dict(stamped) == []

    def test_contestant_author_is_rejected(self):
        with pytest.raises(RuntimeError, match="must never author"):
            stamp_criteria(
                _base_scenario_dict(),
                [dict(c) for c in CRITERIA],
                criteria_author_model="gpt-5.5-2026-04-23",
            )

    def test_invalid_criteria_raise_value_error(self):
        bad = [dict(CRITERIA[0])]  # only 1 criterion (< 3)
        with pytest.raises(ValueError, match="expected 3-6"):
            stamp_criteria(
                _base_scenario_dict(),
                bad,
                criteria_author_model="anthropic/claude-opus-4.8",
            )

    def test_human_review_sets_review_date(self):
        stamped = stamp_criteria(
            _base_scenario_dict(),
            [dict(c) for c in CRITERIA],
            criteria_author_model="human-handwritten",
            human_reviewed_by="Conor Bronsdon",
        )
        assert stamped["criteria_authorship"]["human_reviewed_by"] == "Conor Bronsdon"
        assert stamped["criteria_authorship"]["review_date"]


# --- 8. Artifacts carry the per-criterion evidence ------------------------------


class TestArtifactsSerializeCriteria:
    def test_serialize_judges_includes_criteria_fields(self):
        from eval.artifacts import _serialize_judges

        jr = judge_mod.JudgeResult(
            judge_name="Solo",
            rubric_type="task_completion",
            overall_score=2 / 3,
            reasoning="ok",
            raw_response={"overall_score": 0.9},
            latency_ms=1.0,
            holistic_score=0.9,
            criterion_informed=True,
            criteria_verdicts=[{"id": "verify_before_balance", "met": True, "evidence": "t2"}],
        )

        class _Consensus:
            judge_results = [jr]

        (row,) = _serialize_judges(_Consensus())
        assert row["overall_score"] == pytest.approx(2 / 3)
        assert row["holistic_score"] == pytest.approx(0.9)
        assert row["criterion_informed"] is True
        assert row["criteria_verdicts"] == [
            {"id": "verify_before_balance", "met": True, "evidence": "t2"}
        ]

    def test_legacy_judge_result_serializes_with_defaults(self):
        from eval.artifacts import _serialize_judges

        jr = judge_mod.JudgeResult(
            judge_name="Solo",
            rubric_type="task_completion",
            overall_score=0.9,
            reasoning="ok",
            raw_response={},
            latency_ms=1.0,
        )

        class _Consensus:
            judge_results = [jr]

        (row,) = _serialize_judges(_Consensus())
        assert row["holistic_score"] is None
        assert row["criterion_informed"] is False
        assert row["criteria_verdicts"] is None
