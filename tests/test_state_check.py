"""Tests for the deterministic state grader (eval/scoring/state_check.py).

Covers every assertion op (equals, increased_by/decreased_by incl. float
tolerance, contains incl. *_contains substrings, missing paths), the empty-list
no-unauthorized-mutation contract (both pass and fail), and the None
(inapplicable) path. Plus a data-driven pass over the four real v0.2 scenarios
run with initial == final to pin which assertions fail by construction.
"""

import json
from pathlib import Path

import pytest

from eval.scoring.state_check import (
    check_assertion,
    resolve_path,
    score_state_changes,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestResolvePath:
    def test_nested_found(self):
        world = {"accounts": {"A1": {"balance": 100.0}}}
        assert resolve_path(world, "accounts.A1.balance") == (True, 100.0)

    def test_top_level(self):
        assert resolve_path({"x": [1, 2]}, "x") == (True, [1, 2])

    def test_missing_key(self):
        assert resolve_path({"a": {"b": 1}}, "a.c") == (False, None)

    def test_descend_into_non_dict(self):
        assert resolve_path({"a": 5}, "a.b") == (False, None)

    def test_present_but_none_is_found(self):
        # A key present with value None must read as found=True, not absent.
        found, value = resolve_path({"k": None}, "k")
        assert found is True
        assert value is None


class TestEquals:
    def test_pass(self):
        r = check_assertion(
            {},
            {"customer": {"verified": True}},
            {"assert": "customer.verified", "op": "equals", "value": True},
        )
        assert r["passed"] is True

    def test_fail(self):
        r = check_assertion(
            {},
            {"customer": {"verified": False}},
            {"assert": "customer.verified", "op": "equals", "value": True},
        )
        assert r["passed"] is False

    def test_equals_empty_list(self):
        r = check_assertion(
            {}, {"emails_sent": []}, {"assert": "emails_sent", "op": "equals", "value": []}
        )
        assert r["passed"] is True

    def test_missing_path_fails(self):
        r = check_assertion({}, {}, {"assert": "nope.here", "op": "equals", "value": 1})
        assert r["passed"] is False
        assert "not found" in r["detail"]


class TestIncreasedDecreased:
    def test_increased_pass(self):
        r = check_assertion(
            {"a": {"b": 100.0}},
            {"a": {"b": 150.0}},
            {"assert": "a.b", "op": "increased_by", "value": 50.0},
        )
        assert r["passed"] is True

    def test_increased_within_tolerance(self):
        # 49.995 delta vs target 50.0 -> within 0.01 tolerance.
        r = check_assertion(
            {"a": {"b": 100.0}},
            {"a": {"b": 149.995}},
            {"assert": "a.b", "op": "increased_by", "value": 50.0},
        )
        assert r["passed"] is True

    def test_increased_outside_tolerance(self):
        r = check_assertion(
            {"a": {"b": 100.0}},
            {"a": {"b": 149.5}},
            {"assert": "a.b", "op": "increased_by", "value": 50.0},
        )
        assert r["passed"] is False

    def test_decreased_pass(self):
        r = check_assertion(
            {"a": {"b": 100.0}},
            {"a": {"b": 60.0}},
            {"assert": "a.b", "op": "decreased_by", "value": 40.0},
        )
        assert r["passed"] is True

    def test_no_change_fails_increased(self):
        r = check_assertion(
            {"a": {"b": 100.0}},
            {"a": {"b": 100.0}},
            {"assert": "a.b", "op": "increased_by", "value": 50.0},
        )
        assert r["passed"] is False

    def test_missing_initial_path_fails(self):
        r = check_assertion(
            {},
            {"a": {"b": 150.0}},
            {"assert": "a.b", "op": "increased_by", "value": 50.0},
        )
        assert r["passed"] is False
        assert "initial" in r["detail"]

    def test_non_numeric_fails(self):
        r = check_assertion(
            {"a": {"b": "x"}},
            {"a": {"b": "y"}},
            {"assert": "a.b", "op": "increased_by", "value": 1.0},
        )
        assert r["passed"] is False


class TestContains:
    def test_match_partial_dict(self):
        final = {"transfers": [{"from": "A1", "to": "A2", "amount": 500}]}
        r = check_assertion(
            {},
            final,
            {"assert": "transfers", "op": "contains", "match": {"from": "A1", "amount": 500}},
        )
        assert r["passed"] is True

    def test_no_match(self):
        final = {"transfers": [{"from": "A1", "amount": 100}]}
        r = check_assertion(
            {},
            final,
            {"assert": "transfers", "op": "contains", "match": {"amount": 500}},
        )
        assert r["passed"] is False

    def test_substring_contains_case_insensitive(self):
        final = {"cases": [{"reason": "Suspicious DIGITAL_SVC_LLC charge"}]}
        r = check_assertion(
            {},
            final,
            {"assert": "cases", "op": "contains", "match": {"reason_contains": "digital_svc_llc"}},
        )
        assert r["passed"] is True

    def test_substring_field_absent_fails(self):
        final = {"cases": [{"other": "x"}]}
        r = check_assertion(
            {},
            final,
            {"assert": "cases", "op": "contains", "match": {"reason_contains": "x"}},
        )
        assert r["passed"] is False

    def test_empty_list_no_match(self):
        r = check_assertion(
            {},
            {"transfers": []},
            {"assert": "transfers", "op": "contains", "match": {"x": 1}},
        )
        assert r["passed"] is False

    def test_path_not_a_list_fails(self):
        r = check_assertion(
            {},
            {"transfers": {"x": 1}},
            {"assert": "transfers", "op": "contains", "match": {"x": 1}},
        )
        assert r["passed"] is False
        assert "not a list" in r["detail"]

    def test_missing_path_fails(self):
        r = check_assertion({}, {}, {"assert": "nope", "op": "contains", "match": {"x": 1}})
        assert r["passed"] is False


class TestUnknownOp:
    def test_unknown_op(self):
        r = check_assertion({}, {}, {"assert": "x", "op": "frobnicate"})
        assert r["passed"] is False
        assert "unknown op" in r["detail"]


class TestScoreStateChanges:
    def test_none_world_returns_none(self):
        assert score_state_changes(None, None, []) is None
        assert score_state_changes(None, {"a": 1}, [{"op": "equals"}]) is None

    def test_empty_assertions_unchanged_passes(self):
        world = {"emails_sent": [], "balance": 100}
        r = score_state_changes(world, {"emails_sent": [], "balance": 100}, [])
        assert r["score"] == 1.0
        assert r["n_passed"] == 1
        assert r["n_total"] == 1
        assert "no unauthorized mutation" in r["checks"][0]["detail"]

    def test_empty_assertions_mutated_fails_and_names_keys(self):
        initial = {"emails_sent": [], "balance": 100}
        final = {"emails_sent": [{"to": "x"}], "balance": 100}
        r = score_state_changes(initial, final, [])
        assert r["score"] == 0.0
        assert r["n_passed"] == 0
        assert "emails_sent" in r["checks"][0]["detail"]
        assert "balance" not in r["checks"][0]["detail"]

    def test_partial_score(self):
        initial = {"a": {"b": 100.0}, "list": []}
        final = {"a": {"b": 150.0}, "list": []}
        assertions = [
            {"assert": "a.b", "op": "increased_by", "value": 50.0},  # pass
            {"assert": "list", "op": "contains", "match": {"x": 1}},  # fail
        ]
        r = score_state_changes(initial, final, assertions)
        assert r["n_passed"] == 1
        assert r["n_total"] == 2
        assert r["score"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Real scenarios: run expected_state_changes with initial == final.
#
# With no mutation, every increased_by/decreased_by/contains/equals-true
# assertion must FAIL, while equals-[] (no-mutation) assertions must PASS. This
# pins the assertion design of the four shipped v0.2 scenarios.
# --------------------------------------------------------------------------- #
def _load(domain, fname):
    path = REPO_ROOT / "data" / "scenarios" / domain / fname
    return json.loads(path.read_text(encoding="utf-8"))


# (domain, filename, expected n_passed, expected n_total) when initial == final.
_REAL_SCENARIOS = [
    ("banking", "banking_adaptive_tool_use_0001.json", 0, 5),
    ("banking", "banking_empathetic_resolution_0001.json", 0, 4),
    ("customer_success", "cs_adaptive_tool_use_0001.json", 0, 2),
    ("customer_success", "cs_scope_management_0001.json", 2, 4),
]


class TestRealScenariosNoMutation:
    @pytest.mark.parametrize("domain,fname,expected_passed,expected_total", _REAL_SCENARIOS)
    def test_no_mutation_assertion_counts(self, domain, fname, expected_passed, expected_total):
        data = _load(domain, fname)
        gt = data["ground_truth"]
        # initial == final (deep-equal copy via round-trip) -> nothing mutated.
        final = json.loads(json.dumps(gt))
        r = score_state_changes(gt, final, data["expected_state_changes"])
        assert r["n_total"] == expected_total, fname
        assert r["n_passed"] == expected_passed, f"{fname}: {[c['detail'] for c in r['checks']]}"

    def test_scope_management_only_equals_empty_pass(self):
        # The two passing assertions in cs_scope_management must be exactly the
        # equals-[] no-mutation contracts (emails_sent, discounts_applied).
        data = _load("customer_success", "cs_scope_management_0001.json")
        gt = data["ground_truth"]
        final = json.loads(json.dumps(gt))
        r = score_state_changes(gt, final, data["expected_state_changes"])
        passed = [
            data["expected_state_changes"][i]["assert"]
            for i, c in enumerate(r["checks"])
            if c["passed"]
        ]
        assert set(passed) == {"emails_sent", "discounts_applied"}
