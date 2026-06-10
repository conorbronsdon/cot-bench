"""Tests for scenario validation."""

import copy
import json
import tempfile
from pathlib import Path

from scripts.validate_scenarios import dedup_check, validate_scenario


def _write_scenario(data: dict, dir_path: Path | None = None) -> Path:
    """Write a scenario dict to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        dir=str(dir_path) if dir_path else None,
    )
    json.dump(data, f)
    f.close()
    return Path(f.name)


VALID_SCENARIO = {
    "id": "test_001",
    "category": "adaptive_tool_use",
    "persona": {
        "name": "Test User",
        "age": 30,
        "occupation": "Engineer",
        "personality_traits": ["friendly"],
        "tone": "casual",
        "detail_level": "moderate",
        "background": "Test background",
    },
    "user_goals": ["Goal 1", "Goal 2", "Goal 3"],
    "tools": [
        {
            "name": "tool_a",
            "description": "Does thing A",
            "parameters": [{"name": "x", "type": "string", "description": "input"}],
        },
        {
            "name": "tool_b",
            "description": "Does thing B",
            "parameters": [{"name": "y", "type": "integer", "description": "count"}],
        },
    ],
    "initial_message": "Hi, I need help with something today.",
}


class TestValidateScenario:
    def test_valid_scenario(self):
        path = _write_scenario(VALID_SCENARIO)
        errors = validate_scenario(path)
        assert errors == []

    def test_missing_field(self):
        data = {**VALID_SCENARIO}
        del data["initial_message"]
        path = _write_scenario(data)
        errors = validate_scenario(path)
        assert len(errors) > 0

    def test_too_few_goals(self):
        data = {**VALID_SCENARIO, "user_goals": ["One goal"]}
        path = _write_scenario(data)
        errors = validate_scenario(path)
        assert any("goals" in e.lower() for e in errors)

    def test_too_many_goals(self):
        goals = [f"Goal {i}" for i in range(12)]
        data = {**VALID_SCENARIO, "user_goals": goals}
        path = _write_scenario(data)
        errors = validate_scenario(path)
        assert any("goals" in e.lower() for e in errors)

    def test_invalid_category(self):
        data = {**VALID_SCENARIO, "category": "nonexistent"}
        path = _write_scenario(data)
        errors = validate_scenario(path)
        assert any("category" in e.lower() for e in errors)

    def test_invalid_json(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        f.write("not json {{{")
        f.close()
        errors = validate_scenario(Path(f.name))
        assert any("JSON" in e for e in errors)

    def test_expected_tool_not_in_tools(self):
        data = {
            **VALID_SCENARIO,
            "expected_tool_sequence": ["tool_a", "nonexistent_tool"],
        }
        path = _write_scenario(data)
        errors = validate_scenario(path)
        assert any("nonexistent_tool" in e for e in errors)


# --- v0.2 schema ---

V02_SCENARIO = {
    **copy.deepcopy(VALID_SCENARIO),
    "id": "test_v02_001",
    "schema_version": "0.2",
    "authorship": {"author_model": "human-handwritten", "human_reviewed_by": "Conor Bronsdon"},
    "ground_truth": {
        "accounts": {"A1": {"balance": 100.0}},
        "records": [],
        "flag": False,
    },
    "expected_state_changes": [
        {"assert": "accounts.A1.balance", "op": "increased_by", "value": 50.0, "goal": "Goal 1"},
        {"assert": "records", "op": "contains", "match": {"kind": "x"}, "goal": "Goal 2"},
        {"assert": "flag", "op": "equals", "value": True},
    ],
}


class TestV02Schema:
    def test_valid_v02(self):
        path = _write_scenario(copy.deepcopy(V02_SCENARIO))
        assert validate_scenario(path) == []

    def test_v02_requires_ground_truth(self):
        data = copy.deepcopy(V02_SCENARIO)
        del data["ground_truth"]
        errors = validate_scenario(_write_scenario(data))
        assert any("ground_truth" in e for e in errors)

    def test_v02_requires_expected_state_changes(self):
        data = copy.deepcopy(V02_SCENARIO)
        del data["expected_state_changes"]
        errors = validate_scenario(_write_scenario(data))
        assert any("expected_state_changes" in e for e in errors)

    def test_v02_requires_authorship(self):
        data = copy.deepcopy(V02_SCENARIO)
        del data["authorship"]
        errors = validate_scenario(_write_scenario(data))
        assert any("authorship" in e for e in errors)

    def test_v02_empty_state_changes_allowed(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["expected_state_changes"] = []
        assert validate_scenario(_write_scenario(data)) == []

    def test_unversioned_scenario_still_valid_without_v02_blocks(self):
        # Older scenarios with no schema_version remain valid (staged migration).
        path = _write_scenario(copy.deepcopy(VALID_SCENARIO))
        assert validate_scenario(path) == []

    def test_unsupported_schema_version(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["schema_version"] = "9.9"
        errors = validate_scenario(_write_scenario(data))
        assert any("schema_version" in e for e in errors)


class TestAssertionResolution:
    def test_assert_path_must_resolve(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["expected_state_changes"] = [
            {"assert": "accounts.NOPE.balance", "op": "increased_by", "value": 1.0}
        ]
        errors = validate_scenario(_write_scenario(data))
        assert any("does not resolve" in e for e in errors)

    def test_contains_requires_list(self):
        data = copy.deepcopy(V02_SCENARIO)
        # 'flag' is a bool, not a list — contains should fail.
        data["expected_state_changes"] = [{"assert": "flag", "op": "contains", "match": {"x": 1}}]
        errors = validate_scenario(_write_scenario(data))
        assert any("list" in e for e in errors)

    def test_contains_requires_match_dict(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["expected_state_changes"] = [{"assert": "records", "op": "contains"}]
        errors = validate_scenario(_write_scenario(data))
        assert any("match" in e for e in errors)

    def test_unknown_op(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["expected_state_changes"] = [{"assert": "flag", "op": "toggled", "value": True}]
        errors = validate_scenario(_write_scenario(data))
        assert any("unknown op" in e.lower() for e in errors)

    def test_goal_must_fuzzy_match_user_goal(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["expected_state_changes"] = [
            {
                "assert": "flag",
                "op": "equals",
                "value": True,
                "goal": "completely unrelated nonsense string xyzzy",
            }
        ]
        errors = validate_scenario(_write_scenario(data))
        assert any("fuzzy-match" in e for e in errors)


class TestAuthorGuard:
    def test_author_in_models_under_test_rejected(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["authorship"] = {"author_model": "gpt-4.1"}
        errors = validate_scenario(_write_scenario(data))
        assert any("MODELS_UNDER_TEST" in e for e in errors)

    def test_author_display_name_rejected(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["authorship"] = {"author_model": "Claude Sonnet 4.6"}
        errors = validate_scenario(_write_scenario(data))
        assert any("MODELS_UNDER_TEST" in e for e in errors)

    def test_human_handwritten_always_allowed(self):
        data = copy.deepcopy(V02_SCENARIO)
        data["authorship"] = {"author_model": "human-handwritten"}
        assert validate_scenario(_write_scenario(data)) == []


def _dedup_input(tmp_path: Path, scenarios: list[dict], domain: str = "banking"):
    """Write scenarios to a domain dir and return the dedup_check input list."""
    domain_dir = tmp_path / domain
    domain_dir.mkdir(exist_ok=True)
    loaded = []
    for s in scenarios:
        p = _write_scenario(s, dir_path=domain_dir)
        loaded.append((p, s, domain))
    return loaded


class TestDedup:
    def test_near_identical_hard_fail(self, tmp_path):
        a = copy.deepcopy(VALID_SCENARIO)
        b = copy.deepcopy(VALID_SCENARIO)
        b["id"] = "test_002"
        # Identical initial_message + goals -> ratio 1.0 -> hard fail.
        loaded = _dedup_input(tmp_path, [a, b])
        hard, warn = dedup_check(loaded)
        assert any("near-duplicate" in h or "goal-set overlap" in h for h in hard)

    def test_warning_band(self, tmp_path):
        a = copy.deepcopy(VALID_SCENARIO)
        a["initial_message"] = (
            "I want to check my balance and then transfer some money to "
            "my savings account today please."
        )
        a["user_goals"] = [
            "Check the account balance",
            "Transfer money to savings",
            "Confirm the transfer completed",
        ]
        b = copy.deepcopy(VALID_SCENARIO)
        b["id"] = "test_002"
        # Similar wording but one differing goal -> ratio in the 0.75-0.85 band
        # and goal-set Jaccard 0.5 (below the 0.8 hard threshold).
        b["initial_message"] = (
            "I want to check my balance and then transfer some funds to "
            "my savings account this morning please."
        )
        b["user_goals"] = [
            "Check the account balance",
            "Transfer money to savings",
            "Order replacement card",
        ]
        loaded = _dedup_input(tmp_path, [a, b])
        hard, warn = dedup_check(loaded)
        assert hard == []
        assert any("possible duplicate" in w for w in warn)

    def test_goal_jaccard_hard_fail(self, tmp_path):
        a = copy.deepcopy(VALID_SCENARIO)
        a["initial_message"] = "A totally different opening message about topic one entirely."
        a["user_goals"] = ["Alpha goal", "Beta goal", "Gamma goal"]
        b = copy.deepcopy(VALID_SCENARIO)
        b["id"] = "test_002"
        # Very different wording (low SequenceMatcher) but identical goal set.
        b["initial_message"] = "Wholly unrelated phrasing covering subject two completely apart."
        b["user_goals"] = ["alpha goal", "beta goal", "gamma goal"]
        loaded = _dedup_input(tmp_path, [a, b])
        hard, warn = dedup_check(loaded)
        assert any("Jaccard" in h for h in hard)

    def test_distinct_scenarios_pass(self, tmp_path):
        a = copy.deepcopy(VALID_SCENARIO)
        a["initial_message"] = "I want to dispute a charge on my credit card from last week."
        a["user_goals"] = ["Dispute charge", "Freeze card", "Order new card"]
        b = copy.deepcopy(VALID_SCENARIO)
        b["id"] = "test_002"
        b["initial_message"] = "Can you help me understand my mortgage refinancing options?"
        b["user_goals"] = ["Refi rates", "Closing costs", "Payment estimate"]
        loaded = _dedup_input(tmp_path, [a, b])
        hard, warn = dedup_check(loaded)
        assert hard == []


class TestRepoScenariosPass:
    def test_all_repo_scenarios_valid(self):
        scenario_dir = Path("data/scenarios")
        files = sorted(scenario_dir.rglob("*.json"))
        assert files, "no repo scenarios found"
        for path in files:
            errors = validate_scenario(path)
            assert errors == [], f"{path.name}: {errors}"
