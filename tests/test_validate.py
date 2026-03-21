"""Tests for scenario validation."""

import json
import tempfile
from pathlib import Path

from scripts.validate_scenarios import validate_scenario


def _write_scenario(data: dict) -> Path:
    """Write a scenario dict to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
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
            "parameters": [
                {"name": "x", "type": "string", "description": "input"}
            ],
        },
        {
            "name": "tool_b",
            "description": "Does thing B",
            "parameters": [
                {"name": "y", "type": "integer", "description": "count"}
            ],
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

    def test_invalid_category(self):
        data = {**VALID_SCENARIO, "category": "nonexistent"}
        path = _write_scenario(data)
        errors = validate_scenario(path)
        assert any("category" in e.lower() for e in errors)

    def test_invalid_json(self):
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
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
