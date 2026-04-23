"""Validate scenario JSON files against the expected schema.

Use this after generating scenarios to catch malformed data before
running expensive evaluation runs.
"""

import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


class Tool(BaseModel):
    name: str
    description: str
    parameters: list[ToolParameter]
    response_schema: dict | None = None


class Persona(BaseModel):
    name: str
    age: int
    occupation: str
    personality_traits: list[str]
    tone: str
    detail_level: str
    background: str


class ScenarioSchema(BaseModel):
    id: str
    category: str
    persona: Persona
    user_goals: list[str]
    tools: list[Tool]
    initial_message: str
    difficulty: str | None = None
    expected_tool_sequence: list[str] | None = None


def validate_scenario(path: Path) -> list[str]:
    """Validate a single scenario file. Returns list of error messages."""
    errors = []
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    try:
        scenario = ScenarioSchema(**data)
    except ValidationError as e:
        return [f"Schema validation: {err['msg']} at {err['loc']}" for err in e.errors()]

    # Content validation
    if len(scenario.user_goals) < 3:
        errors.append(f"Only {len(scenario.user_goals)} goals (minimum 3)")
    if len(scenario.user_goals) > 10:
        errors.append(f"{len(scenario.user_goals)} goals (maximum 10)")
    if len(scenario.tools) < 2:
        errors.append(f"Only {len(scenario.tools)} tools (minimum 2)")
    if len(scenario.initial_message) < 10:
        errors.append("Initial message too short")
    if scenario.category not in {
        "adaptive_tool_use",
        "scope_management",
        "empathetic_resolution",
        "extreme_scenario_recovery",
        "adversarial_input_mitigation",
    }:
        errors.append(f"Unknown category: {scenario.category}")

    # Check tool names in expected sequence exist
    if scenario.expected_tool_sequence:
        tool_names = {t.name for t in scenario.tools}
        for tool_name in scenario.expected_tool_sequence:
            if tool_name not in tool_names:
                errors.append(f"Expected tool '{tool_name}' not in available tools")

    return errors


def main():
    scenario_dir = Path("data/scenarios")
    if not scenario_dir.exists():
        logger.error("No scenarios directory found at %s", scenario_dir)
        sys.exit(1)

    total = 0
    valid = 0
    invalid = 0

    for path in sorted(scenario_dir.rglob("*.json")):
        total += 1
        errors = validate_scenario(path)
        if errors:
            invalid += 1
            logger.error("%s:", path.relative_to(scenario_dir))
            for err in errors:
                logger.error("  - %s", err)
        else:
            valid += 1

    print(f"\nValidation complete: {valid}/{total} valid, {invalid} invalid")
    sys.exit(1 if invalid > 0 else 0)


if __name__ == "__main__":
    main()
