"""Synthetic data generation for COT Bench.

Generates tools, personas, and multi-turn scenarios for each domain.
Adapted from Galileo's agent-leaderboard (Apache 2.0) with modifications
for COT Bench's domain structure and scenario design.
"""

import argparse
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ValidationError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Lazy client — only created when generation functions are called
_client: OpenAI | None = None
GENERATION_MODEL = "gpt-4.1"


def _get_client() -> OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY environment variable required for data generation"
            )
        _client = OpenAI(api_key=api_key)
    return _client


# --- Pydantic schemas for validation ---


class ToolParameter(BaseModel):
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None


class ToolDefinition(BaseModel):
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
    detail_level: str  # "brief", "moderate", "verbose"
    background: str


class Scenario(BaseModel):
    id: str
    category: str
    persona: dict
    user_goals: list[str]
    tools: list[dict]
    initial_message: str
    difficulty: str  # "easy", "medium", "hard"
    expected_tool_sequence: list[str] | None = None


# --- Domain tool definitions ---

TOOL_GENERATION_PROMPT = """\
Generate {count} realistic tool/API definitions for a {domain} domain agent.

Tool categories to cover: {categories}

For each tool, provide:
1. A clear, specific name (snake_case)
2. A detailed description of what it does
3. Parameters with types, descriptions, and whether required
4. A response schema showing what the tool returns

Requirements:
- Tools should be realistic — things an actual {domain} system would have
- Include both simple (1-2 params) and complex (4-6 params) tools
- Some tools should have dependencies on others (e.g., must look up account before modifying it)
- Include at least one tool that could be misused if called incorrectly (for adversarial testing)

Return a JSON array of tool definitions. Each tool should have:
{{"name": "...", "description": "...", "parameters": [...], "response_schema": {{...}}}}
"""

PERSONA_GENERATION_PROMPT = """\
Generate {count} diverse user personas for a {domain} domain.

Each persona should have:
- name: realistic full name
- age: between 22-75
- occupation: relevant to someone who'd interact with a {domain} agent
- personality_traits: 3-4 traits that affect communication style
- tone: how they communicate (e.g., "formal and precise", "casual and impatient")
- detail_level: "brief", "moderate", or "verbose"
- background: 1-2 sentences of context relevant to their interaction

Requirements:
- Diverse ages, occupations, and communication styles
- Some should be technically savvy, others not
- Some should be cooperative, others difficult/adversarial
- Include personas that might trigger edge cases (confused, angry, non-native speaker)

Existing personas (avoid duplicating): {existing}

Return a JSON array of persona objects.
"""

SCENARIO_GENERATION_PROMPT = """\
Generate a challenging multi-turn conversation scenario for a {domain} agent.

Category: {category}
Persona: {persona}
Available Tools: {tools}

Category definitions:
- adaptive_tool_use: User needs shift mid-conversation, requiring the agent to adapt its tool selection
- scope_management: User requests things outside the agent's capabilities — agent must recognize and handle gracefully
- empathetic_resolution: Emotionally charged situation requiring both tool use AND empathetic communication
- extreme_scenario_recovery: Something goes wrong (tool error, unexpected data) — agent must recover
- adversarial_input_mitigation: User provides misleading, ambiguous, or adversarial inputs

Generate a scenario with:
1. id: unique identifier (format: "{domain}_{category}_{number}")
2. category: the category above
3. persona: the persona dict
4. user_goals: 5-8 specific, interconnected goals the user wants to accomplish
5. tools: the relevant subset of available tools (as full definitions)
6. initial_message: the user's opening message (in-character for the persona)
7. difficulty: "easy", "medium", or "hard"
8. expected_tool_sequence: the ideal sequence of tool calls (optional, for reference)

Requirements:
- Goals should be interconnected (completing one reveals information needed for another)
- At least one goal should require multiple tool calls
- At least one goal should have potential for the agent to make a mistake
- The initial message should naturally lead toward the goals without stating them all explicitly
- For the given category, include the specific challenge pattern described above

Return a single JSON object (not an array).
"""


def _extract_json(text: str) -> str:
    """Extract JSON from a response that might have markdown formatting."""
    # Try parsing directly first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Extract from code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try finding array or object boundaries
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

    return text


def generate_tools(domain: str, categories: list[str], count: int = 20) -> list[dict]:
    """Generate tool definitions for a domain."""
    logger.info("Generating %d tools for %s", count, domain)

    prompt = TOOL_GENERATION_PROMPT.format(
        count=count,
        domain=domain,
        categories=", ".join(categories),
    )

    response = _get_client().chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=8000,
    )

    raw = _extract_json(response.choices[0].message.content)
    tools_data = json.loads(raw)

    # Validate each tool
    validated = []
    for t in tools_data:
        try:
            tool = ToolDefinition(**t)
            validated.append(tool.model_dump())
        except ValidationError as e:
            logger.warning("Skipping invalid tool: %s", e)

    logger.info("Generated %d valid tools for %s", len(validated), domain)
    return validated


def generate_personas(
    domain: str, count: int = 30, existing: list[dict] | None = None
) -> list[dict]:
    """Generate user personas for a domain."""
    logger.info("Generating %d personas for %s", count, domain)

    existing_names = [p.get("name", "") for p in (existing or [])]

    prompt = PERSONA_GENERATION_PROMPT.format(
        count=count,
        domain=domain,
        existing=", ".join(existing_names) if existing_names else "none",
    )

    response = _get_client().chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=6000,
    )

    raw = _extract_json(response.choices[0].message.content)
    personas_data = json.loads(raw)

    validated = []
    for p in personas_data:
        try:
            persona = Persona(**p)
            validated.append(persona.model_dump())
        except ValidationError as e:
            logger.warning("Skipping invalid persona: %s", e)

    logger.info("Generated %d valid personas for %s", len(validated), domain)
    return validated


def generate_scenario(
    domain: str,
    category: str,
    persona: dict,
    tools: list[dict],
    index: int,
) -> dict | None:
    """Generate a single scenario."""
    # Simplify tools for the prompt (just name + description + param names)
    tools_summary = [
        {
            "name": t["name"],
            "description": t["description"],
            "parameters": [p["name"] for p in t.get("parameters", [])],
        }
        for t in tools
    ]

    prompt = SCENARIO_GENERATION_PROMPT.format(
        domain=domain,
        category=category,
        persona=json.dumps(persona, indent=2),
        tools=json.dumps(tools_summary, indent=2),
        number=index,
    )

    try:
        response = _get_client().chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_tokens=4000,
        )

        raw = _extract_json(response.choices[0].message.content)
        scenario_data = json.loads(raw)

        # Ensure required fields
        scenario_data["id"] = f"{domain}_{category}_{index:04d}"
        scenario_data["category"] = category

        # Attach full tool definitions (not just summaries)
        tool_names = [t["name"] for t in scenario_data.get("tools", tools_summary)]
        scenario_data["tools"] = [t for t in tools if t["name"] in tool_names] or tools

        scenario = Scenario(**scenario_data)
        return scenario.model_dump()

    except Exception as e:
        logger.error("Failed to generate scenario %s_%s_%d: %s", domain, category, index, e)
        return None


def generate_scenarios(
    domain: str,
    categories: list[str],
    personas: list[dict],
    tools: list[dict],
    per_category: int = 20,
    max_workers: int = 5,
) -> list[dict]:
    """Generate scenarios for all categories in a domain."""
    scenarios = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for cat in categories:
            for i in range(per_category):
                persona = personas[i % len(personas)]
                future = executor.submit(generate_scenario, domain, cat, persona, tools, i)
                futures[future] = (cat, i)

        for future in as_completed(futures):
            cat, idx = futures[future]
            result = future.result()
            if result:
                scenarios.append(result)
                logger.info("Generated scenario %s_%s_%d", domain, cat, idx)

    logger.info("Generated %d total scenarios for %s", len(scenarios), domain)
    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation data")
    parser.add_argument("--domain", required=True, help="Domain name")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "adaptive_tool_use",
            "scope_management",
            "empathetic_resolution",
            "extreme_scenario_recovery",
            "adversarial_input_mitigation",
        ],
    )
    parser.add_argument("--tools-count", type=int, default=20)
    parser.add_argument("--personas-count", type=int, default=30)
    parser.add_argument("--scenarios-per-category", type=int, default=20)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()

    output_base = Path(args.output_dir)

    # Step 1: Generate tools
    domain_dir = output_base / "domains" / args.domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    tools_path = domain_dir / "tools.json"
    if tools_path.exists():
        logger.info("Loading existing tools from %s", tools_path)
        with open(tools_path) as f:
            tools = json.load(f)
    else:
        tools = generate_tools(args.domain, args.categories, args.tools_count)
        with open(tools_path, "w") as f:
            json.dump(tools, f, indent=2)
        logger.info("Saved tools to %s", tools_path)

    # Step 2: Generate personas
    personas_path = domain_dir / "personas.json"
    if personas_path.exists():
        logger.info("Loading existing personas from %s", personas_path)
        with open(personas_path) as f:
            personas = json.load(f)
    else:
        personas = generate_personas(args.domain, args.personas_count)
        with open(personas_path, "w") as f:
            json.dump(personas, f, indent=2)
        logger.info("Saved personas to %s", personas_path)

    # Step 3: Generate scenarios
    scenario_dir = output_base / "scenarios" / args.domain
    scenario_dir.mkdir(parents=True, exist_ok=True)

    scenarios = generate_scenarios(
        args.domain,
        args.categories,
        personas,
        tools,
        per_category=args.scenarios_per_category,
        max_workers=args.max_workers,
    )

    for scenario in scenarios:
        path = scenario_dir / f"{scenario['id']}.json"
        with open(path, "w") as f:
            json.dump(scenario, f, indent=2)

    logger.info("Saved %d scenarios to %s", len(scenarios), scenario_dir)


if __name__ == "__main__":
    main()
