"""Validate scenario JSON files against the expected schema.

Use this after generating scenarios to catch malformed data before
running expensive evaluation runs.

Covers the v0.2 schema (ground-truth world state + deterministic state
assertions + authorship), cross-scenario dedup, and a distribution report.
Stdlib-only beyond Pydantic (already a dependency): difflib for fuzzy matching.
"""

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, ValidationError

from eval.config import MODELS_UNDER_TEST

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


VALID_CATEGORIES = {
    "adaptive_tool_use",
    "scope_management",
    "empathetic_resolution",
    "extreme_scenario_recovery",
    "adversarial_input_mitigation",
}

VALID_OPS = {"equals", "increased_by", "decreased_by", "contains"}

# Dedup thresholds (difflib SequenceMatcher ratio, 0-1).
DEDUP_HARD_THRESHOLD = 0.85
DEDUP_WARN_THRESHOLD = 0.75
# Jaccard overlap on lowercased goal sets.
GOAL_JACCARD_HARD_THRESHOLD = 0.8
# Fuzzy match floor for an assertion's `goal` against user_goals.
GOAL_FUZZY_FLOOR = 0.70

# Difficulty mix target (report §3d), used only with --strict-distribution.
# Scaled to whatever total exists; tolerance is +/- 10 percentage points.
DIFFICULTY_TARGET = {"easy": 0.22, "medium": 0.45, "hard": 0.33}
DIFFICULTY_TOLERANCE = 0.10
# Persona reuse cap per domain (prevents 1-persona-per-scenario monoculture).
PERSONA_REUSE_CAP = 5


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


class Authorship(BaseModel):
    author_model: str
    author_run: str | None = None
    human_reviewed_by: str | None = None
    review_date: str | None = None


class ScenarioSchema(BaseModel):
    id: str
    category: str
    persona: Persona
    user_goals: list[str]
    tools: list[Tool]
    initial_message: str
    difficulty: str | None = None
    expected_tool_sequence: list[str] | None = None
    # v0.2 additions (optional unless schema_version == "0.2")
    schema_version: str | None = None
    authorship: Authorship | None = None
    ground_truth: dict | None = None
    expected_state_changes: list[dict] | None = None


def _author_blocklist() -> set[str]:
    """Lowercased set of model ids AND display names from MODELS_UNDER_TEST."""
    blocked: set[str] = set()
    for m in MODELS_UNDER_TEST:
        blocked.add(m["model_id"].lower())
        blocked.add(m["name"].lower())
    return blocked


def _resolve_path(state: dict, dotted: str):
    """Resolve a dotted path into a nested dict. Returns (found, value)."""
    cur = state
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return False, None
    return True, cur


def _best_goal_ratio(text: str, goals: list[str]) -> float:
    """Best SequenceMatcher ratio of `text` against any goal (lowercased)."""
    text_l = text.lower()
    return max(
        (SequenceMatcher(None, text_l, g.lower()).ratio() for g in goals),
        default=0.0,
    )


def _validate_v02_blocks(scenario: ScenarioSchema) -> list[str]:
    """Validate the v0.2-specific blocks. Assumes schema_version == '0.2'."""
    errors: list[str] = []

    # Required blocks
    if scenario.authorship is None:
        errors.append("schema_version 0.2 requires 'authorship' with author_model")
    if scenario.ground_truth is None:
        errors.append("schema_version 0.2 requires 'ground_truth'")
    if scenario.expected_state_changes is None:
        errors.append("schema_version 0.2 requires 'expected_state_changes' (may be [])")

    # Authorship: author must not be a contestant. FAMILY-AWARE (prefix) match,
    # mirroring scripts/generate_data.assert_author_allowed — a different
    # snapshot/display-name of a contestant is still a contestant. (Exact match
    # alone silently misses e.g. author "gpt-4.1" vs contestant snapshot
    # "gpt-4.1-2025-04-14", or a contestant whose display name carries a suffix
    # like "GPT-4.1 (anchor)".)
    if scenario.authorship is not None:
        author = scenario.authorship.author_model.strip().lower()
        if author != "human-handwritten":
            for blocked in _author_blocklist():
                if author == blocked or author.startswith(blocked) or blocked.startswith(author):
                    errors.append(
                        f"authorship.author_model '{scenario.authorship.author_model.strip()}' "
                        f"matches MODELS_UNDER_TEST entry '{blocked}' "
                        "(a contestant must not author its own exam)"
                    )
                    break

    # Assertions resolve against ground_truth + goals fuzzy-match
    if scenario.ground_truth is not None and scenario.expected_state_changes is not None:
        for i, raw in enumerate(scenario.expected_state_changes):
            loc = f"expected_state_changes[{i}]"
            path = raw.get("assert")
            op = raw.get("op")
            if not path:
                errors.append(f"{loc}: missing 'assert' path")
                continue
            if op not in VALID_OPS:
                errors.append(f"{loc}: unknown op '{op}' (allowed: {sorted(VALID_OPS)})")
                continue

            found, value = _resolve_path(scenario.ground_truth, path)
            if op in {"equals", "increased_by", "decreased_by"}:
                if not found:
                    errors.append(f"{loc}: assert path '{path}' does not resolve in ground_truth")
            elif op == "contains":
                if not found:
                    errors.append(f"{loc}: assert path '{path}' does not resolve in ground_truth")
                elif not isinstance(value, list):
                    errors.append(
                        f"{loc}: op 'contains' requires '{path}' to be a list in ground_truth"
                    )
                if not isinstance(raw.get("match"), dict):
                    errors.append(f"{loc}: op 'contains' requires a 'match' partial dict")

            goal = raw.get("goal")
            if goal:
                ratio = _best_goal_ratio(goal, scenario.user_goals)
                if ratio < GOAL_FUZZY_FLOOR:
                    errors.append(
                        f"{loc}: goal '{goal[:50]}...' does not fuzzy-match any user_goal "
                        f"(best ratio {ratio:.2f} < {GOAL_FUZZY_FLOOR})"
                    )

    return errors


def validate_scenario(path: Path) -> list[str]:
    """Validate a single scenario file. Returns list of error messages."""
    errors = []
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    # expected_state_changes assertions are kept as raw dicts (the JSON key
    # `assert` is a Python keyword, so a Pydantic field model would need an
    # alias). They are validated by hand in _validate_v02_blocks.
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
    if scenario.category not in VALID_CATEGORIES:
        errors.append(f"Unknown category: {scenario.category}")

    # Check tool names in expected sequence exist
    if scenario.expected_tool_sequence:
        tool_names = {t.name for t in scenario.tools}
        for tool_name in scenario.expected_tool_sequence:
            if tool_name not in tool_names:
                errors.append(f"Expected tool '{tool_name}' not in available tools")

    # v0.2 conditional requirements
    if scenario.schema_version == "0.2":
        errors.extend(_validate_v02_blocks(scenario))
    elif scenario.schema_version is not None:
        errors.append(f"Unsupported schema_version '{scenario.schema_version}' (expected '0.2')")

    return errors


# --- Cross-scenario checks (operate on loaded scenario dicts) ---


def _comparison_string(scenario: dict) -> str:
    goals = scenario.get("user_goals", [])
    return scenario.get("initial_message", "") + " || " + " ".join(sorted(goals))


def _goal_set(scenario: dict) -> set[str]:
    return {g.lower().strip() for g in scenario.get("user_goals", [])}


def _domain_of(path: Path, scenario_dir: Path) -> str:
    rel = path.relative_to(scenario_dir)
    return rel.parts[0] if len(rel.parts) > 1 else "(root)"


def dedup_check(loaded: list[tuple[Path, dict, str]]) -> tuple[list[str], list[str]]:
    """Cross-scenario dedup within each domain. Returns (hard_fails, warnings).

    ``loaded`` is a list of (path, scenario_dict, domain) tuples.
    """
    hard: list[str] = []
    warn: list[str] = []

    # Group by domain (passed in alongside path).
    by_domain: dict[str, list[tuple[Path, str, set[str]]]] = defaultdict(list)
    for path, scenario, domain in loaded:
        by_domain[domain].append((path, _comparison_string(scenario), _goal_set(scenario)))

    for domain, items in by_domain.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                p_i, cmp_i, goals_i = items[i]
                p_j, cmp_j, goals_j = items[j]
                name_i = p_i.name
                name_j = p_j.name

                ratio = SequenceMatcher(None, cmp_i, cmp_j).ratio()
                if ratio >= DEDUP_HARD_THRESHOLD:
                    hard.append(
                        f"[{domain}] near-duplicate {name_i} ~ {name_j} "
                        f"(similarity {ratio:.2f} >= {DEDUP_HARD_THRESHOLD})"
                    )
                elif ratio >= DEDUP_WARN_THRESHOLD:
                    warn.append(
                        f"[{domain}] possible duplicate {name_i} ~ {name_j} "
                        f"(similarity {ratio:.2f}); spot-check"
                    )

                # Goal-set Jaccard
                if goals_i and goals_j:
                    inter = len(goals_i & goals_j)
                    union = len(goals_i | goals_j)
                    jaccard = inter / union if union else 0.0
                    if jaccard >= GOAL_JACCARD_HARD_THRESHOLD:
                        hard.append(
                            f"[{domain}] goal-set overlap {name_i} ~ {name_j} "
                            f"(Jaccard {jaccard:.2f} >= {GOAL_JACCARD_HARD_THRESHOLD})"
                        )

    return hard, warn


def _load_domain_tools(scenario_dir: Path, domain: str) -> set[str] | None:
    """Load tool names from data/domains/{domain}/tools.json if it exists."""
    tools_path = scenario_dir.parent / "domains" / domain / "tools.json"
    if not tools_path.exists():
        return None
    try:
        with open(tools_path) as f:
            tools = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return {t.get("name", "") for t in tools if isinstance(t, dict)}


def distribution_report(
    loaded: list[tuple[Path, dict, str]], scenario_dir: Path, strict: bool
) -> list[str]:
    """Print a per-domain distribution table. Returns hard-fail messages
    (only non-empty when strict=True and bands are violated)."""
    failures: list[str] = []

    by_domain: dict[str, list[dict]] = defaultdict(list)
    for _, scenario, domain in loaded:
        by_domain[domain].append(scenario)

    print("\n--- Distribution report ---")
    for domain in sorted(by_domain):
        scenarios = by_domain[domain]
        total = len(scenarios)
        cat_counts = Counter(s.get("category", "?") for s in scenarios)
        diff_counts = Counter(s.get("difficulty") or "(none)" for s in scenarios)
        persona_counts = Counter(s.get("persona", {}).get("name", "(unnamed)") for s in scenarios)

        print(f"\n[{domain}] {total} scenarios")
        print("  categories: " + ", ".join(f"{c}={n}" for c, n in sorted(cat_counts.items())))
        print("  difficulty: " + ", ".join(f"{d}={n}" for d, n in sorted(diff_counts.items())))

        reused = {name: n for name, n in persona_counts.items() if n > 1}
        if reused:
            print(
                "  persona reuse: " + ", ".join(f"{name}={n}" for name, n in sorted(reused.items()))
            )

        # Tool-coverage gaps vs domain tools.json (if present)
        domain_tools = _load_domain_tools(scenario_dir, domain)
        if domain_tools:
            used = set()
            for s in scenarios:
                used.update(t.get("name", "") for t in s.get("tools", []))
            gaps = sorted(domain_tools - used)
            if gaps:
                print(f"  tool-coverage gaps ({len(gaps)}): {', '.join(gaps)}")

        # Strict band checks (scaled to total)
        if strict:
            for cat in VALID_CATEGORIES:
                if cat_counts.get(cat, 0) == 0:
                    failures.append(f"[{domain}] category '{cat}' has 0 scenarios")
            for diff, target in DIFFICULTY_TARGET.items():
                frac = (diff_counts.get(diff, 0) / total) if total else 0.0
                if abs(frac - target) > DIFFICULTY_TOLERANCE:
                    failures.append(
                        f"[{domain}] difficulty '{diff}' at {frac:.0%} "
                        f"(target {target:.0%} +/- {DIFFICULTY_TOLERANCE:.0%})"
                    )
            for name, n in persona_counts.items():
                if n > PERSONA_REUSE_CAP:
                    failures.append(
                        f"[{domain}] persona '{name}' reused {n}x (cap {PERSONA_REUSE_CAP})"
                    )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Validate scenario JSON files")
    parser.add_argument(
        "--strict-distribution",
        action="store_true",
        help="Turn distribution band violations into failures (default: informational)",
    )
    parser.add_argument("--scenario-dir", default="data/scenarios")
    args = parser.parse_args()

    scenario_dir = Path(args.scenario_dir)
    if not scenario_dir.exists():
        logger.error("No scenarios directory found at %s", scenario_dir)
        sys.exit(1)

    total = 0
    valid = 0
    invalid = 0
    loaded: list[tuple[Path, dict, str]] = []

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
        # Load for cross-scenario checks even if individually invalid, so dedup
        # still works; skip files that aren't parseable JSON.
        try:
            with open(path) as f:
                loaded.append((path, json.load(f), _domain_of(path, scenario_dir)))
        except (json.JSONDecodeError, OSError):
            pass

    # Cross-scenario dedup
    hard_dupes, warn_dupes = dedup_check(loaded)
    for w in warn_dupes:
        logger.warning(w)
    for h in hard_dupes:
        logger.error(h)

    # Distribution report (+ optional strict failures)
    dist_failures = distribution_report(loaded, scenario_dir, args.strict_distribution)
    for d in dist_failures:
        logger.error(d)

    n_failures = invalid + len(hard_dupes) + len(dist_failures)
    print(f"\nValidation complete: {valid}/{total} schema-valid, {invalid} invalid")
    if hard_dupes:
        print(f"Dedup: {len(hard_dupes)} hard failure(s), {len(warn_dupes)} warning(s)")
    if dist_failures:
        print(f"Distribution: {len(dist_failures)} band failure(s)")
    sys.exit(1 if n_failures > 0 else 0)


if __name__ == "__main__":
    main()
