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
from eval.simulation.dual_control import (
    TRIGGER_AFTER_TURN,
    TRIGGER_AGENT_CALLED,
    TRIGGER_KINDS,
    USER_ACTION_TURN_MAX,
    USER_ACTION_TURN_MIN,
)
from eval.simulation.probes import PROBE_KINDS, PROBE_TURN_MAX, PROBE_TURN_MIN
from eval.templating import (
    DEFAULT_INSTANTIATION_SEED,
    SLOT_TYPES,
    TEMPLATE_SLOTS_KEY,
    find_placeholders,
    instantiate,
    resolve_slots,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


VALID_CATEGORIES = {
    "adaptive_tool_use",
    "scope_management",
    "empathetic_resolution",
    "extreme_scenario_recovery",
    "adversarial_input_mitigation",
}

VALID_OPS = {"equals", "increased_by", "decreased_by", "contains", "not_exists"}

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

# Atomic rubric criteria bounds (issue #54). Criteria may only inform the two
# JUDGE-scored dimensions — Cost/Latency/Reliability and the deterministic state
# check are measured, not judged, so they are not valid targets. Weights are
# positive and capped so no single criterion can dominate a dimension by orders
# of magnitude; the text floor is a cheap guard against placeholder criteria
# ("good", "ok") — true atomicity is a human/author-review property.
VALID_CRITERIA_DIMENSIONS = {"task_completion", "tool_selection"}
CRITERIA_MIN_COUNT = 3
CRITERIA_MAX_COUNT = 6
CRITERIA_MIN_TEXT_LEN = 15
CRITERIA_MAX_WEIGHT = 10.0


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
    # Set by the generation repair loop when a scenario was corrected in one
    # round. Informational only — the author_model is the source of record.
    repaired: bool | None = None


class RubricCriterion(BaseModel):
    id: str
    text: str
    dimension: str
    weight: float = 1.0


class CriteriaAuthorship(BaseModel):
    """Provenance for rubric criteria (issue #54).

    Criteria are typically authored LATER than the scenario, by a different
    model/agent, so they carry their own stamp instead of overloading the
    scenario's ``authorship`` block. ``criteria_author_model`` follows the same
    contamination rule as ``author_model``: a contestant must not write the
    grading criteria for its own exam.
    """

    criteria_author_model: str
    criteria_author_run: str | None = None
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
    # Atomic rubric criteria (issue #54) — optional on any schema version; when
    # present, criteria_authorship is required (see _validate_criteria).
    rubric_criteria: list[RubricCriterion] | None = None
    criteria_authorship: CriteriaAuthorship | None = None
    # Dual control (issue #58) — optional. Kept as a raw dict (its user_actions'
    # ``state_delta`` and the user_tools' ``scope`` are validated by hand against
    # ground_truth in _validate_dual_control, and its assertion-free deltas use
    # the same dotted-path format as expected_state_changes).
    dual_control: dict | None = None
    # Recovery probe (issue #57) — optional deterministic mid-conversation
    # perturbation. Kept as a raw dict (its ``recovery_assertions`` use the
    # ``assert`` key, a Python keyword) and validated by hand in
    # _validate_recovery_probe.
    recovery_probe: dict | None = None


def _author_blocklist() -> set[str]:
    """Lowercased set of model ids AND display names from MODELS_UNDER_TEST."""
    blocked: set[str] = set()
    for m in MODELS_UNDER_TEST:
        blocked.add(m["model_id"].lower())
        blocked.add(m["name"].lower())
    return blocked


def _matching_contestant(author: str) -> str | None:
    """FAMILY-AWARE contestant match for an author string (lowercased input).

    Returns the matched blocklist entry, or None. Mirrors
    scripts/generate_data.assert_author_allowed: a different snapshot or
    display-name of a contestant is still a contestant, so block when either id
    is a prefix of the other. The "human-handwritten" sentinel never matches.
    Shared by the scenario-authorship check and the criteria-authorship check
    (issue #54) so both provenance stamps enforce one rule.
    """
    if author == "human-handwritten":
        return None
    for blocked in _author_blocklist():
        if author == blocked or author.startswith(blocked) or blocked.startswith(author):
            return blocked
    return None


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
        blocked = _matching_contestant(author)
        if blocked is not None:
            errors.append(
                f"authorship.author_model '{scenario.authorship.author_model.strip()}' "
                f"matches MODELS_UNDER_TEST entry '{blocked}' "
                "(a contestant must not author its own exam)"
            )

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


def _validate_criteria(scenario: ScenarioSchema) -> list[str]:
    """Validate atomic rubric criteria + their provenance, when present (#54).

    Presence-gated on any schema version: a scenario without ``rubric_criteria``
    is untouched (the criteria_authorship-without-criteria case is the one
    error reachable then). With criteria: 3-6 items, unique non-empty ids,
    non-placeholder text, a valid judge dimension, a sane positive weight, and a
    required ``criteria_authorship`` stamp whose author is not a contestant.
    """
    errors: list[str] = []

    if scenario.rubric_criteria is None:
        if scenario.criteria_authorship is not None:
            errors.append("criteria_authorship present without rubric_criteria")
        return errors

    n = len(scenario.rubric_criteria)
    if not (CRITERIA_MIN_COUNT <= n <= CRITERIA_MAX_COUNT):
        errors.append(
            f"rubric_criteria has {n} item(s) (expected {CRITERIA_MIN_COUNT}-{CRITERIA_MAX_COUNT})"
        )

    seen_ids: set[str] = set()
    for i, crit in enumerate(scenario.rubric_criteria):
        loc = f"rubric_criteria[{i}]"
        cid = crit.id.strip()
        if not cid:
            errors.append(f"{loc}: empty id")
        elif cid in seen_ids:
            errors.append(f"{loc}: duplicate id '{cid}'")
        seen_ids.add(cid)

        if len(crit.text.strip()) < CRITERIA_MIN_TEXT_LEN:
            errors.append(
                f"{loc}: text too short (min {CRITERIA_MIN_TEXT_LEN} chars; "
                "criteria must be atomic AND checkable, not placeholders)"
            )
        if crit.dimension not in VALID_CRITERIA_DIMENSIONS:
            errors.append(
                f"{loc}: dimension '{crit.dimension}' not in "
                f"{sorted(VALID_CRITERIA_DIMENSIONS)} (criteria inform the judge "
                "dimensions only; state grading and the measured CLEAR dimensions "
                "are not judgeable targets)"
            )
        if not (0 < crit.weight <= CRITERIA_MAX_WEIGHT):
            errors.append(f"{loc}: weight {crit.weight} out of range (0, {CRITERIA_MAX_WEIGHT}]")

    # Provenance: criteria are authored separately from the scenario (often
    # later, by a different model), so they need their own honest stamp.
    if scenario.criteria_authorship is None:
        errors.append("rubric_criteria requires 'criteria_authorship' with criteria_author_model")
    else:
        crit_author = scenario.criteria_authorship.criteria_author_model.strip().lower()
        if not crit_author:
            errors.append("criteria_authorship.criteria_author_model is empty")
        else:
            blocked = _matching_contestant(crit_author)
            if blocked is not None:
                errors.append(
                    "criteria_authorship.criteria_author_model "
                    f"'{scenario.criteria_authorship.criteria_author_model.strip()}' "
                    f"matches MODELS_UNDER_TEST entry '{blocked}' "
                    "(a contestant must not write the grading criteria for its own exam)"
                )

    return errors


def _validate_dual_control(scenario: ScenarioSchema) -> list[str]:
    """Validate a dual_control block when present (issue #58).

    Presence-gated on any schema version: a scenario without ``dual_control`` is
    untouched. With a block: non-empty ``expected_state_changes`` (the empty-
    assertions no-unauthorized-mutation contract is incompatible with a user who
    legitimately mutates the world); at least one user_tool (each with a
    non-empty name and a declared ``scope`` list of top-level keys it may
    mutate) and at least one user_action; and — the authorization boundary, the
    subtle part of #58 — every action must name a DECLARED user_tool and write
    ONLY within that tool's declared scope. The trigger vocabulary and turn
    bounds mirror
    ``DualControl``/``UserAction`` so the on-disk validator and the runtime
    object agree on what a valid block is.
    """
    block = scenario.dual_control
    if block is None:
        return []

    errors: list[str] = []
    if not isinstance(block, dict):
        return ["dual_control must be an object"]

    # A dual-control scenario MUST declare non-empty expected_state_changes.
    # With empty assertions ([] or absent) the state grader evaluates the
    # no-unauthorized-mutation contract instead (final world == initial world,
    # see score_state_changes) — and in a dual-control scenario the USER always
    # mutates the world, so the agent would be charged with an unauthorized
    # mutation for the user's own legitimate self-serve (coordination_ok
    # permanently False and the efficacy grade failing with the user's key
    # named). The two contracts are incompatible by construction, so this is a
    # validation failure, caught before any run.
    if not scenario.expected_state_changes:
        errors.append(
            "dual_control requires non-empty 'expected_state_changes': with empty "
            "assertions the state grader enforces the no-unauthorized-mutation "
            "contract (final world == initial world), which the user's own scripted "
            "mutations always violate — the agent would be charged for the user's "
            "legitimate actions"
        )

    tools_raw = block.get("user_tools")
    if not isinstance(tools_raw, list) or not tools_raw:
        errors.append("dual_control requires a non-empty 'user_tools' list")
        tools_raw = tools_raw if isinstance(tools_raw, list) else []

    # Map tool name -> declared scope (top-level keys it may mutate).
    tool_scopes: dict[str, list[str]] = {}
    for i, t in enumerate(tools_raw):
        loc = f"dual_control.user_tools[{i}]"
        if not isinstance(t, dict):
            errors.append(f"{loc}: must be an object")
            continue
        name = t.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"{loc}: missing non-empty 'name'")
            continue
        scope = t.get("scope") or []
        if not isinstance(scope, list):
            errors.append(f"{loc}: 'scope' must be a list of top-level state keys")
            scope = []
        tool_scopes[name] = [str(s) for s in scope]

    actions_raw = block.get("user_actions")
    if not isinstance(actions_raw, list) or not actions_raw:
        errors.append("dual_control requires a non-empty 'user_actions' list")
        actions_raw = actions_raw if isinstance(actions_raw, list) else []

    for i, a in enumerate(actions_raw):
        loc = f"dual_control.user_actions[{i}]"
        if not isinstance(a, dict):
            errors.append(f"{loc}: must be an object")
            continue

        tool_name = a.get("tool")
        if tool_name not in tool_scopes:
            errors.append(f"{loc}: references undeclared user_tool '{tool_name}'")

        trigger = a.get("trigger")
        if trigger not in TRIGGER_KINDS:
            errors.append(f"{loc}: unknown trigger '{trigger}' (allowed: {sorted(TRIGGER_KINDS)})")
        elif trigger == TRIGGER_AFTER_TURN:
            tv = a.get("trigger_value")
            if not isinstance(tv, int) or isinstance(tv, bool):
                errors.append(f"{loc}: after_turn trigger_value must be an int turn")
            elif not (USER_ACTION_TURN_MIN <= tv <= USER_ACTION_TURN_MAX):
                errors.append(
                    f"{loc}: after_turn turn {tv} out of range "
                    f"[{USER_ACTION_TURN_MIN}, {USER_ACTION_TURN_MAX}]"
                )
        elif trigger == TRIGGER_AGENT_CALLED:
            tv = a.get("trigger_value")
            if not isinstance(tv, str) or not tv.strip():
                errors.append(f"{loc}: agent_called trigger_value must be a non-empty tool name")

        # Authorization boundary: the action's state delta may only write
        # top-level keys within the named tool's declared scope. This is the
        # no-unauthorized-mutation contract on the USER side — a user action
        # cannot reach into agent-only / server-only state.
        delta = a.get("state_delta") or {}
        if not isinstance(delta, dict):
            errors.append(f"{loc}: 'state_delta' must be an object of dotted-path -> value")
            delta = {}
        allowed = set(tool_scopes.get(tool_name, []))
        for dotted in delta:
            top = str(dotted).split(".", 1)[0]
            if top not in allowed:
                errors.append(
                    f"{loc}: state_delta path '{dotted}' writes top-level key '{top}' "
                    f"outside the declared scope {sorted(allowed)} of user_tool '{tool_name}' "
                    "(authorization boundary: user tools may only touch user-legible state)"
                )
            # A delta that targets a key that doesn't exist in ground_truth at all
            # is suspicious for a user self-serve action (the user mutates their
            # OWN existing state), but absence is allowed (e.g. appending the first
            # item to an as-yet-absent list) — so this is a soft check, only the
            # scope breach above is a hard error.

    return errors


def _validate_assertion_block(assertions, ground_truth: dict | None, loc_prefix: str) -> list[str]:
    """Validate a list of state assertions against ground_truth (issue #57).

    Factored out of _validate_v02_blocks' expected_state_changes loop so the
    recovery-probe's ``recovery_assertions`` are held to EXACTLY the same op /
    path-resolution / contains-match rules as the scenario's own assertions —
    one grammar, one validator. ``goal`` fuzzy-matching is deliberately NOT
    applied here: a probe assertion (e.g. "the wrong account never came into
    existence") need not correspond to a user_goal.
    """
    errors: list[str] = []
    if not isinstance(assertions, list):
        return [f"{loc_prefix}: must be a list of assertions"]
    for i, raw in enumerate(assertions):
        loc = f"{loc_prefix}[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{loc}: assertion must be an object")
            continue
        path = raw.get("assert")
        op = raw.get("op")
        if not path:
            errors.append(f"{loc}: missing 'assert' path")
            continue
        if op not in VALID_OPS:
            errors.append(f"{loc}: unknown op '{op}' (allowed: {sorted(VALID_OPS)})")
            continue
        # Path resolution against ground_truth. An ``equals`` assertion MAY name a
        # path absent from ground_truth on purpose — the canonical "the bad entity
        # the probe introduced must NOT exist / be acted on" check (e.g. assert a
        # wrong account id equals null). So absent paths are only an error for the
        # delta ops, which require a resolvable initial value to diff against.
        found, value = _resolve_path(ground_truth, path) if ground_truth else (False, None)
        if op in {"increased_by", "decreased_by"} and not found:
            errors.append(f"{loc}: assert path '{path}' does not resolve in ground_truth")
        elif op == "contains":
            if found and not isinstance(value, list):
                errors.append(
                    f"{loc}: op 'contains' requires '{path}' to be a list in ground_truth"
                )
            if not isinstance(raw.get("match"), dict):
                errors.append(f"{loc}: op 'contains' requires a 'match' partial dict")
    return errors


def _validate_recovery_probe(scenario: ScenarioSchema) -> list[str]:
    """Validate a recovery_probe block when present (issue #57).

    Presence-gated on any schema version: a scenario without ``recovery_probe``
    is untouched. With a probe: a valid kind (the small enum), a turn in
    [4, 5], a non-empty injection string, and — if recovery_assertions are
    present — assertions held to the same grammar as expected_state_changes.
    Mirrors RecoveryProbe.__init__ so the on-disk validator and the runtime
    object agree on what a valid probe is.
    """
    probe = scenario.recovery_probe
    if probe is None:
        return []

    errors: list[str] = []
    if not isinstance(probe, dict):
        return ["recovery_probe must be an object"]

    kind = probe.get("kind")
    if kind not in PROBE_KINDS:
        errors.append(f"recovery_probe.kind '{kind}' not in {sorted(PROBE_KINDS)}")

    turn = probe.get("turn")
    if not isinstance(turn, int) or isinstance(turn, bool):
        errors.append("recovery_probe.turn must be an integer")
    elif not (PROBE_TURN_MIN <= turn <= PROBE_TURN_MAX):
        errors.append(
            f"recovery_probe.turn {turn} out of range [{PROBE_TURN_MIN}, {PROBE_TURN_MAX}]"
        )

    injection = probe.get("injection")
    if not isinstance(injection, str) or not injection.strip():
        errors.append("recovery_probe.injection must be a non-empty string")

    recovery_assertions = probe.get("recovery_assertions")
    if recovery_assertions:
        errors.extend(
            _validate_assertion_block(
                recovery_assertions, scenario.ground_truth, "recovery_probe.recovery_assertions"
            )
        )

    return errors


def _validate_template(data: dict) -> list[str]:
    """Validate a ``template_slots`` declaration + slot/placeholder coherence (#60).

    Checks, all on the RAW template (before instantiation):

    - every slot declaration is an object with a known ``type`` (or a pinned
      ``value``);
    - ``choice`` slots carry a non-empty ``options`` list (the one type whose
      generation can otherwise raise at instantiation);
    - every ``{{slot}}`` referenced ANYWHERE in the scenario is declared, and
      every declared slot is referenced somewhere — so a template can neither
      reference an undeclared slot (instantiation would leave a literal ``{{…}}``)
      nor declare a dead slot.

    The placeholder scan deliberately ignores the ``template_slots`` block itself
    (its keys ARE the slot names, not references) by scanning the rest of the dict.
    """
    errors: list[str] = []
    slot_specs = data.get(TEMPLATE_SLOTS_KEY)
    if not isinstance(slot_specs, dict) or not slot_specs:
        errors.append(f"'{TEMPLATE_SLOTS_KEY}' must be a non-empty object when present")
        return errors

    declared = set(slot_specs)
    for name, spec in slot_specs.items():
        loc = f"{TEMPLATE_SLOTS_KEY}['{name}']"
        if not isinstance(spec, dict):
            errors.append(f"{loc}: declaration must be an object")
            continue
        if "value" in spec and "type" not in spec:
            continue  # pinned literal, no generator
        slot_type = spec.get("type")
        if slot_type not in SLOT_TYPES:
            errors.append(f"{loc}: unknown slot type {slot_type!r} (known: {list(SLOT_TYPES)})")
        elif slot_type == "choice" and not spec.get("options"):
            errors.append(f"{loc}: slot type 'choice' requires a non-empty 'options' list")

    # Coherence: referenced (everywhere except the declaration block) vs declared.
    referenced = find_placeholders({k: v for k, v in data.items() if k != TEMPLATE_SLOTS_KEY})
    for missing in sorted(referenced - declared):
        errors.append(
            f"placeholder '{{{{{missing}}}}}' referenced but not declared in {TEMPLATE_SLOTS_KEY}"
        )
    for dead in sorted(declared - referenced):
        errors.append(f"slot '{dead}' declared in {TEMPLATE_SLOTS_KEY} but never referenced")

    # Stable-identifier guard: slots rotate surface VALUES only. The scenario id,
    # criteria ids, and authorship records are stable keys — file/board identity,
    # judge verdict round-trip keys, and the contamination audit trail — so a
    # placeholder in any of them would silently rotate the key per seed.
    if find_placeholders(data.get("id")):
        errors.append("placeholders are not allowed in 'id' (stable scenario identifier)")
    criteria = data.get("rubric_criteria")
    if isinstance(criteria, list):
        for i, crit in enumerate(criteria):
            if isinstance(crit, dict) and find_placeholders(crit.get("id")):
                errors.append(
                    f"rubric_criteria[{i}]: placeholders are not allowed in criterion "
                    "ids (stable scoring keys)"
                )
    for block in ("authorship", "criteria_authorship"):
        if find_placeholders(data.get(block)):
            errors.append(f"placeholders are not allowed in '{block}' (audit trail)")

    return errors


def validate_scenario_dict(data: dict) -> list[str]:
    """Validate an in-memory scenario dict. Returns list of error messages.

    This is the single source of truth for per-scenario validation. Both the
    on-disk validator (``validate_scenario``) and the generation-time repair
    loop (scripts/generate_data.py) call it, so a scenario is checked against
    EXACTLY the same rules whether it lives in a file or in memory. The error
    strings are stable and human-readable; the repair loop feeds them verbatim
    back to the author model.

    Parameterized templates (issue #60): when ``data`` carries a
    ``template_slots`` block, the declaration and slot/placeholder coherence are
    validated first, then the template is INSTANTIATED with the default seed and
    the rest of the schema/content checks run against the instantiated scenario —
    so a template is held to exactly the same bar as a concrete scenario (its
    assertions resolve, its criteria reference real entities, etc.). A scenario
    with no ``template_slots`` is validated unchanged.
    """
    errors: list[str] = []

    if data.get(TEMPLATE_SLOTS_KEY) is not None:
        errors.extend(_validate_template(data))
        # Don't try to instantiate an ill-formed template — slot resolution could
        # raise and the placeholder errors above are the actionable ones.
        if errors:
            return errors
        try:
            resolve_slots(data.get("id", ""), data[TEMPLATE_SLOTS_KEY], DEFAULT_INSTANTIATION_SEED)
        except ValueError as e:
            return [f"template instantiation: {e}"]
        # Validate the INSTANTIATED scenario against the full schema/content rules.
        data = instantiate(data, DEFAULT_INSTANTIATION_SEED)

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

    # Atomic rubric criteria (issue #54) — presence-gated, any schema version.
    errors.extend(_validate_criteria(scenario))

    # Dual control (issue #58) — presence-gated, any schema version.
    errors.extend(_validate_dual_control(scenario))

    # Recovery probe (issue #57) — presence-gated, any schema version.
    errors.extend(_validate_recovery_probe(scenario))

    return errors


def validate_scenario(path: Path) -> list[str]:
    """Validate a single scenario file. Returns list of error messages."""
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    return validate_scenario_dict(data)


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
