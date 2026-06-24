"""Standing CI guard against the silent fail->pass negative-assertion class (issue #100).

The bug class
-------------
COT Bench grades agent runs by asserting final world state. A refuse/scope
scenario encodes "the agent did NOT do the bad thing" as a VACUOUS NEGATIVE
assertion on some top-level world key K -- ``K equals []`` (or ``{}`` / ``0`` /
``False``) or ``K not_exists``. Those assertions PASS when K never appears in the
final world (``equals []`` against an absent path is False, but an absent
list-typed key that the corpus seeds as ``[]`` stays ``[]``; ``not_exists`` passes
outright on an absent path). So if NO coded transition for the domain ever writes
K, the assertion is satisfied *vacuously* -- it would pass even if the agent DID
perform the violation, because the violating tool wrote a DIFFERENT key.

PR #103 fixed four such holes (``discounts`` / ``tier_changes`` / ``transfers_executed``
and earlier ``data_exports``) by having the violating tool ALSO append the record
under the asserted alias (tool-side mirror-write). This test is the standing guard
so the class cannot return as the corpus grows (issue #100, ahead of the 8-domain
expansion in #95/#96).

The contract this test enforces
-------------------------------
For every scenario, for every VACUOUS-TYPE NEGATIVE assertion on a top-level key K,
K must be EITHER:

  (a) in the domain's WRITTEN-KEY UNIVERSE -- the set of top-level world keys some
      coded transition for that domain can mutate. Then a real violation lands a
      record under K and the ``== []`` / ``not_exists`` assertion catches it. This
      is the only safe state for a key that a capable tool can produce; OR

  (b) in the domain's TRIPWIRE REGISTRY (``data/domains/<domain>/tripwire_keys.json``)
      -- a key that is unviolatable BY DESIGN because NO tool in the domain performs
      that action at all (e.g. banking ``tax_filings`` / ``brokerage_orders``). A
      vacuous negative on such a key is a legitimate out-of-scope tripwire; the
      registry records WHY it can never be written.

A vacuous negative on a key that is neither (a) nor (b) is the dangerous case: a
capable tool may exist that writes a different key -> the assertion false-passes.
This test FAILS loudly, naming the file/key/op, turning the silent grading hole
into a build failure.

Scope (deliberately narrow)
---------------------------
This gate covers ONLY vacuous-type negatives -- ``op == equals`` with value in
``([], {}, 0, False)`` (NOT ``None``: ``equals null`` requires the path to resolve,
so it is not vacuous), OR ``op == not_exists``. Those are the assertions that pass
on an unwritten key, i.e. the fail->pass class. Positive assertions
(``contains`` / ``increased_by`` / a non-vacuous ``equals``) FAIL on an unwritten
key, so they self-report and are out of scope here.

The written-key universe is computed STATICALLY from the source (never
hand-maintained): ``ast`` extracts every ``state_delta`` key literal from each
transition function, reduces dotted paths to their top-level segment
(``accounts.X.balance`` -> ``accounts``), and the ``TRANSITIONS`` registry groups
the functions per domain (mapping ``fn`` -> its keys by ``fn.__name__``). Static
extraction means a tool whose mirror-write key is only ever exercised under a
condition no unit test hits is still counted -- the guard sees the source, not a
sampled execution.

The registry is also checked for staleness: a tripwire key that a tool actually
writes is a contradiction (FAIL -- it belongs to the universe, not the registry),
and a tripwire key asserted by no scenario is unused (FAIL -- dead registry entry).
"""

import ast
import glob
import json
import os

import pytest

from eval.simulation.tool_transitions import TRANSITIONS

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
_TT_SRC = os.path.join(_REPO, "eval", "simulation", "tool_transitions.py")
_SCENARIO_ROOT = os.path.join(_REPO, "data", "scenarios")
_DOMAINS_ROOT = os.path.join(_REPO, "data", "domains")

# A vacuous-type ``equals`` value is one that an absent/seeded-empty key trivially
# satisfies. ``None`` is intentionally EXCLUDED: ``equals null`` requires the path
# to resolve (see eval/scoring/state_check.py), so it is not vacuous. Booleans are
# matched by identity below so 0/False and []/{}/0 are not conflated.
_VACUOUS_EQUALS_VALUES = ([], {}, 0, False)


def _top_segment(dotted: str) -> str:
    """Top-level segment of a (possibly dotted) world path: ``a.b.c`` -> ``a``."""
    return str(dotted).split(".", 1)[0]


def _is_vacuous_negative(assertion: dict) -> bool:
    """True iff ``assertion`` is a vacuous-type negative (the fail->pass class).

    ``not_exists`` (passes on an absent path) or ``equals`` with a value in
    ``([], {}, 0, False)`` (an absent/empty key trivially satisfies it). Matching
    uses ``type``-aware equality so ``0`` and ``False`` -- and ``[]`` and ``0`` --
    are not conflated by Python's ``0 == False`` / ``[] == 0`` semantics.
    """
    op = assertion.get("op")
    if op == "not_exists":
        return True
    if op == "equals":
        value = assertion.get("value")
        return any(value == cand and type(value) is type(cand) for cand in _VACUOUS_EQUALS_VALUES)
    return False


# --- Static written-key universe (computed from source, never hand-maintained) -- #


def _state_delta_keys_in_function(fn_node: ast.FunctionDef) -> set:
    """Top-level world keys a transition's ``state_delta`` can write, via ``ast``.

    Finds every dict literal that is the value for a ``"state_delta"`` key and
    collects the top-level segment of each of its key literals. Handles three
    shapes the transitions use: an inline ``{"state_delta": {<keys>}}`` dict, an
    f-string key (``f"accounts.{id}.balance"`` -> ``accounts``), and a delta dict
    bound to a local name then grown (``delta = {...}; delta[f"..."] = ...`` in
    ``close_account``). A static read sees keys behind conditionals that no unit
    test may exercise -- exactly the coverage the guard needs.
    """
    keys: set = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Dict):
            continue
        for dkey, dval in zip(node.keys, node.values):
            if isinstance(dkey, ast.Constant) and dkey.value == "state_delta":
                keys |= _keys_of_delta_value(dval, fn_node)
    return keys


def _keys_of_delta_value(value: ast.AST, fn_node: ast.FunctionDef) -> set:
    """Top-level keys of a ``state_delta`` value (an inline dict or a local name)."""
    if isinstance(value, ast.Dict):
        out: set = set()
        for k in value.keys:
            out |= _key_literal_top_segments(k)
        return out
    if isinstance(value, ast.Name):
        return _keys_assigned_to_local(value.id, fn_node)
    return set()


def _key_literal_top_segments(key_node: ast.AST) -> set:
    """Top-level segment(s) of a dict-key literal (a string constant or f-string)."""
    if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
        return {_top_segment(key_node.value)}
    if isinstance(key_node, ast.JoinedStr):
        # Rebuild the template with a sentinel for each interpolation, then take the
        # top segment. ``f"accounts.{id}.balance"`` -> ``accounts.\x00.balance`` ->
        # ``accounts``. The top segment is a literal in every transition (the
        # interpolation is always AFTER the first dot), so this is exact.
        parts = [
            str(v.value) if isinstance(v, ast.Constant) else "\x00" for v in key_node.values
        ]
        return {_top_segment("".join(parts))}
    return set()


def _keys_assigned_to_local(name: str, fn_node: ast.FunctionDef) -> set:
    """Keys written into a local dict ``name`` -- via ``name = {...}`` and ``name[k] = ...``.

    Covers ``close_account``'s ``delta = {...}`` then ``delta[f"accounts.{id}.balance"] = ...``
    pattern so its conditionally-added balance keys are counted statically.
    """
    keys: set = set()
    for node in ast.walk(fn_node):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.Dict):
                for k in node.value.keys:
                    keys |= _key_literal_top_segments(k)
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == name
            ):
                keys |= _key_literal_top_segments(target.slice)
    return keys


def _function_written_keys() -> dict:
    """Map each transition ``fn.__name__`` -> its static set of written top-level keys."""
    with open(_TT_SRC, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    return {
        node.name: _state_delta_keys_in_function(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def _written_key_universe() -> dict:
    """Map domain -> set of top-level world keys some coded transition can write.

    Groups the per-function static key sets through the ``TRANSITIONS`` registry
    (which is keyed by ``(domain, tool_name)``), mapping each registered function
    to its keys by ``__name__``.
    """
    fn_keys = _function_written_keys()
    universe: dict = {}
    for (domain, _tool), fn in TRANSITIONS.items():
        universe.setdefault(domain, set()).update(fn_keys.get(fn.__name__, set()))
    return universe


# --- Tripwire registry + scenario corpus loaders ------------------------------- #


def _tripwire_registry(domain: str) -> dict:
    """Load ``data/domains/<domain>/tripwire_keys.json`` ({key: rationale}); {} if absent."""
    path = os.path.join(_DOMAINS_ROOT, domain, "tripwire_keys.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        registry = json.load(f)
    assert isinstance(registry, dict), f"{path}: tripwire registry must be a JSON object"
    for key, rationale in registry.items():
        assert (
            isinstance(rationale, str) and rationale.strip()
        ), f"{path}: tripwire key {key!r} must have a non-empty rationale string"
    return registry


def _domains() -> list:
    """Domains present under data/scenarios/ (one dir per domain)."""
    return sorted(
        d for d in os.listdir(_SCENARIO_ROOT) if os.path.isdir(os.path.join(_SCENARIO_ROOT, d))
    )


def _vacuous_negatives_by_domain() -> dict:
    """Map domain -> list of (scenario_filename, key, op) for every vacuous negative."""
    by_domain: dict = {d: [] for d in _domains()}
    for path in glob.glob(os.path.join(_SCENARIO_ROOT, "**", "*.json"), recursive=True):
        domain = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as f:
            scenario = json.load(f)
        for assertion in scenario.get("expected_state_changes", []) or []:
            if _is_vacuous_negative(assertion):
                key = _top_segment(assertion.get("assert", ""))
                by_domain.setdefault(domain, []).append(
                    (os.path.basename(path), key, assertion.get("op"))
                )
    return by_domain


# --- The gate ------------------------------------------------------------------ #


def test_every_vacuous_negative_key_is_writable_or_a_registered_tripwire():
    """THE GATE: no vacuous negative may assert a key that is neither writable nor a tripwire.

    For every scenario's vacuous-type negative assertion on top-level key K, K must
    be in the domain's written-key universe (a real violation lands a record there)
    OR in the domain's tripwire registry (unviolatable by design). Anything else is
    a potential silent fail->pass hole and fails the build with the file/key/op.
    """
    universe = _written_key_universe()
    by_domain = _vacuous_negatives_by_domain()
    assert any(items for items in by_domain.values()), (
        "no vacuous-negative assertions found across the corpus -- the guard has "
        "nothing to check; scenario discovery or the vacuity test is broken"
    )

    violations = []
    for domain, items in by_domain.items():
        written = universe.get(domain, set())
        tripwires = set(_tripwire_registry(domain))
        for filename, key, op in items:
            if key not in written and key not in tripwires:
                violations.append(
                    f"  {domain}/{filename}: asserts {key!r} ({op}) -- not written by any "
                    f"coded transition and not in tripwire_keys.json. Either a tool must "
                    f"mirror-write {key!r} (a real hole, fix like PR #103) or register it as "
                    f"an out-of-scope tripwire."
                )
    assert not violations, (
        "Vacuous negative assertion(s) on keys no tool writes and no tripwire registry "
        "covers -- these false-pass even when the agent performs the violation:\n"
        + "\n".join(sorted(violations))
    )


@pytest.mark.parametrize("domain", _domains())
def test_tripwire_registry_has_no_writable_key(domain):
    """A tripwire that a tool actually writes is a stale/contradictory entry.

    A registered tripwire claims "no tool can ever write this." If a transition DOES
    write it, the entry is wrong -- the key belongs to the written-key universe, and
    leaving it in the registry would let a future vacuous negative on it pass without
    the universe-membership check that proves it is catchable. Fail loudly.
    """
    universe = _written_key_universe().get(domain, set())
    tripwires = set(_tripwire_registry(domain))
    contradictory = sorted(tripwires & universe)
    assert not contradictory, (
        f"{domain}: tripwire_keys.json registers key(s) a coded transition DOES write: "
        f"{contradictory}. Remove them from the registry -- they are writable, so the gate "
        f"already covers them via the written-key universe."
    )


@pytest.mark.parametrize("domain", _domains())
def test_tripwire_registry_has_no_unused_key(domain):
    """A tripwire asserted by no scenario is a dead registry entry.

    The registry exists to whitelist out-of-scope keys that scenarios actually
    assert. A key no vacuous negative references is dead weight that can drift out of
    sync with the corpus -- fail so the registry stays a faithful mirror of the
    asserted out-of-scope surface.
    """
    asserted_keys = {key for _f, key, _op in _vacuous_negatives_by_domain().get(domain, [])}
    tripwires = set(_tripwire_registry(domain))
    unused = sorted(tripwires - asserted_keys)
    assert not unused, (
        f"{domain}: tripwire_keys.json registers key(s) no scenario asserts as a vacuous "
        f"negative: {unused}. Remove the dead entries (or add the scenario that needs them)."
    )
