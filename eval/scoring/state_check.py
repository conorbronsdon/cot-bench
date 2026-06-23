"""Deterministic state grader for COT Bench (schema v0.2).

This is the judge-independent half of Efficacy. Where the LLM judges score
*opinion* ("did the agent handle this well?"), this module scores *fact* ("did
the checking balance actually increase by $2,500?") by comparing the canonical
world state before and after a conversation against a scenario's
``expected_state_changes`` assertions.

The assertion vocabulary is intentionally tiny and fully deterministic:

- ``equals`` — the final value at the path equals ``value``. Numeric comparisons
  are tolerance-aware: when BOTH the actual and expected value are numbers
  (int/float, not bool) they match iff ``abs(a - b) <= 0.01`` (the same
  ``_TOLERANCE`` used by ``increased_by``/``decreased_by``), so a float money
  field does not require bit-exact equality. Non-numeric values (strings, etc.)
  still compare with exact ``==``.
- ``increased_by`` / ``decreased_by`` — the final value differs from the initial
  value by ``value`` (float tolerance 0.01). The path must resolve in both worlds.
- ``contains`` — the path resolves to a list that, after the conversation,
  contains an item matching every key in ``match``. A ``<key>_contains`` match
  key means a case-insensitive substring test on ``str(item[<key>])`` instead of
  an equality test. A plain (non-``_contains``) match key uses the same
  tolerance-aware numeric comparison as ``equals``.
- ``not_exists`` — the path does NOT resolve in the final world (added for
  recovery probes, issue #57). This is the one assertion that PASSES on an absent
  path: it encodes "the bad entity the probe introduced was never acted on /
  created" — e.g. a wrong-account probe must not produce ``accounts.BUS-CHK-999``.
  ``equals null`` cannot express this (it requires the path to resolve), so this
  is the minimal, deterministic op that makes the wrong-entity recovery check
  expressible in the existing assertion grammar.

An **empty** assertion list (``[]``) encodes the no-unauthorized-mutation
contract: it scores 1.0 iff the world is unchanged, else 0.0 with the differing
top-level keys named. ``None``/missing ground_truth makes state grading
inapplicable and returns ``None`` so callers can fall back to judge-only Efficacy.
"""

# Numeric tolerance for increased_by / decreased_by float comparisons.
_TOLERANCE = 0.01

# Detail-string prefix for a failed no-unauthorized-mutation check. A named
# constant (not an inline literal) because the failure-mode classifier
# (eval/scoring/failure_modes.py) keys on this exact prefix to deterministically
# tag such failures as policy violations — single source, no drift.
UNAUTHORIZED_MUTATION_DETAIL = "unauthorized mutation in top-level key(s): "

_MISSING = object()


def _values_equal(a, b) -> bool:
    """Equality test that is tolerance-aware for numbers, exact otherwise.

    When BOTH sides are numeric (int/float, excluding bool) they compare equal
    iff ``abs(a - b) <= _TOLERANCE`` — the same tolerance applied to
    ``increased_by``/``decreased_by`` — so a float money field does not require
    bit-exact equality (e.g. 250.0 == 250.000001). For any non-numeric operand
    (strings, lists, dicts, bools, None) the comparison falls back to ``==``.
    ``bool`` is excluded from the numeric branch so ``True``/``1`` and
    ``False``/``0`` keep strict ``==`` semantics.
    """
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    ):
        return abs(a - b) <= _TOLERANCE
    return a == b


def resolve_path(world, dotted: str):
    """Resolve a dotted path into ``world``.

    Returns ``(found, value)``. ``found`` is False (and ``value`` is None) when
    any segment is missing or an intermediate value is not a dict. A path may
    legitimately resolve to ``None``; ``found`` distinguishes "present and None"
    from "absent".
    """
    node = world
    for key in str(dotted).split("."):
        if not isinstance(node, dict):
            return False, None
        nxt = node.get(key, _MISSING)
        if nxt is _MISSING:
            return False, None
        node = nxt
    return True, node


def _match_item(item: dict, match: dict) -> bool:
    """True when ``item`` satisfies every key in ``match`` (partial-dict match).

    A ``<key>_contains`` match key is a case-insensitive substring test on
    ``str(item[<key>])``; any other key is an equality test against
    ``item[<key>]``.
    """
    if not isinstance(item, dict):
        return False
    for mkey, mval in match.items():
        if mkey.endswith("_contains"):
            field = mkey[: -len("_contains")]
            if field not in item:
                return False
            if str(mval).lower() not in str(item[field]).lower():
                return False
        else:
            got = item.get(mkey, _MISSING)
            if got is _MISSING or not _values_equal(got, mval):
                return False
    return True


def check_assertion(initial_world, final_world, assertion: dict) -> dict:
    """Evaluate one ``expected_state_changes`` assertion.

    Returns ``{"passed": bool, "detail": str}``.
    """
    op = assertion.get("op")
    path = assertion.get("assert", "")

    if op == "equals":
        found, value = resolve_path(final_world, path)
        if not found:
            return {"passed": False, "detail": f"{path}: path not found in final world"}
        expected = assertion.get("value")
        passed = _values_equal(value, expected)
        return {
            "passed": passed,
            "detail": f"{path}: expected {expected!r}, got {value!r}",
        }

    if op in ("increased_by", "decreased_by"):
        i_found, i_val = resolve_path(initial_world, path)
        f_found, f_val = resolve_path(final_world, path)
        if not i_found:
            return {"passed": False, "detail": f"{path}: path not found in initial world"}
        if not f_found:
            return {"passed": False, "detail": f"{path}: path not found in final world"}
        try:
            delta = float(f_val) - float(i_val)
        except (TypeError, ValueError):
            return {
                "passed": False,
                "detail": f"{path}: non-numeric values ({i_val!r} -> {f_val!r})",
            }
        expected = float(assertion.get("value", 0.0))
        target = expected if op == "increased_by" else -expected
        passed = abs(delta - target) <= _TOLERANCE
        return {
            "passed": passed,
            "detail": (
                f"{path}: changed by {delta:+.2f}, expected {op} {expected} (target {target:+.2f})"
            ),
        }

    if op == "not_exists":
        # Passes iff the path is ABSENT from the final world — the recovery-probe
        # "bad entity was never created / acted on" check (issue #57). The only
        # op for which a non-resolving path is the SUCCESS case.
        found, _ = resolve_path(final_world, path)
        return {
            "passed": not found,
            "detail": (f"{path}: {'absent (ok)' if not found else 'present (should not exist)'}"),
        }

    if op == "contains":
        found, value = resolve_path(final_world, path)
        if not found:
            return {"passed": False, "detail": f"{path}: path not found in final world"}
        if not isinstance(value, list):
            return {"passed": False, "detail": f"{path}: not a list (got {type(value).__name__})"}
        match = assertion.get("match", {})
        passed = any(_match_item(item, match) for item in value)
        return {
            "passed": passed,
            "detail": (
                f"{path}: {'found' if passed else 'no'} item matching {match} "
                f"({len(value)} item(s) present)"
            ),
        }

    return {"passed": False, "detail": f"unknown op {op!r}"}


def _changed_top_level_keys(initial_world: dict, final_world: dict) -> list[str]:
    """Top-level keys whose value differs between initial and final worlds."""
    keys = set(initial_world) | set(final_world)
    return sorted(k for k in keys if initial_world.get(k) != final_world.get(k))


def score_state_changes(initial_world, final_world, assertions):
    """Score a run's final world against its ``expected_state_changes``.

    Returns ``{"score", "checks", "n_passed", "n_total"}`` where ``score`` is
    ``n_passed / n_total`` in [0, 1].

    Special cases:

    - ``None`` initial/final world (no ground_truth): returns ``None`` — state
      grading is not applicable; the caller falls back to judge-only Efficacy.
    - Empty ``assertions`` (``[]``): the no-unauthorized-mutation contract.
      Score 1.0 if ``final_world == initial_world``, else 0.0 with the differing
      top-level keys named in the single check's detail.
    """
    if initial_world is None or final_world is None:
        return None

    if not assertions:
        unchanged = final_world == initial_world
        if unchanged:
            detail = "no unauthorized mutation: final world == initial world"
        else:
            changed = _changed_top_level_keys(initial_world, final_world)
            detail = UNAUTHORIZED_MUTATION_DETAIL + ", ".join(changed)
        return {
            "score": 1.0 if unchanged else 0.0,
            "checks": [{"passed": unchanged, "detail": detail}],
            "n_passed": 1 if unchanged else 0,
            "n_total": 1,
        }

    checks = [check_assertion(initial_world, final_world, a) for a in assertions]
    n_total = len(checks)
    n_passed = sum(1 for c in checks if c["passed"])
    return {
        "score": n_passed / n_total,
        "checks": checks,
        "n_passed": n_passed,
        "n_total": n_total,
    }
