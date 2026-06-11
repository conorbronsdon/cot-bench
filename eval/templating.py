"""Parameterized scenario templates — anti-memorization instantiation (issue #60).

The public scenario corpus is static and on GitHub, so a model that memorized a
published scenario (or its transcript) could recognize the exact account IDs,
names, amounts, and dates and short-circuit the task. This module lets a scenario
declare a **template**: surface values (names, account IDs, amounts, dates,
company names, …) become ``{{slot}}`` placeholders backed by a ``template_slots``
declaration, and each evaluation run instantiates **fresh** concrete values from a
per-run seed. The *logical* task — goals, state assertions, tool sequence, and
the semantics of the rubric criteria — is invariant; only the surface changes, so
memorizing the published surface gains nothing.

Design invariants (all tested in ``tests/test_templating.py``):

1. **Deterministic instantiation.** ``instantiate(template, seed)`` is a pure
   function of the template content and the seed: same template + same seed ->
   byte-identical instantiated scenario. This is what makes a run resumable and a
   published result reproducible from the pre-registered seed.
2. **Coherence.** A slot is resolved to one concrete value ONCE, then substituted
   everywhere that slot's placeholder appears — across ``user_goals``,
   ``persona``, ``initial_message``, ``ground_truth`` (values AND the dotted
   keys/paths that name entities), ``expected_state_changes`` (assert paths,
   match values, and the ``goal`` echo), tool argument literals, AND
   ``rubric_criteria`` text. An account ID changed in the goal is the same value
   changed in the assertion and in the criterion, so the grader still grades the
   right thing.
3. **Validator-clean output.** The instantiated scenario is an ordinary v0.2
   scenario (the ``template_slots`` declaration is stripped on instantiation), so
   it passes ``validate_scenario_dict`` and is graded by the unchanged state
   checker and judges.
4. **Pre-registration honesty.** The run pre-registers the *template* corpus hash,
   the instantiation seed, and the *instantiated* corpus hash, so the exact
   surface a run committed to is tamper-evident and recomputable (see
   ``eval/pre_registration.py``).
5. **Backwards compatible.** A scenario WITHOUT ``template_slots`` is not a
   template: ``instantiate`` returns it unchanged (minus the absent declaration),
   and the existing 92 scenarios run exactly as today. Templating is opt-in per
   scenario.

This module is mechanism-only and dependency-free (stdlib ``hashlib`` +
``random`` for a seeded, reproducible generator). It does NOT convert any
existing scenario; see ``docs/parameterized-templates.md`` for the conversion
decision.
"""

from __future__ import annotations

import hashlib
import random
import re
import string
from typing import Any

# Default per-run instantiation seed. A run that does not pass --instantiation-seed
# uses this fixed value, so default behavior is fully deterministic and CI is
# reproducible. A real published run SHOULD pass a fresh seed (and it is recorded
# in the pre-registration); the fixed default exists only so the common path and
# the test suite are deterministic without requiring the flag.
DEFAULT_INSTANTIATION_SEED = 0

# Key under which a scenario declares its substitutable slots. Its ABSENCE is the
# backwards-compatible signal: a scenario without this key is not a template and
# is returned unchanged by ``instantiate``.
TEMPLATE_SLOTS_KEY = "template_slots"

# Placeholder syntax: ``{{slot_name}}``. Double braces are used deliberately —
# single braces collide with the ``str.format`` templates in
# ``eval/scoring/rubrics.py``; double braces never appear in the existing corpus.
# Slot names are restricted to ``[A-Za-z0-9_]`` so the pattern is unambiguous and
# a name can be embedded inside other text without delimiter ambiguity.
_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z0-9_]+)\}\}")

# Slot generator types. Each maps to a deterministic generator below. Adding a
# type is a one-line entry in ``_GENERATORS`` plus a generator function; the
# validator (scripts/validate_scenarios.py) is data-driven off ``SLOT_TYPES``.
SLOT_TYPES = (
    "name",
    "first_name",
    "account_id",
    "customer_id",
    "amount",
    "integer",
    "date",
    "company",
    "digits",
    "choice",
)

# Surface pools. Intentionally generic and large enough that an instantiation is
# unlikely to reproduce a published value, but the POINT is variation, not
# secrecy — the seed and instantiated hash are pre-registered, so the surface is
# auditable, just not memorizable in advance.
_FIRST_NAMES = (
    "Marcus",
    "Priya",
    "Eleanor",
    "Devon",
    "Yuki",
    "Amara",
    "Sofia",
    "Omar",
    "Nadia",
    "Theo",
    "Lena",
    "Rafael",
    "Imani",
    "Bjorn",
    "Wei",
    "Carmen",
    "Ezra",
    "Tariq",
    "Greta",
    "Hassan",
    "Ingrid",
    "Mateo",
    "Saanvi",
    "Cyrus",
)
_LAST_NAMES = (
    "Whitfield",
    "Okonkwo",
    "Castellanos",
    "Nakamura",
    "Adeyemi",
    "Sandoval",
    "Petrova",
    "Fitzgerald",
    "Mwangi",
    "Hollander",
    "Vasquez",
    "Delacroix",
    "Ramachandran",
    "Andersson",
    "Bukowski",
    "Esposito",
    "Nguyen",
    "Haddad",
)
_COMPANIES = (
    "Northwind Logistics",
    "Meridian Foods",
    "Apex Dynamics",
    "Cedar & Bloom",
    "Vantage Analytics",
    "Brightline Media",
    "Sterling Components",
    "Harborview Retail",
    "Quill & Co.",
    "Ironwood Manufacturing",
    "Lakeside Clinics",
    "Pinnacle Freight",
)


def find_placeholders(value: Any) -> set[str]:
    """Return the set of slot names referenced by ``{{slot}}`` anywhere in a value.

    Walks strings, lists, and dicts (BOTH dict keys and values), so a placeholder
    embedded in a ``ground_truth`` account key (e.g. ``"{{acct}}"``) is found just
    like one in free text. Used by the validator to require that every referenced
    slot is declared and every declared slot is referenced.
    """
    found: set[str] = set()
    if isinstance(value, str):
        found.update(_PLACEHOLDER_RE.findall(value))
    elif isinstance(value, dict):
        for k, v in value.items():
            found.update(find_placeholders(k))
            found.update(find_placeholders(v))
    elif isinstance(value, list):
        for item in value:
            found.update(find_placeholders(item))
    return found


def _seeded_rng(corpus_seed: int, scenario_id: str, slot_name: str) -> random.Random:
    """A ``random.Random`` seeded deterministically per (run seed, scenario, slot).

    Seeding per-slot (not once per run) makes the instantiation order-independent:
    adding a slot to a scenario does not perturb the values drawn for its other
    slots, so re-instantiating after an edit changes only the edited slot. The
    seed is the sha256 of the joined key, so it is stable across processes and
    platforms (unlike ``hash()``).
    """
    key = f"{corpus_seed}\x00{scenario_id}\x00{slot_name}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _gen_first_name(rng: random.Random, spec: dict) -> str:
    return rng.choice(_FIRST_NAMES)


def _gen_name(rng: random.Random, spec: dict) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _gen_company(rng: random.Random, spec: dict) -> str:
    return rng.choice(_COMPANIES)


def _gen_digits(rng: random.Random, spec: dict) -> str:
    """A run of decimal digits. ``length`` (default 4) controls the count.

    Returned as a STRING so a leading zero is preserved and so it can be
    concatenated/embedded; account/customer ids build on this.
    """
    length = int(spec.get("length", 4))
    return "".join(rng.choice(string.digits) for _ in range(length))


def _gen_account_id(rng: random.Random, spec: dict) -> str:
    """An account id of the form ``{prefix}{digits}`` (e.g. ``PERS-CHK-4417``).

    ``prefix`` (default ``"ACCT-"``) and ``length`` (default 4) are declared on
    the slot. The prefix carries the human-meaningful entity label (the test's
    coherence does not depend on the prefix, only on consistent substitution).
    """
    prefix = str(spec.get("prefix", "ACCT-"))
    length = int(spec.get("length", 4))
    return prefix + "".join(rng.choice(string.digits) for _ in range(length))


def _gen_customer_id(rng: random.Random, spec: dict) -> str:
    prefix = str(spec.get("prefix", "CUST-"))
    length = int(spec.get("length", 5))
    return prefix + "".join(rng.choice(string.digits) for _ in range(length))


def _gen_amount(rng: random.Random, spec: dict):
    """A monetary amount in ``[min, max]`` rounded to ``step`` (default 0.01).

    Returned as a NUMBER (not a string) so it substitutes into numeric
    ground-truth / match fields as a number (see ``_coerce``). When the bounds and
    step are whole numbers the value is returned as an ``int`` (so a $400 amount
    renders as ``400``, not ``400.0``, and equality-matches an int in a
    ``contains`` assertion exactly as the hand-authored corpus does); otherwise a
    2-dp ``float``. ``step`` lets a scenario constrain to whole dollars (step 1) or
    round values (step 25).
    """
    lo = float(spec.get("min", 1.0))
    hi = float(spec.get("max", 1000.0))
    step = float(spec.get("step", 0.01))
    n_steps = int(round((hi - lo) / step))
    chosen = lo + rng.randint(0, max(n_steps, 0)) * step
    chosen = round(chosen, 2)
    if chosen.is_integer() and float(step).is_integer():
        return int(chosen)
    return chosen


def _gen_integer(rng: random.Random, spec: dict) -> int:
    lo = int(spec.get("min", 0))
    hi = int(spec.get("max", 100))
    return rng.randint(lo, hi)


def _gen_date(rng: random.Random, spec: dict) -> str:
    """An ISO ``YYYY-MM-DD`` date in ``[start, end]`` inclusive.

    Pure stdlib date arithmetic (ordinal proleptic Gregorian) — no third-party
    date lib. ``start``/``end`` default to a 2026 window if unspecified.
    """
    from datetime import date

    start = date.fromisoformat(str(spec.get("start", "2026-01-01")))
    end = date.fromisoformat(str(spec.get("end", "2026-12-31")))
    span = end.toordinal() - start.toordinal()
    chosen = start.toordinal() + rng.randint(0, max(span, 0))
    return date.fromordinal(chosen).isoformat()


def _gen_choice(rng: random.Random, spec: dict) -> Any:
    """Pick one of an explicit ``options`` list — for slots that must stay within
    a closed vocabulary (e.g. a merchant name drawn from a fixed set so the
    fraud-report criterion still matches)."""
    options = spec.get("options")
    if not options:
        raise ValueError("slot type 'choice' requires a non-empty 'options' list")
    return rng.choice(list(options))


_GENERATORS = {
    "name": _gen_name,
    "first_name": _gen_first_name,
    "account_id": _gen_account_id,
    "customer_id": _gen_customer_id,
    "amount": _gen_amount,
    "integer": _gen_integer,
    "date": _gen_date,
    "company": _gen_company,
    "digits": _gen_digits,
    "choice": _gen_choice,
}


def resolve_slots(scenario_id: str, slot_specs: dict, corpus_seed: int) -> dict[str, Any]:
    """Resolve every declared slot to one concrete value, deterministically.

    Returns ``{slot_name: value}`` where value is a str/int/float depending on the
    slot type. ``slot_specs`` maps slot name -> ``{"type": ..., **params}``. Each
    slot gets its own per-(seed, scenario, slot) RNG so the mapping is stable and
    order-independent. A literal value can be pinned with ``{"value": ...}`` (no
    ``type``), which is occasionally useful to hold one field fixed while
    varying its neighbors.
    """
    resolved: dict[str, Any] = {}
    for name in sorted(slot_specs):  # sorted: deterministic, declaration-order-free
        spec = slot_specs[name]
        if not isinstance(spec, dict):
            raise ValueError(f"slot '{name}': declaration must be an object")
        if "value" in spec and "type" not in spec:
            resolved[name] = spec["value"]
            continue
        slot_type = spec.get("type")
        gen = _GENERATORS.get(slot_type)
        if gen is None:
            raise ValueError(
                f"slot '{name}': unknown type {slot_type!r} (known: {sorted(_GENERATORS)})"
            )
        rng = _seeded_rng(corpus_seed, scenario_id, name)
        resolved[name] = gen(rng, spec)
    return resolved


def _coerce(text: str, mapping: dict[str, Any]) -> Any:
    """Substitute ``{{slot}}`` placeholders in one string.

    If the ENTIRE string is a single placeholder for a non-string slot (e.g.
    ``"{{amount}}"`` with a float value), the native typed value is returned so a
    numeric ground-truth/match field stays a number. Otherwise placeholders are
    rendered into the surrounding text as ``str(value)``. An unmapped placeholder
    is left intact (the validator forbids that case, so it only surfaces in
    direct unit tests).
    """
    m = _PLACEHOLDER_RE.fullmatch(text)
    if m is not None and m.group(1) in mapping:
        return mapping[m.group(1)]

    def repl(match: re.Match) -> str:
        slot = match.group(1)
        if slot not in mapping:
            return match.group(0)
        return str(mapping[slot])

    return _PLACEHOLDER_RE.sub(repl, text)


def substitute(value: Any, mapping: dict[str, Any]) -> Any:
    """Recursively substitute slot values through a JSON-like structure.

    Strings are coerced via ``_coerce``; dict KEYS are substituted too (so an
    account-id placeholder used as a ``ground_truth.accounts`` key or inside a
    dotted ``assert`` path is rewritten consistently with its uses elsewhere);
    lists and dict values recurse. Non-string scalars pass through unchanged.

    Dotted paths (``"accounts.{{acct}}.balance"``) work because the placeholder
    is a whole path SEGMENT, not the whole string — ``_coerce``'s in-text branch
    renders ``str(value)`` into the path, and the resolved account-id value has no
    dot, so the path structure is preserved.
    """
    if isinstance(value, str):
        return _coerce(value, mapping)
    if isinstance(value, dict):
        return {_coerce(k, mapping): substitute(v, mapping) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(item, mapping) for item in value]
    return value


def is_template(scenario_data: dict) -> bool:
    """True iff the scenario declares a non-empty ``template_slots`` block."""
    return bool(scenario_data.get(TEMPLATE_SLOTS_KEY))


def instantiate(scenario_data: dict, corpus_seed: int) -> dict:
    """Instantiate a scenario for a run, returning a NEW dict (input untouched).

    - Non-template scenario (no ``template_slots``): returns a shallow copy with
      the (absent) declaration removed — byte-for-content identical to today, so
      the existing corpus is unaffected.
    - Template scenario: resolves every slot from ``corpus_seed`` (deterministic),
      substitutes the values coherently through every field, strips the
      ``template_slots`` declaration (the instantiated scenario is an ordinary
      v0.2 scenario), and returns it. Same input + same seed -> byte-identical
      output (the substitution is pure and the slot RNG is seed-derived).

    The scenario ``id`` is NOT changed here: it remains the template's id so a
    run's per-scenario artifacts/reliability group across instantiations. The
    instantiated CONTENT hash (pre-registration) is what records the surface a run
    actually used.
    """
    if not is_template(scenario_data):
        out = dict(scenario_data)
        out.pop(TEMPLATE_SLOTS_KEY, None)
        return out

    slot_specs = scenario_data[TEMPLATE_SLOTS_KEY]
    scenario_id = scenario_data.get("id", "")
    mapping = resolve_slots(scenario_id, slot_specs, corpus_seed)

    instantiated: dict = {}
    for key, val in scenario_data.items():
        if key == TEMPLATE_SLOTS_KEY:
            continue  # declaration is consumed, not emitted
        instantiated[key] = substitute(val, mapping)
    return instantiated
