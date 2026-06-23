# Deterministic Coded Tool Transitions (issue #87)

**Status: phase 1a — foundation only.** This PR ships the registry, the
interface contract, three exemplar transitions, the unit tests, and this doc.
It does NOT touch the runner. Runner wiring (with an LLM fallback) is phase 1b,
a separate PR. The foundation is published first so the pattern can be reviewed
before the remaining ~17 tools are built against it.

## The problem

COT Bench's whole identity is *deterministic state grading*: the grader reads a
single canonical world, recovery probes apply byte-identical scripted
perturbations (issue #57), and dual-control attribution (issue #58) must be
reproducible run to run. There is exactly one place where that determinism
leaks — the tool simulator.

Today, when an agent calls a registered tool, its effect on the graded world is
whatever an LLM narrates. `runner._simulate_tool_stateful` builds a prompt
("you are simulating a tool backend… emit `{response, state_delta}`"), an LLM
reads the world and *invents* both the response and the state delta, and the
runner applies that delta via `apply_state_delta`. So the authority over the
graded world is an LLM. "The agent moved \$2,500 from savings to checking" is
true only if the sim chose to emit a matching decrement/increment. It usually
does — but not deterministically, and the bench's value proposition is that the
state grade is trustworthy. The transfer case (S2) is where this bites hardest.

## The fix (tau-bench architecture)

Replace the LLM's authority over registered tools with **deterministic pure
Python transition functions**. A coded transition returns the *same*
`{response, state_delta}` shape the sim produced — so every downstream consumer
keeps working unchanged — but it computes that result as a pure function of its
inputs, so the same `(args, world)` always yields a byte-identical result.

### The interface contract

A tool transition is a deterministic pure function:

```python
def transition(args: dict, world: dict) -> dict:
    # returns {"state_delta": dict, "response": <json-serializable>}
```

Rules every transition MUST obey:

- **Pure & deterministic.** Same `(args, world)` → byte-identical return, every
  call. A transition MUST NOT mutate `world` in place — it treats the world as
  read-only and describes every mutation through `state_delta`. No clock, no
  randomness, no `id()` / hash-of-object. Any generated id is a deterministic
  function of `(args, world)` (see ticket ids below) — never random.
- **`state_delta` is the existing dotted-path format.** Keys are dotted paths
  into the world (`"accounts.BUS-CHK-001.balance"`); a plain value replaces the
  value at that path; a list grows via the `{"__append__": <item>}` value
  convention. This is exactly what `runner.apply_state_delta` already consumes —
  verified against that function, not assumed. A read-only tool returns
  `{"state_delta": {}, ...}`.
- **`response`** is what the agent sees as the tool result (a string or any
  json-serializable value). For a registered tool this REPLACES the
  LLM-narrated result.
- **In-task errors are returned, not raised.** Insufficient funds, unknown
  account/id, missing required args → return
  `{"state_delta": {}, "response": {"error": "...realistic message..."}}`. This
  mirrors tau-bench's `observation = f"Error: {e}"`: the agent reads the error
  and may recover. Real exceptions are reserved for genuine programmer error.
- **Scope discipline.** A transition's `state_delta` top-level keys should fall
  within that tool's declared `writes` allow-list (Option A — see below). The
  exemplars here are written to respect it; phase 1b's runner clamp enforces it
  as a backstop.

### The registry

`eval/simulation/tool_transitions.py` holds `TRANSITIONS`, a dict keyed by
`(domain, tool_name)` → transition function, and:

```python
def get_transition(domain: str, tool_name: str) -> Transition | None: ...
```

`None` means there is no coded transition for that tool — phase 1b's runner
falls back to the LLM sim. The module is import-light by design (no runner
import) so the runner and the scenario loader can both depend on it without an
import cycle — the same shape as `profiles.py` and `dual_control.py`.

## The three exemplars

They span the three patterns the full migration will repeat.

1. **`get_account_balance` (banking, READ).** Reads
   `world["accounts"][account_id]`, returns `current_balance` /
   `available_balance` (mirroring the scenario's declared `response_schema`),
   `state_delta = {}`. Unknown account → in-task error.
2. **`initiate_transfer` (banking, MUTATE).** Validates both accounts and
   sufficient funds, then emits a `state_delta` that sets the source balance to
   its new total and the destination balance to its new total
   (`accounts.<from>.balance` / `accounts.<to>.balance`). Balances are written
   as absolute new values, not deltas, because `apply_state_delta` *replaces*
   the value at a path — computing the new totals in the transition keeps the
   applier a dumb setter. Insufficient funds / unknown account / non-positive
   amount → error with an empty delta (world untouched). This is the case S2
   most needs to be deterministic.
3. **`create_ticket` (customer_success, CREATE).** Appends a new ticket to
   `world["tickets"]` (a top-level list) via `{"tickets": {"__append__": …}}`,
   with a deterministic id. Response returns the new ticket id.

Arg and field names follow the real scenarios (`account_id`,
`from_account_id` / `to_account_id` / `amount`, `subject` / `description` /
`priority` / `category`). Where a detail is ambiguous it is documented as an
assumption in the function docstring; phase 2 reconciles every tool against its
scenarios.

## Deterministic id generation (`create_ticket`)

A created ticket needs an id that is (a) a pure function of `(args, world)` and
(b) collision-free against every existing ticket. Random ids and clock-based
ids both break determinism, so the scheme is:

1. Collect the set of existing `ticket_id` values from `world["tickets"]`.
2. Candidate sequence number `n = 10000 + len(existing)`, formatted `TCK-<n>`
   (the corpus uses the `TCK-` prefix).
3. If `TCK-<n>` is already present (a pre-seeded id could land on the
   count-derived candidate), deterministically probe forward — `n+1`, `n+2`, … —
   until a free id is found.

Because both the starting candidate (the set's size) and the forward probe (the
set's membership) are pure functions of the existing-id set, the result is
byte-identical run to run and can never collide with a pre-seeded id. The unit
tests pin both the no-collision case and the forced-probe case.

## How it composes with `apply_state_delta` and the Option A `writes`-clamp

Nothing downstream changes. A coded transition emits the *same* `state_delta`
format the LLM sim emits, so:

- `runner.apply_state_delta(world, delta)` applies it unchanged — dotted-path
  set and `{"__append__": …}` list-growth both already supported.
- The Option A `writes`-clamp in `runner._simulate_tool_stateful` (issue #58)
  drops any delta path whose top-level key is not in the called tool's declared
  `writes` allow-list. A coded transition that respects scope discipline passes
  the clamp untouched; the clamp remains a backstop against a future transition
  that strays.
- Dual-control attribution (`_agent_mutated_keys`) records the same top-level
  keys regardless of whether the delta came from the LLM or from coded Python —
  so the coordination verdict is unaffected, and becomes *more* reproducible
  because the delta is now deterministic.

## Phased migration plan

- **Phase 1a (this PR) — foundation.** Registry + interface contract + 3
  exemplars + pure unit tests + this doc. No runner change.
- **Phase 1b — runner wiring with LLM fallback.** Wire `get_transition(domain,
  tool_name)` into `_simulate_tool_stateful`: if a coded transition exists, call
  it and use its `{response, state_delta}`; otherwise fall back to the LLM sim
  exactly as today. Tools without a coded transition behave identically to now,
  so the wiring is safe to land before the full tool set is coded. (Phase 1b
  edits the runner; phase 1a deliberately does not, so the two reviews stay
  independent.)
- **Phase 2 — remaining mutating tools.** Build coded transitions for the rest
  of the mutating tools (transfers, ticket/escalation, subscription, meeting
  scheduling, recurring transfers, fraud reports, …), reconciling each against
  its scenarios' arg names, world shape, and `expected_state_changes`.
- **Phase 3 — state spine trusts only coded mutations.** Once every
  graded-state-mutating tool has a coded transition, tighten the state grade so
  it trusts only coded mutations for graded keys; the LLM sim stays only for
  read-only narration of tools that do not touch graded state.

## References

- Issue #87 (this work).
- `eval/simulation/runner.py` — `apply_state_delta`, `_simulate_tool_stateful`
  (the existing format and the LLM authority being replaced).
- `eval/simulation/dual_control.py` — the Option A `writes` / scope concept.
- `docs/dual-control.md` — the determinism posture this work extends.
