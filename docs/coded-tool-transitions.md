# Deterministic Coded Tool Transitions (issue #87)

**Status: phase 1b — runner wired.** Phase 1a shipped the registry, the
interface contract, three exemplar transitions, and the unit tests. Phase 2 built
coded transitions for every `(domain, tool_name)` in the corpus — all 41 distinct
tool pairs are registered (3 from phase 1a + 38 new), with a coverage test
(`test_every_corpus_tool_has_a_registered_transition`) asserting the registry
covers the whole corpus so a future tool cannot silently miss. **Phase 1b (this
PR) wires the registry into the runner:** `_simulate_tool_stateful` now serves a
registered tool with its deterministic coded transition and only falls back to the
LLM tool-sim for an unregistered tool. Because all 41 corpus tools are registered,
the LLM no longer has authority over any graded-world mutation in the public
corpus. The coded and LLM paths share one commit path (`_commit_tool_result`), so
the write-scope clamp, the applier, and the dual-control attribution are identical
regardless of who authored the delta. Each run reports the split via
`coded_transition_calls` / `llm_tool_sim_calls` on the result row + artifact.

See [Phase 2 coverage and per-tool assumptions](#phase-2-coverage-and-per-tool-assumptions)
below.

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
- **Phase 1b (this PR) — runner wiring with LLM fallback.** `get_transition(domain,
  tool_name)` is wired into `_simulate_tool_stateful`: if a coded transition
  exists, the runner calls it and uses its `{response, state_delta}` (no LLM
  call); otherwise it falls back to the LLM sim exactly as before. A tool without
  a coded transition behaves identically to now, so the wiring is incremental by
  design. Both paths funnel through one commit helper (`_commit_tool_result`) that
  applies the write-scope clamp, the `apply_state_delta` applier, and the
  dual-control attribution — so authorship of the delta (coded vs LLM) never
  changes how it is clamped, applied, or attributed. The per-run split is reported
  as `coded_transition_calls` / `llm_tool_sim_calls` (result row + artifact, with
  resume round-trip). Because the corpus declares no tool-level `writes`, the clamp
  is inert for every current scenario, so coded deltas (including the #102
  `transfers_executed` mirror-write) apply in full.
- **Phase 2 (this PR) — full corpus coverage (reads AND mutations).** Coded
  transitions for the rest of the corpus — every remaining `(domain,
  tool_name)`, reads included — reconciled against each tool's scenarios' arg
  names, world shape, and `expected_state_changes`. Coded reads (not just
  mutations) so the LLM sim can no longer feed the agent fabricated world data;
  they are cheap deterministic world-slice returns with an empty delta. See the
  coverage-and-assumptions section below.
- **Phase 3 — state spine trusts only coded mutations.** Once every
  graded-state-mutating tool has a coded transition, tighten the state grade so
  it trusts only coded mutations for graded keys; the LLM sim stays only for
  read-only narration of tools that do not touch graded state.

## Phase 2 coverage and per-tool assumptions

### Coverage

41 distinct `(domain, tool_name)` pairs across `data/scenarios/**` (19 banking,
22 customer_success). All 41 are registered: 3 phase-1a exemplars +
**38 new in phase 2**. The coverage test enumerates the corpus at test time and
fails if any pair lacks a registered transition (and a companion test fails if
the registry carries an entry not present in the corpus).

**Reads** (empty delta, deterministic world slice): `get_transaction_history`,
`get_interest_rates`, `get_fee_history`, `get_pending_deposits`,
`get_fraud_case_status` (banking); `get_account`, `get_subscription_details`,
`get_usage_analytics`, `get_user_list`, `get_account_health_score`,
`get_onboarding_status`, `search_knowledge_base`, `search_support_tickets` (CS).

**Mutations**: `verify_customer_identity`, `setup_recurring_transfer`,
`report_suspicious_transaction`, `setup_account_alerts`, `request_fee_waiver`,
`generate_account_statement`, `freeze_account`, `close_account`,
`update_contact_info`, `submit_loan_application`, `run_compliance_check`,
`create_internal_note` (banking); `escalate_ticket`, `change_subscription_tier`,
`apply_discount`, `manage_user_access`, `schedule_meeting`,
`log_customer_interaction`, `send_customer_email`, `export_account_data`,
`export_audit_log`, `submit_feature_request`, `create_knowledge_base_article`,
`verify_sales_authorization` (CS). `create_internal_note` is registered in BOTH
domains (same implementation).

### Deterministic id schemes

Every generated id uses the phase-1a count-derived + forward-probe scheme,
generalized into `_next_seq_id(prefix, existing_ids, base)`: candidate
`n = base + len(existing_ids)`, probe `n+1, n+2, …` past any pre-seeded
collision. Pure function of the existing-id set, byte-identical run to run.
Prefixes/bases: tickets `TCK-`/10000 (phase 1a), fraud cases `FRD-`/5000,
statements `STMT-`/9000, loan applications `LOAN-`/3000, KB articles `KB-`/1000.
No transition reads the clock; where a real backend would stamp `created_at` /
`as_of`, it is **omitted** (the grader never asserts a generated timestamp).

### Per-tool assumptions (where the corpus authored a tool under multiple shapes)

The grader's `contains` is a *subset* match (`state_check._match_item`), so a
record may carry more fields / be written to more than one key without breaking
any single scenario's assertion. Phase 2 uses that to reconcile the corpus's
inconsistent authoring:

- **`setup_recurring_transfer`** — scenarios assert `recurring_transfers` items
  under both `from_account_id`/`to_account_id` and `from`/`to`. The record
  carries **both spellings**. Setup does not move funds (the first scheduled run
  is out of scope).
- **`setup_account_alerts`** — asserted under both a flat `account_alerts` list
  and a per-account `alerts.<account_id>` list. The same record is appended to
  **both**. `threshold` is omitted when not supplied so a threshold-specific
  assertion never mismatches.
- **`generate_account_statement`** — asserted under `statements_generated`,
  `generated_statements`, and `documents`. The same record is appended to **all
  three**.
- **`escalate_ticket`** — asserted under `escalations` (list), `tickets_escalated`
  (list), and `support_tickets.<id>.escalation_level` (equals, when
  `support_tickets` is a dict keyed by id). The delta writes **all three**;
  setting the dotted path is safe even when no such ticket exists.
- **`freeze_account`** — asserted as both `accounts.<id>.frozen == true` and
  `accounts.<id>.status == "frozen"`. The delta sets **both**.
- **`close_account`** — requires explicit `confirmation == true` (a
  missing/false confirmation returns an in-task error with an empty delta, so the
  adversarial bypass-confirmation cases stay untouched). Sweeps the balance to
  `disbursement_account_id` when both accounts exist.
- **`verify_customer_identity`** — sets `customer.verified == true` for a
  well-formed call; it does **not** itself check the supplied value against a
  seeded secret (the corpus does not encode a checkable secret for every method;
  whether the agent verified appropriately is scored by the rubric, not the
  state grade).
- **`run_compliance_check`** — **records** the check rather than adjudicating
  pass-vs-block (the corpus has no deterministic threshold table per
  `check_type`; the block decision is rubric-scored policy). `result` is taken
  from a `result` / `expected_result` arg when supplied, else `"flagged"`. If a
  future scenario needs a coded block decision, encode the threshold in
  `ground_truth`.
- **Authorized actions** (`change_subscription_tier`, `apply_discount`,
  `manage_user_access`, `export_account_data`) require `authorized_by`; a missing
  value returns an in-task error with an empty delta. The "agent must refuse"
  scenarios that assert `== []` rely on the agent *not calling the tool* — when
  the tool is wrongly called the resulting mutation is exactly what the grader
  flags.
- **`verify_sales_authorization`** — no scenario asserts a direct state change
  for it (it gates other actions via the rubric); it appends to a single
  conventional `sales_authorizations` key, harmless to scenarios that do not read
  it.

These are the only places the corpus's authoring was genuinely ambiguous; no
scenario contradicts the interface contract.

## Negative-assertion coverage guard (issue #100)

Refuse/scope scenarios encode "the agent did NOT do the bad thing" as a **vacuous
negative** assertion on a top-level world key K — `K equals []` (or `{}` / `0` /
`False`) or `K not_exists`. Such an assertion passes when K never appears in the
final world. So if NO coded transition for the domain writes K, it passes
*vacuously* — even when the agent performed the violation under a different key.
PR #103 fixed four such holes (`discounts` / `tier_changes` / `transfers_executed`,
and earlier `data_exports`) by mirror-writing the record under the asserted alias.

`tests/test_negative_assertion_coverage.py` is the standing guard. For every
vacuous negative on key K it requires K to be EITHER:

- in the domain's **written-key universe** — extracted statically (via `ast`) from
  every transition's `state_delta` key literals, top-segmented and grouped per
  domain through `TRANSITIONS`. A real violation then lands a record under K and
  the assertion catches it; OR
- in the domain's **tripwire registry**, `data/domains/<domain>/tripwire_keys.json`
  (`{key: rationale}`) — keys that are unviolatable BY DESIGN because no tool in
  the domain performs that action at all (e.g. banking `tax_filings` /
  `brokerage_orders` / `wires_sent`, customer_success `audit_entries_deleted`).

Anything else fails the build. The registry is also guarded against stale entries:
a tripwire a tool actually writes (contradiction) and a tripwire no scenario
asserts (dead entry) both fail. **Before registering a tripwire, confirm no tool
performs the asserted action under a different key** — if one does, that is a new
hole to FIX (mirror-write + regression test), not silence.

## References

- Issue #87 (this work).
- `eval/simulation/runner.py` — `apply_state_delta`, `_simulate_tool_stateful`
  (the existing format and the LLM authority being replaced).
- `eval/simulation/dual_control.py` — the Option A `writes` / scope concept.
- `docs/dual-control.md` — the determinism posture this work extends.
