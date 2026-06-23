# Dual-Control Extension — Decision Doc (issue #58)

**Status: STAGED for an adopt/defer decision (this PR). DO NOT MERGE without
Conor's call.** The complete mechanism, two demonstration fixtures, and the
scoring / hash / aggregation wiring are in this PR so the decision is
*adopt-or-defer*, not a work item. Nothing in the published v1 corpus changes:
the demo fixtures live in `tests/fixtures/dual_control/`, never in
`data/scenarios/`, so the mechanism is inert until a scenario declares a
`dual_control` block.

## What it solves

From the best-in-class agent-eval research (Delta 5; source: **tau2-bench**):
the dominant real-world complexity that single-control simulation misses is
**dual control** — both the agent AND the user invoke tools against the *same*
shared world. A customer self-serves mid-conversation: they update their own
contact info, they approve or deny a request the agent sent, they read their own
notifications. That creates a coordination problem the agent must navigate — it
must **wait for an approval it does not control**, and it must **not re-apply a
change the user already made**. tau2-bench reports that every model tested
degrades significantly under dual control, and banking / customer-success (COT
Bench's two domains) are exactly where it bites.

Today COT Bench is single-control: only the agent (via the tool simulator) ever
mutates the world. The user simulator emits text and a completion signal — it
never acts. Dual control adds a *second actor* on the same world, which is the
real differentiator the issue calls out: 5-10 dual-control scenarios in v1.1
would be a capability no comparable open bench has, measured the way the rest of
COT Bench measures — **deterministic state grading, not judge opinion**.

## tau2 provenance — what tau2 actually does

tau2-bench gives the **user simulator its own tool set** acting on the same
backend the agent uses. The user-sim is an LLM prompted with its tools and its
goals; it *decides* when to call them. The thing under test is coordination: the
agent and a second autonomous actor must converge on a correct shared end state.
Source verification: tau2's design is "the user has tools too, against one
shared environment," and the headline finding is the across-the-board
degradation under that condition. This PR borrows the *idea* (a second actor on
the shared world) and the two domains, and deliberately diverges on *how the
user acts* — argued next.

## Design

A scenario MAY carry an optional `dual_control` block with two parts:

```jsonc
"dual_control": {
  "user_tools": [
    {
      "name": "approve_pending_request",
      "description": "Customer-side: approve a pending high-value request.",
      "parameters": [ /* same shape as an agent tool */ ],
      "scope": ["pending_requests"]        // top-level state keys the user may write
    }
  ],
  "user_actions": [
    {
      "tool": "approve_pending_request",
      "trigger": "agent_called",           // or "after_turn"
      "trigger_value": "request_wire_approval",
      "arguments": {"request_id": "WIRE-REQ-5001"},
      "state_delta": {                     // same dotted-path format as the tool sim
        "pending_requests": [ {"request_id": "WIRE-REQ-5001", "status": "approved"} ]
      },
      "user_message": "I just approved the wire on my end. Go ahead and send it."
    }
  ]
}
```

- **`user_tools`** declare *what the user side may touch*. Each carries a
  `scope`: the set of top-level state keys it is authorized to mutate. Scope is
  the load-bearing field — the authorization boundary is enforced against it.
- **`user_actions`** bind a **trigger** (when the user self-serves) to a specific
  user-tool call with a specific `state_delta`. Two trigger kinds (the single
  source is `eval/simulation/dual_control.py`):
  - `after_turn` — the user acts at a fixed turn (turns 1-9). The
    self-serve-first case ("I already changed my email myself").
  - `agent_called` — the user reacts the first time the agent calls a named tool
    (the agent sends the approval request; the user approves it next turn). The
    coordination / handoff case.

When an action fires, the runner applies its `state_delta` through the **same
`apply_state_delta`** the agent's tools use (one world, one delta mechanism), and
if the action carries a `user_message`, that scripted text becomes the next user
turn — so the agent *sees* the coordination signal, exactly like a recovery-probe
injection (issue #57). A silent action (no `user_message`) mutates the world
only and the user sim still speaks that turn.

### The scripted-vs-prompted decision (and its defense)

**The choice: scripted, deterministic `user_actions` — NOT a prompted user-sim
tool loop.** This is the one genuinely contested design call, so it gets the
full argument.

tau2's higher-fidelity approach is to give the user simulator its tools and let
the **LLM decide** when and how to call them. The honest case *for* that: it is
what real users do, and it tests the agent against an unscripted second actor.

The case *against* it for **this** bench is decisive:

1. **Reproducibility.** A prompted user-sim emitting tool calls is
   nondeterministic. The user's mutations would differ run-to-run, so the shared
   world — and therefore the agent's scoring conditions — would stop being
   reproducible. COT Bench's entire identity is determinism: the deterministic
   state grader (`eval/scoring/state_check.py`), tamper-evident pre-registration
   (the corpus hash), and the byte-identical scripted perturbations of recovery
   probes. A nondeterministic user action would be a foreign body in that design.
2. **Attribution (the fatal one).** The no-unauthorized-mutation contract and
   recovery/policy grading depend on knowing **who** mutated **what**. If the
   user-sim freely calls tools through the shared tool simulator, the grader can
   no longer cleanly separate an agent-side mutation from a user-side one —
   they flow through the same path with no reliable tag. Get attribution wrong
   and the policy-violation classifier (which keys on
   `state_check.UNAUTHORIZED_MUTATION_DETAIL`) starts charging the agent for the
   user's actions, or crediting it for them. Scripted actions let us tag every
   user mutation at application time.
3. **Cost and flakiness.** A prompted user tool loop adds simulator calls and a
   new flake surface (the user-sim mis-formatting a tool call) to every
   dual-control run, for a behavior we want to be a *controlled* variable.

**The trade we are accepting:** the user's behavior is fixed per scenario, not
free. We lose tau2's open-ended user. We gain a reproducible, attributable,
zero-extra-API-cost coordination test that fits COT Bench's deterministic spine.
For a *benchmark* — where the same controlled condition must be applied to every
model on every run — that is the right side of the trade. The same reasoning that
made recovery probes scripted (issue #57) makes user actions scripted here. If a
future version wants tau2-fidelity free user simulation, it is a separable,
larger piece of work; this staging does not preclude it (the trigger/action
abstraction is exactly where a prompted policy would slot in).

### Attribution semantics (the subtle part — get this right or grading breaks)

The shared world is **one** world, but every mutation is **tagged at application
time**:

- **Agent-side** deltas (applied by the tool simulator when the agent calls a
  tool) accumulate their top-level keys into `self._agent_mutated_keys`.
- **User-side** deltas (the scripted user actions) accumulate into
  `user_mutated_keys`.

The **coordination verdict** (`_coordination_verdict`) is `True` iff:

1. **Correct end state** — the scenario's normal `expected_state_changes` all
   pass (the agent reached the right end state *given* the user's concurrent
   actions), AND
2. **Attribution holds** — the agent did NOT itself mutate any **user-owned**
   top-level key. "User-owned" = any key in any declared `user_tool` scope (plus,
   defensively, any key a user action actually wrote). A user-owned key the agent
   also wrote is the canonical **double-apply** — "the agent re-applied the
   approval / update the customer already made."

Both halves use the **existing** state grader for the end-state check (no new
scoring machinery); the attribution half is a pure set intersection
(`agent_mutated_keys & user_owned`). A partial — end state right but the agent
trespassed, or no trespass but the task abandoned — is a **non-coordination**.

`coordination_ok` is `True`/`False` only on rows where **at least one user
action fired**; it is `None` when the scenario isn't dual-control, has no
ground_truth, or no action fired (the conversation ended before any trigger).
Why `None` and not `False`: grading a coordination that never happened would
score the row by base-state alone — counting a never-triggered run as a
non-coordination it never had the chance to attempt, or (worse) crediting one if
the base task happened to already match. `user_actions_fired` is recorded per row
so the declared-vs-fired gap is auditable, exactly like the recovery-probe
`probe_fired` flag.

**The delivery gate (the #74 fired-but-not-delivered lesson):** an action only
fires when at least one delivery turn remains. An `agent_called` action whose
watched tool is first called on the loop's **final** outer turn would stage a
user message that is never delivered — the agent would never see the
coordination signal and would get no turn to act on it — so firing there would
apply the user's delta and grade `coordination_ok` for a harness-timing reason,
not an agent failure. Such an action is treated as **not fired**: no delta, not
counted in `user_actions_fired`, no verdict (`coordination_ok` stays `None` when
nothing else fired, keeping the coordination-rate denominator honest). It is
counted in `user_actions_suppressed` on the row/artifact so the
trigger-met-but-undeliverable case is auditable. `after_turn` triggers cannot
normally reach the gate (`USER_ACTION_TURN_MAX` = 9 < the default `max_turns` =
10); `agent_called` can.

**Authoring constraint — non-empty `expected_state_changes` (validator-enforced):**
the empty-assertions form (`expected_state_changes: []`) means the
no-unauthorized-mutation contract — score 1.0 iff `final world == initial world`
— which a dual-control scenario violates **by construction**: the user's own
scripted mutations always change the world, so the agent would be charged with
an unauthorized mutation for the user's legitimate self-serve. The validator
rejects a `dual_control` block whose scenario does not declare non-empty
`expected_state_changes`.

**Checked write-scope contract (Option A) — replaces the old author-discipline
rule.** Agent-side mutated keys come from the LLM tool-sim's *generated*
`state_delta`, and the tool-sim can nondeterministically touch a shared key
(e.g. while simulating an unrelated tool it hallucinates a write into the
user-owned `contact` or `pending_requests` key) — manufacturing a false
double-apply against a perfectly-coordinating agent, run-to-run
nondeterministically. The original mitigation was an unenforced authoring
discipline ("keep user-owned keys in a top-level key no agent tool writes"). That
is now a **checked, deterministic contract**: every agent tool declares an
optional `writes` allow-list — the top-level state keys it is permitted to mutate
— and the runner **clamps** the tool-sim's `state_delta` to that list before
applying it (`SimulationRunner._simulate_tool_stateful`). An out-of-scope path is
dropped (with a logged warning) from **both** the world and the `agent_mutated_keys`
attribution set, so a hallucinated write can no longer fabricate a trespass. The
coordination verdict (`agent_mutated_keys & user_owned`) is unchanged — it is now
deterministic because its input is clamped.

A read-only tool declares `writes: []`. A tool that legitimately performs a
double-apply you want the metric to catch declares the user-owned key in its own
`writes` (e.g. an agent-side `update_contact_email` declares `writes: ["contact"]`):
its write is in scope, kept, recorded, and so a **genuine** double-apply is still
flagged. `writes` is **optional** in general (a tool without it is not clamped,
so the single-control public corpus runs and hashes identically), but in a
`dual_control` scenario the validator **requires** every agent tool to declare it,
so the clamp is always in force where the coordination metric depends on it.

### The authorization boundary

`user_tools` declare a `scope` of top-level state keys. The validator and the
`DualControl` constructor both enforce that **every user action writes only
within its tool's declared scope** — a user action that reaches into agent-only /
server-only state (e.g. a "update my contact info" action trying to write
`accounts.*.balance`) is a **validation failure**, caught before any run. This
keeps "user tools can only touch user-legible state" a checked contract, not a
convention. An empty scope means read-only (the user can call the tool but must
produce no delta — e.g. reading their own notifications).

## Hash handling (the #54 lesson)

`dual_control` **is** hashed scenario content when present (it changes what a run
does), added to the canonical dict **conditionally** — exactly like
`rubric_criteria` (#54) and `recovery_probe` (#57). A single-control scenario's
canonical dict contains **no** `dual_control` key, so its digest is
**byte-identical** to before this field existed (verified by test:
`test_single_control_digest_unchanged`, `test_single_control_digest_matches_legacy_object`,
and a corpus-hash-unchanged assertion in CI). Adding/changing a dual_control
block (including changing a `user_message` or a `state_delta`) moves the scenario
digest and the corpus hash, so the second actor is tamper-evident and
pre-registered like any other content. The whole public corpus (92 scenarios)
hashes to the same value as on `master`.

## Aggregation — additive, conditional, never moves public efficacy

Dual-control rows are a **new additive tier**, exactly like recovery probes
(#57) and the persona-stratified robustness table (#59):

- `compute_dual_control_rates` computes a per-model `coordination_rate` over
  **fired-action rows only** (`coordination_ok` non-null), with row and scenario
  counts.
- `compute_leaderboard` reads it from the full frame **before** the cooperative
  exclusion (dual control is orthogonal to the sim profile — it can ride any
  profile), with holdout and null-agent rows already dropped.
- It is emitted into `leaderboard.json` under `dual_control_robustness`
  **only when dual-control rows exist** — the same conditional-emission pattern
  as `recovery_probe_robustness`, `sim_profile_robustness`, and the holdout
  `present` block. A normal run on the v1 corpus (no dual_control) ships **no
  empty surface**; the public efficacy / CLEAR rankings are **untouched** (pinned
  by `test_absent_on_normal_run` and `test_dual_control_rows_additive_not_replacing`).

## Exclusions / corpus interaction (the recommended, simpler v1)

**v1 corpus has NO dual_control blocks.** The two demonstration fixtures live in
`tests/fixtures/dual_control/` only. Consequences:

- Existing public aggregates are **untouched** — no dual-control rows exist to
  exclude, so `coordination_rate` is purely **additive**.
- If adopted, dual-control scenarios are **NEW scenarios (a tier)** — authored
  copies of (or new) real scenarios with a `dual_control` block and a distinct
  id — added under `data/scenarios/` as their own set, not edits to existing
  files. Because the corpus hash binds id→content, adding a tier changes the
  corpus set (a deliberate, pre-registered change at adoption), while every
  existing scenario keeps its exact digest.

## Adoption cost

- Authoring a dual-control **tier**: write N scenarios with a `dual_control`
  block — declare the user tools + scope, script the user actions (trigger +
  delta + message), and the `expected_state_changes` that encode correct
  coordination. This is **plan-token subagent** work (no API cost) following the
  corpus-authoring pattern, plus Conor's review. Each scenario needs a human eye
  on "is this a realistic self-serve, and does the verdict actually distinguish
  coordinate-correctly from double-apply?" — semantic coherence is a review
  property, like criteria atomicity (#54) and probe realism (#57).
- Mechanism cost is already paid (this PR): runner firing + injection,
  attribution tracking, the coordination verdict, validator (with the
  authorization-scope check), hash coverage, aggregation, and 50 tests.

## v1.1 framing vs adopt-now

- **Adopt now (author a small dual-control tier before the first published
  run):** the coordination dimension ships in v1; no version-bump cost later.
  The tier is additive (new scenarios), so it does not disturb the
  single-control corpus or its digests. Cost: the authoring + review pass now.
- **Defer to v1.1 (recommended in the issue — post-launch, effort L):** v1 ships
  on the known-good single-control path; the mechanism sits inert and tested on
  master, ready to light up when a dual-control tier is authored. Because
  dual-control scenarios are a separate tier (new ids, additive), adopting them
  in v1.1 is **not** a forced version bump of the existing corpus — it adds a
  tier and a column. This is the cheap, low-risk path and matches the issue's
  "post-launch (v1.1)" framing.

The mechanism being inert-until-declared is what makes deferral free: merging
this PR (if Conor chooses) adds zero dual-control rows and zero published
surface; it only makes the capability available.

## Interaction with #57 (recovery probes) and #59 (profiles)

- **#57 recovery probes — sibling pattern, orthogonal.** Dual control borrows the
  scripted-deterministic-injection conventions of recovery probes (the
  `injected_message`-style override, the `fired`/`None` grading gate, the
  conditional-hash and conditional-emission patterns) but is independent code. A
  scenario could in principle carry both; v1 keeps them separate tiers. This PR
  does NOT depend on PR #74's unmerged probe code — it re-implements the shared
  conventions locally.
- **#59 profiles — orthogonal.** A user action fires regardless of the active sim
  profile (the firing is deterministic on trigger; the profile only shapes the
  *generated* user turns). A dual-control scenario can ride any profile; the
  coordination rate is reported per model across whatever profiles ran. No
  coupling either way.

## Open questions for the decision

1. **Scripted vs prompted user actions** — accept the scripted, deterministic
   design (this PR) as the right trade for a reproducible, attributable bench, or
   is tau2-fidelity free user simulation worth the determinism/attribution cost?
   (The decision doc argues scripted; this is the headline call.)
2. **Trigger vocabulary** — are `after_turn` + `agent_called` enough, or do we
   want a `state_condition` trigger (fire when the world reaches some state)?
   `state_condition` is more expressive but couples firing to mid-run state
   evaluation; deferred unless a scenario needs it.
3. **Tier size and mix** — how many dual-control scenarios, and what
   domain/shape mix? Suggest a small balanced set: ~2-3 approve-mid-flow and
   ~2-3 act-first-no-double-apply per domain, so the rate is not single-scenario
   noise (mirrors the recovery-probe sizing question).
4. **One action per scenario, or a sequence?** This PR supports multiple
   `user_actions` per scenario (each fires at most once), but the demo fixtures
   use one each. A multi-action conversation is a richer coordination test but
   harder to attribute and grade; recommend starting with one action per
   scenario and expanding only if needed.
5. **Attribution granularity** — the contract is at the **top-level key**. A
   finer-grained (dotted-path) ownership model would let the agent and user share
   a top-level key writing different sub-paths. Top-level is simpler and matches
   how scopes are declared; revisit only if a scenario genuinely needs shared
   ownership of one top-level object. The practical consequence for tier authors
   is now handled by the checked write-scope clamp (Option A) in the attribution
   section above: each agent tool declares `writes`, the tool-sim's delta is
   clamped to it, and a hallucinated shared-key write can no longer fabricate a
   trespass — so this is a deterministic contract, not an author-discipline rule.
6. **Independent review bar** — mirror the #54 / #57 bar: each dual-control
   scenario gets an independent adversarial review that the coordination is
   realistic and the verdict actually distinguishes coordinate-correctly from
   double-apply, before it joins the corpus.
