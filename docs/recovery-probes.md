# Recovery-Probe Tier — Decision Doc (issue #57)

**Status: STAGED for an adopt/defer decision (this PR). DO NOT MERGE without
Conor's call.** The complete mechanism, three demonstration probes, and the
scoring/hash wiring are in this PR so the decision is *adopt-or-defer*, not a
work item. Nothing in the published v1 corpus changes: the demo probes live in
`tests/fixtures/recovery_probes/`, never in `data/scenarios/`, so the mechanism
is inert until a scenario declares a `recovery_probe`.

## What it solves

From the game-benchmarks research (idea 4; source: **Vending-Bench**): long-
horizon *coherence* failures are **goal-maintenance failures triggered by
unrecovered errors**, with near-zero correlation to context saturation. Agents
don't drift because their context filled up — they drift because something went
wrong mid-task and they never got back on track. COT Bench's 10-turn cap is too
short to surface that with long contexts, but it is exactly long enough to
surface it with an **injected** fault: drop a deliberate perturbation at turn
4-5 and check — deterministically, by state grading — whether the agent has
recovered the correct end state by the time the conversation closes.

This gives COT Bench a **long-horizon-robustness dimension without long
contexts**, measured the way the rest of the bench measures: deterministic state
checking, not judge opinion.

## Design

A scenario MAY carry an optional `recovery_probe` block:

```jsonc
"recovery_probe": {
  "turn": 4,                       // 4 or 5 — mid-conversation, with room to recover
  "kind": "wrong_entity",          // one of a small enum (below)
  "injection": "Actually, send that $2,500 to BUS-CHK-999 instead.",
  "recovery_assertions": [         // OPTIONAL, same grammar as expected_state_changes
    {"assert": "accounts.BUS-CHK-999", "op": "not_exists"}
  ]
}
```

**Probe kinds** (a deliberately small enum, mirroring the four sim profiles —
`eval/simulation/probes.py` is the single source):

- `contradictory_reference` — the user cites a policy/fact contradicting what was
  established earlier ("your colleague said the limit was $10k"). A coherent
  agent holds the verified truth; a drifting one adopts the contradiction.
- `wrong_entity` — the user supplies a wrong account/id/record in a follow-up. A
  coherent agent catches the mismatch; a drifting one acts on the wrong entity
  (the canonical unauthorized-mutation failure).
- `incomplete_action_claim` — the user asserts a prior step was already done when
  it was not ("you already filed that, right?"). A coherent agent
  verifies/completes; a drifting one accepts the false premise and skips it.

**Scripted, not generated.** At the declared user turn the runner REPLACES the
user simulator's generated message with `injection` *verbatim*. The perturbation
is byte-identical for every model on every run — that determinism is the whole
point of a controlled fault. The user sim still drives every other turn; only
the probe turn is overridden.

### Injection point in the sim loop

In `SimulationRunner.run`, at the bottom of each outer turn iteration, before the
next user turn is generated:

```
... agent <-> tool rounds for turn_num ...
if probe and probe.turn == turn_num + 1:
    current_user_message = probe.injected_message()   # scripted, deterministic
    continue                                          # skip the user-sim call this turn
else:
    current_user_message = self._simulate_user_turn(...)   # normal generated turn
```

The probe at turn N is set as the message for turn N at the end of iteration
N-1, so the user sim is simply not called that one turn. Everything else — the
agent loop, tool simulation, token accounting, completion detection — is
unchanged.

### Recovery scoring (deterministic-first)

Recovery is graded by the **existing** state grader (`eval/scoring/state_check.py`)
— no new scoring machinery. `_recovery_verdict` AND-s two checks:

1. **Base task still done** — the scenario's normal `expected_state_changes` all
   pass (the agent reached the correct end state DESPITE the fault).
2. **Bad entity not acted on** — the probe's `recovery_assertions` all pass
   (typically "the wrong account never came into existence / received a
   transfer").

A partial — task done but acted on the wrong entity, or wrong entity avoided but
task abandoned — counts as a **non-recovery**. `recovered` is `True`/`False` on
probe rows, `None` everywhere else.

**One small grammar addition: the `not_exists` op.** The wrong-entity recovery
check needs to assert *absence* — "`accounts.BUS-CHK-999` must never exist". The
existing `equals null` can't express this (it requires the path to resolve), and
the empty-assertion-list unauthorized-mutation check is whole-world (it would be
tripped by the legitimate transfer). `not_exists` is the minimal, deterministic
op that makes the canonical wrong-entity check expressible — it is the one op for
which an absent path is the *success* case. It's a general assertion, usable in
`expected_state_changes` too, not probe-only machinery.

## Scoring semantics & where probe results are published

Probe rows carry two extra columns: `recovery_probe_kind` and `recovered`.
Aggregation computes a **per-model `recovery_rate` over probe-carrying rows
only**, with a `by_kind` breakdown, and emits it into `leaderboard.json` under
`recovery_probe_robustness` **only when probe rows exist** — the same
conditional-emission pattern as `sim_profile_robustness` (#59 / PR #70) and the
holdout `present` block. A normal run on the v1 corpus (no probes) ships no
empty surface; the public efficacy/CLEAR rankings are untouched.

**Flag, not category — but the rows are a separate dimension.** The issue asks
"distinct scenario category or a flag on existing ones". Staging implements the
**flag** mechanics (a block on a scenario) and reports probe results as a
**separate published dimension** (`recovery_probe_robustness`), not folded into
efficacy. This is the best of both: authoring is a flag on a copy of an existing
scenario (cheap), and the result is a clean, separable robustness number.

## Exclusions / corpus interaction (the recommended, simpler v1)

**v1 corpus has NO probes.** The three demonstration probes live in
`tests/fixtures/recovery_probes/` only. Consequences:

- Existing public aggregates are **untouched** — no probe rows exist to exclude,
  so `recovery_rate` is purely **additive**.
- If adopted, probe scenarios are **NEW scenarios (a tier)** — copies of real
  scenarios with a probe block and a distinct id — added under `data/scenarios/`
  as their own set, not edits to existing files. Because the corpus hash binds
  id→content, adding a tier changes the corpus set (a deliberate, pre-registered
  change at adoption), while every existing scenario keeps its exact digest.

This sidesteps the harder "do probe rows join or split the public aggregates?"
question by construction: a probe scenario is a different scenario, graded on a
different (recovery) dimension, reported in its own column.

## Hash handling (the #54 lesson)

`recovery_probe` **is** hashed scenario content when present (it changes what a
run does), added to the canonical dict **conditionally** — exactly like
`rubric_criteria` was. A probe-less scenario's canonical dict contains **no**
`recovery_probe` key, so its digest is **byte-identical** to before this field
existed (verified by test: `test_probeless_digest_matches_legacy_object` and a
corpus-hash-unchanged assertion). Adding/changing a probe (including changing its
injection text) moves the scenario digest and the corpus hash, so the
perturbation is tamper-evident and pre-registered like any other content.

## Interaction with #59 (profiles) and #60 (templates)

- **#59 profiles — orthogonal.** The probe text is injected regardless of the
  active sim profile (the override is unconditional on the probe turn; the
  profile only shapes the *generated* turns). A probe can ride any profile; the
  recovery rate is reported per model across whatever profiles ran (probe rows
  are not excluded from the recovery table by profile). No coupling either way.
- **#60 templates — integration requirement, documented not coded.** A probe
  whose `injection` references an entity (e.g. an account id) MUST be rewritten
  under template instantiation, the same way goals/ground-truth/assert-paths are
  — otherwise the scripted text would name a stale id. `RecoveryProbe.injected_message()`
  is a method (not a bare attribute read) precisely so a future templating pass
  can rewrite entity references at that single call site. #60's `find_placeholders`
  machinery is **not on master** (PR #69 is unmerged), so this PR does not depend
  on it; the integration is a one-line hook when #60 lands. Recorded here as the
  requirement.

## Adoption cost

- Authoring a probe **tier**: copy N real scenarios, add a probe block + distinct
  id, write the `recovery_assertions`. This is **plan-token subagent** work (no
  API cost) following the corpus-authoring pattern, plus Conor's review — the
  same shape as the atomic-rubrics (#54) authoring. Each probe needs a human eye
  on "is this injection a realistic fault, and does the recovery assertion
  actually encode 'did not act on the bad entity'?" — semantic coherence is a
  review property, like criteria atomicity.
- Mechanism cost is already paid (this PR): runner injection, the `not_exists`
  op, validator, hash coverage, aggregation, tests.

## v1.1 framing vs adopt-now

- **Adopt now (author a small probe tier before the first published run):** the
  recovery dimension ships in v1; no version-bump cost later. The probe tier is
  additive (new scenarios), so it does not disturb the non-probe corpus or its
  digests. Cost: the authoring + review pass now.
- **Defer to v1.1 (recommended in the issue — post-launch, effort M):** v1 ships
  on the known-good path; the mechanism sits inert and tested on master, ready to
  light up when a probe tier is authored. Because probe scenarios are a separate
  tier (new ids, additive), adopting them in v1.1 is **not** a forced version
  bump of the existing corpus — it adds a tier and a column. This is the cheap,
  low-risk path and matches the issue's "post-launch (v1.1)" framing.

The mechanism being inert-until-declared is what makes deferral free: merging
this PR (if Conor chooses) adds zero probe rows and zero published surface; it
only makes the capability available.

## Open questions for the decision

1. **`not_exists` op** — accept the one-op grammar addition as the way to encode
   "bad entity not created"? (The alternative — relying solely on base
   `expected_state_changes` to catch a misdirected transfer — works for
   destination-swap probes but cannot express "this record must not exist" for
   probes that aren't a simple transfer redirect.)
2. **Probe tier size** — how many probe scenarios, and what domain/kind mix?
   (Suggest a small balanced set: ~2-3 per kind per domain so `by_kind` rates are
   not single-scenario noise.)
3. **One probe per scenario, or allow a sequence?** v1 is exactly one probe per
   scenario (one fault, one recovery window). Multiple faults per conversation is
   a richer test but harder to attribute and grade; defer.
4. **Recovery window** — is "recovered by end of conversation" the right bar, or
   should recovery be required by a specific later turn (the issue's "turn 8-9")?
   State grading is only done at end today; a by-turn-N check would need
   mid-conversation grading. End-of-conversation is the cheaper, current-machinery
   choice; document the looser bar.
5. **Profile crossing** — should the published recovery table stratify by sim
   profile (recovery under cooperative vs adversarial), or pool? v1 pools (one
   recovery_rate per model). Stratifying is a `by_kind`-style extension if wanted.
6. **Independent review bar** — mirror the #54 criteria-authoring bar: each probe
   gets an independent adversarial review that the injection is a realistic fault
   and the recovery assertion is correct, before it joins the corpus.
