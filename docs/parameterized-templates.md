# Parameterized Scenario Templates — Decision Doc (issue #60)

**Status: STAGED, mechanism-only + one demonstration template. Not adopted.**
This branch carries a complete, backwards-compatible templating mechanism
(`eval/templating.py`), its wiring into the loader / pre-registration / validator,
its test suite, and TWO converted demonstration templates (under
`data/scenarios_templates/`, one banking + one customer_success) that live
OUTSIDE the evaluated corpus. **No existing scenario file is modified**, and the
current 92-scenario corpus runs exactly as today. This document exists so the
decision reduces to **adopt the mechanism for v1 (and convert the corpus)** or
**defer to v1.1 / v2**.

## What problem this solves

The public corpus is static and on GitHub. As the leaderboard gains visibility,
a model could be trained on the published scenarios (or on published transcripts,
which echo the scenario surface verbatim) and recognize the exact account IDs,
names, amounts, and dates — short-circuiting the *task* by recall instead of
*tool-use skill*. The private holdout (issue #31) is the committed defense
against this, but it covers only ~10 scenarios; the 92 public ones remain
memorizable forever once published.

Parameterization is **cheap insurance now, expensive retrofit later**: a template
holds the logical task fixed while the *surface* (which name, which account
number, which dollar amount, which date) is regenerated fresh on every run from a
per-run seed. Memorizing the published surface then buys nothing — the model that
saw `PERS-CHK-4417 / $600 / 2026-06-14` in training meets `PERS-CHK-0019 / $550 /
2026-06-22` at eval time, with the *same* verify-before-access, one-time-vs-
recurring, fraud-report, and statement-generation skills under test. This is
idea 1 from the game-benchmarks research (clembench, EcoGym logic-mutation):
randomize surface features, keep the skill test identical.

## The design

A scenario opts in by adding a `template_slots` block and replacing surface
values with `{{slot}}` placeholders. Everything else about the scenario is
unchanged.

### Slot syntax

```jsonc
"template_slots": {
  "first_name":      {"type": "first_name"},
  "chk_num":         {"type": "digits", "length": 4},
  "transfer_amount": {"type": "amount", "min": 200, "max": 900, "step": 50},
  "recurring_start": {"type": "date", "start": "2026-06-14", "end": "2026-07-12"},
  "fraud_merchant":  {"type": "choice",
                      "options": ["PRESTIGE MEMBERSHIPS", "SUMMIT REWARDS CLUB"]}
}
```

`{{slot}}` (double braces — single braces collide with the `str.format` rubric
templates) can appear anywhere a surface value appears: `user_goals`, `persona`,
`initial_message`, `ground_truth` (values **and** dict keys), the dotted
`expected_state_changes` assert paths, `match` values, and **`rubric_criteria`
text**. Slot types: `name`, `first_name`, `account_id`, `customer_id`, `amount`,
`integer`, `date`, `company`, `digits`, `choice` (and a literal `{"value": …}` to
pin a field). Account/customer IDs are built compositionally — `PERS-CHK-{{chk_num}}`
— so one digit slot drives the key, the goal text, the assert path, and the
criterion together.

### Instantiation

`instantiate(template, seed)` resolves each slot **once** (a per-`(seed,
scenario_id, slot)` sha256-seeded RNG, so resolution is order-independent and
stable across processes/platforms), substitutes the resolved values coherently
through every field, strips the `template_slots` declaration, and returns an
ordinary v0.2 scenario. A non-template scenario passes through unchanged.

```
template + seed  --instantiate-->  concrete v0.2 scenario (validator-clean)
```

### Criteria-rewrite (the load-bearing coherence property)

Because criteria reference specific entities (an account ID, the fraud txn ID, a
dollar amount), parameterization is only correct if a slot rewrites the criterion
text in lock-step with the goal it grades and the assertion that checks it. It
does: a single resolved value is substituted into the goal, the
`expected_state_changes` assert path / match, **and** the criterion text, so
"agent moved `${{transfer_amount}}` from savings `{{sav_num}}` …" becomes the same
concrete numbers the goal asked for and the state check verifies. A template
whose criteria reference an undeclared slot (or that declares a slot it never
uses) fails validation.

### Pre-registration integration (honesty)

A templated run pins **three** values in `pre_registration.json` under a
`templating` block:

- `template_corpus.sha256` — hash of the RAW authored templates (seed-invariant);
  pins *which templates* the run used.
- `instantiation_seed` — the per-run seed.
- `instantiated_corpus_sha256` — equals `scenario_set.sha256` (the scenarios are
  already instantiated before hashing); pins the *exact surface* the run used.

The surface is therefore tamper-evident **and recomputable**: anyone can
re-instantiate `(templates, seed)` and confirm they get the recorded instantiated
hash byte-for-byte. `--resume` reuses the original seed (read from the
pre-registration, not the CLI), so a resumed run reproduces the same surface and
the existing corpus-unchanged guard (governance §3) passes only when both the
templates and the seed are unchanged. The block is omitted entirely when no
scenario is templated — a non-templated run pre-registers exactly as today.

### Guaranteed invariants (all tested, `tests/test_templating.py`)

1. **Deterministic** — same template + seed → byte-identical scenario.
2. **Coherent** — every slot reference rewrites consistently across goals,
   persona, ground-truth keys/values, assert paths, match values, and criteria.
3. **Validator-clean** — instantiated output passes `validate_scenario_dict`;
   state grading and judges are unchanged.
4. **Pre-registration honest** — template hash + seed + instantiated hash recorded
   and recomputable.
5. **Backwards compatible** — the 92 non-templated scenarios are byte-for-content
   unchanged; templating is opt-in per scenario.

## Conversion decision: mechanism-plus-demonstration, not corpus conversion

This PR **deliberately does not convert the real corpus.** Converting all 92
scenarios is invasive (every surface value in every field, including the
criteria authored under #54, must become a coherent slot), error-prone to review
in one pass, and — most importantly — a **corpus change** (see below) that is its
own decision. So this branch ships the mechanism plus two converted *copies*
of real scenarios — one banking, one customer_success — clearly marked (`_tmpl`
id suffix, in `data/scenarios_templates/` outside the loader's path), as
concrete, reviewable, test-backed demonstrations across two domains and two
coherence shapes (account id as a dict KEY vs as a VALUE). Full conversion is a
mechanical follow-up *if* the mechanism is adopted.

## Version-bump implication (the actual decision)

Per governance §5, **adopting templating changes scenario content** — a converted
scenario's canonical bytes differ from its static form, and the instantiated
surface differs per run. Adopted after a published run, it forces a benchmark
version bump and breaks score comparability across the boundary.

- **Adopt for v1 (before the first published run):** zero comparability cost —
  there are no published numbers to invalidate. This is the only free window. The
  mechanism is done; conversion of the ~102-scenario corpus (92 public + ~10
  holdout) is plan-token subagent work following the #54 authoring pattern, plus a
  review pass to confirm coherence on each. The first published run then commits a
  seed and is reproducible from it.
- **Defer to v1.1 / v2:** v0.1 ships sooner on the static corpus, and the first
  published run is a clean static baseline. Cost: the published 92 scenarios are
  memorizable for the life of v1, the holdout is the only contamination tripwire
  until v2, and conversion later becomes a mandatory version bump.

A middle path exists and is cheap: **adopt the mechanism now** (it is inert until
a scenario declares `template_slots`), ship v1 on the static corpus, and convert
incrementally — each converted scenario is a corpus change tracked into the next
version, but the harness never needs to change again.

## What is staged on this branch

- `eval/templating.py` — slot types, deterministic seeded instantiation, coherent
  substitution (including dict keys / dotted paths), `instantiate` / `is_template`.
- Loader wiring (`scripts/run_eval.py`): `load_scenarios` / `load_holdout_scenarios`
  instantiate against an `--instantiation-seed` (default 0, deterministic) and
  return the raw templates for hashing; resume reuses the original seed.
- Pre-registration (`eval/pre_registration.py`): `template_corpus_hash` +
  `templating` block (template hash, seed, instantiated hash); run-manifest mirror.
- Validator (`scripts/validate_scenarios.py`): template-declaration + slot/
  placeholder coherence checks, then full v0.2/criteria validation of the
  *instantiated* scenario, so a template is held to the concrete bar.
- Two demonstration templates (`data/scenarios_templates/`, outside the corpus,
  one banking + one customer_success) and the `tests/test_templating.py` suite;
  the no-template path is asserted to pass through unchanged.

**Not done (deliberately):** no existing scenario converted; no corpus or holdout
conversion; no methodology.md update (that lands with adoption, not staging).

## Open questions for the adopt/defer decision

1. **Adopt-now-convert-later, or convert-with-adoption?** The mechanism is inert
   until a scenario declares `template_slots`, so it can land in v1 with zero
   behavior change and the corpus converted scenario-by-scenario afterward. Each
   conversion is still a corpus change (version-tracked), but the harness never
   changes again. Is the incremental path preferred over a single big-bang
   conversion?
2. **Default seed policy for published runs.** The default seed is `0`
   (deterministic, for CI). A published leaderboard run SHOULD pass a fresh seed
   so the surface is unmemorizable, but then re-running the exact leaderboard
   requires reading the seed back from the pre-registration. Do we want a
   convention (e.g. seed = run date) or a `--random-instantiation-seed` flag that
   draws and records one?
3. **Pool size / collision tolerance.** The name/company pools are small (~12-24
   entries). The point is variation, not secrecy (the seed + instantiated hash are
   pre-registered), but if we want a low probability that two published runs draw
   the same surface, the pools should grow. How large is large enough?
4. **Should the holdout be templated too?** The loader already instantiates holdout
   templates and the design supports it, but a private holdout is already
   unmemorizable by virtue of being private. Templating it adds per-run surface
   variation on top — worth the authoring cost, or redundant?
5. **Conversion review bar.** A mis-converted template (a slot that rewrites the
   goal but not the criterion, say) is a silent grading bug. The validator catches
   undeclared/dead slots and re-validates the instantiated scenario, but semantic
   coherence ("this criterion still grades the right entity") is a human-review
   property. Do we require independent review per converted scenario, mirroring the
   #54 criteria-authoring bar?
