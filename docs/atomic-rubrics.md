# Atomic Rubric Criteria — Decision Doc (issue #54)

**Status: ADOPTED — merged to master in PR #67, before the first published
run.** All 92 public scenarios carry authored criteria (4–6 per scenario,
stamped `criteria_authorship`: `anthropic/claude-opus-4.8`, run
`2026-06-11-atomic-rubrics-batch`), and the 10 private holdout scenarios carry
criteria with the same schema and author stamp (under their own run id,
`2026-06-11-atomic-rubrics-holdout-batch`) in the private repo. The sections
below are preserved as the decision record; the adopted behavior is documented
in [methodology.md](methodology.md) §3 and
[scenario-schema.md](scenario-schema.md).

## What changes

Scenarios may carry a `rubric_criteria` array: 3–6 atomic, checkable,
instance-specific criteria (the validator floor is 3; the authored corpus
uses 4–6), each mapped to one of the two judge-scored dimensions
(`task_completion` / `tool_selection`) with a positive weight.
Example:

```jsonc
"rubric_criteria": [
  {"id": "verify_before_balance",
   "text": "Agent verified the customer's identity before disclosing any balance.",
   "dimension": "task_completion",
   "weight": 2},
  ...
],
"criteria_authorship": {
  "criteria_author_model": "anthropic/claude-opus-4.8",
  "criteria_author_run": "2026-06-11-atomic-rubrics-batch"
}
```

When present, the combined judge prompt gains a per-criterion section (the
criteria-less prompt is **byte-identical** to today's template — asserted by
test) and the judge returns a per-criterion met/unmet verdict with brief
evidence alongside the existing holistic dimension scores. Scoring then uses:

> **dimension score = weighted fraction of met criteria mapped to that
> dimension.** The judge's holistic template score is still recorded
> (`holistic_score` in artifacts) for comparison and audit, but it no longer
> feeds consensus for a criteria-bearing dimension. A dimension with no mapped
> criteria keeps the holistic score.

Per-criterion verdicts land in the per-run artifacts, so the criterion-based vs
holistic comparison (the halo-effect measurement) is analyzable from any run.
Deterministic state grading, pass/fail, Cost, Latency, and Reliability are
untouched — criteria inform the judge dimensions only.

## Why

Generic rubric templates let judge verdicts be dominated by halo effects: a
confident, fluent, verbose answer reads as a good answer, and the judge's
holistic dimension score absorbs that impression. HealthBench (OpenAI, 2025)
showed that decomposing grading into instance-specific, independently-checked
criteria substantially tightens judge–expert agreement; Autorubric-style
per-criterion grading shows the same mechanism: a judge asked "did the agent
verify identity before disclosing the balance — cite the turn" has far less
room for tone to leak into the verdict than one asked "score task completion
0–1". This complements the length-bias regression (#30, merged): #30 measures
the verbosity confound; atomic criteria remove much of its attack surface.

## Version-bump implication (the actual decision)

Per governance §5, this is a **rubric change** (scoring semantics) *and*, once
criteria are authored, a **corpus change** (scenario content — the corpus
sha256 covers `rubric_criteria` when present). Adopted after a published run,
it forces a benchmark version bump and breaks score comparability.

- **Adopt now (before the first published run):** zero comparability cost —
  there are no published numbers to invalidate. Criteria authoring for the
  ~102-scenario corpus is plan-token subagent work (no API cost) following the
  corpus authoring pattern, plus your review pass. This is the only window
  where adoption is free.
- **Defer to v2:** v0.1 ships sooner on the known-good template path, and the
  first published run doubles as a clean holistic baseline to compare the
  criterion-based scores against. Cost: v1 numbers carry the halo-effect
  weakness the best-in-class review flagged, and v2 becomes a mandatory bump.

A middle path is cheap because the artifacts record **both** scores: adopt the
harness (this branch), author criteria, and run one A/B rehearsal — the
holistic-vs-criterion delta per judge is then measurable before committing the
published scoring semantics.

## What merged (PR #67)

- Schema + validator: `rubric_criteria` (3–6 items, unique ids, valid judge
  dimension, weight in (0, 10]), `criteria_authorship` provenance required,
  contestant criteria-authors blocked (family-aware, same rule as scenario
  authors).
- Judge integration on BOTH paths (combined default and
  `--separate-judge-calls`), with strict verdict parsing (missing/partial
  verdict block = parse failure, one retry, then excluded from consensus).
- Criterion-informed scoring with holistic scores + per-criterion verdicts
  persisted to artifacts (and reconstructed on `--resume`).
- Corpus-hash coverage: criteria are hashed when present; criteria-less
  scenarios hash identically to today (no spurious corpus-hash change).
- `stamp_criteria()` in `scripts/generate_data.py` for authoring agents.
- 51 new tests; the no-criteria path is asserted byte-identical.

**Still open after adoption:** no A/B rehearsal run yet — the
holistic-vs-criterion delta is measurable from any run's artifacts, since both
scores are recorded on every judged row.
