# Governance

This document is a public commitment. It states the rules COT Bench holds itself
to so that a published score means what it appears to mean. For a small,
single-maintainer benchmark the two credibility risks that matter most are silent
judge-version drift and run cherry-picking: re-running an evaluation until the
numbers look good, or quietly swapping a judge model out from under a published
ranking. The policies below exist to make both visible and to make corrections
auditable rather than invisible.

Each policy is written to be exactly true. Where a mechanism is enforced in code,
this document points to the file that enforces it. Where a mechanism is intended
but not yet built, it is marked **planned** and linked to the tracking issue.
Nothing here claims a property the code does not have.

## 1. No silent retraction or rerun

A published run is never deleted and never quietly re-run. Once results are
published, they stay published, including runs that turned out to be unflattering
or that revealed a problem with the benchmark itself.

Corrections are append-only. A problem with a published run is fixed by computing
a **new, dated run** and recording the reason in a changelog entry, not by
overwriting or removing the original. The old numbers remain visible alongside the
correction so anyone can see what changed and why. This is the benchmark's
defense against the "re-run until it looks right" failure mode: there is no clean
slate to re-run toward, because nothing is ever silently replaced.

Every run is timestamped and self-describing. `scripts/run_eval.py` writes a
`run_manifest.json` next to each run's results, and the methodology document's
"What every run publishes (for audit)" section enumerates the per-evaluation
artifacts (full transcripts and raw judge outputs) that let any published number
be traced back to its evidence.

## 2. Judge pinning

Judge model versions are pinned per run, and the model a provider *actually
served* is recorded — not just the model that was *requested*. This matters
because hosted model IDs (the OpenRouter slugs used for the open-weight judges, in
particular) can be silently re-pointed to a different snapshot or quantization
without the requesting code changing at all. A pinned request ID alone would not
catch that drift.

The recorded field is `resolved_model`. For every judge call it is serialized into
the per-evaluation artifact by `eval/artifacts.py` (`_serialize_judges`, which
emits a `resolved_model` value per judge), and it is also recorded for the agent
under test (`build_artifact`'s `sim_meta.resolved_model`). The same value appears
in the flat results row (`scripts/run_eval.py`, `build_result_row`, the
`resolved_model` column). So if the provider serving a judge changes between runs,
the change is a recorded value on disk, not a hidden variable — and a judge
version bump is therefore a detectable, documentable event rather than silent
drift.

A judge version change is treated as a methodology change. See §5: scores produced
under a different judge belong to a different benchmark version and are not
comparable across that boundary.

## 3. Pre-registration of runs

The goal of pre-registration is to remove the maintainer's freedom to choose,
after seeing results, which run "counts." The honest version of this commits the
run's definition — the models under test, the exact scenario set, the judge panel,
and any seeds — to the record **before** results are computed.

**Current mechanism (partial).** Today `scripts/run_eval.py` writes
`run_manifest.json` *after* the run completes (it records `run_id`, the models
requested / completed / failed, the domains and per-domain scenario counts, the
reliability-run count, and the artifact and trace directories). That manifest is
genuinely useful for auditing what a run contained, and the publish gate
(`scripts/check_publish_ready.py`) reads its `models_failed` field to block a
leaderboard commit that would silently ship missing models. But because it is
written post-hoc, it is an *after-the-fact record*, not a pre-registration: it
does not, on its own, prove the run's definition was fixed before the numbers were
known. It also does not yet capture a scenario-set hash, the judge panel, or run
seeds — the fields a true pre-registration needs.

What pinning does exist today is partial and worth stating precisely. Each
scenario's ID embeds an 8-character content hash of the scenario JSON
(`scripts/generate_data.py`), so individual scenario contents are tamper-evident,
but there is no committed corpus-level hash over the whole scenario set. The
aggregation bootstrap uses a fixed seed (`BOOTSTRAP_SEED = 42` in
`scripts/aggregate_results.py`) for reproducible confidence intervals; the agent
under test runs at temperature 0.0, while the user and tool simulators run at
temperature > 0 and are not seeded, so runs are not bit-for-bit reproducible (the
methodology document says as much).

**Planned.** Writing the manifest — extended with a corpus/scenario-set hash, the
pinned judge panel, and seeds — *before* results are computed, and committing it as
the pre-registration of the run, is planned. Tracking: the pre-first-run
statistics work in issue #25. Until that lands, this section describes the real
post-hoc manifest, not an aspirational pre-registration, and the gap is stated
openly here.

## 4. Contamination policy

**Scenario authors are never contestants.** A model that authored an exam must
never sit for it. This is enforced in code, not by convention:
`scripts/generate_data.py` exposes `assert_author_allowed`, a hard guard that
refuses to generate scenarios when the author model matches any entry in
`MODELS_UNDER_TEST`. The match is **family-aware**, not exact-string: contestants
are pinned to dated snapshots (for example `gpt-4.1-2025-04-14`), so the guard
blocks any author ID where either ID is a prefix of the other — a different
snapshot of a contestant is still that contestant. An author that is also a judge
is permitted (author and judge are different roles); only author-equals-contestant
is treated as contamination.

**Authorship records are never altered.** Every scenario carries an `authorship`
block (`author_model`, plus batch information) stamped at generation time by the
pipeline. These records are part of the published corpus and are not edited after
the fact; if authorship was wrong, the fix is a new scenario (see §1 and §5), not
a rewrite of the record.

**Synthetic scenarios, with a public-exposure caveat.** Scenarios are synthetic,
which is a strong native contamination defense — there is no scraped public
dataset to have leaked into training. But the scenarios are themselves public on
GitHub, so a future model could in principle train on them.

**Private holdout (planned).** The committed defense against the public-exposure
risk is a private holdout: a fresh scenario subset that is never published, run
alongside the public set so a gap between public and holdout scores acts as an
overfitting tripwire. This is planned, not yet built. Tracking: issue #31.

## 5. Versioning

A benchmark version is the unit of comparability. Scores are only comparable
*within* a version; across a version boundary they are not.

A version bump is required for any of the following:

- **Corpus change** — adding, removing, or modifying scenarios. Following the
  lesson learned by other maintained benchmarks (MTEB's), a published scenario is
  immutable: once a scenario ships in version *N* it is not edited in place. New or
  changed scenarios go into version *N+1*.
- **Judge change** — changing the judge panel, or a judge's model version changing
  underneath it (see §2). A different judge is a different measuring instrument.
- **Rubric change** — changing the scoring rubrics, the dimension weights, or the
  efficacy composition.

**Cross-version scores are not comparable.** This is not only a versioning rule; it
is already structural in the scoring. The CLEAR composite is min-max normalized
across the set of models in a run, so adding or removing a model rescales every
other model's composite — which is why the methodology document already instructs
readers to compare models within a single run and to use raw per-dimension scores,
not the composite, for any cross-run view. Versioning makes the same caveat
explicit at the corpus, judge, and rubric level: when any of those change, the
benchmark changed, and old numbers describe a different benchmark.

---

*This policy is versioned with the repository. Changes to it are themselves part
of the public record via git history.*
