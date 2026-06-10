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

**Mechanism (implemented, issue #38).** `scripts/run_eval.py` writes a
pre-registration file, `pre_registration.json`, to the results directory **before
the first agent, simulator, or judge call** — before any number is known. The
write is a single function, `write_pre_registration` (in
`eval/pre_registration.py`), called at the top of the run after the scenario set
and model roster are resolved and before the evaluation loop starts; a test
(`tests/test_pre_registration.py::TestWrittenBeforeRun`) asserts the file exists
on disk at the moment the first model is dispatched, so the ordering is enforced,
not merely intended. Because the definition is fixed on disk ahead of the results,
the maintainer cannot retroactively choose which run "counts."

The pre-registration records:

- **Run identity** — `run_id` and a UTC `timestamp` (when the pre-registration
  was written).
- **Models under test** — the *requested* roster (name, model_id, provider).
  Requested only: which models *complete* is a post-run fact and stays in the
  completion record.
- **Scenario set** — the domains, the per-domain scenario IDs, and a
  corpus-level **`sha256` over the canonical serialized scenario set**. Each
  scenario is re-serialized with sorted keys, the per-scenario digests are sorted
  by scenario ID, and they are folded into one corpus digest — so the hash is
  deterministic and independent of on-disk file order or whitespace. A
  `scenario_index` (a sorted list of `{domain, scenario_id, sha256}` entries) is
  included so an auditor can recompute
  it. Any change to any scenario's content, or adding/removing a scenario,
  changes the corpus hash. This is the corpus-level hash the per-scenario ID
  fragments (`scripts/generate_data.py`) did not previously provide.
- **Judge panel** — the configured judges (name + configured `model_id` +
  provider + temperature). `resolved_model` — the model a provider *actually
  served* — is deliberately **not** pre-registered: it is only knowable at call
  time and is recorded per call in the post-run artifacts (see §2). The
  pre-registration commits to the panel *as configured*, which is what is fixable
  in advance.
- **Reliability-run count** and **seeds / temperatures** — the bootstrap seed
  (`BOOTSTRAP_SEED = 42`, `scripts/aggregate_results.py`), the agent temperature
  (0.0), and the user/tool simulator temperatures. The record states explicitly
  that the user and tool simulators run **unseeded** and that the user simulator
  runs at temperature > 0, so runs are **not bit-for-bit reproducible** — the
  honest caveat is part of the artifact, not a footnote elsewhere.
- **Judge-prompt mode** — combined single-prompt (default) or the legacy
  separate-prompt path.

**Completion record.** The existing `run_manifest.json` remains the post-run
completion record (it records the models requested / completed / failed, domains,
per-domain scenario counts, the reliability-run count, and the artifact and trace
directories; the publish gate, `scripts/check_publish_ready.py`, reads its
`models_failed` field to block a leaderboard commit that would silently ship
missing models). It now carries a `pre_registration` block linking back to the
pre-registration by **path and sha256** (plus the corpus hash), so the two files
are a verifiable pair: anyone can confirm that the run's definition fixed before
the run matches the one the completion record points at.

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

**Private holdout (implemented).** The committed defense against the
public-exposure risk is a private holdout: a fresh scenario subset that is
**never published**, run alongside the public set so a gap between public and
holdout scores acts as an overfitting tripwire. The harness support is built
(issue #31); the held-out scenarios themselves are authored and stored *outside*
this repository.

How it works, and how privacy is guaranteed:

- **External, opt-in.** The holdout corpus lives in a separate location and is
  pulled in at evaluation time via `--holdout-dir` (or the `COT_BENCH_HOLDOUT_DIR`
  environment variable). The public CI never sets it, so the holdout is only ever
  exercised from a private run. Its scenarios are laid out exactly like
  `data/scenarios/` (one subdirectory per domain) and use the identical v0.2
  schema, so they are graded by the same simulator, judges, and deterministic
  state checks as the public corpus.
- **Hash-pinned, content never revealed.** When a holdout is run it gets its own
  entry in `pre_registration.json`: a `holdout_set` block carrying the corpus
  `sha256` **and the count only** — deliberately no scenario IDs and no
  per-scenario index, unlike the public `scenario_set` (which publishes both).
  The hash is computed by the same machinery as the public corpus hash, so it is
  tamper-evident — any change to the held-out scenarios (a different scenario, an
  added or removed one) changes the hash — yet the hash reveals nothing about the
  scenarios' content. The set is *pinned without being exposed*.
- **Public score is unaffected; the gap is published.** Holdout result rows are
  tagged `holdout: true` and split out before aggregation, so the headline
  efficacy and CLEAR rankings are computed over the **public corpus only**. The
  leaderboard then publishes, per model, the public score, the holdout score, and
  the gap (`gap = public - holdout`). A materially positive gap is the
  overfitting signal: strong on the public scenarios a model could have trained
  on, weaker on the never-seen holdout.
- **No holdout scenario detail in any published output.** `leaderboard.json` and
  `latest.csv` carry only the per-model holdout aggregates; no holdout scenario
  ID, text, ground truth, or per-scenario score reaches the published surfaces.
  (Per-scenario results, transcripts, and the public scenario index live only in
  the run's local/CI artifacts, which are gitignored and not part of the
  published leaderboard.)

Count is publishable; content is not. Tracking: issue #31.

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
