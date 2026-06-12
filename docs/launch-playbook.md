# Launch Playbook

Tracking issue: [#34](https://github.com/conorbronsdon/cot-bench/issues/34). This
document is the execution plan from the private rehearsal through the public
launch and the late-August Snorkel Open Benchmarks application. The playbook
merges now; every number that depends on a run that has not happened is
bracketed `[TODO: from run]`. Nothing in this file is a result.

Companion documents: `docs/rehearsal-runbook.md` (the run that precedes all of
this), `docs/governance.md` (the policies the risk register leans on),
`docs/methodology.md` (what every published number means).

---

## 1. Sequence

```
rehearsal  ->  review gates  ->  full run go/no-go  ->  full private run
           ->  publish  ->  launch day  ->  Snorkel application (late Aug)
```

Each step is a gate for the next. No step starts until the previous one is
signed off, and the launch-day content is written only after the full run's
numbers exist (that is what the brackets in section 2 are for).

### 1.1 Rehearsal

`docs/rehearsal-runbook.md`, in full: one contestant (Claude Haiku 4.5) across
the 92 public scenarios plus the 10-scenario private holdout, 3 reliability
runs, full 3-judge panel, roughly $30, hard-capped at $50. Plus the null-agent
subset pass and the sim-sensitivity subset pass. Results stay local. Nothing
publishes.

### 1.2 Review gates

These are the conditions that must be true before any money goes into the full
run. They come directly from the runbook's post-run analysis checklist
(sections 3, 4, and 5 there); the runbook has the exact commands and the full
list. The four hard gates:

1. **Null-agent floor is sane.** The deterministic do-nothing agent scores
   near zero: mean efficacy and state_score at or below ~0.1, every row
   carrying a failure mode. The null agent scoring materially above ~0.1 to
   0.15 means the judges reward doing nothing politely. That is a scoring
   problem and it blocks the full run.
2. **Judge alpha clears threshold.** Krippendorff's alpha on both judged
   dimensions (`judge_alpha.task_completion`, `judge_alpha.tool_selection` in
   `leaderboard.json`): at or above 0.8 is solid, 0.667 to 0.8 is tentative,
   **below 0.667 blocks the full run**. Below that line the panel is not
   measuring one thing.
3. **Halo delta reviewed.** The per-judge criterion-vs-holistic delta
   (runbook 5e): `criterion_informed` true on essentially every
   non-parse-failed judge entry (false everywhere = criteria are not reaching
   the judges, a harness bug, blocks), small positive mean deltas expected.
   One judge with a much larger absolute delta than the others, or large
   negative deltas (criteria raising scores), gets investigated before the
   full run. Record the per-judge numbers in the rehearsal notes.
4. **Cost within estimate.** `cost.actual_usd` at or below
   `cost.estimate_usd` in the run manifest (the priors are deliberately
   conservative). If the smoke-run ratio came in above 1.5, or mean agent
   input tokens ran well above the 9,000 prior, `PER_EVAL_TOKEN_PRIORS` in
   `eval/config.py` must be recalibrated from measured tokens before the full
   run, or the budget math below is fiction.

Secondary checks, same source, all of which must at least be looked at:
degraded judge rows under ~5% of the parquet (more blocks), same-lab check
(Opus delta on the Claude contestant in line with the open judges, a large
negative delta blocks), length bias (no significant positive slope with real
r-squared), premature-end rate low (roughly under 0.15), holdout gap small
(absolute gap roughly under 0.1), sim-sensitivity delta under ~0.05 (at or
above ~0.09 it must be understood and disclosed before publishing leaderboard
claims), `reliability_pass_hat_k` decaying gently rather than cliffing, and
the never-passed scenario IDs from the aggregation log reviewed for broken
scenarios.

### 1.3 Full run go/no-go

The full run is the launch run: all 11 contestants in `MODELS_UNDER_TEST`,
92 public scenarios plus the 10-scenario private holdout, 3 reliability runs,
full 3-judge panel, combined judge path. It runs **locally and privately**
(issue #34: full eval run privately before any announcement), not in CI. The
public CI refuses holdout runs by design, and the announcement should never
race the numbers.

**Cost, computed from the repo's own estimator** (`eval/cost.estimate_run_cost`
over the `PER_EVAL_TOKEN_PRIORS` and `TOKEN_COSTS` in `eval/config.py`, as of
the priors calibrated 2026-06-09):

- 11 models x 102 scenarios x 3 reliability runs = **3,366 evaluations**
- **Total estimate: ~$319** (public corpus only, 92 scenarios: ~$288)
- Breakdown: agents ~$90.53, simulators ~$26.93, judges ~$201.93. The judge
  panel is ~63% of the bill, and most of that is Opus.

Per-model agent cost over the full 306-evaluation slate (estimator output,
rounded):

| Model | Agent cost |
|---|---|
| GPT-5.5 | $27.54 |
| Claude Sonnet 4.6 | $15.15 |
| Gemini 3.1 Pro | $11.02 |
| GPT-4.1 (anchor) | $9.18 |
| Gemini 3.5 Flash | $8.26 |
| Qwen3.7-Max | $5.16 |
| Claude Haiku 4.5 | $5.05 |
| GPT-5.4-mini | $4.13 |
| Mistral Large 3 | $2.07 |
| DeepSeek-V4 Pro | $1.60 |
| MiniMax M3 | $1.38 |

Notes on this number:

- The earlier working ballpark for the full run (~$620, or ~$400 with batch
  pricing) predates the calibrated priors and is superseded by the ~$319
  figure above. **Batches API integration is not built**, so no batch discount
  applies either way; budget at the full synchronous figure.
- The estimate is conservative by construction (the rehearsal's job is to
  confirm that with a measured ratio). Set `--max-cost` with headroom anyway:
  ~$480 (1.5x the estimate) is a sane cap that still stops a runaway.
- Wall clock: the runbook budgets 5 to 12 hours for the rehearsal's single
  model over 306 evaluations (a prior, not a measurement; replace with the
  rehearsal's actual time once it has run). `run_eval` evaluates models concurrently
  (`--parallel-models`, default 2), so plan for the full run to span days at
  the default, not hours. It must be **one run**, resumed across sessions with
  `--resume` if needed, never stitched from separate runs: the
  pre-registration, the corpus hash check, and the within-run CLEAR
  normalization all assume a single run.

**Go** when: all four hard gates in 1.2 passed, the secondary checks reviewed,
priors confirmed or recalibrated, keys funded, and a multi-day local window
exists. **No-go** if any hard gate failed; fix and re-run the affected
rehearsal pass first.

### 1.4 Publish

This is the first real exercise of the publish path that the H1 fix repaired
(the `weekly-eval.yml` `git add` used to die on gitignored
`leaderboard.json`; the three published surfaces are now un-ignored and pinned
by `tests/test_published_surfaces_committable.py`). Because the launch run is
local, the publish is a local commit, not a CI run, but it follows the same
gate and touches exactly the same surfaces:

1. `python -m scripts.aggregate_results` immediately after the full run
   completes, before any other run overwrites the newest parquet. Keep the
   console output (never-passed scenario IDs log there and nowhere else).
2. `python -m scripts.check_publish_ready` against the full run's
   `run_manifest.json`. It blocks on: any failed model, any domain below the
   30-scenario minimum, a reduced judge panel, non-default reliability runs,
   or a non-cooperative sim profile. All five must pass clean; `--allow-partial`
   is not for launch.
3. `git status` discipline before staging: confirm nothing under
   `data/results/artifacts/` (full transcripts, **including holdout content**),
   no parquet, no manifest, no `.env` is staged. The only files that publish
   are `data/results/leaderboard.json`, `data/results/latest.csv`, and
   `data/results/history.jsonl`.
4. Commit those three files to `main` and push.
5. The push triggers `.github/workflows/pages.yml` (its path filter watches
   `data/results/leaderboard.json` and `frontend/**`), which copies
   `frontend/index.html` and the leaderboard JSON into a static site and
   deploys it. See 2.2 for the go-live verification steps.

The leaderboard JSON publishes per-model holdout aggregates (score and gap)
and never any holdout scenario ID, text, or per-scenario result
(governance.md section 4).

### 1.5 Launch day

Section 2. Launch day happens only after 1.4 is live and spot-checked, so no
post ever points at an empty board.

### 1.6 Snorkel application (late August)

The application ships when there is a live leaderboard URL and a methods
writeup to attach (the `paper/` skeleton is the long-form version; the
methodology doc is the short one). Per the grant strategy
(cot-production `production/cot-bench-grant-strategy.md`) and the application
skeleton (cot-production PR #43):

- The optional supporting materials are the application: live repo, live
  board, methodology, governance. A live v0.1 beats prose.
- Position on the program's three axes (environment complexity, autonomy
  horizon, output complexity) and frame v2 as what their expert-data credits
  enable (scenario review at scale, judge calibration, domain expansion).
- Warm path: Alex Ratner announced the program on COT Ep 57. Grant ask goes to
  the benchmarks program; the separate Snorkel sponsorship conversation stays
  with marketing. Never bundled.
- Never describe Terminal-Bench 2.0 or any Snorkel-affiliated benchmark as
  grant-funded. Snorkel contributed expert data and verification to TB2.0; it
  did not fund it. The committee will know.
- Application timing inside August is flexible (rolling review, quarterly
  selections), but it should land ahead of the fall summit cohort.

---

## 2. Launch-day checklist (dual channel)

Two channels, one day, one source of truth: every claim in every post traces
to `leaderboard.json` or a run artifact. Anything bracketed below cannot be
written until the full run is done.

**Collision rule with the June tools launch:** the tools-launch-kit sprint
(cot-production, `outputs/sprints/2026-06-11-tools-launch/`) posts one tool per
day and its day-one post (eval-integrity) already uses cot-bench's test count
and the H1 bug as its story. The cot-bench launch gets its own clear window,
no overlap with an active tools-launch posting day, and its posts lead with
run results, not the already-spent repo-hardening story.

### 2.1 Podcast channel

- [ ] **Launch episode or segment** (pick one, this is a content option, not a
  commitment):
  - *Single-thesis episode* anchored on ONE number (issue #34: the same-lab
    judge delta or the cost-reliability Pareto):
    `[TODO: from run: same-lab delta / Pareto headline]`.
  - *Benchmark trial segment*: 3 to 4 scenarios narrated, with the
    judge-vs-state-grader divergence as the story.
  - *Head-to-head format* (from the game-benchmark research, marked as an
    option): the same scenario run on two or three models back to back,
    reasoning narrated, listeners predict which one passes state grading
    before the reveal. Pick 3 to 4 high-divergence scenarios from the run
    `[TODO: from run: scenario IDs where models diverge most]`. Frame it as
    "what each model did differently", not a clean ranking; scaffolding and
    simulator caveats apply and get said out loud.
- [ ] **Substack post**, argument-driven, one thesis, the same single number
  as the episode. Not a listicle of leaderboard rows.
- [ ] Content fact-check note for any script or post: do not cite
  Terminal-Bench 2.0 "milestone rewards" as a partial-credit precedent. Its
  scoring is binary pass/fail per third-party analysis of the TB2.0 harness
  (research notes, confirmed as a non-recommendation in issue #34). Nobody on
  this project has read the grader code itself, which is one more reason the
  claim never goes in a script, post, or the application.

### 2.2 Developer channel

- [ ] **Leaderboard go-live** (the actual mechanism, verified against the
  repo): the site is GitHub Pages, already configured with the
  GitHub Actions build type, serving
  `https://conorbronsdon.github.io/cot-bench/`. `frontend/index.html` is a
  single static page that fetches `data/results/leaderboard.json` relative to
  itself and renders an explicit pre-launch empty state when the JSON is
  absent. `pages.yml` stages `frontend/index.html` plus
  `data/results/leaderboard.json` into `_site/` and deploys on any `main`
  push touching either path (or manual `workflow_dispatch`). Go-live steps:
  - [ ] Push the publish commit (1.4). Watch the `pages` workflow run green.
  - [ ] Load the URL, confirm the empty state is gone and the board renders
        all 11 models with CIs and rank bands.
  - [ ] Confirm the holdout column shows aggregates only.
- [ ] **README update**: replace the pre-results placeholder ("Results will
  appear here after the first evaluation run") with the live headline:
  `[TODO: from run: top-line result, e.g. best model + efficacy + CI]`.
- [ ] **llms.txt**: add the leaderboard URL and a one-line benchmark
  description to `cot-site/public/llms.txt` (the chainofthought.show AEO
  surface). cot-bench itself has no llms.txt; if one is added, it points at
  the README and methodology.
- [ ] **HN post** (Show HN): `[needs run results: headline number]` in the
  title or first sentence. Link the leaderboard, not the repo root.
- [ ] **X post**: `[needs run results: headline number]`, 280 chars, no
  invented @-handles.
- [ ] **LinkedIn post**: leads with a concrete result
  `[needs run results: headline number or the sharpest disagreement-log
  case]`, never "I built a benchmark". Tag real entities. Closes with
  `#notanofficialspokesperson`.
- [ ] **Bluesky variant**: under 300 graphemes, written fresh, not trimmed
  from X.
- [ ] All posts run through the AI-writing filter. No em dashes anywhere.
- [ ] **Labs-cite-it flywheel** (issue #34): frontier models covered at launch
  (the roster already is), self-serve harness instructions linked, one
  headline metric per model usable in a model card, and the "no model judges
  its own lab without disclosure" stance stated plainly.

---

## 3. Disagreement-log publishing (spec)

The research finding behind this: failure transcripts outdraw metrics
(Vending-Bench's FBI-call transcript beat its own leaderboard for attention).
cot-bench's native equivalent is the case where the LLM judges passed an agent
but deterministic state grading failed it. The state grader catching a 2x
money error the judge missed is the prototype story. This section specs the
published artifact; **the generation script is a post-launch build item**, not
launch scope.

### 3.1 The artifact

A periodic digest, "The Disagreement Log", published as a markdown page
(repo `docs/` or chainofthought.show, decide at first issue) containing 5 to
10 annotated cases per issue. Each case shows:

- The scenario premise, paraphrased in two sentences (**public scenarios
  only, never holdout**).
- A short transcript excerpt: only the turns where the divergence happened,
  trimmed, with synthetic entity names left as-is (the corpus is synthetic;
  there is no real PII, but excerpts stay minimal anyway).
- What the judges said: per-judge scores per dimension, the consensus, and
  the dissenting judge where there is one, with one line of each judge's
  reasoning.
- What the state grader said: the deterministic verdict and which specific
  check failed or passed.
- One paragraph of editorial: why the judges missed it, or why they disagreed.

### 3.2 Case selection and source fields

All fields below exist today and were verified against the code. Two sources:

**Per-evaluation artifacts** (`data/results/artifacts/<run_id>/`, written by
`eval/artifacts.py`):

- `judges.task_completion[]` and `judges.tool_selection[]`: per judge,
  `judge_name`, `overall_score` (criterion-informed, the score that counts),
  `holistic_score` (the pre-criteria template score), `criteria_verdicts`
  (per-criterion `{id, met, evidence}`), `reasoning`, `parse_failed`,
  `resolved_model`.
- `state`: `score` plus per-assertion `checks` (and the `final_world` it was
  computed from).
- `transcript`: the full serialized conversation, for excerpting.
- `holdout`: the boolean the privacy filter keys on.

**Result rows** (the run parquet, built in `scripts/run_eval.py`): per-judge
score columns (`tc_<judge>` / `ts_<judge>`), `tc_max_disagreement` /
`ts_max_disagreement`, and `high_disagreement` (true when either dimension's
judge spread exceeds 0.3), plus `failure_mode` and `state_score` for the
selection queries.

Selection, in priority order:

1. **Judge-pass, state-fail**: consensus judge scores above the pass bar,
   `state.score` failing. The flagship case type.
2. **High inter-judge disagreement**: `high_disagreement` rows (spread > 0.3),
   showing consensus vs dissenting judge with the state grader as the
   tiebreaking evidence.
3. **Halo cases**: large `holistic_score` minus `overall_score` gaps, where
   the atomic criteria changed a judge's verdict.

### 3.3 Cadence

One digest per published benchmark run, drawn only from that run's artifacts,
released within the week after the run publishes. At the weekly eval cadence
that is at most weekly; in practice expect the editorial pass to make it
roughly monthly, and that is fine. Versioned releases each get one regardless.

### 3.4 Privacy rules

- **Public scenarios only. Never holdout.** The generator hard-filters on the
  artifact's `holdout` flag (`holdout: false` only), and a human verifies the
  filter on every issue before publishing, because the artifacts directory
  contains full holdout transcripts whenever a holdout run happened.
- Stated again because it is the rule that cannot fail even once: **no holdout
  scenario, transcript, ID, or excerpt is ever published in a disagreement
  digest.** One leak burns the overfitting tripwire permanently
  (governance.md section 4).
- Excerpts are minimal: the divergent turns, not full transcripts. Full
  transcripts stay in the local artifacts.
- No raw judge `raw_response` dumps (they can quote the whole transcript);
  publish scores, verdicts, and trimmed reasoning only.

### 3.5 Build item

`scripts/disagreement_digest.py` (post-launch): query the parquet for the
selection classes, join the artifacts, emit a markdown draft with the fields
above pre-filled and the holdout filter applied, for human annotation. Not
needed for launch day; the first digest can be assembled by hand from the
full run.

---

## 4. Risk register

| Risk | Answer | Where it is written down |
|---|---|---|
| A model lab disputes its score publicly | The run was pre-registered before any number existed (roster, corpus hash, judge panel, seeds), every score traces to a per-evaluation artifact (transcript, per-judge output, state checks), and the served model is recorded per call (`resolved_model`), so the dispute is answerable with evidence. If the lab is right, the correction is a new dated run plus a changelog entry; the original numbers stay visible. No silent retraction, no quiet re-run, in either direction. | governance.md sections 1, 2, 3 |
| Someone finds a real scoring bug after launch | The fix ships as a new benchmark version; corpus, judge, or rubric changes all force a version bump, published scenarios are immutable once shipped, and cross-version scores are declared non-comparable. Old numbers stay up as version N, fixed numbers are version N+1. | governance.md section 5 |
| Traffic spike on the leaderboard | The site is one static HTML file and one JSON file behind GitHub Pages' CDN. There is no backend to fall over. A front-page HN spike is within Pages' soft bandwidth limits for a page this size. No action needed. | `pages.yml`, `frontend/index.html` |
| OpenRouter re-points a judge slug mid-version | `resolved_model` is recorded per judge call, so the drift is a detectable recorded event, and a judge version change is a methodology change forcing a version bump. | governance.md sections 2, 5 |
| Launch posts go out before the board is live | The checklist ordering in section 2 (publish, verify, then post) plus the standing rule that no announcement points at a 404. Posts fire only after 2.2's go-live verification. | this document |
| The weekly scheduled run publishes something unreviewed post-launch | The scheduled path always publishes, gated by `check_publish_ready` (failed models, scenario minimum, full panel, default reliability, cooperative profile). Decide deliberately at launch whether the schedule is enabled at all; until it is, publishes are manual and reviewed. | `weekly-eval.yml`, `scripts/check_publish_ready.py` |
| Holdout content leaks | CI refuses to start with a holdout dir configured; artifacts are gitignored; published surfaces carry holdout aggregates only; the disagreement digest hard-filters and re-verifies per issue (3.4). | governance.md section 4, `weekly-eval.yml` |
