# Methodology

This document explains how COT Bench evaluates AI agents, why we made the design choices we did, and how to interpret results.

## Overview

COT Bench evaluates AI agents on multi-turn, tool-calling tasks across real-world domains. Unlike benchmarks that test single-shot tool selection, we simulate complete conversations where an agent must:

- Understand user intent across multiple turns
- Select and sequence tools correctly
- Handle ambiguity, errors, and scope boundaries
- Maintain conversation coherence

Every evaluation produces four CLEAR-aligned dimensions (Efficacy, Cost, Reliability, Latency), scored by three independent judges.

## Evaluation Pipeline

### 1. Scenario Design

Each scenario consists of:

- **Persona**: A synthetic user with specific personality traits, communication style, and background context. Personas range from cooperative to adversarial, tech-savvy to confused.
- **Goals**: 3-10 interconnected goals the user wants to accomplish (typically 5-8). Goals are designed so completing one reveals information needed for another.
- **Tools**: A subset of domain-specific tools with full JSON schemas, including parameter types, required fields, and response formats.
- **Initial message**: An in-character opening message that naturally leads toward the goals.

Scenarios are organized into five categories that test distinct failure modes:

| Category | What it tests |
|----------|--------------|
| **Adaptive tool use** | User needs shift mid-conversation — agent must adapt |
| **Scope management** | User requests things outside agent capabilities |
| **Empathetic resolution** | Emotionally charged situations requiring tool use AND empathy |
| **Extreme scenario recovery** | Tool errors or unexpected data — agent must recover |
| **Adversarial input mitigation** | Misleading, ambiguous, or manipulative inputs |

**Ground-truth world state (schema v0.2).** Each v0.2 scenario carries a
`ground_truth` block — a canonical world (balances, records, customer state) that
the simulated tools answer from and mutate — plus `expected_state_changes`, a list
of deterministic assertions over the post-conversation state (e.g. "the checking
balance increased by $2,500", "a fraud case was created for this transaction").
These assertions use a tiny, judge-independent vocabulary (`equals`,
`increased_by`, `decreased_by`, `contains`) and are verified without an LLM, so
"did the transfer actually happen?" becomes an objective check rather than a
judge's opinion. For judgment calls we grade the **action**, not the **outcome**
(assert a fee-waiver request was *submitted*, not that it was approved). A scenario
with no legitimate state change sets `expected_state_changes: []`, which asserts
that **no unauthorized mutation** occurred — a strong objective signal for the
refusal-heavy scope-management and adversarial categories. The full schema is
documented in [scenario-schema.md](scenario-schema.md).

**Authorship and dedup gates.** Every v0.2 scenario records an `authorship` block
(`author_model`, optional batch id and human reviewer). The validator rejects any
`author_model` that appears in `MODELS_UNDER_TEST` — a model under test must never
author its own exam — and supports multi-author generation across labs to avoid a
single-author monoculture. It also runs a cross-scenario dedup gate within each
domain (near-duplicate `initial_message` + `user_goals` similarity, and goal-set
Jaccard overlap) plus a distribution report over categories, difficulty, persona
reuse, and tool coverage.

### 2. Simulation

The simulation runs for up to 10 **user turns**. Each user turn drives an inner
agent loop that lets the agent act on tool results *before* the user replies:

1. **User simulator** (GPT-4.1-mini, temperature 0.7) generates a message in-character for the persona, pursuing unmet goals
2. **Agent under test** (the model being evaluated, temperature 0.0) responds. Tools are provided through **native function calling** (LangChain `bind_tools`): each scenario tool definition is converted to an OpenAI-style JSON Schema function, and tool calls are read from the model's structured `tool_calls`. The agent is not asked to emit a bespoke JSON-in-text convention.
3. If the agent calls one or more tools, the **tool simulator** (GPT-4.1-mini, temperature 0.0) generates realistic responses conforming to each tool's schema. Results are returned to the agent as `ToolMessage`s, and **the agent is re-invoked** on those results.
4. Steps 2–3 repeat within the turn until the agent produces a user-facing message (no tool calls), or an inner cap of **5 tool rounds per user turn** is reached (a safeguard against runaway tool loops).
5. The user simulator then decides whether it is **done talking**. If it signals completion the conversation ends; otherwise the outer loop continues with a new user turn. Crucially, the user sim does **not** get to declare the goals *met* — see "Decoupling completion from goal-completion" below.

**Stateful tool simulation (schema v0.2).** For scenarios that carry a
`ground_truth`, the tool simulator is no longer a stateless response generator.
At the start of each run the world is seeded from a deep copy of `ground_truth`
(one world per run, reset between reliability repeats so mutations never leak
across runs). On every tool call the simulator is shown the *current* world and
instructed to **answer ONLY from that state** — never to invent balances, IDs, or
records not present in it — and to return a structured
`{"response": …, "state_delta": …}` object. Read-only calls return an empty
delta; mutating calls (a transfer, a fraud report, a feature request) return a
dotted-path `state_delta` that the runner applies deterministically to the world,
so a later "check my balance" reflects the earlier transfer. Only the `response`
part is fed back to the agent; it never sees the delta or the world. This removes
the simulator-incoherence failure mode where asking a balance twice yields two
different numbers and the judge then penalizes the *agent* for the *simulator's*
drift. When a scenario has no `ground_truth`, the simulator keeps the original
stateless behavior.

**Decoupling completion from goal-completion (user-sim independence).** A single
LLM plays the user simulator, and the published literature is clear that
simulated users are miscalibrated and over-cooperative: which model plays the
user can swing agent success by up to ~9 points ([Lost in Simulation, arXiv
2601.17087](https://arxiv.org/abs/2601.17087); [Sim2Real, arXiv
2603.11245](https://arxiv.org/pdf/2603.11245)). If that same simulator both plays
the customer **and** decides the goals are met, its bias contaminates the stop
condition: a sim that gets tired or is too easily satisfied can end a
conversation early, and the agent is then credited (or blamed) for an ending the
simulator chose. We therefore split the two signals:

- The user simulator only declares that it is **done talking** (it emits the
  conversation-complete token). This is recorded as `ended_by = "user_sim"`.
- Whether the goals were actually accomplished is the **deterministic state
  check** (`score_state_changes`, the same grader used for the post-run
  `state_score`), evaluated against the mutated world at the exact moment the
  conversation ended and recorded as `state_progress_at_end` (the pass fraction
  in `[0, 1]`, or null for legacy scenarios with no `ground_truth`).

When the sim ends the conversation while the state check is still below 1.0, the
run is flagged `premature_end = True`: the simulator quit before the goals were
verifiably met. We do **not** silently treat that ending as a success, and we do
not extend the conversation either — extending would let an already-miscalibrated
sim keep driving turns and would change the agent's scoring conditions. Instead
the discrepancy is made **visible and aggregable**: `ended_by`,
`state_progress_at_end`, and `premature_end` are written to every per-run
artifact (`sim_meta`) and to the results parquet, and `aggregate_results` reports
a per-model **premature-ending rate** on the leaderboard. The deterministic
state check — not the simulator's satisfaction — remains the source of truth for
goal completion (it already drives 30% of Efficacy via `state_score`); this
change ensures a biased sim can no longer launder an early exit into an apparent
pass. Scenarios without a `ground_truth` have no deterministic check to gate on,
so for those the sim signal is still all the harness has; `state_progress_at_end`
is null and `premature_end` is false for them, which is the honest accounting.

This is part 1 of the user-simulator de-risking. A planned part 2 (not yet run)
is a sim-sensitivity check that re-runs a subset under a second simulator model
and publishes the agent-score delta, directly quantifying the swing the
literature warns about.

This agent→tool→agent iteration means the model sees and reasons over tool output within the same turn, rather than only on the following turn. The transcript preserves true conversational order — user → agent (with tool calls) → tool results → agent follow-up → … → user — so judges read each tool call before its result.

Native tool calling (rather than a regex-parsed JSON-in-text protocol) measures real tool-calling ability and avoids penalizing models tuned for function-calling APIs. A content-embedded fallback parser exists only for providers that return no native tool calls; it logs a warning whenever it fires so its use is measurable.

Temperature 0.0 for the agent under test ensures reproducibility. The user simulator uses 0.7 for natural variation in conversation flow. Token and latency accounting cover every agent invocation, including inner tool-loop rounds; latency sums agent wall-times only (simulators excluded).

### 3. Scoring

After simulation, the full transcript is sent to three independent judges.

#### Task Completion (40% of Efficacy, or 50% for legacy scenarios)

Each judge evaluates whether the agent accomplished the user's goals:

- **COMPLETE** (1.0): Goal fully addressed with correct tools and information
- **PARTIAL** (0.5): Meaningful progress but not fully resolved
- **FAILED** (0.0): Goal not addressed, wrong tools used, or incorrect information

Judges also assess: appropriate clarifying questions, graceful error recovery, scope awareness, and multi-step dependency handling. These factors adjust the score ±0.1.

The full rubric is in [`eval/scoring/rubrics.py`](../eval/scoring/rubrics.py).

#### Tool Selection Quality (30% of Efficacy, or 50% for legacy scenarios)

Each judge evaluates every tool call on five dimensions:

1. **Selection correctness**: Was this the right tool?
2. **Parameter accuracy**: Were parameters correct and complete?
3. **Sequencing**: Were tools called in logical order?
4. **Necessity**: Was the call needed, or was it redundant?
5. **Omissions**: Were there tool calls the agent should have made?

#### State Verification (30% of Efficacy, deterministic; v0.2 scenarios only)

For scenarios with a `ground_truth`, the final world state (after the
conversation) is checked against the scenario's `expected_state_changes` with no
LLM in the loop. Each assertion uses a tiny vocabulary — `equals`,
`increased_by`/`decreased_by` (float tolerance 0.01), `contains` (partial-dict
match with optional `*_contains` substrings) — and the state score is simply the
fraction of assertions that pass. An empty assertion list encodes the
no-unauthorized-mutation contract (score 1.0 iff the world is unchanged), which is
the objective signal for refusal-heavy scope-management and adversarial scenarios.
This component is judge-independent and free to compute. See
[`eval/scoring/state_check.py`](../eval/scoring/state_check.py).

#### Combined judge prompt (default)

Task completion and tool selection are scored in a **single judge call per
judge**, not two. The combined prompt (`COMBINED_RUBRIC` in
[`eval/scoring/rubrics.py`](../eval/scoring/rubrics.py)) presents the shared
context — domain, user goals, available tools — and the transcript **once**,
then asks for both dimensions in one JSON response nesting the two per-rubric
shapes under `task_completion` and `tool_selection` keys. The per-dimension
scoring criteria are the same published rubric text used by the two-prompt path,
reorganized under one shared context rather than rewritten.

**Why this is equivalent, and what it costs.** Each dimension is still scored
against its own criteria, and the parsed object for each dimension is the same
shape the separate prompts produced — so the consensus math, per-judge score
columns, artifacts, aggregation, confidence intervals, and the same-lab check
are all unchanged. The win is twofold: the judge **call count is halved** (one
call per judge instead of two), and because the transcript is the bulk of the
prompt and is no longer duplicated across two calls, judge **input tokens drop
by roughly 45%**.

**Failure coupling (the honest tradeoff).** Because both dimensions ride on a
single API call and a single parsed JSON body, a failure now affects both
dimensions for that judge:

- An **API failure** drops the judge from both panels (there is no
  per-dimension fallback within one call).
- A **parse failure** — unrecoverable JSON after one retry, **or** a parsed body
  in which *either* dimension is missing or has an invalid/absent
  `overall_score` — flags the judge as parse-failed for **both** dimensions.
  Treating a half-valid response as a whole-judge parse failure is the simplest
  honest rule: a judge that could not produce one valid dimension gives us no
  reason to trust the other from the same generation. (The retry semantics are
  otherwise identical to the two-call path: one fresh API call on a parse
  failure before giving up.)

In exchange for that coupling, a single transient glitch now costs at most one
judge across both dimensions rather than potentially failing them
independently, and the per-row panel accounting (`tc_n_judges` / `ts_n_judges`,
parse/api failure counts, `degraded`) reflects the coupling transparently.

The legacy two-call path remains available behind `run_eval
--separate-judge-calls` for A/B validation; it sends each dimension's prompt
separately (context + transcript twice) and is otherwise identical.

#### Multi-Judge Consensus

All three judges score independently. We report:

- **Consensus score**: **Median** of all *valid* judge scores
- **Inter-judge reliability**: **Krippendorff's alpha** (ordinal/interval level), the primary chance-corrected agreement metric, with the within-0.2 pairwise rate kept as a secondary readout
- **Individual scores**: Every judge's score is published for transparency

When judges disagree significantly (>0.3 spread), this often indicates genuine ambiguity in the scenario — these cases are flagged in the results.

**Why median, not mean.** For an n=3 panel the median is the middle judge's score, so a single rogue or leniency-drifted judge cannot drag the consensus the way a mean would; for n=2 the median equals the mean of the two. (Bradley-Terry was rejected: it models pairwise preference data, but cot-bench produces absolute pointwise rubric scores, so it does not fit the data.) Every individual judge score is still published, so a mean (or any other aggregation) can be recomputed by anyone.

**Why Krippendorff's alpha for agreement.** A raw "within-0.2 rate" is not chance-corrected, not comparable across score distributions, and its 0.2 threshold is arbitrary. Krippendorff's alpha is the field-standard chance-corrected metric for 3+ raters on ordinal/interval labels (Cohen's kappa is only for 2 raters and assumes nominal categories). We compute alpha at the interval level (squared-difference distance, appropriate for the continuous 0-1 rubric scores) over the published per-judge score columns: each result row is a *unit*, each judge a *rater*, and a judge that parse-failed on a row is simply a missing value (alpha is defined for incomplete data and uses only units with 2+ present scores). Alpha is published per dimension both overall (`judge_alpha` in `leaderboard.json`) and per model (`judge_alpha` in each model entry). We implement alpha in-repo (a small, test-validated function checked against worked examples from Krippendorff's reference material) rather than adding a dependency. Alpha is `null` when undefined — fewer than two pairable scores, or no usable variation.

##### Judge-failure handling

A judge can fail in two ways, and neither is allowed to silently contaminate the consensus:

- **Parse failure** — the judge returns text we cannot recover valid JSON from (often a truncated or malformed generation). We do *not* treat this as a genuine 0.0 grade, because a fabricated 0.0 would drag the consensus and crater the agreement rate, indistinguishable from a real low score. Instead, the judge is called **once more** (a fresh API call); transient format glitches usually clear on the retry. If parsing still fails, the result is flagged and **excluded** from the consensus score, agreement rate, and max-disagreement. It is still kept in the per-row record for transparency.
- **API failure** — the judge call raises. The judge name is recorded and the judge is dropped from that row's consensus.

Because failures shrink the panel, every row publishes explicit accounting so consensus quality is auditable rather than hidden:

- `tc_n_judges` / `ts_n_judges` — number of *valid* (scored and parsed) judges that contributed to consensus
- `tc_parse_failures` / `ts_parse_failures` — count of judges excluded after a failed retry
- `tc_api_failures` / `ts_api_failures` — count of judges whose API call raised
- `tc_degraded` / `ts_degraded` — true when fewer than 2 valid judges remained despite 2+ being requested (the consensus is still computed from what's valid, but flagged as low-confidence)

**Agreement is undefined with fewer than 2 valid judges.** Agreement rate and max-disagreement are reported as **null** in that case — a single grader is not "perfect agreement." Downstream aggregation skips these nulls (a model whose rows are all single-judge will show a null judge-agreement rather than a misleading 1.0).

##### Length-bias check

LLM judges can favor verbose outputs (the AlpacaEval length-bias lesson). Rather than assert we are immune, we measure it: the per-row agent `output_tokens` already exists, so at aggregation we fit a simple ordinary-least-squares regression of each judge dimension (`task_completion` and `tool_selection` consensus) on `output_tokens` across all rows — the bias is a property of the judge panel, not of any one model, so the regression pools all rows. The deterministic `state_score` is judge-independent and therefore excluded. The regression is plain math (no new dependency): for each dimension we publish the slope, intercept, R², the slope's standard error, its t-statistic, the sample size, and a simple significance flag (`|t| > 1.96`, the ~5% two-sided threshold) under the top-level `length_bias` key in `leaderboard.json`. A materially positive, significant slope is direct evidence that longer agent outputs earn higher judge scores independent of correctness; publishing it makes the confound measurable rather than deniable, and is the trigger for adding an explicit length-neutrality line to the rubrics.

### 4. CLEAR Dimensions

#### Efficacy (weight: 35%)

Efficacy is **hybrid**: two LLM-judge dimensions plus one deterministic,
judge-independent dimension (state verification, see §2). For a v0.2 scenario
that carries a `ground_truth` world:

```
efficacy = 0.4 × task_completion_consensus
         + 0.3 × tool_selection_consensus
         + 0.3 × state_verification   (deterministic, from expected_state_changes)
```

A full third of Efficacy is therefore objective — "did the transfer actually
happen?" is checked against the post-conversation world, not inferred by a judge.
For **legacy scenarios with no `ground_truth`**, state verification is
inapplicable and Efficacy degrades gracefully, renormalizing to the original
equal split over the two judge dimensions:

```
efficacy = 0.5 × task_completion_consensus + 0.5 × tool_selection_consensus
```

#### Reliability (weight: 25%)

Each scenario runs 3 times (pass@3). We measure:
- **Pass rate**: Fraction of runs scoring above 0.7 threshold
- **Consistency**: 1.0 minus the spread between highest and lowest scores
- **Variance**: Statistical variance across runs
- **pass^k**: The probability that *all* k trials succeed (tau-bench style)

A model that scores 0.9, 0.85, 0.88 is more reliable than one that scores 0.95, 0.4, 0.8 — even though the latter has a higher peak.

**pass^k (all-k-succeed).** Borrowed from tau-bench, `pass^k` is the probability that *every one* of k independent trials of a task succeeds — the opposite of `pass@k` (at least one succeeds). For i.i.d. trials it decays as p^k, so a model that passes 2 of 3 runs has `pass^1 = 0.67` but `pass^3 = 0.0`: a single failure across the repeats sinks it. This is a sharper, harder-to-game reliability construct than a pass-rate-above-threshold and directly measures the *autonomy horizon* — can you trust the agent to do the same task right every time, not just once. We estimate it empirically per scenario as the average over all C(n, k) size-k subsets of the n collected trials of the all-pass indicator, which has the closed form `C(c, k) / C(n, k)` for c passing runs out of n (`compute_pass_hat_k` in [`eval/scoring/rubrics.py`](../eval/scoring/rubrics.py)). `pass^1` equals the ordinary pass rate. Each model's leaderboard entry publishes `reliability_pass_hat_k` (one value per k, the mean of the per-scenario estimates) **alongside** — not replacing — `reliability` (pass@3) and `reliability_consistency`. With the current 3 reliability runs, k ranges over 1, 2, 3; running more repeats extends the horizon.

#### Cost (weight: 20%)
```
cost_per_task = (input_tokens × input_price_per_M / 1,000,000)
              + (output_tokens × output_price_per_M / 1,000,000)
```

Token prices are from published provider pricing as of the evaluation date. Prices are tracked in [`eval/config.py`](../eval/config.py) and updated with each evaluation run.

#### Latency (weight: 20%)

Wall-clock time across all agent turns in a scenario (excludes user simulator and tool simulator time). This measures the actual wait time a user would experience.

### 5. CLEAR Composite Score

Dimensions are min-max normalized across all evaluated models, then weighted:

```
CLEAR = 0.35 × efficacy_norm + 0.25 × reliability_norm
      + 0.20 × (1 - cost_norm) + 0.20 × (1 - latency_norm)
```

Cost and latency are inverted because lower is better. When only one model is evaluated, CLEAR equals raw efficacy.

Because the min-max normalization is relative to the set of models in the run, adding or removing a model rescales every other model's normalized dimensions and therefore changes every CLEAR score. CLEAR scores are only comparable within a single run. To compare a model across runs, use its per-dimension raw scores (efficacy, reliability, cost, latency) rather than the composite.

### Uncertainty

Every published dimension carries a 95% confidence interval, and the leaderboard refuses to publish below a minimum scenario count. With few scenarios, model orderings are dominated by sampling noise rather than real capability differences — an honest board has to say so.

**Bootstrap over scenarios.** We estimate uncertainty with the nonparametric bootstrap: **B = 2000** replicates, fixed seed **42** for reproducibility. The unit of resampling is the **scenario**, not the individual row. All runs of a scenario share the same task and persona, so their scores are correlated; resampling rows would treat correlated repeats as independent draws and understate the true uncertainty. Each replicate draws a sample of scenario IDs with replacement (keeping every run/row of a resampled scenario) and recomputes the per-model means. The 2.5th and 97.5th percentiles of the replicate distribution give each model's `ci_low` / `ci_high`. We publish CIs for **efficacy** and the **CLEAR composite** (`efficacy_ci`, `clear_score_ci`).

**Paired normalization.** Because the CLEAR composite is *field-relative* (min-max normalized across the evaluated models), its uncertainty must include normalization variance, not just the variance of each raw dimension. So the **same** resampled set of scenario IDs is applied to every model within a replicate (a *paired* bootstrap), and the full min-max normalization plus weighted composite is recomputed inside each replicate across all models. This keeps the field consistent per replicate and propagates normalization variance into the CLEAR interval.

**Rank bands.** Orderings are only meaningful where intervals separate. We cluster models into rank bands by CLEAR-score CI overlap using a simple greedy algorithm: walk the models sorted by CLEAR descending; the current band's leader is the highest unbanded model; every following model whose `clear_score_ci` overlaps the leader's joins that band; the first model that does *not* overlap starts a new band. Models within a band are not statistically distinguishable from one another, and the frontend marks them with a tied rank (`1=`). The leaderboard JSON also carries a top-level `statistical_note` stating the scenario count and the band caveat.

**Publish minimum.** A leaderboard does not publish at all when any evaluated domain has fewer than **`MIN_SCENARIOS_FOR_PUBLISH` = 30** scenarios (see [`eval/config.py`](../eval/config.py)). The publish gate ([`scripts/check_publish_ready.py`](../scripts/check_publish_ready.py)) enforces this alongside the run-completeness check; `--allow-partial` downgrades it to a warning for deliberate previews. The project is scaling toward ~80 scenarios; the gate keeps the current state honest in the meantime.

Edge cases are handled without crashing: a single scenario produces a degenerate (zero-width) CI equal to the point estimate, and a single-model run skips cross-model normalization so its CLEAR CI mirrors its efficacy CI.

## Design Decisions

### Why synthetic scenarios instead of real conversations?

Real conversation data is hard to obtain, contains PII, and can't be shared openly. Synthetic scenarios give us:
- Full control over difficulty and coverage
- Reproducibility — same scenarios for every model
- Ground truth goals for scoring
- No privacy concerns

### Contamination and the private holdout

Synthetic scenarios mean there is no scraped public dataset that could have leaked into a model's training set. But the public corpus is itself on GitHub, so a future model could in principle train on it. The defense is a **private holdout** (issue #31): a fresh scenario subset, authored to the same quality bar and the same v0.2 schema, that is **never published**. It is run alongside the public set via `--holdout-dir` / `COT_BENCH_HOLDOUT_DIR`, and the leaderboard reports each model's public score, holdout score, and the gap between them (`gap = public - holdout`). A model that is strong on the public corpus but weaker on the never-seen holdout shows a positive gap — the overfitting tripwire.

The holdout is **hash-pinned but content-private**. When a run includes a holdout, `pre_registration.json` records the holdout corpus `sha256` and count only — no scenario IDs, no per-scenario index — so the held-out set is fixed and tamper-evident before any score is known, without revealing what it contains. Holdout result rows are split out before aggregation, so the public efficacy/CLEAR rankings are unaffected by it; only the per-model gap is published. No holdout scenario ID, text, ground truth, or per-scenario score ever appears in `leaderboard.json` or `latest.csv`. The mechanism and privacy guarantees are documented in [governance.md §4](governance.md).

### Why three judges instead of one?

A single judge introduces systematic bias. Using three judges from different labs (Moonshot's Kimi, Zhipu's GLM, Anthropic's Claude) means:
- Shared biases are less likely to go undetected
- Agreement rate is itself a useful metric
- Individual scores let users weight judges by their own trust preferences

### Why these judges?

Two principles. First, **no judge is also a model under test** — if a model graded itself the conflict of interest would be structural, so judges are drawn only from models that aren't on the leaderboard (an earlier panel violated this with Qwen3-235B and DeepSeek-V3 judging while also competing). Second, **lab diversity**: two open-weight judges from different labs plus one frontier reference means a disagreement signals genuine ambiguity rather than a shared blind spot. Open judges run through OpenRouter (OpenAI-compatible), so the full panel needs no GPU or self-hosted inference.

One same-lab pairing remains: the Claude Opus 4.6 judge shares a lab (Anthropic) with the Claude Sonnet 4.6 and Haiku 4.5 contestants. We don't claim this is eliminated. The mitigation is that the other two of three judges are from unrelated labs (Moonshot and Zhipu), so no Anthropic model can dominate consensus, and every per-judge score is published — anyone who wants a fully arm's-length view can recompute consensus with the same-lab judge excluded. The leaderboard does this recomputation itself: each Anthropic contestant's entry carries a `same_lab_check` block with task-completion and tool-selection means over the open judges only, plus the delta vs the full panel — so any same-lab inflation is a published number, not a hypothetical.

**Per-judge-vs-consensus deltas (every judge).** The same-lab check is a special case of a more general diagnostic: for *every* contestant and *every* judge on the panel, the leaderboard publishes that judge's mean task-completion / tool-selection for the model and its delta against the full-panel consensus (`judge_deltas` in each model entry; `delta = consensus_mean - judge_mean`, so a positive delta means the judge rates the model *lower* than its peers and a negative delta means the judge is more generous than the panel). This makes any judge's systematic favoritism toward a model — not only the same-lab pairing — a published number anyone can inspect.

### Why CLEAR weights of 35/25/20/20?

These weights reflect production priorities: efficacy matters most (a wrong answer at any speed is useless), reliability matters next (inconsistent agents can't be trusted), and cost/latency are important but secondary. Users can re-weight by using per-dimension scores directly.

### Anti-gaming: the do-nothing agent check

The [Berkeley RDI agentic-benchmark audit](https://arxiv.org/abs/2507.10325)
showed that trivial agents — ones that make no real tool calls and return
boilerplate — can game several published agentic benchmarks to near-perfect
scores, usually because the scoring rewards plausible-looking output rather than
verified task completion. A benchmark that can be gamed this way is not measuring
agent capability.

COT Bench includes a **do-nothing ("null") agent** as a standing sanity check
against exactly that failure mode. The null agent
([`eval/providers/null_agent.py`](../eval/providers/null_agent.py)) is a
deterministic contestant that makes **no tool calls** and returns a single
trivial deflecting message every turn (*"I'm not able to help with that right
now…"*). It requires no API key and spends nothing — that determinism is the
point: a do-nothing baseline should not move from run to run.

The expectation is that **the bench scores it near zero on both halves of
Efficacy**:

- **Deterministic state checks → 0.0.** Because the null agent never calls a
  tool, the canonical world is never mutated. Any v0.2 scenario that expects a
  state change (a transfer that should land, a recurring transfer that should be
  created, a fraud case that should be filed, an identity flag that should flip)
  scores 0.0 — every assertion fails against an unchanged world. This half is
  judge-independent and not gameable by fluent text. (The lone case where doing
  nothing is *correct* is a no-unauthorized-mutation scenario with an empty
  assertion list; there the null agent trivially "passes" precisely because it
  changed nothing — which is the desired contract, not a gamed score.)
- **LLM judges → floor.** The transcript shows no tools used and no goals
  resolved, so task-completion and tool-selection consensus land at the bottom of
  their rubrics.

The null agent is deliberately **not** in `MODELS_UNDER_TEST` and is excluded
from the published leaderboard, history, and bootstrap/rank-band math (see
`exclude_non_contestants` in
[`scripts/aggregate_results.py`](../scripts/aggregate_results.py)) — it is a
validation probe, not a ranked model, so it can never appear as a contestant. Run
it on demand with `python -m scripts.run_eval --include-null-agent` (or
`--models null-agent`). Publishing the result of a real judged run is a
methodology credential: evidence that a near-zero floor exists and that the
bench's scores reflect verified work, not the appearance of it.

## Limitations

- **Synthetic scenarios**: While carefully designed, synthetic conversations don't capture the full messiness of real-world interactions
- **LLM-as-judge**: Judge quality is bounded by the judge models' capabilities. We mitigate with multi-judge consensus but this is an inherent limitation
- **Tool simulation**: The tool simulator still generates the response *surface* with an LLM, so edge cases in real API behavior may not be perfectly represented. For v0.2 scenarios, however, the underlying *values* are pinned to a canonical `ground_truth` world the simulator must answer from and mutate, and the resulting state is verified deterministically — so balances, IDs, and records stay coherent across a run rather than being freely invented
- **Temperature 0.0**: Deterministic agent responses miss potential failure modes at higher temperatures
- **Domain coverage**: V1 covers 2 domains. Agent performance varies significantly by domain
- **Cost approximation**: Token costs use published pricing; actual costs may vary with volume discounts or cached tokens
- **Public corpus exposure**: The 92-scenario public corpus is on GitHub and could be trained on. The mitigation is the hash-pinned private holdout and the published public-vs-holdout gap (issue #31; see "Contamination and the private holdout" above), but the holdout caps overfitting *detection*, not its possibility

## Reproducing Results

All evaluation code, scenarios, and rubrics are open source. To reproduce:

```bash
git clone https://github.com/conorbronsdon/cot-bench.git
cd cot-bench && pip install -e .
python -m scripts.run_eval --domains banking --models "GPT-5.5"
```

Results may vary slightly due to:
- Non-determinism in user/tool simulators (temperature > 0)
- API-side variation even at temperature 0.0
- Judge model updates over time

We recommend comparing models within the same evaluation run rather than across runs.

**Model pinning.** Model IDs are pinned to dated snapshots wherever the provider publishes one (OpenAI models and simulators; Claude Haiku). Where no dated snapshot exists, the listed ID is the provider's canonical identifier — Anthropic's `claude-sonnet-4-6` and `claude-opus-4-6` have no dated variants, and Google's `gemini-2.5-pro`/`-flash` are the stable (non-preview) IDs. OpenRouter slugs pin the model but not the serving provider or quantization, so every results row and audit artifact also records the **resolved model** the provider reported actually serving — for the agent under test and for each judge call — making any drift between "requested" and "served" visible after the fact.

### What every run publishes (for audit)

A published score is only trustworthy if you can inspect the evidence behind it.
Each run persists exactly four things, and nothing is claimed that isn't on disk:

0. **Pre-registration (written before any model call).**
   `data/results/pre_registration.json` commits the run's definition — `run_id`
   and timestamp, the requested models, the domains and per-domain scenario IDs
   with a corpus-level `sha256` over the canonical serialized scenario set, the
   configured judge panel, the reliability-run count, and the seeds/temperatures
   (with the explicit caveat that the unseeded simulators make runs not
   bit-for-bit reproducible) — to disk before the first agent/simulator/judge
   call. The post-run manifest (item 3) links back to it by path and hash. See
   [governance.md §3](governance.md) for why this ordering matters.

1. **Per-evaluation artifacts.** For every `(scenario, model, run_index)` the run
   writes a JSON file to
   `data/results/artifacts/{run_id}/{model-slug}/{scenario_id}_run{run_index}.json`.
   Each file contains:
   - `transcript` — the full conversation: every `ConversationTurn` with its role,
     content, turn number, tool calls (name + arguments), tool results, and tool
     call ids, in true conversational order.
   - `judges` — for both `task_completion` and `tool_selection`, every judge's
     output: `judge_name`, `rubric_type`, `overall_score`, `reasoning`,
     `parse_failed`, and the raw parsed `raw_response`. Parse-failed judges are
     retained here for transparency (they are still excluded from consensus math).
   - `sim_meta` — `completed`, `total_turns`, token counts, `latency_ms`, and any
     simulation `error`.

   `run_id` is the stem of the results parquet, so a run's artifacts sit alongside
   the results they explain. Artifact persistence is **on by default**; pass
   `--no-artifacts` to disable it.

2. **Trace export (OpenInference → JSONL).** Agent turns and judge evaluations are
   emitted as OpenTelemetry spans carrying
   [OpenInference](https://github.com/Arize-ai/openinference) semantic-convention
   attributes (e.g. span kind `AGENT`/`EVALUATOR`, model name, scores). These are
   written to `data/results/traces/{run_id}/spans.jsonl` — one span per line — by a
   dependency-free file exporter built on the OpenTelemetry SDK. The JSONL can be
   loaded into [Arize Phoenix](https://github.com/Arize-ai/phoenix) (via its
   OTLP/file import) or any OTel/OpenInference reader. Set `COT_BENCH_TRACE_DIR` to
   write spans elsewhere. (Spans are exported to a file on disk; they are not held
   only in memory.)

3. **Run manifest (completion record).** `data/results/run_manifest.json` records
   the `run_id`, the models requested/completed/failed, domains, per-domain
   scenario counts, the reliability-run count, and the artifact/trace directories
   for the run. It also carries a `pre_registration` block linking back to the
   pre-registration (item 0) by path and `sha256` (plus the corpus hash), so the
   pre-registration and completion record are a verifiable pair.

All four are uploaded as workflow artifacts on every weekly evaluation run.

The policies governing how these runs are published, corrected, and versioned — no silent retraction or rerun, judge pinning, run pre-registration, contamination handling, and what triggers a benchmark version bump — are documented in [governance.md](governance.md).
