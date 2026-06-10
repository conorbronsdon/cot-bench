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
5. The user simulator then evaluates whether all goals are met. If yes, the conversation ends. Otherwise the outer loop continues with a new user turn.

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

- **Consensus score**: Mean of all *valid* judge scores
- **Agreement rate**: Fraction of valid judge pairs within 0.2 of each other
- **Individual scores**: Every judge's score is published for transparency

When judges disagree significantly (>0.3 spread), this often indicates genuine ambiguity in the scenario — these cases are flagged in the results.

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

A model that scores 0.9, 0.85, 0.88 is more reliable than one that scores 0.95, 0.4, 0.8 — even though the latter has a higher peak.

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

### Why three judges instead of one?

A single judge introduces systematic bias. Using three judges from different labs (Moonshot's Kimi, Zhipu's GLM, Anthropic's Claude) means:
- Shared biases are less likely to go undetected
- Agreement rate is itself a useful metric
- Individual scores let users weight judges by their own trust preferences

### Why these judges?

Two principles. First, **no judge is also a model under test** — if a model graded itself the conflict of interest would be structural, so judges are drawn only from models that aren't on the leaderboard (an earlier panel violated this with Qwen3-235B and DeepSeek-V3 judging while also competing). Second, **lab diversity**: two open-weight judges from different labs plus one frontier reference means a disagreement signals genuine ambiguity rather than a shared blind spot. Open judges run through OpenRouter (OpenAI-compatible), so the full panel needs no GPU or self-hosted inference.

One same-lab pairing remains: the Claude Opus 4.6 judge shares a lab (Anthropic) with the Claude Sonnet 4.6 and Haiku 4.5 contestants. We don't claim this is eliminated. The mitigation is that the other two of three judges are from unrelated labs (Moonshot and Zhipu), so no Anthropic model can dominate consensus, and every per-judge score is published — anyone who wants a fully arm's-length view can recompute consensus with the same-lab judge excluded. The leaderboard does this recomputation itself: each Anthropic contestant's entry carries a `same_lab_check` block with task-completion and tool-selection means over the open judges only, plus the delta vs the full panel — so any same-lab inflation is a published number, not a hypothetical.

### Why CLEAR weights of 35/25/20/20?

These weights reflect production priorities: efficacy matters most (a wrong answer at any speed is useless), reliability matters next (inconsistent agents can't be trusted), and cost/latency are important but secondary. Users can re-weight by using per-dimension scores directly.

## Limitations

- **Synthetic scenarios**: While carefully designed, synthetic conversations don't capture the full messiness of real-world interactions
- **LLM-as-judge**: Judge quality is bounded by the judge models' capabilities. We mitigate with multi-judge consensus but this is an inherent limitation
- **Tool simulation**: The tool simulator still generates the response *surface* with an LLM, so edge cases in real API behavior may not be perfectly represented. For v0.2 scenarios, however, the underlying *values* are pinned to a canonical `ground_truth` world the simulator must answer from and mutate, and the resulting state is verified deterministically — so balances, IDs, and records stay coherent across a run rather than being freely invented
- **Temperature 0.0**: Deterministic agent responses miss potential failure modes at higher temperatures
- **Domain coverage**: V1 covers 2 domains. Agent performance varies significantly by domain
- **Cost approximation**: Token costs use published pricing; actual costs may vary with volume discounts or cached tokens

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
Each run persists exactly three things, and nothing is claimed that isn't on disk:

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

3. **Run manifest.** `data/results/run_manifest.json` records the `run_id`, the
   models requested/completed/failed, domains, per-domain scenario counts, the
   reliability-run count, and the artifact/trace directories for the run.

All three are uploaded as workflow artifacts on every weekly evaluation run.

The policies governing how these runs are published, corrected, and versioned — no silent retraction or rerun, judge pinning, run pre-registration, contamination handling, and what triggers a benchmark version bump — are documented in [governance.md](governance.md).
