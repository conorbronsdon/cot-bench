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

### 2. Simulation

The simulation runs for up to 10 **user turns**. Each user turn drives an inner
agent loop that lets the agent act on tool results *before* the user replies:

1. **User simulator** (GPT-4.1-mini, temperature 0.7) generates a message in-character for the persona, pursuing unmet goals
2. **Agent under test** (the model being evaluated, temperature 0.0) responds. Tools are provided through **native function calling** (LangChain `bind_tools`): each scenario tool definition is converted to an OpenAI-style JSON Schema function, and tool calls are read from the model's structured `tool_calls`. The agent is not asked to emit a bespoke JSON-in-text convention.
3. If the agent calls one or more tools, the **tool simulator** (GPT-4.1-mini, temperature 0.0) generates realistic responses conforming to each tool's schema. Results are returned to the agent as `ToolMessage`s, and **the agent is re-invoked** on those results.
4. Steps 2–3 repeat within the turn until the agent produces a user-facing message (no tool calls), or an inner cap of **5 tool rounds per user turn** is reached (a safeguard against runaway tool loops).
5. The user simulator then evaluates whether all goals are met. If yes, the conversation ends. Otherwise the outer loop continues with a new user turn.

This agent→tool→agent iteration means the model sees and reasons over tool output within the same turn, rather than only on the following turn. The transcript preserves true conversational order — user → agent (with tool calls) → tool results → agent follow-up → … → user — so judges read each tool call before its result.

Native tool calling (rather than a regex-parsed JSON-in-text protocol) measures real tool-calling ability and avoids penalizing models tuned for function-calling APIs. A content-embedded fallback parser exists only for providers that return no native tool calls; it logs a warning whenever it fires so its use is measurable.

Temperature 0.0 for the agent under test ensures reproducibility. The user simulator uses 0.7 for natural variation in conversation flow. Token and latency accounting cover every agent invocation, including inner tool-loop rounds; latency sums agent wall-times only (simulators excluded).

### 3. Scoring

After simulation, the full transcript is sent to three independent judges.

#### Task Completion (50% of Efficacy)

Each judge evaluates whether the agent accomplished the user's goals:

- **COMPLETE** (1.0): Goal fully addressed with correct tools and information
- **PARTIAL** (0.5): Meaningful progress but not fully resolved
- **FAILED** (0.0): Goal not addressed, wrong tools used, or incorrect information

Judges also assess: appropriate clarifying questions, graceful error recovery, scope awareness, and multi-step dependency handling. These factors adjust the score ±0.1.

The full rubric is in [`eval/scoring/rubrics.py`](../eval/scoring/rubrics.py).

#### Tool Selection Quality (50% of Efficacy)

Each judge evaluates every tool call on five dimensions:

1. **Selection correctness**: Was this the right tool?
2. **Parameter accuracy**: Were parameters correct and complete?
3. **Sequencing**: Were tools called in logical order?
4. **Necessity**: Was the call needed, or was it redundant?
5. **Omissions**: Were there tool calls the agent should have made?

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

One same-lab pairing remains: the Claude Opus 4.6 judge shares a lab (Anthropic) with the Claude Sonnet 4.6 and Haiku 4.5 contestants. We don't claim this is eliminated. The mitigation is that the other two of three judges are from unrelated labs (Moonshot and Zhipu), so no Anthropic model can dominate consensus, and every per-judge score is published — anyone who wants a fully arm's-length view can recompute consensus with the same-lab judge excluded.

### Why CLEAR weights of 35/25/20/20?

These weights reflect production priorities: efficacy matters most (a wrong answer at any speed is useless), reliability matters next (inconsistent agents can't be trusted), and cost/latency are important but secondary. Users can re-weight by using per-dimension scores directly.

## Limitations

- **Synthetic scenarios**: While carefully designed, synthetic conversations don't capture the full messiness of real-world interactions
- **LLM-as-judge**: Judge quality is bounded by the judge models' capabilities. We mitigate with multi-judge consensus but this is an inherent limitation
- **Tool simulation**: Tool responses are LLM-generated, not from real APIs. Edge cases in real API behavior may not be represented
- **Temperature 0.0**: Deterministic agent responses miss potential failure modes at higher temperatures
- **Domain coverage**: V1 covers 2 domains. Agent performance varies significantly by domain
- **Cost approximation**: Token costs use published pricing; actual costs may vary with volume discounts or cached tokens

## Reproducing Results

All evaluation code, scenarios, and rubrics are open source. To reproduce:

```bash
git clone https://github.com/conorbronsdon/cot-bench.git
cd cot-bench && pip install -e .
python -m scripts.run_eval --domains banking --models "GPT-4.1"
```

Results may vary slightly due to:
- Non-determinism in user/tool simulators (temperature > 0)
- API-side variation even at temperature 0.0
- Judge model updates over time

We recommend comparing models within the same evaluation run rather than across runs.
