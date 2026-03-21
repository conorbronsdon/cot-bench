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

The simulation loop runs for up to 10 turns:

1. **User simulator** (GPT-4.1-mini, temperature 0.7) generates a message in-character for the persona, pursuing unmet goals
2. **Agent under test** (the model being evaluated, temperature 0.0) responds and optionally makes tool calls
3. **Tool simulator** (GPT-4.1-mini, temperature 0.0) generates realistic tool responses conforming to the tool's schema
4. The user simulator evaluates whether all goals are met. If yes, the conversation ends. Otherwise, loop continues.

Temperature 0.0 for the agent under test ensures reproducibility. The user simulator uses 0.7 for natural variation in conversation flow.

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

- **Consensus score**: Mean of all judge scores
- **Agreement rate**: Fraction of judge pairs within 0.2 of each other
- **Individual scores**: Every judge's score is published for transparency

When judges disagree significantly (>0.3 spread), this often indicates genuine ambiguity in the scenario — these cases are flagged in the results.

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

## Design Decisions

### Why synthetic scenarios instead of real conversations?

Real conversation data is hard to obtain, contains PII, and can't be shared openly. Synthetic scenarios give us:
- Full control over difficulty and coverage
- Reproducibility — same scenarios for every model
- Ground truth goals for scoring
- No privacy concerns

### Why three judges instead of one?

A single judge introduces systematic bias. Using three judges from different training paradigms (Qwen, DeepSeek, Claude) means:
- Shared biases are less likely to go undetected
- Agreement rate is itself a useful metric
- Individual scores let users weight judges by their own trust preferences

### Why open-source judges on MAX?

If GPT-4o judges GPT-4o, there's an inherent conflict of interest. Running open-source judges on neutral infrastructure (Modular MAX) eliminates vendor bias in scoring.

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
