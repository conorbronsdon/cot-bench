# COT Bench

Open agent evaluation leaderboard with multi-judge scoring, CLEAR-aligned metrics, and full transparency.

From the makers of [Chain of Thought](https://www.yourpodcast.com).

## What makes COT Bench different

- **Multi-dimensional scoring** — not just accuracy. We measure Efficacy, Cost, Reliability, and Latency ([CLEAR framework](https://arxiv.org/abs/2511.14136)-aligned).
- **Vendor-neutral judging** — three independent judges (Qwen3-235B, DeepSeek-V3, Claude Opus) score every scenario. Open-source judges run on [Modular MAX](https://www.modular.com/max). All scores published.
- **Published rubrics** — our scoring criteria are in the code, not a black box. See [`eval/scoring/rubrics.py`](eval/scoring/rubrics.py).
- **Automated & fresh** — weekly evaluation runs keep results current as models update.
- **OpenInference traces** — every eval run emits traces compatible with [Arize Phoenix](https://github.com/Arize-ai/phoenix) and any OTel backend.

## Metrics

| Dimension | What it measures | How |
|-----------|-----------------|-----|
| **Efficacy** | Task completion + tool selection accuracy | Multi-judge LLM evaluation |
| **Cost** | Dollars per task | Token count × published pricing |
| **Reliability** | Consistency across runs | Pass@k over repeated evaluations |
| **Latency** | Speed of completion | Wall-clock time per scenario |

## Models evaluated

V1 targets 10 models across 2 domains (banking, customer success):

GPT-4.1, GPT-4.1-mini, Claude Sonnet 4.6, Claude Haiku 4.5, Gemini 2.5 Pro, Gemini 2.5 Flash, DeepSeek-V3, Qwen3-235B, Llama 4 Maverick, Mistral Large

## Quick start

```bash
# Install
pip install -e .

# Set API keys
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

# Start MAX judge servers (requires GPU)
python -m infra.max_serve

# Run evaluation
python -m scripts.run_eval --domains banking --models "GPT-4.1" "Claude Sonnet 4.6"
```

## Architecture

```
Simulation Loop (per scenario):
  User Simulator (GPT-4.1-mini) → Agent Under Test → Tool Simulator (GPT-4.1-mini)
  └── repeats up to 10 turns until goals met

Scoring (per scenario):
  Transcript → 3 independent judges → consensus score + agreement rate
  ├── Qwen3-235B    (via MAX, localhost:8010)
  ├── DeepSeek-V3   (via MAX, localhost:8011)
  └── Claude Opus   (via API)

Metrics:
  Efficacy  = 0.5 × task_completion + 0.5 × tool_selection (judge consensus)
  Cost      = input_tokens × $/M + output_tokens × $/M
  Reliability = pass@3 (3 runs per scenario, score consistency)
  Latency   = total wall-clock ms across all agent turns
```

## Project structure

```
cot-bench/
├── eval/
│   ├── config.py              # Domains, models, judges, costs
│   ├── tracing.py             # OpenInference trace emission
│   ├── scoring/
│   │   ├── rubrics.py         # Published evaluation rubrics
│   │   └── judge.py           # Multi-judge orchestration
│   ├── simulation/
│   │   └── runner.py          # Multi-turn conversation simulation
│   └── providers/
│       └── registry.py        # Model provider registry
├── data/
│   ├── domains/               # Domain configurations
│   ├── scenarios/             # Generated test scenarios
│   └── results/               # Evaluation results (parquet + csv)
├── infra/
│   └── max_serve.py           # MAX judge server management
├── scripts/
│   └── run_eval.py            # CLI entry point
└── frontend/                  # Leaderboard UI (coming soon)
```

## Methodology

COT Bench uses synthetic multi-turn conversations to evaluate agents. Each scenario has:
- A **persona** with specific communication style and context
- **5-8 interconnected goals** that require tool use
- **Domain-specific tools** with realistic schemas

The simulation runs the agent through the conversation, then three independent judges score the transcript on task completion and tool selection quality. Scenarios run 3× for reliability measurement.

Synthetic data generation approach adapted from [Galileo's agent-leaderboard](https://github.com/rungalileo/agent-leaderboard) (Apache 2.0).

## License

Apache 2.0
