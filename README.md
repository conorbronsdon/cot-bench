<p align="center">
  <h1 align="center">COT Bench</h1>
  <p align="center">
    Open agent evaluation leaderboard with multi-judge scoring, CLEAR-aligned metrics, and full transparency.
    <br />
    <a href="https://conorbronsdon.github.io/cot-bench">View Leaderboard</a> · <a href="docs/methodology.md">Methodology</a> · <a href="docs/contributing.md">Contribute</a>
  </p>
</p>

<p align="center">
  <a href="https://github.com/conorbronsdon/cot-bench/actions/workflows/ci.yml"><img src="https://github.com/conorbronsdon/cot-bench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/conorbronsdon/cot-bench/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/conorbronsdon/cot-bench/blob/master/eval/scoring/rubrics.py"><img src="https://img.shields.io/badge/rubrics-published-brightgreen.svg" alt="Rubrics"></a>
</p>

---

Most agent benchmarks measure one thing — accuracy — and publish results once. COT Bench measures what actually matters for production agents, stays fresh with automated weekly runs, and publishes every score from every judge so you can verify our work.

From the [Chain of Thought](https://chainofthought.show/) podcast. Open-weight judge models served via [OpenRouter](https://openrouter.ai/).

## Why COT Bench?

| Problem with existing benchmarks | How COT Bench solves it |
|----------------------------------|------------------------|
| Only measure accuracy | **4 CLEAR dimensions**: Efficacy, Cost, Reliability, Latency |
| Single judge = single point of bias | **3 independent judges** (2 open-source + 1 frontier), all scores published |
| Scoring criteria are a black box | **Rubrics are code** — read them in [`eval/scoring/rubrics.py`](eval/scoring/rubrics.py) |
| Results go stale within weeks | **Automated weekly runs** via GitHub Actions |
| OpenAI judges OpenAI models | **Vendor-neutral**: open-weight judges, none also under test (no self-grading) |
| No way to audit a published score | **Full artifacts per run** — every evaluation's transcript and raw judge outputs are saved to disk, plus OpenInference-attributed spans exported to JSONL ([loadable into Arize Phoenix](https://github.com/Arize-ai/phoenix)) |

## Latest Results

> Results will appear here after the first evaluation run. See the [live leaderboard](https://conorbronsdon.github.io/cot-bench) for interactive results. Every score publishes with a 95% bootstrap confidence interval, models whose intervals overlap share a rank band (their ordering is not statistically distinguishable), and the board does not publish below a 30-scenario-per-domain minimum.

| Rank | Model | CLEAR Score | Efficacy | $/Task | Reliability | Latency |
|------|-------|-------------|----------|--------|-------------|---------|
| — | *Evaluation pending* | — | — | — | — | — |

## Metrics

COT Bench aligns with the [CLEAR framework](https://arxiv.org/abs/2511.14136), which showed that accuracy-only evaluation correlates just 0.41 with production success, while multi-dimensional evaluation achieves 0.83.

| Dimension | Weight | What it measures | How |
|-----------|--------|-----------------|-----|
| **Efficacy** | 35% | Task completion + tool selection accuracy + state verification | Hybrid: multi-judge LLM consensus (3 judges) plus deterministic state verification against each scenario's ground-truth world (did the transfer actually happen?) |
| **Reliability** | 25% | Consistency across repeated runs | Pass@3 — same scenario run 3×, measuring score variance |
| **Cost** | 20% | Dollars per task | Token count × published per-token pricing |
| **Latency** | 20% | Speed of task completion | Wall-clock time across all agent turns |

**CLEAR Score** = weighted composite of all four dimensions, normalized across evaluated models. See [`scripts/aggregate_results.py`](scripts/aggregate_results.py) for the exact calculation.

## Judge Panel

Every scenario is scored independently by three judges. We publish all individual scores plus consensus and agreement rates.

| Judge | Type | Served via | Purpose |
|-------|------|------------|---------|
| **Kimi K2.6** | Open-weight | OpenRouter | Primary open judge |
| **GLM-4.6** | Open-weight | OpenRouter | Second open judge (different training paradigm) |
| **Claude Opus 4.6** | Frontier | Anthropic API | Reference judge for calibration |

Two open-weight judges from different labs (Kimi from Moonshot AI, GLM from Zhipu AI) mean disagreements surface genuine ambiguity rather than shared biases. **No judge is also a model under test** — judges never grade themselves. Open judges run through OpenRouter (OpenAI-compatible), so the full three-judge consensus needs no GPU or self-hosted inference.

## Models Evaluated

V1 targets 10 models across 2 domains:

| Model | Provider | Category |
|-------|----------|----------|
| GPT-4.1 | OpenAI | Frontier |
| GPT-4.1-mini | OpenAI | Efficient |
| Claude Sonnet 4.6 | Anthropic | Frontier |
| Claude Haiku 4.5 | Anthropic | Efficient |
| Gemini 2.5 Pro | Google | Frontier |
| Gemini 2.5 Flash | Google | Efficient |
| DeepSeek-V3 | DeepSeek (via OpenRouter) | Open-weight |
| Qwen3-235B | Alibaba (via OpenRouter) | Open-weight |
| Llama 4 Maverick | Meta (via OpenRouter) | Open-weight |
| Mistral Large | Mistral (via OpenRouter) | Open-weight |

## Domains

| Domain | Description | Why it's interesting |
|--------|-------------|---------------------|
| **Banking** | Account management, transactions, fraud detection, compliance | Structured, tool-heavy, strict correctness requirements |
| **Customer Success** | CRM, ticketing, health scoring, escalation, onboarding | Ambiguous goals, empathy matters, multi-system coordination |

Each domain has 5 scenario categories: adaptive tool use, scope management, empathetic resolution, extreme scenario recovery, and adversarial input mitigation.

## Quick Start

### Installation

```bash
git clone https://github.com/conorbronsdon/cot-bench.git
cd cot-bench
pip install -e ".[dev]"
```

### Set API keys

```bash
cp .env.example .env
# Edit .env with your API keys
source .env  # or use direnv/dotenv
```

### Generate evaluation scenarios

```bash
# Generate tools, personas, and scenarios for a domain
python -m scripts.generate_data --domain banking --scenarios-per-category 20
python -m scripts.generate_data --domain customer_success --scenarios-per-category 20
```

### Run evaluation

```bash
# Quick run — a subset of models on one domain (one frontier judge)
python -m scripts.run_eval \
  --domains banking \
  --models "GPT-4.1" "Claude Sonnet 4.6" \
  --judges opus

# Full evaluation — all models, all three judges (no GPU; open judges via OpenRouter)
python -m scripts.run_eval

# Evaluate models in parallel (2 at a time)
python -m scripts.run_eval --parallel-models 2
```

All three judges run over hosted APIs — Kimi K2.6 and GLM-4.6 through OpenRouter, Opus through Anthropic — so the full consensus needs only API keys, no GPU.

### Generate leaderboard

```bash
python -m scripts.aggregate_results
# Outputs: data/results/leaderboard.json + data/results/latest.csv
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Simulation Loop (per scenario, up to 10 turns)                  │
│                                                                  │
│  User Simulator ──→ Agent Under Test ──→ Tool Simulator         │
│  (GPT-4.1-mini)     (model being         (GPT-4.1-mini)        │
│                      evaluated)                                  │
│  ← ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ → │
│  Repeats until all user goals met or max turns reached          │
└─────────────────────────┬───────────────────────────────────────┘
                          │ transcript
┌─────────────────────────▼───────────────────────────────────────┐
│ Scoring (concurrent, per scenario)                               │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Kimi K2.6   │  │   GLM-4.6    │  │ Claude Opus  │          │
│  │ (OpenRouter) │  │ (OpenRouter) │  │   (API)      │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └──────────────────┼──────────────────┘                  │
│                    consensus score                                │
│                  + agreement rate                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
cot-bench/
├── eval/                          # Core evaluation library
│   ├── config.py                  # Domains, models, judges, token costs
│   ├── tracing.py                 # OpenInference/OTel trace emission
│   ├── scoring/
│   │   ├── rubrics.py             # Published evaluation rubrics (the IP)
│   │   └── judge.py              # Concurrent multi-judge orchestration
│   ├── simulation/
│   │   └── runner.py             # Multi-turn conversation engine
│   └── providers/
│       └── registry.py           # Config-driven model provider registry
├── data/
│   ├── domains/                   # Domain tool + persona definitions
│   ├── scenarios/                 # Test scenarios (generated JSON)
│   └── results/                   # Evaluation outputs (parquet + csv + json)
│       ├── artifacts/             # Per-run transcripts + raw judge outputs (audit trail)
│       └── traces/                # OpenInference spans as JSONL (Phoenix-loadable)
├── scripts/
│   ├── run_eval.py               # Main evaluation CLI
│   ├── generate_data.py          # Synthetic scenario generation
│   └── aggregate_results.py      # Leaderboard computation
├── frontend/
│   └── index.html                # Leaderboard UI (GitHub Pages)
├── tests/                         # Tests covering scoring, parsing, config, validation
├── docs/                          # Detailed documentation
└── .github/workflows/             # CI + weekly eval + GitHub Pages deploy
```

## Documentation

- **[Methodology](docs/methodology.md)** — detailed explanation of evaluation approach, scoring rubrics, and statistical methods
- **[Contributing](docs/contributing.md)** — how to add models, domains, or improve the evaluation
- **[Roadmap](docs/roadmap.md)** — planned improvements and feature priorities

## How It Works

1. **Scenario generation**: LLM-powered synthetic data creates realistic multi-turn conversations with 5-8 interconnected user goals per scenario, across diverse personas and difficulty levels.

2. **Simulation**: Each model runs through every scenario in a multi-turn loop. A user simulator drives the conversation, the model under test responds and makes tool calls, and a tool simulator returns realistic responses.

3. **Scoring**: Three independent judges evaluate the transcript against published rubrics for task completion and tool selection quality. Scores are averaged for consensus, with agreement rates published for transparency.

4. **Reliability**: Each scenario runs 3 times to measure consistency. A model that scores 0.9 once but 0.3 the next time is less useful than one that consistently scores 0.7.

5. **Aggregation**: All four CLEAR dimensions are normalized and weighted into a composite score, broken down by domain, category, and individual judge.

## Reproducibility & Audit

Every run publishes the evidence behind its scores, not just the numbers:

- **Per-evaluation artifacts** — for each `(scenario, model, run)`, a JSON file under `data/results/artifacts/{run_id}/{model-slug}/{scenario_id}_run{n}.json` contains the full conversation transcript (every turn, tool call, tool result, and call id) plus each judge's raw output for both rubrics (score, reasoning, `parse_failed` flag, and the raw parsed response). This lets anyone see *why* a published score is what it is. On by default; disable with `--no-artifacts`.
- **Trace export** — agent turns and judge evaluations are emitted as [OpenInference](https://github.com/Arize-ai/openinference)-attributed OpenTelemetry spans and written to `data/results/traces/{run_id}/spans.jsonl` (one span per line). The JSONL is loadable into [Arize Phoenix](https://github.com/Arize-ai/phoenix) or any OTel/OpenInference reader. Set `COT_BENCH_TRACE_DIR` to override the output location.
- **Run manifest** — `data/results/run_manifest.json` records the run id, which models were requested/completed/failed, domains, scenario counts, and the artifact/trace directories.

All of the above are uploaded as workflow artifacts on every weekly run.

## Acknowledgments

- Evaluation methodology inspired by [Galileo's agent-leaderboard](https://github.com/rungalileo/agent-leaderboard) (Apache 2.0)
- Metrics framework aligned with the [CLEAR paper](https://arxiv.org/abs/2511.14136) (Simmering et al., 2025)
- Open-weight judge inference served via [OpenRouter](https://openrouter.ai/)
- Trace format follows [OpenInference](https://github.com/Arize-ai/openinference) semantic conventions

---

## Disclaimer

*All views, opinions, and statements expressed on this account are solely my own and are made in my personal capacity. They do not reflect, and should not be construed as reflecting, the views, positions, or policies of Modular. This account is not affiliated with, authorized by, or endorsed by Modular in any way.*

## License

[Apache 2.0](LICENSE)

---

Built by [Conor Bronsdon](https://github.com/conorbronsdon) · [Chain of Thought](https://chainofthought.show/)
