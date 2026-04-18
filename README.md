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

From the [Chain of Thought](https://www.linkedin.com/in/conorbronsdon/) podcast. Open-source judge models served on [Modular MAX](https://www.modular.com/max).

## Why COT Bench?

| Problem with existing benchmarks | How COT Bench solves it |
|----------------------------------|------------------------|
| Only measure accuracy | **4 CLEAR dimensions**: Efficacy, Cost, Reliability, Latency |
| Single judge = single point of bias | **3 independent judges** (2 open-source + 1 frontier), all scores published |
| Scoring criteria are a black box | **Rubrics are code** — read them in [`eval/scoring/rubrics.py`](eval/scoring/rubrics.py) |
| Results go stale within weeks | **Automated weekly runs** via GitHub Actions |
| OpenAI judges OpenAI models | **Vendor-neutral**: open-source judges on [Modular MAX](https://www.modular.com/max) |
| No way to reproduce results | **OpenInference traces** for every run, compatible with [Arize Phoenix](https://github.com/Arize-ai/phoenix) |

## Latest Results

> Results will appear here after the first evaluation run. See the [live leaderboard](https://conorbronsdon.github.io/cot-bench) for interactive results.

| Rank | Model | CLEAR Score | Efficacy | $/Task | Reliability | Latency |
|------|-------|-------------|----------|--------|-------------|---------|
| — | *Evaluation pending* | — | — | — | — | — |

## Metrics

COT Bench aligns with the [CLEAR framework](https://arxiv.org/abs/2511.14136), which showed that accuracy-only evaluation correlates just 0.41 with production success, while multi-dimensional evaluation achieves 0.83.

| Dimension | Weight | What it measures | How |
|-----------|--------|-----------------|-----|
| **Efficacy** | 35% | Task completion + tool selection accuracy | Multi-judge LLM evaluation (consensus of 3 judges) |
| **Reliability** | 25% | Consistency across repeated runs | Pass@3 — same scenario run 3×, measuring score variance |
| **Cost** | 20% | Dollars per task | Token count × published per-token pricing |
| **Latency** | 20% | Speed of task completion | Wall-clock time across all agent turns |

**CLEAR Score** = weighted composite of all four dimensions, normalized across evaluated models. See [`scripts/aggregate_results.py`](scripts/aggregate_results.py) for the exact calculation.

## Judge Panel

Every scenario is scored independently by three judges. We publish all individual scores plus consensus and agreement rates.

| Judge | Type | Served via | Purpose |
|-------|------|------------|---------|
| **Qwen3-235B** | Open-source | [Modular MAX](https://www.modular.com/max) | Primary open judge |
| **DeepSeek-V3** | Open-source | [Modular MAX](https://www.modular.com/max) | Second open judge (different training paradigm) |
| **Claude Opus 4.6** | Frontier | Anthropic API | Reference judge for calibration |

Using two open-source judges from different training paradigms (Qwen from Alibaba, DeepSeek from DeepSeek AI) means disagreements surface genuine ambiguity rather than shared biases.

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
| DeepSeek-V3 | DeepSeek | Open-source |
| Qwen3-235B | Alibaba | Open-source |
| Llama 4 Maverick | Meta (via Together) | Open-source |
| Mistral Large | Mistral | Open-source |

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
# Evaluate specific models on one domain (frontier judge only — no GPU needed)
python -m scripts.run_eval \
  --domains banking \
  --models "GPT-4.1" "Claude Sonnet 4.6" \
  --judges opus

# Full evaluation with all judges (requires MAX + GPU for open-source judges)
python -m scripts.run_eval

# Evaluate models in parallel (2 at a time)
python -m scripts.run_eval --parallel-models 2
```

### Generate leaderboard

```bash
python -m scripts.aggregate_results
# Outputs: data/results/leaderboard.json + data/results/latest.csv
```

### Start MAX judge servers (for open-source judges)

```bash
# Requires GPU — see docs/max-setup.md for hardware requirements
pip install modular
max serve --model Qwen/Qwen3-235B --port 8010 &
max serve --model deepseek-ai/DeepSeek-V3-0324 --port 8011 &
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
│  │  Qwen3-235B  │  │ DeepSeek-V3  │  │ Claude Opus  │          │
│  │   (MAX)      │  │   (MAX)      │  │   (API)      │          │
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
├── scripts/
│   ├── run_eval.py               # Main evaluation CLI
│   ├── generate_data.py          # Synthetic scenario generation
│   └── aggregate_results.py      # Leaderboard computation
├── infra/
│   └── max_serve.py              # MAX judge server lifecycle management
├── frontend/
│   └── index.html                # Leaderboard UI (GitHub Pages)
├── tests/                         # 30 tests covering scoring, parsing, config
├── docs/                          # Detailed documentation
└── .github/workflows/             # CI + weekly eval + GitHub Pages deploy
```

## Documentation

- **[Methodology](docs/methodology.md)** — detailed explanation of evaluation approach, scoring rubrics, and statistical methods
- **[Contributing](docs/contributing.md)** — how to add models, domains, or improve the evaluation
- **[MAX Setup](docs/max-setup.md)** — hardware requirements and setup for running open-source judge models
- **[Roadmap](docs/roadmap.md)** — planned improvements and feature priorities

## How It Works

1. **Scenario generation**: LLM-powered synthetic data creates realistic multi-turn conversations with 5-8 interconnected user goals per scenario, across diverse personas and difficulty levels.

2. **Simulation**: Each model runs through every scenario in a multi-turn loop. A user simulator drives the conversation, the model under test responds and makes tool calls, and a tool simulator returns realistic responses.

3. **Scoring**: Three independent judges evaluate the transcript against published rubrics for task completion and tool selection quality. Scores are averaged for consensus, with agreement rates published for transparency.

4. **Reliability**: Each scenario runs 3 times to measure consistency. A model that scores 0.9 once but 0.3 the next time is less useful than one that consistently scores 0.7.

5. **Aggregation**: All four CLEAR dimensions are normalized and weighted into a composite score, broken down by domain, category, and individual judge.

## Acknowledgments

- Evaluation methodology inspired by [Galileo's agent-leaderboard](https://github.com/rungalileo/agent-leaderboard) (Apache 2.0)
- Metrics framework aligned with the [CLEAR paper](https://arxiv.org/abs/2511.14136) (Simmering et al., 2025)
- Open-source judge inference powered by [Modular MAX](https://www.modular.com/max)
- Trace format follows [OpenInference](https://github.com/Arize-ai/openinference) semantic conventions

---

## Disclaimer

*All views, opinions, and statements expressed on this account are solely my own and are made in my personal capacity. They do not reflect, and should not be construed as reflecting, the views, positions, or policies of Modular. This account is not affiliated with, authorized by, or endorsed by Modular in any way.*

## License

[Apache 2.0](LICENSE)

---

Built by [Conor Bronsdon](https://github.com/conorbronsdon) · [Chain of Thought](https://www.linkedin.com/in/conorbronsdon/)
