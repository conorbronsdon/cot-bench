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
  <a href="https://github.com/conorbronsdon/cot-bench/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://github.com/conorbronsdon/cot-bench/blob/main/eval/scoring/rubrics.py"><img src="https://img.shields.io/badge/rubrics-published-brightgreen.svg" alt="Rubrics"></a>
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

The launch roster spans 11 models across 2 domains — 10 current frontier/
efficient/open-weight models plus one legacy cross-generation anchor (verified
against live provider sources 2026-06-10):

| Model | Provider | Category |
|-------|----------|----------|
| GPT-5.5 | OpenAI | Frontier closed |
| Gemini 3.1 Pro | Google | Frontier closed |
| GPT-5.4-mini | OpenAI | Efficient closed |
| Claude Sonnet 4.6 | Anthropic | Mid closed |
| Claude Haiku 4.5 | Anthropic | Efficient closed |
| Gemini 3.5 Flash | Google | Efficient closed |
| DeepSeek-V4 Pro | DeepSeek (via OpenRouter) | Open-weight |
| Qwen3.7-Max | Alibaba (via OpenRouter) | Open-weight |
| MiniMax M3 | MiniMax (via OpenRouter) | Open-weight |
| Mistral Large 3 | Mistral (via OpenRouter) | Open-weight |
| GPT-4.1 (anchor) | OpenAI | Legacy cross-gen anchor |

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
  --models "GPT-5.5" "Claude Sonnet 4.6" \
  --judges opus

# Full evaluation — all models, all three judges (no GPU; open judges via OpenRouter)
python -m scripts.run_eval

# Evaluate models in parallel (2 at a time)
python -m scripts.run_eval --parallel-models 2
```

All three judges run over hosted APIs — Kimi K2.6 and GLM-4.6 through OpenRouter, Opus through Anthropic — so the full consensus needs only API keys, no GPU.

Other `run_eval` options:

```bash
# Run an external private-holdout scenario set alongside the public corpus, so
# the board shows a public-vs-holdout efficacy gap (overfitting tripwire, #31).
# The holdout content is never stored in this repo; falls back to the
# COT_BENCH_HOLDOUT_DIR env var when the flag is omitted.
python -m scripts.run_eval --holdout-dir /path/to/private/scenarios

# Also run the deterministic do-nothing null agent (anti-gaming validation).
# It should score near zero; it never appears on the leaderboard.
python -m scripts.run_eval --include-null-agent

# Use the legacy two-call judge path (one call per rubric) instead of the
# default combined single-call path. Kept for A/B validation only.
python -m scripts.run_eval --separate-judge-calls
```

By default each run also writes a `pre_registration.json` (the run's frozen
config, model list, and scenario-corpus hash, committed *before* any model call)
and per-run artifacts; disable artifacts with `--no-artifacts`.

### Generate leaderboard

```bash
python -m scripts.aggregate_results
# Outputs: data/results/leaderboard.json + data/results/latest.csv
```

### Validate scenarios and calibrate judges

```bash
# Validate the scenario corpus against the schema and distribution bands
python -m scripts.validate_scenarios            # informational distribution check
python -m scripts.validate_scenarios --strict-distribution  # bands become failures

# Human judge-calibration (issue #33): sample a blind labeling workbook from a
# run's artifacts, then score filled-in human labels against the key. The key is
# written as a sibling of the workbook dir (workbook_key.json) so it can't be
# opened by accident while labeling.
python -m scripts.calibration sample --artifacts data/results/artifacts/<run_id> --out workbook/
python -m scripts.calibration score --workbook workbook/ --key workbook_key.json
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
│   ├── artifacts.py               # Per-run artifact (transcript + judge output) writer
│   ├── pre_registration.py        # Pre-run config + corpus-hash artifact (frozen before any call)
│   ├── scoring/
│   │   ├── rubrics.py             # Published evaluation rubrics (the IP)
│   │   ├── judge.py               # Concurrent multi-judge orchestration
│   │   ├── agreement.py           # Inter-judge agreement (Krippendorff alpha)
│   │   └── state_check.py         # Deterministic ground-truth state verification
│   ├── simulation/
│   │   └── runner.py              # Multi-turn conversation engine
│   └── providers/
│       ├── registry.py            # Config-driven model provider registry
│       └── null_agent.py          # Deterministic do-nothing agent (anti-gaming)
├── data/
│   ├── domains/                   # Domain tool + persona definitions
│   ├── scenarios/                 # Test scenarios (generated JSON)
│   └── results/                   # Evaluation outputs (parquet + csv + json)
│       ├── leaderboard.json       # Aggregated leaderboard (the frontend reads this)
│       ├── pre_registration.json  # Per-run frozen config + corpus hash
│       ├── run_manifest.json      # Per-run requested/completed/failed models + counts
│       ├── artifacts/             # Per-run transcripts + raw judge outputs (audit trail)
│       └── traces/                # OpenInference spans as JSONL (Phoenix-loadable)
├── scripts/
│   ├── run_eval.py                # Main evaluation CLI
│   ├── generate_data.py           # Synthetic scenario generation
│   ├── aggregate_results.py       # Leaderboard computation
│   ├── validate_scenarios.py      # Scenario schema + distribution validation
│   ├── calibration.py             # Human judge-calibration workbook + scoring
│   ├── preflight.py               # API-key / environment preflight check
│   └── first_run.sh               # Minimal end-to-end first-run script
├── frontend/
│   └── index.html                 # Leaderboard UI (GitHub Pages)
├── tests/                         # Tests covering scoring, parsing, config, validation
├── docs/                          # Detailed documentation
└── .github/workflows/             # CI + weekly eval + GitHub Pages deploy
```

## Documentation

- **[Methodology](docs/methodology.md)** — detailed explanation of evaluation approach, scoring rubrics, and statistical methods
- **[Governance](docs/governance.md)** — no-retraction, judge pinning, pre-registration, contamination, and versioning policy
- **[Contributing](docs/contributing.md)** — how to add models, domains, or improve the evaluation
- **[Roadmap](docs/roadmap.md)** — planned improvements and feature priorities
- **[Paper](paper/)** — arXiv-ready methods paper scaffold (LaTeX); methods sections drafted, results sections populate from the first eval run

## How It Works

1. **Scenario generation**: LLM-powered synthetic data creates realistic multi-turn conversations with 5-8 interconnected user goals per scenario, across diverse personas and difficulty levels.

2. **Simulation**: Each model runs through every scenario in a multi-turn loop. A user simulator drives the conversation, the model under test responds and makes tool calls, and a tool simulator returns realistic responses.

3. **Scoring**: Three independent judges evaluate the transcript against published rubrics for task completion and tool selection quality. Scores are averaged for consensus, with agreement rates published for transparency.

4. **Reliability**: Each scenario runs 3 times to measure consistency. A model that scores 0.9 once but 0.3 the next time is less useful than one that consistently scores 0.7.

5. **Aggregation**: All four CLEAR dimensions are normalized and weighted into a composite score, broken down by domain, category, and individual judge.

## Reproducibility & Audit

Every run publishes the evidence behind its scores, not just the numbers:

- **Pre-registration** — before any model or simulator call, `data/results/pre_registration.json` records the run's frozen config (models, judges, reliability runs, seeds, temperatures) and a SHA-256 hash of the public scenario corpus plus its scenario index. A private holdout (if run) is recorded as a hash and count only — never its scenario IDs. This commits to the evaluation setup up front so results can't be selectively reported after the fact.
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
