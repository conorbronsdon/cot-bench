# COT Bench Roadmap

Prioritized improvements informed by analysis of leading leaderboards (HuggingFace Open LLM Leaderboard, Chatbot Arena, PinchBench, BFCL, Artificial Analysis) and a deep code audit.

## v0.2 — Hardening & First Results

**Goal:** Run the first real evaluation and publish results.

- [ ] Generate scenarios at scale (20 per category × 5 categories × 2 domains = 200 scenarios)
- [ ] Run first evaluation with Opus-only judging (no GPU needed)
- [ ] Set up GPU access for MAX judge models (Lambda Labs / Vast.ai)
- [ ] Run full 3-judge evaluation
- [ ] Publish first leaderboard results
- [ ] Enable GitHub Pages deployment

## v0.3 — Evaluation Quality

**Goal:** Match the quality bar of BFCL and PinchBench.

- [ ] **Combined judge prompt**: score task completion + tool selection in a single judge call instead of two (50% reduction in judge API calls — currently 6 calls per scenario, could be 3)
- [ ] **Confidence intervals**: bootstrap over repeated runs to show uncertainty on scores (inspired by Chatbot Arena's Bradley-Terry model). Never show bare numbers without uncertainty.
- [ ] **Hallucination/refusal detection**: dedicated metric for when models call tools they shouldn't or refuse valid tool use (from BFCL)
- [ ] **Hybrid grading**: add deterministic checks alongside LLM judge — verify tool call JSON is valid, parameters match schema, required fields present (from PinchBench's atomic verifiable criteria)
- [ ] **Checkpoint/recovery**: save results incrementally per scenario so interrupted runs can resume instead of restarting from scratch

## v0.4 — Frontend & Engagement

**Goal:** Make the leaderboard a destination people return to.

- [ ] **Category-specific sub-leaderboards**: separate rankings by scenario type (adaptive tool use, scope management, etc.) — from Chatbot Arena and BFCL
- [ ] **Historical trend charts**: show how models improve across versions over time — a gap across all existing leaderboards
- [ ] **Model detail pages**: click a model to see per-domain, per-category, per-judge breakdown — from Artificial Analysis
- [ ] **Embeddable ranking badges**: `![Ranked #3 on COT Bench](badge-url)` for model providers to embed in their READMEs — free marketing, no existing leaderboard does this well
- [ ] **Data download**: results as downloadable CSV/JSON + HuggingFace Dataset for programmatic access (from HF Open LLM Leaderboard)
- [ ] **Changelog page**: dedicated log of every methodology change, model addition, and scoring update (from Chatbot Arena)

## v0.5 — Community & Scale

**Goal:** Enable community contributions and scale to 20+ models.

- [ ] **Model submission pipeline**: GitHub PR-based model submission with automated evaluation (start with BFCL's Tier 2 approach, evolve toward HF's self-service)
- [ ] **pip-installable eval harness**: `pip install cot-bench` + `cot-bench run --model my-model` for local reproducibility (from BFCL — this is their gold standard for reproducibility)
- [ ] **New domains**: healthcare, e-commerce, IT support (community-driven)
- [ ] **Provider comparison**: same agent framework, different LLM backends — who performs best? (from Artificial Analysis)
- [ ] **Community voting**: let users vote on which models to evaluate next (from HF Open LLM Leaderboard)

## v1.0 — Production Grade

**Goal:** Become the reference agent evaluation leaderboard.

- [ ] **Academic paper**: detailed methodology writeup for credibility (from BFCL — published at OpenReview)
- [ ] **Progressive difficulty tiers**: simple tool calls → parallel → multi-turn → full agentic (from BFCL v1-v4 evolution)
- [ ] **Rolling live measurement**: 72-hour windows for speed/latency metrics with real-time updates (from Artificial Analysis)
- [ ] **Self-reported data labeling**: flag when metrics are provider-reported vs independently measured (from Artificial Analysis)
- [ ] **Multi-agent evaluation**: test agents that delegate to sub-agents

## Architecture Improvements (Ongoing)

From the code audit — address as capacity allows:

- [ ] **Environment-based config**: move operational config (endpoints, timeouts, worker counts) to environment variables with defaults. Keep business logic (rubrics, metric weights) in code.
- [ ] **Rate limiting**: add exponential backoff and queue management for API calls — current parallel execution has no rate limit strategy
- [ ] **Scenario versioning**: compute SHA256 of scenario JSON, include in results for full reproducibility
- [ ] **Result idempotency**: detect and skip already-evaluated scenario/model/run combinations
- [ ] **Stronger type hints**: replace generic `dict` for tool definitions, personas, scenarios with proper dataclass/Pydantic types throughout (currently only in generate_data.py)
