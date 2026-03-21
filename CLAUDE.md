# COT Bench

Open agent evaluation leaderboard. Part of the Chain of Thought brand.

## Architecture
- `eval/` — core evaluation pipeline (config, scoring, simulation, providers, tracing)
- `data/` — domains, scenarios (generated JSON), results (parquet/csv)
- `infra/` — MAX judge server management
- `scripts/` — CLI entry points (run_eval, generate_data, aggregate_results, validate_scenarios)
- `docs/` — methodology, contributing guide, MAX setup guide
- `.github/workflows/` — CI, weekly eval runs, GitHub Pages deploy
- `frontend/` — static leaderboard UI (deployed to GitHub Pages)
- `tests/` — 38 tests covering scoring, parsing, config, validation

## Key design decisions
- **Multi-judge scoring**: Qwen3-235B + DeepSeek-V3 (on Modular MAX) + Claude Opus (frontier reference)
- **CLEAR-aligned metrics**: Efficacy (35%), Reliability (25%), Cost (20%), Latency (20%)
- **Concurrent execution**: judges run in parallel via ThreadPoolExecutor, models can run in parallel with --parallel-models
- **OpenInference traces**: every eval emits OTel-compatible spans for Phoenix/Arize interop
- **Published rubrics**: scoring criteria in eval/scoring/rubrics.py, not a black box
- **Config-driven providers**: eval/providers/registry.py, no if/elif chains

## Running
```bash
pip install -e ".[dev]"
python -m scripts.run_eval --domains banking --models "GPT-4.1" --judges opus
python -m scripts.aggregate_results
```

## Generating and validating data
```bash
python -m scripts.generate_data --domain banking --scenarios-per-category 20
python -m scripts.validate_scenarios
```

## Testing
```bash
pytest tests/ -v
ruff check eval/ scripts/ tests/
```

## Important files
- `eval/config.py` — token costs (update when pricing changes), model list, judge config
- `eval/scoring/rubrics.py` — the scoring rubrics (most important code in the repo)
- `eval/scoring/judge.py` — multi-judge orchestration with concurrent execution
- `eval/simulation/runner.py` — multi-turn conversation simulation engine
- MAX judges: localhost:8010 (Qwen3), localhost:8011 (DeepSeek)
