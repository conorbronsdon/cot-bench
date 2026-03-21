# COT Bench

Open agent evaluation leaderboard. Part of the Chain of Thought brand.

## Architecture
- `eval/` — core evaluation pipeline (config, scoring, simulation, providers, tracing)
- `data/` — domains, scenarios (generated JSON), results (parquet/csv)
- `infra/` — MAX judge server management
- `scripts/` — CLI entry points (run_eval, generate_data, aggregate_results)
- `.github/workflows/` — weekly automated eval runs

## Key design decisions
- **Multi-judge scoring**: Qwen3-235B + DeepSeek-V3 (on Modular MAX) + Claude Opus (frontier reference)
- **CLEAR-aligned metrics**: Efficacy, Cost, Reliability, Latency — not just accuracy
- **OpenInference traces**: every eval emits OTel-compatible spans for Phoenix/Arize interop
- **Published rubrics**: scoring criteria in eval/scoring/rubrics.py, not a black box
- **Config-driven providers**: eval/providers/registry.py, no if/elif chains

## Running
```bash
pip install -e .
python -m scripts.run_eval --domains banking --models "GPT-4.1"
python -m scripts.aggregate_results
```

## Generating data
```bash
python -m scripts.generate_data --domain banking --scenarios-per-category 20
```

## Notes
- Token costs in eval/config.py — update when pricing changes
- MAX judges run on localhost:8010 (Qwen3) and localhost:8011 (DeepSeek)
- Sample scenarios in data/scenarios/ — hand-crafted for testing, generated at scale via scripts/generate_data.py
