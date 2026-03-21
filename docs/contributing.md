# Contributing to COT Bench

We welcome contributions that make agent evaluation more comprehensive, fair, and useful. Here's how to help.

## Ways to Contribute

### Add a new model

1. Add the model config to `MODELS_UNDER_TEST` in [`eval/config.py`](../eval/config.py)
2. Add token pricing to `TOKEN_COSTS`
3. If the provider isn't supported, add a factory function in [`eval/providers/registry.py`](../eval/providers/registry.py)
4. Run the evaluation on at least one domain to verify it works
5. Open a PR with the config changes and a sample result

### Add a new domain

1. Add the domain enum value to `Domain` in [`eval/config.py`](../eval/config.py)
2. Add a domain config to `DOMAIN_CONFIGS` with system prompt, tool categories, and scenario categories
3. Generate tools and scenarios: `python -m scripts.generate_data --domain your_domain`
4. Review and hand-edit at least 2-3 scenarios for quality
5. Open a PR with the domain config and sample scenarios

### Improve evaluation rubrics

The rubrics in [`eval/scoring/rubrics.py`](../eval/scoring/rubrics.py) are the most impactful thing to improve. If you find scoring blind spots:

1. Describe the failure mode (what does a rubric miss?)
2. Propose specific rubric changes with reasoning
3. If possible, include a scenario transcript showing the issue
4. Open an issue first to discuss before submitting a PR

### Write scenarios by hand

Generated scenarios are good but hand-crafted ones are often better at targeting specific failure modes. We especially want:

- Scenarios that expose real production failures you've seen
- Edge cases that synthetic generation misses
- Scenarios in underrepresented difficulty levels

Format: see any JSON file in `data/scenarios/` for the schema.

## Development Setup

```bash
git clone https://github.com/conorbronsdon/cot-bench.git
cd cot-bench
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check eval/ scripts/ tests/

# Format code
ruff format eval/ scripts/ tests/
```

## Code Standards

- **Python 3.11+** with type hints
- **Ruff** for linting and formatting (config in `pyproject.toml`)
- **Tests required** for new functionality — add to `tests/`
- **No unnecessary dependencies** — check if an existing dep covers your need
- Keep scoring rubrics readable by non-engineers — they're the project's most public-facing code

## Pull Request Guidelines

- One concern per PR (don't mix a new model with a rubric change)
- Include test results if changing evaluation logic
- Update docs if changing user-facing behavior
- PRs that improve test coverage are always welcome

## Reporting Issues

- **Scoring seems wrong**: Include the scenario ID, model, and what you think the correct score should be
- **Model missing**: Open an issue with the model name, provider API, and why it should be included
- **Domain request**: Describe the domain and what failure modes it would test that existing domains don't

## Code of Conduct

Be constructive. This is an evaluation project — objectivity and fairness are core values. Advocacy for specific models or providers in scoring methodology discussions should be backed by evidence.
