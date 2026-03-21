"""Tests for configuration integrity."""

import importlib

from eval.config import (
    DOMAIN_CONFIGS,
    JUDGES,
    MODELS_UNDER_TEST,
    TOKEN_COSTS,
    Domain,
    Metric,
)

# All modules that must import cleanly (catches broken dependencies)
_ALL_MODULES = [
    "eval.config",
    "eval.tracing",
    "eval.scoring.rubrics",
    "eval.scoring.judge",
    "eval.simulation.runner",
    "eval.providers.registry",
    "scripts.run_eval",
    "scripts.aggregate_results",
    "scripts.validate_scenarios",
    "scripts.generate_data",
]


class TestImports:
    def test_all_modules_import(self):
        for mod in _ALL_MODULES:
            try:
                importlib.import_module(mod)
            except Exception as e:
                raise AssertionError(f"Failed to import {mod}: {e}") from e


class TestConfig:
    def test_all_domains_have_configs(self):
        for domain in Domain:
            assert domain in DOMAIN_CONFIGS, f"Missing config for {domain}"

    def test_domain_configs_have_required_fields(self):
        required = ["description", "system_prompt", "tool_categories", "scenario_categories"]
        for domain, config in DOMAIN_CONFIGS.items():
            for field in required:
                assert field in config, f"{domain} missing '{field}'"

    def test_judges_configured(self):
        assert len(JUDGES) >= 3, "Need at least 3 judges"
        # Must have both MAX-served and API judges
        providers = {j.provider for j in JUDGES.values()}
        assert "max" in providers, "Need at least one MAX-served judge"
        assert "anthropic" in providers, "Need frontier reference judge"

    def test_models_have_costs(self):
        for model in MODELS_UNDER_TEST:
            model_id = model["model_id"]
            assert model_id in TOKEN_COSTS, f"Missing cost data for {model_id}"

    def test_token_costs_have_both_directions(self):
        for model_id, costs in TOKEN_COSTS.items():
            assert "input" in costs, f"{model_id} missing input cost"
            assert "output" in costs, f"{model_id} missing output cost"
            assert costs["input"] >= 0
            assert costs["output"] >= 0

    def test_clear_metrics_defined(self):
        expected = {"efficacy", "cost", "reliability", "latency"}
        actual = {m.value for m in Metric}
        assert expected == actual
