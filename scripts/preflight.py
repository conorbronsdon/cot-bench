"""Pre-flight check — verify everything is ready for an evaluation run.

Run this before your first eval to catch config, dependency, and API key
issues before spending money on API calls.
"""

import importlib
import os
import sys
from pathlib import Path


def check(name: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    return ok


def main():
    print("COT Bench Pre-flight Check\n")
    all_ok = True

    # 1. Module imports
    print("1. Module imports")
    modules = [
        "eval.config",
        "eval.tracing",
        "eval.scoring.rubrics",
        "eval.scoring.judge",
        "eval.simulation.runner",
        "eval.providers.registry",
        "scripts.run_eval",
        "scripts.aggregate_results",
    ]
    for mod in modules:
        try:
            importlib.import_module(mod)
            check(mod, True)
        except Exception as e:
            all_ok = check(mod, False, str(e)) and all_ok

    # 2. API keys
    print("\n2. API keys")
    key_checks = {
        "OPENAI_API_KEY": "Required — user/tool simulators + OpenAI models",
        "ANTHROPIC_API_KEY": "Required for Opus judge + Anthropic models",
        "GOOGLE_API_KEY": "Optional — only for Gemini models",
        "DEEPSEEK_API_KEY": "Optional — only for DeepSeek models via API",
        "TOGETHER_API_KEY": "Optional — only for Llama 4 Maverick",
        "MISTRAL_API_KEY": "Optional — only for Mistral Large",
    }
    for key, desc in key_checks.items():
        val = os.environ.get(key, "")
        is_set = len(val) > 5
        required = "Required" in desc
        if required:
            all_ok = check(key, is_set, desc if not is_set else "set") and all_ok
        else:
            check(key, True, f"{'set' if is_set else 'not set'} ({desc})")

    # 3. Scenarios
    print("\n3. Scenarios")
    scenario_dir = Path("data/scenarios")
    for domain in ["banking", "customer_success"]:
        domain_dir = scenario_dir / domain
        if domain_dir.exists():
            count = len(list(domain_dir.glob("*.json")))
            all_ok = check(f"{domain} scenarios", count > 0, f"{count} found") and all_ok
        else:
            all_ok = check(f"{domain} scenarios", False, "directory missing") and all_ok

    # 4. Scenario validity
    print("\n4. Scenario validation")
    from scripts.validate_scenarios import validate_scenario

    for path in sorted(scenario_dir.rglob("*.json")):
        errors = validate_scenario(path)
        name = str(path.relative_to(scenario_dir))
        if errors:
            all_ok = check(name, False, "; ".join(errors)) and all_ok
        else:
            check(name, True)

    # 5. Config integrity
    print("\n5. Config integrity")
    from eval.config import JUDGES, MODELS_UNDER_TEST, TOKEN_COSTS

    check("Models configured", True, f"{len(MODELS_UNDER_TEST)} models")
    check("Judges configured", True, f"{len(JUDGES)} judges")

    missing_costs = [m["model_id"] for m in MODELS_UNDER_TEST if m["model_id"] not in TOKEN_COSTS]
    all_ok = (
        check(
            "Token costs complete",
            len(missing_costs) == 0,
            f"missing: {missing_costs}" if missing_costs else "all models have pricing",
        )
        and all_ok
    )

    # 6. Quick API connectivity test
    print("\n6. API connectivity (quick check)")
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if len(openai_key) > 5:
        try:
            from openai import OpenAI

            client = OpenAI()
            client.models.list()
            check("OpenAI API", True, "connected")
        except Exception as e:
            all_ok = check("OpenAI API", False, str(e)[:80]) and all_ok
    else:
        all_ok = check("OpenAI API", False, "no key set") and all_ok

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if len(anthropic_key) > 5:
        try:
            import anthropic

            client = anthropic.Anthropic()
            # Just verify we can create a client — don't make a real call
            check("Anthropic API", True, "client created")
        except Exception as e:
            all_ok = check("Anthropic API", False, str(e)[:80]) and all_ok
    else:
        all_ok = check("Anthropic API", False, "no key set") and all_ok

    # Summary
    print(f"\n{'=' * 40}")
    if all_ok:
        print("All checks passed! Ready to run:")
        print()
        print("  # Quick test (1 model, 1 scenario, Opus judge only):")
        print("  python -m scripts.run_eval \\")
        print("    --domains banking \\")
        print('    --models "GPT-4.1-mini" \\')
        print("    --judges opus \\")
        print("    --scenario-limit 1 \\")
        print("    --reliability-runs 1")
        print()
        print("  # Full test (2 models, all scenarios, Opus judge):")
        print("  python -m scripts.run_eval \\")
        print('    --models "GPT-4.1-mini" "Claude Haiku 4.5" \\')
        print("    --judges opus")
    else:
        print("Some checks failed. Fix the issues above before running.")
        sys.exit(1)


if __name__ == "__main__":
    main()
