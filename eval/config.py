"""COT Bench configuration — domains, models, metrics, and judge settings."""

from dataclasses import dataclass
from enum import Enum


class Domain(str, Enum):
    BANKING = "banking"
    CUSTOMER_SUCCESS = "customer_success"


class Metric(str, Enum):
    # CLEAR-aligned dimensions
    EFFICACY = "efficacy"  # Task completion + tool selection accuracy
    COST = "cost"  # $/task (tokens * price)
    RELIABILITY = "reliability"  # Pass@k consistency across repeated runs
    LATENCY = "latency"  # Wall-clock seconds per scenario


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for a single judge model."""

    name: str
    model_id: str
    provider: str  # "openrouter" (open judges) or "anthropic" (frontier)
    endpoint: str | None = None  # Optional base_url override (else provider default)
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class SimulationConfig:
    """Settings for the multi-turn simulation loop."""

    max_turns: int = 10
    user_simulator_model: str = "gpt-4.1-mini-2025-04-14"
    tool_simulator_model: str = "gpt-4.1-mini-2025-04-14"
    user_simulator_temperature: float = 0.7
    tool_simulator_temperature: float = 0.0


# --- Judge Panel ---
# Multi-judge setup: 2 open-weight + 1 frontier reference.
#
# Judges are deliberately chosen from models that are NOT under test, so no
# model grades itself — the self-judging bias the multi-judge design exists to
# avoid. (An earlier panel used Qwen3-235B and DeepSeek-V3 as judges while both
# were also contestants.) Open judges are served through OpenRouter
# (OpenAI-compatible); no GPU or self-hosted inference required.

JUDGES = {
    "kimi": JudgeConfig(
        name="Kimi K2.6",
        model_id="moonshotai/kimi-k2.6",
        provider="openrouter",
    ),
    "glm": JudgeConfig(
        name="GLM-4.6",
        model_id="z-ai/glm-4.6",
        provider="openrouter",
    ),
    "opus": JudgeConfig(
        name="Claude Opus 4.6",
        model_id="claude-opus-4-6",
        provider="anthropic",
    ),
}


# --- Models Under Test ---
# Roster target: 10-12 current top models across frontier-closed, efficient-
# closed, and open-weight tiers.
#
# Pinning policy (reproducibility): pin to a dated snapshot wherever the
# provider publishes one. Where no dated snapshot exists, the listed ID is the
# provider's canonical (most-pinned-available) identifier.
# Verified against live provider sources on 2026-06-10:
#   - OpenAI: developers.openai.com/api/docs/models — dated snapshots exist and
#     are pinned below (gpt-5.5-2026-04-23, gpt-5.4-mini-2026-03-17).
#   - Anthropic: claude-sonnet-4-6 / claude-opus-4-8 have no dated snapshot —
#     the alias IS the canonical ID (appending a date 404s). Haiku has one and
#     is pinned. Pricing from the Anthropic models cache (Opus 4.8 $5/$25,
#     Sonnet 4.6 $3/$15, Haiku 4.5 $1/$5).
#   - Google: ai.google.dev/gemini-api/docs/models + /pricing. gemini-3.5-flash
#     is the stable current-gen Flash. The current-gen Pro is preview-only
#     (gemini-3.1-pro-preview) — Google has no GA Gemini 3 Pro yet, so the
#     preview ID is the only non-dated Gemini 3 Pro; shipping gemini-2.5-pro
#     instead would make the board look a generation behind.
#   - OpenRouter: openrouter.ai/api/v1/models. Slugs pin the model but NOT the
#     serving provider or quantization (routing varies per call). The resolved
#     upstream model is recorded per run in results/artifacts for audit. Every
#     open-weight slug below was confirmed to advertise `tools` in
#     supported_parameters (the bench uses native tool calling).

MODELS_UNDER_TEST = [
    # --- Frontier closed ---
    {"name": "GPT-5.5", "model_id": "gpt-5.5-2026-04-23", "provider": "openai"},
    {"name": "Claude Opus 4.8", "model_id": "claude-opus-4-8", "provider": "anthropic"},
    {"name": "Gemini 3.1 Pro", "model_id": "gemini-3.1-pro-preview", "provider": "google"},
    # --- Efficient / mid closed ---
    {"name": "GPT-5.4-mini", "model_id": "gpt-5.4-mini-2026-03-17", "provider": "openai"},
    {"name": "Claude Sonnet 4.6", "model_id": "claude-sonnet-4-6", "provider": "anthropic"},
    {"name": "Claude Haiku 4.5", "model_id": "claude-haiku-4-5-20251001", "provider": "anthropic"},
    {"name": "Gemini 3.5 Flash", "model_id": "gemini-3.5-flash", "provider": "google"},
    # --- Open-weight (via OpenRouter) ---
    # Routed through OpenRouter so one OPENROUTER_API_KEY covers them all (was
    # four separate provider keys: deepseek/qwen/together/mistral). GPT/Claude/
    # Gemini stay on their native APIs.
    {"name": "DeepSeek-V4 Pro", "model_id": "deepseek/deepseek-v4-pro", "provider": "openrouter"},
    {"name": "Qwen3.7-Max", "model_id": "qwen/qwen3.7-max", "provider": "openrouter"},
    {"name": "MiniMax M3", "model_id": "minimax/minimax-m3", "provider": "openrouter"},
    {
        "name": "Mistral Large 3",
        "model_id": "mistralai/mistral-large-2512",
        "provider": "openrouter",
    },
    # --- Legacy cross-generation anchor ---
    # One prior-generation model kept so the board shows how far the frontier
    # moved in ~14 months. NOTE: because GPT-4.1 is now a contestant, it can no
    # longer be a scenario author — the default AUTHOR_MODELS author was moved
    # off the gpt-4.1 family (see scripts/generate_data.GENERATION_MODEL).
    {"name": "GPT-4.1 (anchor)", "model_id": "gpt-4.1-2025-04-14", "provider": "openai"},
]


# --- Cost per million tokens (input/output) in USD ---
# Verified 2026-06-10 against the live provider sources listed in the pinning
# policy above. Used for the CLEAR Cost dimension.

TOKEN_COSTS = {
    # Frontier / efficient closed
    "gpt-5.5-2026-04-23": {"input": 5.00, "output": 30.00},
    "gpt-5.4-mini-2026-03-17": {"input": 0.75, "output": 4.50},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    # OpenRouter slugs + pricing (per million tokens, OpenRouter list rates).
    "deepseek/deepseek-v4-pro": {"input": 0.435, "output": 0.87},
    "qwen/qwen3.7-max": {"input": 1.25, "output": 3.75},
    "minimax/minimax-m3": {"input": 0.30, "output": 1.20},
    "mistralai/mistral-large-2512": {"input": 0.50, "output": 1.50},
    # Legacy anchor
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 8.00},
}


# --- Domain Configurations ---

DOMAIN_CONFIGS = {
    Domain.BANKING: {
        "description": "Financial services agent handling account management, transactions, compliance checks, fraud detection, and loan processing.",
        "system_prompt": (
            "You are a banking assistant agent. You help customers with account inquiries, "
            "transactions, loan applications, fraud alerts, and compliance-related questions. "
            "You have access to the bank's internal tools and must follow regulatory guidelines. "
            "Always verify customer identity before accessing account information. "
            "Never share sensitive financial data without proper authorization."
        ),
        "tool_categories": [
            "account_lookup",
            "transaction_processing",
            "loan_management",
            "fraud_detection",
            "compliance_check",
            "customer_verification",
        ],
        "scenario_categories": [
            "adaptive_tool_use",
            "scope_management",
            "empathetic_resolution",
            "extreme_scenario_recovery",
            "adversarial_input_mitigation",
        ],
    },
    Domain.CUSTOMER_SUCCESS: {
        "description": "Customer success agent managing onboarding, churn risk, product adoption, escalations, and cross-functional coordination across CRM, ticketing, and knowledge base systems.",
        "system_prompt": (
            "You are a customer success agent. You help manage customer relationships, "
            "monitor account health, handle escalations, drive product adoption, and coordinate "
            "across internal teams. You have access to CRM, ticketing, analytics, and knowledge "
            "base tools. Prioritize customer retention while balancing business objectives. "
            "Escalate to human agents when situations require judgment beyond your capabilities."
        ),
        "tool_categories": [
            "crm_management",
            "ticket_handling",
            "health_scoring",
            "usage_analytics",
            "knowledge_base",
            "escalation_routing",
            "onboarding_workflow",
        ],
        "scenario_categories": [
            "adaptive_tool_use",
            "scope_management",
            "empathetic_resolution",
            "extreme_scenario_recovery",
            "adversarial_input_mitigation",
        ],
    },
}


# --- Simulation defaults ---

DEFAULT_SIMULATION = SimulationConfig()

# Number of repeated runs for reliability measurement (pass@k)
RELIABILITY_RUNS = 3

# Number of scenarios per domain/category combination
SCENARIOS_PER_CATEGORY = 20

# Minimum number of scenarios per evaluated domain before a leaderboard may be
# published. Below this, model orderings are dominated by sampling noise rather
# than real capability differences: with only a handful of scenarios the
# bootstrap confidence intervals on every dimension overlap almost completely,
# so any ranking is indistinguishable from chance. The methodological review
# set this bar so the published board stays honest while the scenario set scales
# toward its ~80-scenario target. The publish gate (scripts/check_publish_ready.py)
# refuses to publish any run whose per-domain scenario count falls below this,
# unless --allow-partial is passed for a deliberate preview.
MIN_SCENARIOS_FOR_PUBLISH = 30
