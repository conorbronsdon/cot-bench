"""COT Bench configuration — domains, models, metrics, and judge settings."""

from dataclasses import dataclass, field
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
    provider: str  # "max" for local MAX serving, "anthropic" for API
    endpoint: str | None = None  # Override for MAX-served models
    temperature: float = 0.0
    max_tokens: int = 4096


@dataclass(frozen=True)
class SimulationConfig:
    """Settings for the multi-turn simulation loop."""

    max_turns: int = 10
    user_simulator_model: str = "gpt-4.1-mini"
    tool_simulator_model: str = "gpt-4.1-mini"
    user_simulator_temperature: float = 0.7
    tool_simulator_temperature: float = 0.0


# --- Judge Panel ---
# Multi-judge setup: 2 open-source on MAX + 1 frontier reference

JUDGES = {
    "qwen3": JudgeConfig(
        name="Qwen3-235B",
        model_id="Qwen/Qwen3-235B",
        provider="max",
        endpoint="http://localhost:8010/v1",
    ),
    "deepseek": JudgeConfig(
        name="DeepSeek-V3",
        model_id="deepseek-ai/DeepSeek-V3-0324",
        provider="max",
        endpoint="http://localhost:8011/v1",
    ),
    "opus": JudgeConfig(
        name="Claude Opus 4.6",
        model_id="claude-opus-4-6",
        provider="anthropic",
    ),
}


# --- Models Under Test ---
# V1 target: 8-10 top models

MODELS_UNDER_TEST = [
    {"name": "GPT-4.1", "model_id": "gpt-4.1", "provider": "openai"},
    {"name": "GPT-4.1-mini", "model_id": "gpt-4.1-mini", "provider": "openai"},
    {"name": "Claude Sonnet 4.6", "model_id": "claude-sonnet-4-6", "provider": "anthropic"},
    {"name": "Claude Haiku 4.5", "model_id": "claude-haiku-4-5-20251001", "provider": "anthropic"},
    {"name": "Gemini 2.5 Pro", "model_id": "gemini-2.5-pro", "provider": "google"},
    {"name": "Gemini 2.5 Flash", "model_id": "gemini-2.5-flash", "provider": "google"},
    {"name": "DeepSeek-V3", "model_id": "deepseek-chat", "provider": "deepseek"},
    {"name": "Qwen3-235B", "model_id": "qwen3-235b", "provider": "qwen"},
    {"name": "Llama 4 Maverick", "model_id": "meta-llama/Llama-4-Maverick-17B-128E", "provider": "together"},
    {"name": "Mistral Large", "model_id": "mistral-large-latest", "provider": "mistral"},
]


# --- Cost per million tokens (input/output) in USD ---
# Updated March 2026. Used for CLEAR Cost dimension.

TOKEN_COSTS = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "qwen3-235b": {"input": 0.50, "output": 2.00},
    "meta-llama/Llama-4-Maverick-17B-128E": {"input": 0.27, "output": 0.85},
    "mistral-large-latest": {"input": 2.00, "output": 6.00},
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
