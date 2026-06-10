"""Model provider registry — clean, config-driven model instantiation.

Replaces the if/elif chain pattern from the Galileo codebase with a
registry that maps provider names to client factories.
"""

import os
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel


@dataclass
class ModelSpec:
    """Specification for a model under test or simulator model."""

    name: str
    model_id: str
    provider: str
    temperature: float = 0.0
    max_tokens: int = 4096
    endpoint: str | None = None


def _create_openai(spec: ModelSpec) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": spec.model_id,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }
    if spec.endpoint:
        kwargs["base_url"] = spec.endpoint
    return ChatOpenAI(**kwargs)


def _create_anthropic(spec: ModelSpec) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model=spec.model_id,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
    )


def _create_google(spec: ModelSpec) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=spec.model_id,
        temperature=spec.temperature,
        max_output_tokens=spec.max_tokens,
    )


def _create_openrouter(spec: ModelSpec) -> BaseChatModel:
    """Open-weight models via OpenRouter — one OpenAI-compatible endpoint and
    one key for every open model (judges + open contestants). Routes across
    OpenRouter's neutral provider pool; we don't pin a single provider."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=spec.model_id,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        base_url=spec.endpoint or "https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )


def _create_deepseek(spec: ModelSpec) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=spec.model_id,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    )


def _create_qwen(spec: ModelSpec) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=spec.model_id,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.environ.get("QWEN_API_KEY", ""),
    )


def _create_together(spec: ModelSpec) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=spec.model_id,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        base_url="https://api.together.xyz/v1",
        api_key=os.environ.get("TOGETHER_API_KEY", ""),
    )


def _create_mistral(spec: ModelSpec) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=spec.model_id,
        temperature=spec.temperature,
        max_tokens=spec.max_tokens,
        base_url="https://api.mistral.ai/v1",
        api_key=os.environ.get("MISTRAL_API_KEY", ""),
    )


def _create_null(spec: ModelSpec) -> BaseChatModel:
    """Deterministic do-nothing agent — no API calls. Anti-gaming validation.

    Imported lazily so the null-agent module (and its langchain_core message
    imports) is only pulled in when actually requested.
    """
    from eval.providers.null_agent import create_null_agent

    return create_null_agent(spec)


# Provider name -> factory function
_REGISTRY: dict[str, Any] = {
    "openai": _create_openai,
    "null": _create_null,
    "anthropic": _create_anthropic,
    "google": _create_google,
    "openrouter": _create_openrouter,
    "deepseek": _create_deepseek,
    "qwen": _create_qwen,
    "together": _create_together,
    "mistral": _create_mistral,
}


def infer_provider(model_id: str) -> str:
    """Best-effort provider name for a bare model id, for CLI sim overrides (#50).

    The sim-model override flags take a model id but no provider; this maps the id
    to a registered provider so the override routes through the same registry as
    everything else. Rules, in order:

    - an explicit ``provider/slug`` form (a slash) -> ``openrouter`` (the
      OpenAI-compatible gateway every open-weight slug already uses);
    - ``claude*`` -> ``anthropic``;
    - ``gemini*`` -> ``google``;
    - ``gpt*`` / ``o1*`` / ``o3*`` / ``o4*`` -> ``openai``;
    - anything else -> ``openai`` (the historical sim default), so an unrecognized
      id behaves exactly as before rather than failing.

    A caller that needs a provider this heuristic gets wrong can always pass the
    model on a slug form or extend SimulationConfig directly.
    """
    mid = model_id.strip().lower()
    if "/" in mid:
        return "openrouter"
    if mid.startswith("claude"):
        return "anthropic"
    if mid.startswith("gemini"):
        return "google"
    if mid.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    return "openai"


def create_model(spec: ModelSpec) -> BaseChatModel:
    """Create a LangChain chat model from a ModelSpec.

    Args:
        spec: Model specification with provider, model_id, and settings.

    Returns:
        A LangChain BaseChatModel ready for invocation.

    Raises:
        ValueError: If the provider is not registered.
    """
    factory = _REGISTRY.get(spec.provider)
    if factory is None:
        raise ValueError(
            f"Unknown provider '{spec.provider}'. Registered providers: {list(_REGISTRY.keys())}"
        )
    return factory(spec)


def register_provider(name: str, factory: Any) -> None:
    """Register a custom provider factory.

    Args:
        name: Provider name to register.
        factory: Callable that takes a ModelSpec and returns a BaseChatModel.
    """
    _REGISTRY[name] = factory
