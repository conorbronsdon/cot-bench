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


# Provider name -> factory function
_REGISTRY: dict[str, Any] = {
    "openai": _create_openai,
    "anthropic": _create_anthropic,
    "google": _create_google,
    "openrouter": _create_openrouter,
    "deepseek": _create_deepseek,
    "qwen": _create_qwen,
    "together": _create_together,
    "mistral": _create_mistral,
}


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
