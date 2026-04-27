"""
providers.py

LLM provider factory for the SHP pipeline.

Supports Groq (free-tier) and OpenAI. Returns a unified BaseChatModel
interface so every other module stays provider-agnostic.

Ragas compatibility note
------------------------
Groq's API rejects n > 1. Ragas internally requests n=3 for AnswerRelevancy.
get_ragas_llm() returns a _FixedChatGroq shim that silently strips this
parameter when the Groq provider is selected. OpenAI supports n natively.
"""

import os
import logging

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper

from config import (
    AGENT_LLM_TEMPERATURE,
    AGENT_LLM_MAX_TOKENS,
    EVAL_LLM_TEMPERATURE,
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_MAX_RETRIES,
    PROVIDER_MODELS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Groq n-parameter shim (Ragas compatibility)
# ─────────────────────────────────────────────────────────────
class _FixedChatGroq(ChatGroq):
    """
    Thin ChatGroq subclass that strips the ``n`` parameter before every
    API call. Groq rejects n > 1; this keeps all Ragas metrics functional.
    """

    def _create_message_dicts(self, messages, stop):
        message_dicts, params = super()._create_message_dicts(messages, stop)
        params.pop("n", None)
        return message_dicts, params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────
def get_llm(
    provider: str,
    model: str,
    temperature: float = AGENT_LLM_TEMPERATURE,
    max_tokens: int = AGENT_LLM_MAX_TOKENS,
) -> BaseChatModel:
    """
    Return a configured chat LLM for the given provider and model.

    Args:
        provider:    "groq" or "openai".
        model:       Model identifier string (must exist in PROVIDER_MODELS).
        temperature: Sampling temperature — 0.0 for deterministic outputs.
        max_tokens:  Maximum completion tokens.

    Returns:
        A Langchain BaseChatModel ready for .invoke() calls.

    Raises:
        ValueError:        Unknown provider.
        EnvironmentError:  Required API key missing from environment.
    """
    _validate(provider, model)

    if provider == "groq":
        api_key = _require_key("GROQ_API_KEY", provider)
        logger.debug("Building ChatGroq model=%s", model)
        return ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    if provider == "openai":
        api_key = _require_key("OPENAI_API_KEY", provider)
        logger.debug("Building ChatOpenAI model=%s", model)
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    raise ValueError(f"Unknown provider '{provider}'.")


def get_ragas_llm(provider: str, model: str) -> LangchainLLMWrapper:
    """
    Return a Ragas-compatible LangchainLLMWrapper for the given provider.

    For Groq:  wraps _FixedChatGroq to strip the n parameter.
    For OpenAI: wraps standard ChatOpenAI (supports n natively).
    """
    _validate(provider, model)

    if provider == "groq":
        api_key = _require_key("GROQ_API_KEY", provider)
        return LangchainLLMWrapper(_FixedChatGroq(
            model=model,
            temperature=EVAL_LLM_TEMPERATURE,
            max_tokens=EVAL_LLM_MAX_TOKENS,
            max_retries=EVAL_LLM_MAX_RETRIES,
            api_key=api_key,
        ))

    if provider == "openai":
        api_key = _require_key("OPENAI_API_KEY", provider)
        return LangchainLLMWrapper(ChatOpenAI(
            model=model,
            temperature=EVAL_LLM_TEMPERATURE,
            max_tokens=EVAL_LLM_MAX_TOKENS,
            api_key=api_key,
        ))

    raise ValueError(f"Unknown provider '{provider}'.")


def list_providers() -> dict[str, list[str]]:
    """Return the provider → model options map (for the frontend /api/models endpoint)."""
    return PROVIDER_MODELS


# ─────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────
def _require_key(env_var: str, provider: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise EnvironmentError(
            f"{env_var} is not set. This is required for the '{provider}' provider. "
            f"Add it to your .env file."
        )
    return key


def _validate(provider: str, model: str) -> None:
    if provider not in PROVIDER_MODELS:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported providers: {list(PROVIDER_MODELS)}"
        )
    if model not in PROVIDER_MODELS[provider]:
        raise ValueError(
            f"Model '{model}' is not available for provider '{provider}'. "
            f"Available: {PROVIDER_MODELS[provider]}"
        )
