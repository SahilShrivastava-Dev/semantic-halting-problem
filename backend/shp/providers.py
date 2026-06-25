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

from shp.config import (
    AGENT_LLM_TEMPERATURE,
    AGENT_LLM_MAX_TOKENS,
    EVAL_LLM_TEMPERATURE,
    EVAL_LLM_MAX_TOKENS,
    EVAL_LLM_MAX_RETRIES,
    PROVIDER_MODELS,
)
from shp.token_meter import METER

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Token-metering mixin (single chokepoint for ALL LLM traffic)
# ─────────────────────────────────────────────────────────────
def _record_result_usage(result) -> None:
    """Pull usage_metadata off every generation in a ChatResult into METER."""
    try:
        for gen in getattr(result, "generations", []) or []:
            msg = getattr(gen, "message", None)
            usage = getattr(msg, "usage_metadata", None)
            METER.record_usage(usage)
    except Exception:  # metering must never break a real run
        pass


class _MeteredMixin:
    """
    Records token usage after every (sync or async) generation. Mixed into the
    concrete chat classes below so Writer, Critic, and the RAGAS judge are all
    metered through one code path — the current role is set by METER.scope().
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        _record_result_usage(result)
        return result

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        _record_result_usage(result)
        return result


# ─────────────────────────────────────────────────────────────
# Groq n-parameter shim (Ragas compatibility) + metering
# ─────────────────────────────────────────────────────────────
class _FixedChatGroq(_MeteredMixin, ChatGroq):
    """
    ChatGroq subclass that (a) strips the ``n`` parameter Groq rejects (n > 1,
    which Ragas requests for AnswerRelevancy) and (b) meters token usage.
    """

    def _create_message_dicts(self, messages, stop):
        message_dicts, params = super()._create_message_dicts(messages, stop)
        params.pop("n", None)
        return message_dicts, params

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("n", None)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _MeteredChatGroq(_MeteredMixin, ChatGroq):
    """Plain metered ChatGroq for Writer/Critic (no n-stripping needed)."""


class _MeteredChatOpenAI(_MeteredMixin, ChatOpenAI):
    """Metered ChatOpenAI for both agents and judge."""


# ─────────────────────────────────────────────────────────────
# NVIDIA build endpoint (OpenAI-compatible) + metering
# ─────────────────────────────────────────────────────────────
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class _MeteredChatNVIDIA(_MeteredMixin, ChatOpenAI):
    """NVIDIA build endpoint via the OpenAI-compatible client; metered. Agents."""


class _FixedChatNVIDIA(_MeteredChatNVIDIA):
    """
    Judge variant that strips the ``n`` parameter — Ragas requests n=3 for
    AnswerRelevancy, which the NVIDIA OpenAI-compatible endpoint rejects.
    Stripped at the request-payload layer (langchain_openai >= 1.x hook).
    """

    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        payload.pop("n", None)
        return payload


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
        return _MeteredChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    if provider == "openai":
        api_key = _require_key("OPENAI_API_KEY", provider)
        logger.debug("Building ChatOpenAI model=%s", model)
        return _MeteredChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )

    if provider == "nvidia":
        api_key = _require_key("NVIDIA_API_KEY", provider)
        logger.debug("Building NVIDIA chat model=%s", model)
        return _MeteredChatNVIDIA(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=NVIDIA_BASE_URL,
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
        return LangchainLLMWrapper(_MeteredChatOpenAI(
            model=model,
            temperature=EVAL_LLM_TEMPERATURE,
            max_tokens=EVAL_LLM_MAX_TOKENS,
            api_key=api_key,
        ))

    if provider == "nvidia":
        api_key = _require_key("NVIDIA_API_KEY", provider)
        return LangchainLLMWrapper(_FixedChatNVIDIA(
            model=model,
            temperature=EVAL_LLM_TEMPERATURE,
            max_tokens=EVAL_LLM_MAX_TOKENS,
            api_key=api_key,
            base_url=NVIDIA_BASE_URL,
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
