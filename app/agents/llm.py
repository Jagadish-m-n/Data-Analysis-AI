"""The chat models the agents run on: LLM_PROVIDER's model and the LLM_FALLBACK model.

The fallback is used through ModelFallbackMiddleware (see app/agents/middleware.py).
"""

from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from app.config.settings import settings
from app.exception.custom_exception import AppError
from app.logger.custom_logger import get_logger

logger = get_logger(__name__)

PROVIDERS = ("groq", "gemini")


def build_model() -> BaseChatModel:
    """The chat model chosen by LLM_PROVIDER."""
    if not settings.llm_api_key_set:
        raise AppError(
            500,
            f"{settings.llm_provider.upper()}_API_KEY is not set. Add it to your .env file.",
            code="MISSING_API_KEY",
        )
    return _chat_model(settings.llm_provider, settings.llm_model)


def build_fallback_model() -> BaseChatModel | None:
    """The model in LLM_FALLBACK ("provider:model", or "model" for LLM_PROVIDER), None if empty."""
    spec = settings.llm_fallback.strip()
    if not spec:
        return None
    provider, _, name = spec.rpartition(":")
    provider = provider or settings.llm_provider
    if provider not in PROVIDERS or not name:
        raise AppError(
            500,
            f"LLM_FALLBACK='{spec}' is not valid. Use 'provider:model' (provider groq or gemini) or just 'model'.",
            code="INVALID_FALLBACK",
        )
    if not _api_key(provider).strip():
        # Optional, so the agent still runs - just without a fallback
        logger.warning("llm_fallback_disabled", fallback=spec, reason=f"{provider.upper()}_API_KEY is not set")
        return None
    return _chat_model(provider, name)


def _chat_model(provider: str, name: str) -> BaseChatModel:
    if provider == "groq":
        return ChatGroq(
            model=name,
            api_key=settings.groq_api_key,
            timeout=settings.groq_timeout,
            # Only for reasoning models (gpt-oss, qwen3); empty = not sent. It is set for GROQ_MODEL,
            # so a Groq fallback (maybe llama, which rejects it) does not get it.
            reasoning_effort=(settings.groq_reasoning_effort.strip() or None) if name == settings.groq_model else None,
        )
    return ChatGoogleGenerativeAI(model=name, google_api_key=settings.gemini_api_key, timeout=settings.gemini_timeout)


def _api_key(provider: str) -> str:
    return settings.groq_api_key if provider == "groq" else settings.gemini_api_key
