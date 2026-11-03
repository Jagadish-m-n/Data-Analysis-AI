"""All configuration in one place.

Every setting is read from an environment variable or from the .env file in the project root.
Example: GROQ_MODEL=llama-3.3-70b-versatile in .env sets settings.groq_model.

Use it anywhere like this:

    from app.config.settings import settings
    print(settings.groq_model)
"""

import re
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The project folder (two levels above this file: app/config/settings.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    # pydantic-settings matches each field to the env variable with the same name, ignoring case
    # (groq_model <- GROQ_MODEL). Unknown variables in .env are ignored.
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    # --- Which LLM runs the agent ---
    llm_provider: Literal["groq", "gemini"] = "groq"
    # Model to switch to when the main one fails, as "provider:model",
    # e.g. "gemini:gemini-3.5-flash-lite". Empty = no fallback.
    llm_fallback: str = ""

    # --- Groq ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_timeout: int = 90  # seconds one call may take
    # Shorter reasoning = fewer tokens. Only for reasoning models (gpt-oss, qwen3); else leave empty
    groq_reasoning_effort: str = "low"

    # --- Gemini ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_timeout: int = 90  # seconds one call may take (Gemini needs at least 10)

    # --- LangSmith (runs the code sandbox) ---
    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = "data-analysis-ai"
    sandbox_create_timeout: int = 120  # seconds to wait for a new sandbox
    # "sse" sends commands over plain HTTPS - use it when a proxy/firewall blocks WebSockets
    sandbox_transport: Literal["websocket", "sse"] = "websocket"
    # Snapshot new sandboxes boot from, with pandas/numpy/matplotlib/seaborn/scipy pre-installed:
    # the id `uv run python -m scripts.build_snapshot` writes to .env, or "name:tag" / a bare name
    # (e.g. data-analysis:2026-11-02). Empty = the default image
    sandbox_snapshot: str = ""

    # --- Application ---
    app_name: str = "Data Analysis AI"
    app_env: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    # If set, clients must send it in the X-API-Key header. Empty = no login (fine on localhost)
    api_auth_key: str = ""

    # --- Limits ---
    enable_ai_agent: bool = True
    max_agent_iterations: int = 40  # model + tool calls before the agent is stopped
    context_max_tokens: int = 5000  # older messages are summarized above this size
    tool_output_max_tokens: int = 1000  # longer script output is saved to a file instead
    max_question_chars: int = 2000
    max_upload_files: int = 10
    max_upload_mb: int = 50
    rate_limit_per_minute: int = 5  # analyses per client IP per minute, 0 = no limit

    @field_validator("sandbox_snapshot")
    @classmethod
    def check_sandbox_snapshot(cls, value: str) -> str:
        """A snapshot UUID, "name:tag" or a bare name - the forms LangSmith accepts. Empty is fine."""
        value = value.strip()
        # No spaces, at most one ":" and something on both sides of it
        if value and not re.fullmatch(r"[^\s:]+(:[^\s:]+)?", value):
            raise ValueError(
                f"SANDBOX_SNAPSHOT={value!r} is not a snapshot: use the id from "
                "`uv run python -m scripts.build_snapshot`, a name:tag, a bare name, or leave it empty"
            )
        return value

    @property
    def llm_model(self) -> str:
        """The main model name."""
        return self.groq_model if self.llm_provider == "groq" else self.gemini_model

    @property
    def llm_api_key_set(self) -> bool:
        key = self.groq_api_key if self.llm_provider == "groq" else self.gemini_api_key
        return bool(key.strip())


settings = Settings()
