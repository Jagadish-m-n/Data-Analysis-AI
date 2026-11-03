"""Tests for app/config/settings.py.

Run with:  uv run pytest tests/test_config.py -v
"""

import pytest
from pydantic import ValidationError

from app.config.settings import Settings, settings


def make_settings(**values) -> Settings:
    """Settings built from the given values only - the .env file is not read."""
    return Settings(_env_file=None, **values)


def test_settings_load_from_env():
    assert settings.llm_provider in ("groq", "gemini")
    assert settings.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    assert settings.max_agent_iterations > 0
    assert settings.max_upload_files > 0
    assert settings.max_upload_mb > 0


def test_llm_model_follows_provider():
    assert make_settings(llm_provider="groq", groq_model="groq-model").llm_model == "groq-model"
    assert make_settings(llm_provider="gemini", gemini_model="gemini-model").llm_model == "gemini-model"


def test_llm_api_key_set_checks_the_main_provider():
    assert make_settings(llm_provider="groq", groq_api_key="abc").llm_api_key_set
    assert not make_settings(llm_provider="groq", groq_api_key="   ").llm_api_key_set
    # A Gemini key does not count when Groq is the provider
    assert not make_settings(llm_provider="groq", groq_api_key="", gemini_api_key="abc").llm_api_key_set


def test_invalid_values_are_rejected():
    with pytest.raises(ValidationError):
        make_settings(llm_provider="openai")
    with pytest.raises(ValidationError):
        make_settings(log_level="VERBOSE")


@pytest.mark.parametrize("snapshot", ["", "data-analysis", "data-analysis:2026-11-02", "  name:tag  "])
def test_sandbox_snapshot_accepts_valid_values(snapshot):
    assert make_settings(sandbox_snapshot=snapshot).sandbox_snapshot == snapshot.strip()


@pytest.mark.parametrize("snapshot", ["bad name", "a:b:c", ":tag", "name:"])
def test_sandbox_snapshot_rejects_invalid_values(snapshot):
    with pytest.raises(ValidationError):
        make_settings(sandbox_snapshot=snapshot)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
