"""Tests for app/config/settings.py.

Run with:  uv run pytest tests/test_config.py -v
"""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config.settings import Settings, export_tracing_env, settings


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


TRACING_VARS = ("LANGSMITH_TRACING", "LANGSMITH_PROJECT", "LANGSMITH_API_KEY")


@pytest.fixture
def clean_env():
    """The tracing variables removed from os.environ, and everything restored after the test."""
    with patch.dict(os.environ):
        for name in TRACING_VARS:
            os.environ.pop(name, None)
        yield


def test_export_tracing_env_copies_the_settings(clean_env):
    export_tracing_env(make_settings(langsmith_tracing=True, langsmith_project="proj", langsmith_api_key="key"))

    assert [os.environ[name] for name in TRACING_VARS] == ["true", "proj", "key"]


def test_export_tracing_env_keeps_existing_variables(clean_env):
    os.environ["LANGSMITH_PROJECT"] = "from-env"
    export_tracing_env(make_settings(langsmith_tracing=True, langsmith_project="proj"))

    assert os.environ["LANGSMITH_PROJECT"] == "from-env"
    assert "LANGSMITH_API_KEY" not in os.environ  # an empty key is not exported


def test_export_tracing_env_does_nothing_when_tracing_is_off(clean_env):
    export_tracing_env(make_settings(langsmith_tracing=False, langsmith_api_key="key"))

    assert not any(name in os.environ for name in TRACING_VARS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
