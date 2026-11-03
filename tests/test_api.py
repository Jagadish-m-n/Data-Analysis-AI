"""Tests for the REST API in app/api/.

Run with:  uv run pytest tests/test_api.py -v

The endpoint tests run offline: every request is rejected before a sandbox is created.
The last two tests call the LLM provider and LangSmith for real, so they check that the
keys in .env work. They are skipped when the key is not set.
"""

import pytest
from fastapi.testclient import TestClient

from app.agents.llm import build_model
from app.api import endpoints
from app.api.main import app
from app.config.settings import settings
from app.exception.custom_exception import AppError

CSV = b"region,amount\nNorth,10\n"


@pytest.fixture
def client(monkeypatch):
    # No X-API-Key needed, agent ready, no rate limit. Undone after each test.
    monkeypatch.setattr(settings, "api_auth_key", "")
    monkeypatch.setattr(settings, "enable_ai_agent", True)
    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setattr(settings, "groq_api_key", "test-groq-key")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    return TestClient(app)


def error_code(response) -> str:
    return response.json()["error"]["code"]


# ---------------------------------------------------------------------------
# Health, status and auth
# ---------------------------------------------------------------------------

def test_health(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_never_returns_api_keys(client):
    response = client.get("/api/status")

    assert response.status_code == 200
    assert response.json()["agent"]["api_key_set"] is True
    assert "test-groq-key" not in response.text


def test_api_key_is_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_auth_key", "test-auth-key")

    assert client.get("/api/status").status_code == 401
    assert client.get("/api/status", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/status", headers={"X-API-Key": "test-auth-key"}).status_code == 200
    # /health stays open
    assert client.get("/api/health").status_code == 200


# ---------------------------------------------------------------------------
# Upload and question checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "filename, content",
    [
        ("notes.txt", CSV),  # not .csv
        ("image.csv", b"\x89PNG\x00\x00"),  # named .csv but binary
    ],
)
def test_non_csv_uploads_are_rejected(client, filename, content):
    response = client.post("/api/threads", files=[("files", (filename, content, "text/csv"))])

    assert response.status_code == 415
    assert error_code(response) == "INVALID_FILE_TYPE"


def test_duplicate_file_names_are_rejected(client):
    files = [("files", ("sales.csv", CSV, "text/csv"))] * 2
    response = client.post("/api/threads", files=files)

    assert response.status_code == 400
    assert error_code(response) == "FILE_UPLOAD_ERROR"


def test_disabled_agent_is_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "enable_ai_agent", False)
    response = client.post("/api/threads", files=[("files", ("sales.csv", CSV, "text/csv"))])

    assert response.status_code == 503
    assert error_code(response) == "AGENT_DISABLED"


def test_analyze_rejects_empty_question(client):
    response = client.post(
        "/api/agent/analyze", data={"question": "   "}, files=[("files", ("sales.csv", CSV, "text/csv"))]
    )

    assert response.status_code == 400
    assert error_code(response) == "BAD_REQUEST"


def test_analyze_requires_question_and_files(client):
    response = client.post("/api/agent/analyze")

    assert response.status_code == 422
    assert error_code(response) == "INVALID_REQUEST"


@pytest.mark.parametrize("filename", ["../../sales.csv", r"C:\temp\sales.csv"])
def test_safe_filename_drops_folders(filename):
    assert endpoints._safe_filename(filename) == "sales.csv"


@pytest.mark.parametrize("filename", ["", ".", "..", "-option.csv", "bad\nname.csv"])
def test_safe_filename_rejects_unusable_names(filename):
    with pytest.raises(AppError):
        endpoints._safe_filename(filename)


def test_personal_id_columns_are_pseudonymized():
    content = b"customer_id,amount\nC9,10\nC7,20\nC9,30\n"

    [(name, result)], pseudonyms = endpoints._pseudonymize_id_columns([("sales.csv", content)])

    # Equal IDs get equal pseudonyms, other columns are untouched
    assert result == b"customer_id,amount\nANON_000001,10\nANON_000002,20\nANON_000001,30\n"
    # The mapping goes from pseudonym back to the real ID
    assert pseudonyms == {"ANON_000001": "C9", "ANON_000002": "C7"}


def test_ids_of_things_are_kept():
    content = b"product_id,order_id,units\nWIDGET-A,O1,5\n"

    [(name, result)], pseudonyms = endpoints._pseudonymize_id_columns([("sales.csv", content)])

    # Returned byte for byte, so answers can name the product
    assert result == content
    assert pseudonyms == {}


@pytest.mark.parametrize(
    "column",
    ["customer_id", "CustomerID", "Employee No", "user_uuid", "pan", "PAN Card No", "aadhaar_no", "customer_pan", "ssn"],
)
def test_personal_id_column_names(column):
    assert endpoints._is_personal_id_column(column)


@pytest.mark.parametrize(
    "column",
    ["product_id", "order_id", "id", "customer_zip_code", "customer_name", "pan_size", "Passport Country", "Japan"],
)
def test_other_column_names(column):
    assert not endpoints._is_personal_id_column(column)


def test_restore_ids_puts_real_ids_back():
    pseudonyms = {"ANON_000001": "C9", "ANON_000002": "C7"}

    assert endpoints._restore_ids("Top: ANON_000001, then ANON_000002", pseudonyms) == "Top: C9, then C7"
    # Unknown pseudonyms and text without any are left alone
    assert endpoints._restore_ids("ANON_000099 and plain text", pseudonyms) == "ANON_000099 and plain text"


def test_restore_file_only_changes_text_files():
    pseudonyms = {"ANON_000001": "C9"}

    assert endpoints._restore_file("report.md", b"Top customer: ANON_000001", pseudonyms) == b"Top customer: C9"
    # A chart can't be changed after it was drawn
    assert endpoints._restore_file("chart.png", b"ANON_000001", pseudonyms) == b"ANON_000001"


def test_restore_event_keeps_scripts_unchanged():
    pseudonyms = {"ANON_000001": "C9"}
    event = {"type": "tool_result", "output": "ANON_000001  1200", "args": {"command": "ANON_000001"}}

    restored = endpoints._restore_event(event, pseudonyms)

    assert restored["output"] == "C9  1200"
    assert restored["args"] == {"command": "ANON_000001"}


# ---------------------------------------------------------------------------
# Live API key checks (real network calls)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not settings.llm_api_key_set, reason="LLM API key is not set in .env")
def test_llm_api_key_works():
    response = build_model().invoke("Reply with exactly: API is working")

    assert response.text.strip()


@pytest.mark.skipif(not settings.langsmith_api_key, reason="LANGSMITH_API_KEY is not set in .env")
def test_langsmith_api_key_works():
    from langsmith import Client

    # Raises if the key is wrong or LangSmith can't be reached
    list(Client(api_key=settings.langsmith_api_key).list_projects(limit=1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))