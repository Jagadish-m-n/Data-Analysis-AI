"""Response models for the REST endpoints (shown in /docs and used to validate output).

There is no request model for /api/agent/analyze: it takes multipart form data
(`question` + CSV `files`), which FastAPI reads with Form(...) / File(...).
"""

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class AppInfo(BaseModel):
    name: str
    env: str
    uptime: str = Field(examples=["01h 02m 03s"])
    python_version: str
    platform: str


class AgentInfo(BaseModel):
    enabled: bool
    provider: str
    model: str
    fallback: str | None  # LLM_FALLBACK, None = no fallback
    api_key_set: bool  # key of the main provider
    max_iterations: int
    max_question_chars: int
    rate_limit_per_minute: int  # 0 = no limit


class SandboxInfo(BaseModel):
    langsmith_api_key_set: bool
    max_upload_mb: int
    max_upload_files: int


class TracingInfo(BaseModel):
    enabled: bool
    project: str


class StatusResponse(BaseModel):
    app: AppInfo
    agent: AgentInfo
    sandbox: SandboxInfo
    tracing: TracingInfo


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class OutputFile(BaseModel):
    """A chart or report the agent saved in /output/ inside the sandbox."""

    name: str = Field(examples=["sales_by_region.png"])
    media_type: str = Field(examples=["image/png"])
    content_base64: str


class AnalyzeResponse(BaseModel):
    thread_id: str
    status: Literal["completed"] = "completed"
    answer: str = ""
    files: list[OutputFile] = []
    # Output files that could not be downloaded (the answer is still returned)
    warnings: list[str] = []


class CancelResponse(BaseModel):
    request_id: str
    # "cancelled" = a running analysis was stopped; "pending" = it hasn't started yet and will not run
    status: str = Field(examples=["cancelled"])


# ---------------------------------------------------------------------------
# Errors - the JSON shape of every AppException (see AppException.to_dict)
# ---------------------------------------------------------------------------

class ErrorDetail(BaseModel):
    code: str = Field(examples=["INVALID_FILE_TYPE"])
    message: str = Field(examples=["Only CSV files are supported."])
    details: dict = {}


class ErrorResponse(BaseModel):
    error: ErrorDetail
