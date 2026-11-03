"""REST endpoints. Each route is thin: validate input, call the real logic, return JSON.
The real logic - upload checks, pseudonyms, conversations, cancels - is in app/api/services.py.

    /api/health          - is the server up? (no API key needed)
    /api/status          - configuration summary (makes no calls to the LLM provider or LangSmith)

Conversations - used by the web UI. Each one has its own sandbox, kept until the conversation
is deleted or idle for SANDBOX_IDLE_TTL_SECONDS:

    POST   /api/threads                     - upload CSV files: starts a sandbox + agent
    GET    /api/threads/{thread_id}         - the conversation so far (restores it after a reload)
    POST   /api/threads/{thread_id}/messages
                                            - ask a question; the agent's steps stream back live
                                              as Server-Sent Events, ending with the answer
    POST   /api/threads/{thread_id}/stop    - stop the question being answered
    DELETE /api/threads/{thread_id}         - end the conversation and delete its sandbox
    GET    /api/threads/{thread_id}/tree    - the files in /data, /output and /skills
    GET    /api/threads/{thread_id}/file?path=/output/report.md
                                            - the raw content of one file in the sandbox

One-shot analysis - for API clients. A fresh sandbox that is deleted when the answer is ready:

    POST   /api/agent/analyze               - CSV files + a question -> answer + charts / report
    POST   /api/agent/cancel/{request_id}   - stop the analysis started with that request_id

Auth: when API_AUTH_KEY is set, everything except /health requires it in the X-API-Key header.

Errors are raised as AppError (app/exception/custom_exception.py) and turned into clean JSON by
the handlers in register_exception_handlers. While a question streams, they arrive as an
"error" event instead.
"""

import json
import platform
import secrets
import sys
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import datetime, timezone

from fastapi import APIRouter, Body, Depends, File, Form, Query, Request, Response, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from app.api import services
from app.api.schema import AnalyzeResponse, CancelResponse, ErrorResponse, HealthResponse, StatusResponse
from app.config.settings import settings
from app.exception.custom_exception import AppError

# auto_error=False so a missing key gives our JSON error, not FastAPI's 403
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Reject the request unless it carries API_AUTH_KEY. Does nothing when no key is configured."""
    if not settings.api_auth_key:
        return
    # compare_digest takes the same time for every wrong key, so it can't be guessed byte by byte
    if not api_key or not secrets.compare_digest(api_key.encode(), settings.api_auth_key.encode()):
        raise AppError(401, "Missing or invalid API key. Send it in the X-API-Key header.")


async def rate_limit(request: Request) -> None:
    """Allow RATE_LIMIT_PER_MINUTE analyses per client IP. async, so it runs on the event loop - no lock needed."""
    services.check_rate_limit(_client_id(request))


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# /health stays open so load balancers / the UI can check the server without a key
health_router = APIRouter(prefix="/api", tags=["Health"])
status_router = APIRouter(prefix="/api", tags=["Status"], dependencies=[Depends(require_api_key)])
thread_router = APIRouter(prefix="/api/threads", tags=["Conversations"], dependencies=[Depends(require_api_key)])
agent_router = APIRouter(prefix="/api/agent", tags=["One-shot analysis"], dependencies=[Depends(require_api_key)])

_server_start_time = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Health & status
# ---------------------------------------------------------------------------

@health_router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@status_router.get("/status", response_model=StatusResponse)
def system_status():
    """Return configuration and uptime. Never returns the API keys themselves."""
    uptime_seconds = int((datetime.now(timezone.utc) - _server_start_time).total_seconds())
    hours, remainder = divmod(uptime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return {
        "app": {
            "name": settings.app_name,
            "env": settings.app_env,
            "uptime": f"{hours:02d}h {minutes:02d}m {seconds:02d}s",
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
        },
        "agent": {
            "enabled": settings.enable_ai_agent,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "fallback": settings.llm_fallback.strip() or None,
            "api_key_set": settings.llm_api_key_set,
            "max_iterations": settings.max_agent_iterations,
            "max_question_chars": settings.max_question_chars,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
        },
        "sandbox": {
            "langsmith_api_key_set": bool(settings.langsmith_api_key),
            "max_upload_mb": settings.max_upload_mb,
            "max_upload_files": settings.max_upload_files,
        },
        "tracing": {
            "enabled": settings.langsmith_tracing,
            "project": settings.langsmith_project,
        },
    }


# ---------------------------------------------------------------------------
# Conversations (web UI)
# ---------------------------------------------------------------------------

@thread_router.post(
    "",
    responses={
        400: {"model": ErrorResponse, "description": "No files, too many files, or a bad file name"},
        413: {"model": ErrorResponse, "description": "File larger than MAX_UPLOAD_MB"},
        415: {"model": ErrorResponse, "description": "Not a text CSV file"},
        502: {"model": ErrorResponse, "description": "The sandbox could not be started"},
        503: {"model": ErrorResponse, "description": "Agent disabled, or too many open conversations"},
    },
)
async def create_thread(files: list[UploadFile] = File(...)):
    """Start a conversation: a sandbox with the CSV files in /data/ and an agent to ask about them."""
    thread_id, thread = await services.start_thread(files)
    return {"thread_id": thread_id, "sandbox_id": thread.backend.id, "data_files": thread.data_files}


@thread_router.get("/{thread_id}")
async def get_thread(thread_id: str):
    """The conversation so far, as the same events /messages streams, plus the questions and answers."""
    thread = await services.get_thread(thread_id)
    events = await services.thread_history(thread_id, thread)
    return {
        "thread_id": thread_id,
        "sandbox_id": thread.backend.id,
        "data_files": thread.data_files,
        "busy": thread.lock.locked(),
        "events": events,
    }


@thread_router.post(
    "/{thread_id}/messages",
    dependencies=[Depends(rate_limit)],
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "The agent's steps, then the answer"},
        400: {"model": ErrorResponse, "description": "Empty or too long question"},
        404: {"model": ErrorResponse, "description": "The conversation ended or expired"},
        409: {"model": ErrorResponse, "description": "The previous question is still being answered"},
        429: {"model": ErrorResponse, "description": "RATE_LIMIT_PER_MINUTE reached"},
    },
)
async def send_message(thread_id: str, question: str = Body(..., embed=True)):
    """Ask a question. The response is a stream of Server-Sent Events, one JSON object each:

        {"type": "note", "text"}                       - what the agent says while it works
        {"type": "tool_call", "id", "name", "args", "agent"}
        {"type": "tool_result", "id", "name", "ok", "output", "agent"}
                                                       - agent is null, or the subagent's name
        {"type": "answer", "text"}                     - last event when it succeeds
        {"type": "error", "code", "message"}           - last event when it fails
        {"type": "stopped"}                            - last event after /stop
    """
    events = await services.ask_question(thread_id, question)
    return StreamingResponse(
        _server_sent_events(events),
        media_type="text/event-stream",
        # no-transform / X-Accel-Buffering: proxies must pass every event on at once
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


async def _server_sent_events(events: AsyncGenerator[dict, None]) -> AsyncGenerator[str, None]:
    # aclosing: when this stream ends early (the browser disconnected), the events stream is
    # closed with it, which stops the agent's run
    async with aclosing(events):
        async for event in events:
            yield f"data: {json.dumps(event, default=str)}\n\n"


@thread_router.post("/{thread_id}/stop")
async def stop_thread(thread_id: str):
    """Stop the question being answered. The stream then ends with a "stopped" event."""
    return {"thread_id": thread_id, "status": await services.stop_question(thread_id)}


@thread_router.delete("/{thread_id}")
async def delete_thread(thread_id: str):
    """End the conversation and delete its sandbox."""
    await services.end_thread(thread_id)
    return {"thread_id": thread_id, "status": "deleted"}


@thread_router.get("/{thread_id}/tree")
async def get_tree(thread_id: str):
    """The files and folders in /data, /output and /skills of the conversation's sandbox."""
    thread = await services.get_thread(thread_id)
    entries = await services.thread_tree(thread_id, thread)
    return {"thread_id": thread_id, "sandbox_id": thread.backend.id, "entries": entries}


@thread_router.get(
    "/{thread_id}/file",
    response_class=Response,
    responses={
        200: {"content": {"application/octet-stream": {}}, "description": "The file's bytes"},
        404: {"model": ErrorResponse, "description": "No such file (or conversation)"},
        413: {"model": ErrorResponse, "description": "File too large to view"},
    },
)
async def get_file(thread_id: str, path: str = Query(..., max_length=1024)):
    """The raw content of one file in the sandbox (any absolute path)."""
    thread = await services.get_thread(thread_id)
    content = await services.thread_file(thread_id, thread, path)
    return Response(
        content,
        # Always a download type + CSP sandbox, so the browser never runs an HTML/SVG file the agent wrote
        media_type="application/octet-stream",
        headers={"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox", "Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------------
# One-shot analysis (API clients)
# ---------------------------------------------------------------------------

@agent_router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    # Documents the error JSON in /docs
    responses={
        400: {"model": ErrorResponse, "description": "Empty or too long question, bad upload, too many files"},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
        409: {"model": ErrorResponse, "description": "Stopped with /api/agent/cancel/{request_id}"},
        413: {"model": ErrorResponse, "description": "File larger than MAX_UPLOAD_MB"},
        415: {"model": ErrorResponse, "description": "Not a text CSV file"},
        422: {"model": ErrorResponse, "description": "Missing form fields, or refused by the model's safety filters"},
        429: {"model": ErrorResponse, "description": "RATE_LIMIT_PER_MINUTE reached"},
        500: {"model": ErrorResponse, "description": "Agent, sandbox or configuration error"},
        503: {"model": ErrorResponse, "description": "Agent disabled or the server is shutting down"},
    },
)
async def analyze(
    request: Request,
    question: str = Form(...),
    files: list[UploadFile] = File(...),
    request_id: str | None = Form(None),
    _: None = Depends(rate_limit),
):
    """Run the data-analysis agent on the uploaded CSV files.

    Each request gets its own sandbox, which is deleted afterwards, so every output file
    is downloaded and returned in the response (base64-encoded). Output files that could
    not be downloaded are listed in "warnings" instead of failing the whole request.

    Pass your own `request_id` (8-64 letters, digits, "-" or "_") to be able to stop the
    analysis with /api/agent/cancel/{request_id}; it then ends with ANALYSIS_CANCELLED (409).
    """
    return await services.analyze(question, files, request_id, client=_client_id(request))


@agent_router.post(
    "/cancel/{request_id}",
    response_model=CancelResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Malformed request_id"},
        401: {"model": ErrorResponse, "description": "Missing or invalid X-API-Key"},
    },
)
async def cancel_analysis(request_id: str):
    """Stop the analysis started with this request_id and delete its sandbox.

    "cancelled" means a running analysis was stopped. "not_running" means none is running
    with this id: if it arrives within the next few minutes it is refused, so a cancel sent
    while the files are still uploading still works.
    """
    return {"request_id": request_id, "status": services.cancel_analysis(request_id)}
