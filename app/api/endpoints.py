"""REST endpoints. Each route is thin: validate input, call the real logic, return JSON.

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

import asyncio
import base64
import codecs
import csv
import io
import json
import math
import platform
import posixpath
import re
import secrets
import sys
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from deepagents.backends import LangSmithSandbox
from fastapi import APIRouter, Body, Depends, File, Form, Query, Request, Response, Security, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from app.agents.agent import arun_agent, build_agent
from app.api.schema import AnalyzeResponse, CancelResponse, ErrorResponse, HealthResponse, StatusResponse
from app.config.settings import settings
from app.exception.custom_exception import AppError
from app.logger.custom_logger import get_logger
from app.sandbox.sandboxbackend import (
    SANDBOX_IDLE_TTL_SECONDS,
    create_sandbox,
    delete_sandbox,
    download_output_files,
    list_tree,
    output_type,
    read_file,
    upload_data_files,
    upload_skills,
)

logger = get_logger(__name__)

# auto_error=False so a missing key gives our JSON error, not FastAPI's 403
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str | None = Security(_api_key_header)) -> None:
    """Reject the request unless it carries API_AUTH_KEY. Does nothing when no key is configured."""
    if not settings.api_auth_key:
        return
    # compare_digest takes the same time for every wrong key, so it can't be guessed byte by byte
    if not api_key or not secrets.compare_digest(api_key.encode(), settings.api_auth_key.encode()):
        raise AppError(401, "Missing or invalid API key. Send it in the X-API-Key header.")


# client -> start times of their analyses in the last minute (in memory: per process, reset on restart)
_recent_requests: dict[str, deque[float]] = defaultdict(deque)
_RATE_WINDOW_SECONDS = 60


async def rate_limit(request: Request) -> None:
    """Allow RATE_LIMIT_PER_MINUTE analyses per client IP. async, so it runs on the event loop - no lock needed."""
    limit = settings.rate_limit_per_minute
    if limit == 0:
        return
    client = _client_id(request)
    now = time.monotonic()
    starts = _recent_requests[client]
    while starts and now - starts[0] >= _RATE_WINDOW_SECONDS:
        starts.popleft()
    if len(starts) >= limit:
        retry_after = max(1, math.ceil(_RATE_WINDOW_SECONDS - (now - starts[0])))
        logger.warning("rate_limit_exceeded", client=client, limit=limit)
        raise AppError(
            429,
            f"Too many questions: the limit is {limit} per minute. Try again in {retry_after} seconds.",
            code="RATE_LIMIT_EXCEEDED",
        )
    starts.append(now)
    # Drop clients with no recent requests, so the dict can't grow forever
    for key in [k for k, v in _recent_requests.items() if not v or now - v[-1] >= _RATE_WINDOW_SECONDS]:
        del _recent_requests[key]


def _client_id(request: Request) -> str:
    return request.client.host if request.client else "unknown"


# /health stays open so load balancers / the UI can check the server without a key
health_router = APIRouter(prefix="/api", tags=["Health"])
status_router = APIRouter(prefix="/api", tags=["Status"], dependencies=[Depends(require_api_key)])
thread_router = APIRouter(prefix="/api/threads", tags=["Conversations"], dependencies=[Depends(require_api_key)])
agent_router = APIRouter(prefix="/api/agent", tags=["One-shot analysis"], dependencies=[Depends(require_api_key)])

_server_start_time = datetime.now(timezone.utc)

# Bytes inspected to decide whether an upload is text (a CSV) or binary
_SNIFF_BYTES = 64 * 1024
# Columns that identify a PERSON are pseudonymized at upload (see _is_personal_id_column).
# IDs of things - product_id, order_id, store_id - are left alone, so answers can name them.
# 1. A person word followed by an ID word: customer_id, CustomerID, "Employee No", user_uuid
_PERSON_WORDS = {
    "customer", "cust", "client", "user", "member", "employee", "emp", "staff", "patient",
    "student", "person", "contact", "account", "acct", "subscriber", "applicant",
}
_ID_WORDS = {"id", "uid", "uuid", "guid", "no", "num", "number", "code"}
# 2. A national ID number, alone or followed only by ID words: pan, pan_card_no, aadhaar_no, ssn,
#    passport_number, customer_pan. Not pan_size or passport_country.
_NATIONAL_ID_WORDS = {"aadhaar", "aadhar", "pan", "ssn", "passport"}
# What a pseudonym looks like. A rare prefix, so a real value in the data is never mistaken for one.
_PSEUDONYM_PREFIX = "ANON_"
_PSEUDONYM_RE = re.compile(r"\bANON_\d{6,}\b")
# Text files whose pseudonyms are turned back into real IDs before the user sees them
_TEXT_FILE_TYPES = {".csv", ".md", ".txt", ".json", ".py", ".log"}
# Cells pandas reads as missing keep their value, so the agent's missing-value counts stay right
_MISSING_CELLS = {"", "na", "n/a", "nan", "null", "none", "<na>", "#n/a"}
# One CSV cell may be as big as a whole upload (the csv module's default limit is 128 KB)
csv.field_size_limit(settings.max_upload_mb * 1024 * 1024)
# How much of each question goes into the logs (enough to audit, not the whole thing)
_QUESTION_LOG_CHARS = 200
# Finish reasons that mean "blocked by safety filters" (Gemini; content_filter = OpenAI-style APIs)
_BLOCKED_FINISH_REASONS = {
    "SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "RECITATION", "IMAGE_SAFETY", "content_filter"
}
# Added to every question, so the user never has to type a path (same text as arun_agent adds)
_DATA_FILES_NOTE = "\n\n(Uploaded data files: "


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

@dataclass
class _Thread:
    """One conversation: its sandbox and agent stay alive between questions."""

    backend: LangSmithSandbox
    agent: CompiledStateGraph
    data_files: list[str]
    # pseudonym -> real ID (ANON_000001 -> "C9"). Kept only in this server's memory, never sent to
    # the model, and gone when the conversation ends.
    pseudonyms: dict[str, str] = field(default_factory=dict)
    last_used: float = field(default_factory=time.monotonic)
    # Held while a question is answered: one question at a time per conversation
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # The run answering the current question, so /stop can cancel it
    task: asyncio.Task | None = None


# thread_id -> conversation. In memory and event loop only: per process, no lock needed
_threads: dict[str, _Thread] = {}
# Every open conversation keeps a sandbox running (and costing money)
_MAX_THREADS = 10
# Files larger than this are not sent to the file viewer
_MAX_VIEW_BYTES = 5_000_000
# Tool call arguments (e.g. a whole script in write_file) and tool output are cut to this for the UI
_TOOL_ARG_CHARS = 300
_TOOL_OUTPUT_CHARS = 600


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
    _check_agent_ready()
    await _close_idle_threads()
    if len(_threads) >= _MAX_THREADS:
        raise AppError(
            503, f"Too many open conversations ({_MAX_THREADS}). End one and try again.", code="TOO_MANY_THREADS"
        )
    data_files, pseudonyms = await _read_uploads(files)

    try:
        backend = await create_sandbox()
    except Exception as e:
        raise _sandbox_error(e, "create_sandbox") from e
    try:
        await upload_skills(backend)
        uploaded = await upload_data_files(backend, data_files)
        agent = build_agent(backend)
    except BaseException as e:  # cancels too: the sandbox must not be left running
        await delete_sandbox(backend)
        if isinstance(e, Exception) and not isinstance(e, AppError):
            raise _sandbox_error(e, "upload_files") from e
        raise

    thread_id = uuid.uuid4().hex
    _threads[thread_id] = _Thread(backend, agent, uploaded, pseudonyms)
    logger.info("thread_started", thread_id=thread_id, sandbox=backend.id, files=uploaded)
    return {"thread_id": thread_id, "sandbox_id": backend.id, "data_files": uploaded}


@thread_router.get("/{thread_id}")
async def get_thread(thread_id: str):
    """The conversation so far, as the same events /messages streams (see _stream_answer)."""
    thread = await _get_thread(thread_id)
    state = await thread.agent.aget_state(_run_config(thread_id))
    events: list[dict] = []
    seen: set[str] = set()
    for message in state.values.get("messages", []):
        events += [_restore_event(event, thread.pseudonyms) for event in _history_events(message, seen)]
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
    thread = await _get_thread(thread_id)
    question = _check_question(question)
    if thread.lock.locked():
        raise AppError(409, "The agent is still answering the previous question.", code="THREAD_BUSY")
    logger.info(
        "question_asked",
        thread_id=thread_id,
        question_chars=len(question),
        question_excerpt=question[:_QUESTION_LOG_CHARS].replace("\n", " "),
    )
    return StreamingResponse(
        _stream_answer(thread_id, thread, question),
        media_type="text/event-stream",
        # no-transform / X-Accel-Buffering: proxies must pass every event on at once
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@thread_router.post("/{thread_id}/stop")
async def stop_thread(thread_id: str):
    """Stop the question being answered. The stream then ends with a "stopped" event."""
    thread = await _get_thread(thread_id)
    if thread.task is None or thread.task.done():
        return {"thread_id": thread_id, "status": "not_running"}
    thread.task.cancel()
    logger.info("question_stopped", thread_id=thread_id)
    return {"thread_id": thread_id, "status": "stopped"}


@thread_router.delete("/{thread_id}")
async def delete_thread(thread_id: str):
    """End the conversation and delete its sandbox."""
    thread = _threads.pop(thread_id, None)
    if thread is None:
        raise _thread_not_found()
    await _close_thread(thread_id, thread)
    return {"thread_id": thread_id, "status": "deleted"}


@thread_router.get("/{thread_id}/tree")
async def get_tree(thread_id: str):
    """The files and folders in /data, /output and /skills of the conversation's sandbox."""
    thread = await _get_thread(thread_id)
    try:
        entries = await list_tree(thread.backend)
    except Exception as e:
        raise _sandbox_error(e, "list_tree", thread_id=thread_id) from e
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
    thread = await _get_thread(thread_id)
    if not path.startswith("/") or "\x00" in path:
        raise AppError(400, "path must be an absolute path in the sandbox, e.g. /output/report.md.")
    path = posixpath.normpath(path)
    try:
        content = await read_file(thread.backend, path, _MAX_VIEW_BYTES)
    except FileNotFoundError:
        raise AppError(404, f"{path} does not exist in the sandbox.", code="FILE_NOT_FOUND") from None
    except ValueError as e:
        raise AppError(413, str(e), code="FILE_TOO_LARGE") from None
    except Exception as e:
        raise _sandbox_error(e, "read_file", thread_id=thread_id) from e
    content = _restore_file(path, content, thread.pseudonyms)
    return Response(
        content,
        # Always a download type + CSP sandbox, so the browser never runs an HTML/SVG file the agent wrote
        media_type="application/octet-stream",
        headers={"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox", "Cache-Control": "no-store"},
    )


async def _stream_answer(thread_id: str, thread: _Thread, question: str) -> AsyncIterator[str]:
    """Answer one question, yielding each event as a Server-Sent Event.

    The agent runs as its own task that fills a queue, so /stop - or the browser closing the
    connection, which ends this generator - can cancel it at once instead of after its next step.
    """
    async with thread.lock:
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        task = thread.task = asyncio.create_task(_answer(thread_id, thread, question, queue))
        try:
            while (event := await queue.get()) is not None:
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            task.cancel()  # does nothing if it already finished
            await asyncio.wait([task])  # the lock is only released once the run has really stopped
            thread.task = None
            thread.last_used = time.monotonic()


async def _answer(thread_id: str, thread: _Thread, question: str, queue: asyncio.Queue) -> None:
    """Run the agent on one question and put its events on the queue, then None when done."""
    config = _run_config(thread_id)
    try:
        history = (await thread.agent.aget_state(config)).values.get("messages", [])
        # Steps of earlier questions must not be sent again (summarization can re-emit old messages)
        seen: set[str] = set()
        for message in history:
            _message_events(message, seen)
        # A stopped run can leave tool calls without a result, which the model provider would reject
        messages: list[Any] = _close_dangling_tool_calls(history)
        messages.append({"role": "user", "content": f"{question}{_DATA_FILES_NOTE}{', '.join(thread.data_files)})"})

        subagent = None  # name of the subagent the last `task` call started
        async for namespace, chunk in thread.agent.astream(
            {"messages": messages}, config=config, stream_mode="updates", subgraphs=True
        ):
            for update in chunk.values():
                for message in _update_messages(update):
                    for event in _message_events(message, seen, agent=(subagent or "subagent") if namespace else None):
                        if event["type"] == "tool_call" and event["name"] == "task" and not namespace:
                            subagent = event["args"].get("subagent_type")
                        # The model works with pseudonyms; the user sees the real IDs
                        queue.put_nowait(_restore_event(event, thread.pseudonyms))

        final = await thread.agent.aget_state(config)
        answer = _final_answer(final.values, thread_id)
        queue.put_nowait({"type": "answer", "text": _restore_ids(answer, thread.pseudonyms)})
    except asyncio.CancelledError:
        queue.put_nowait({"type": "stopped"})
        raise
    except GraphRecursionError:
        queue.put_nowait(_error_event(AppError(
            422,
            f"The agent stopped after {settings.max_agent_iterations} steps (MAX_AGENT_ITERATIONS) "
            "without finishing. Ask a simpler question or raise the limit.",
            code="MAX_ITERATIONS",
        )))
    except AppError as e:
        queue.put_nowait(_error_event(e))
    except Exception as e:  # LLM or sandbox failure
        logger.error("agent_error", thread_id=thread_id, cause=type(e).__name__, cause_message=str(e), exc_info=e)
        queue.put_nowait(_error_event(AppError(500, "The AI agent failed. See the server logs for details.", code="AGENT_ERROR")))
    finally:
        queue.put_nowait(None)


def _error_event(error: AppError) -> dict:
    return {"type": "error", "code": error.code, "message": error.message}


def _update_messages(update: Any) -> list[BaseMessage]:
    """The messages in one node's update (an update can hold none, one, or a wrapped list)."""
    messages = update.get("messages") if isinstance(update, dict) else None
    if messages is None:
        return []
    if not isinstance(messages, (list, tuple)):
        messages = getattr(messages, "value", None) or [messages]  # Overwrite(...) wraps a list
    return [m for m in messages if isinstance(m, BaseMessage)]


def _message_events(message: BaseMessage, seen: set[str], agent: str | None = None) -> list[dict]:
    """The tool calls / tool result in one message as UI events, each only once (tracked in seen)."""
    events = []
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            if call.get("id") in seen:
                continue
            seen.add(call.get("id"))
            args = {k: _cut(v, _TOOL_ARG_CHARS) if isinstance(v, str) else v for k, v in (call.get("args") or {}).items()}
            events.append({"type": "tool_call", "id": call.get("id"), "name": call["name"], "args": args, "agent": agent})
        if events and agent is None and message.text.strip():
            events.insert(0, {"type": "note", "text": message.text.strip()})
    elif isinstance(message, ToolMessage) and f"result:{message.tool_call_id}" not in seen:
        seen.add(f"result:{message.tool_call_id}")
        # The end of the output is the useful part: the result, or the last line of a traceback
        output = message.text.strip()
        if len(output) > _TOOL_OUTPUT_CHARS:
            output = "…" + output[-_TOOL_OUTPUT_CHARS:]
        events.append({
            "type": "tool_result",
            "id": message.tool_call_id,
            "name": message.name,
            "ok": not _tool_failed(message),
            "output": output,
            "agent": agent,
        })
    return events


def _history_events(message: BaseMessage, seen: set[str]) -> list[dict]:
    """Like _message_events, plus the questions and answers, for GET /api/threads/{id}."""
    if isinstance(message, HumanMessage):
        return [{"type": "user", "text": message.text.split(_DATA_FILES_NOTE)[0]}]
    if isinstance(message, AIMessage) and not message.tool_calls:
        return [{"type": "answer", "text": message.text}] if message.text.strip() else []
    return _message_events(message, seen)


def _tool_failed(result: ToolMessage) -> bool:
    # A script that exits non-zero is not a tool error: execute keeps its exit code in the artifact
    artifact = result.artifact if isinstance(result.artifact, dict) else {}
    return result.status == "error" or artifact.get("exit_code") not in (None, 0)


def _close_dangling_tool_calls(history: list[BaseMessage]) -> list[ToolMessage]:
    """A result for each tool call of the last model turn that never got one (the run was stopped)."""
    last_ai = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    if last_ai is None:
        return []
    answered = {m.tool_call_id for m in history if isinstance(m, ToolMessage)}
    return [
        ToolMessage(content="Stopped by the user before this finished.", tool_call_id=call["id"], name=call["name"], status="error")
        for call in last_ai.tool_calls
        if call.get("id") and call["id"] not in answered
    ]


def _cut(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _run_config(thread_id: str) -> RunnableConfig:
    return {
        "configurable": {"thread_id": thread_id},
        # Each model call + tool call is roughly 2 graph steps
        "recursion_limit": settings.max_agent_iterations * 2,
    }


async def _get_thread(thread_id: str) -> _Thread:
    thread = _threads.get(thread_id)
    if thread is None:
        raise _thread_not_found()
    # LangSmith has deleted an idle sandbox by now (SANDBOX_IDLE_TTL_SECONDS), so the conversation is over
    if not thread.lock.locked() and time.monotonic() - thread.last_used > SANDBOX_IDLE_TTL_SECONDS:
        del _threads[thread_id]
        await _close_thread(thread_id, thread)
        raise _thread_not_found()
    thread.last_used = time.monotonic()
    return thread


def _thread_not_found() -> AppError:
    return AppError(404, "This conversation has ended or expired. Start a new one.", code="THREAD_NOT_FOUND")


async def _close_thread(thread_id: str, thread: _Thread) -> None:
    if thread.task is not None:
        thread.task.cancel()
    await delete_sandbox(thread.backend)
    logger.info("thread_closed", thread_id=thread_id)


async def _close_idle_threads() -> None:
    now = time.monotonic()
    for thread_id, thread in list(_threads.items()):
        if not thread.lock.locked() and now - thread.last_used > SANDBOX_IDLE_TTL_SECONDS:
            del _threads[thread_id]
            await _close_thread(thread_id, thread)


def _sandbox_error(cause: Exception, action: str, **context) -> AppError:
    logger.error("sandbox_error", action=action, cause=type(cause).__name__, cause_message=str(cause), exc_info=cause, **context)
    return AppError(502, "The sandbox did not respond. Try again, or start a new conversation.", code="SANDBOX_ERROR")


# ---------------------------------------------------------------------------
# One-shot analysis (API clients)
# ---------------------------------------------------------------------------

# request_id (chosen by the client) -> the task running that analysis, so /cancel can stop it.
# In memory and only touched on the event loop: per process, no lock needed.
_running: dict[str, asyncio.Task] = {}
# request_id -> time a cancel arrived before its analysis started (e.g. while the files were
# still uploading). The analysis is refused when it arrives. Oldest first, pruned on insert.
_early_cancels: dict[str, float] = {}
_EARLY_CANCEL_TTL_SECONDS = 300
_MAX_EARLY_CANCELS = 1000
_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,64}")


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
    if request_id is None:
        request_id = uuid.uuid4().hex
    else:
        _check_request_id(request_id)
        _check_not_running(request_id)
    question = _check_question(question)
    _check_agent_ready()

    data_files, pseudonyms = await _read_uploads(files)

    thread_id = uuid.uuid4().hex
    # Audit trail: who asked what. thread_id links this line to every later log line and error
    logger.info(
        "analysis_started",
        thread_id=thread_id,
        request_id=request_id,
        client=_client_id(request),
        question_chars=len(question),
        question_excerpt=question[:_QUESTION_LOG_CHARS].replace("\n", " "),
        files=[name for name, _ in data_files],
    )
    # Checked and registered with no await in between, so a cancel can't slip through the gap
    if _early_cancels.pop(request_id, None) is not None:
        logger.info("analysis_cancelled", thread_id=thread_id, request_id=request_id, before_start=True)
        raise _cancelled()

    return await _run_tracked(request_id, thread_id, _run_analysis(question, data_files, thread_id, pseudonyms))


async def _run_tracked(request_id: str, thread_id: str, work: Coroutine) -> dict:
    """Run the analysis as a task /cancel can stop, and turn cancels into our errors."""
    task = asyncio.ensure_future(work)
    _running[request_id] = task
    try:
        return await task
    except asyncio.CancelledError:
        # This request itself was cancelled: uvicorn does that on shutdown once
        # SHUTDOWN_TIMEOUT runs out. Answer with a clean 503 instead of letting uvicorn
        # log "Exception in ASGI application"; uncancel() marks the cancel as handled.
        current = asyncio.current_task()
        if current.cancelling():
            current.uncancel()
            logger.info("analysis_cancelled", thread_id=thread_id, request_id=request_id, reason="server_shutdown")
            raise AppError(503, "The server is shutting down. Try again in a minute.", code="SERVER_SHUTTING_DOWN") from None
        # Otherwise /cancel stopped the task; the sandbox was already deleted on the way out
        logger.info("analysis_cancelled", thread_id=thread_id, request_id=request_id)
        raise _cancelled() from None
    finally:
        _running.pop(request_id, None)


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
    _check_request_id(request_id)
    task = _running.get(request_id)
    if task is None or task.done():
        _remember_early_cancel(request_id)
        return {"request_id": request_id, "status": "not_running"}
    task.cancel()
    logger.info("analysis_cancel_requested", request_id=request_id)
    return {"request_id": request_id, "status": "cancelled"}


async def _run_analysis(
    question: str, data_files: list[tuple[str, bytes]], thread_id: str, pseudonyms: dict[str, str]
) -> dict:
    """Create a sandbox, run the agent in it, and return its answer and output files.

    The sandbox is deleted on the way out - on errors and cancels too. Pseudonyms in the answer
    and in text output files are turned back into the real IDs.
    """
    backend = await create_sandbox()
    try:
        await upload_skills(backend)
        uploaded = await upload_data_files(backend, data_files)
        agent = build_agent(backend)
        result = await arun_agent(agent, question, thread_id=thread_id, data_files=uploaded)
        answer = _restore_ids(_final_answer(result, thread_id), pseudonyms)
        # The analysis already succeeded, so a failed download must not lose the answer
        outputs, warnings = await _collect_outputs(backend, pseudonyms)
        logger.info("analysis_complete", thread_id=thread_id, inputs=uploaded, outputs=[f["name"] for f in outputs])
        return {"thread_id": thread_id, "status": "completed", "answer": answer, "files": outputs, "warnings": warnings}
    finally:
        await delete_sandbox(backend)


def _cancelled() -> AppError:
    return AppError(409, "The analysis was stopped.", code="ANALYSIS_CANCELLED")


def _check_request_id(request_id: str) -> None:
    if not _REQUEST_ID_PATTERN.fullmatch(request_id):
        raise AppError(400, "request_id must be 8-64 letters, digits, '-' or '_'.")


def _check_not_running(request_id: str) -> None:
    if request_id in _running:
        raise AppError(400, "An analysis with this request_id is already running.")


def _remember_early_cancel(request_id: str) -> None:
    now = time.monotonic()
    _early_cancels.pop(request_id, None)  # re-insert so the dict stays ordered oldest first
    _early_cancels[request_id] = now
    while _early_cancels:
        oldest_id, cancelled_at = next(iter(_early_cancels.items()))
        if now - cancelled_at < _EARLY_CANCEL_TTL_SECONDS and len(_early_cancels) <= _MAX_EARLY_CANCELS:
            break
        del _early_cancels[oldest_id]


async def _collect_outputs(backend: LangSmithSandbox, pseudonyms: dict[str, str]) -> tuple[list[dict], list[str]]:
    """Download the files the agent saved in /output/. Never raises: failures become warnings."""
    try:
        files, warnings = await download_output_files(backend)
    except Exception as e:  # sandbox unreachable, timed out, ...
        logger.error("output_collection_failed", sandbox=backend.id, cause=type(e).__name__, exc_info=e)
        return [], ["The output files could not be retrieved from the sandbox."]
    outputs = [
        {
            "name": name,
            "media_type": output_type(name),
            "content_base64": base64.b64encode(_restore_file(name, content, pseudonyms)).decode("ascii"),
        }
        for name, content in files
    ]
    return outputs, warnings


# ---------------------------------------------------------------------------
# Shared checks
# ---------------------------------------------------------------------------

def _check_agent_ready() -> None:
    """Fail before creating a sandbox that would only be thrown away."""
    if not settings.enable_ai_agent:
        raise AppError(503, "The AI agent is disabled (ENABLE_AI_AGENT=false).", code="AGENT_DISABLED")
    if not settings.llm_api_key_set:
        raise AppError(
            500, f"{settings.llm_provider.upper()}_API_KEY is not set. Add it to your .env file.", code="MISSING_API_KEY"
        )


def _check_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise AppError(400, "The question is empty.")
    if len(question) > settings.max_question_chars:
        raise AppError(
            400, f"The question is too long ({len(question)} characters). The limit is {settings.max_question_chars}."
        )
    return question


async def _read_uploads(files: list[UploadFile]) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    """Check name, type and size of each upload, then read and pseudonymize it.

    Returns the (filename, content) pairs and the pseudonym -> real ID mapping.
    Raises AppError, never raw errors.
    """
    if not files:
        raise AppError(400, "No files were provided.", code="FILE_UPLOAD_ERROR")
    # Checked before reading, so a huge batch is rejected without loading it into memory
    if len(files) > settings.max_upload_files:
        raise AppError(
            400,
            f"Too many files: {len(files)} were provided, the limit is {settings.max_upload_files}.",
            code="FILE_UPLOAD_ERROR",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data_files: list[tuple[str, bytes]] = []
    for upload in files:
        filename = _safe_filename(upload.filename or "")
        if not filename.lower().endswith(".csv"):
            raise AppError(415, f"'{filename}' is not a CSV file. Only .csv files are supported.", code="INVALID_FILE_TYPE")
        if any(name == filename for name, _ in data_files):
            raise AppError(400, f"Two files are named '{filename}'. Rename one of them.", code="FILE_UPLOAD_ERROR")
        # upload.size can be None, so the length is checked again after reading
        if (upload.size or 0) > max_bytes:
            raise _too_large(filename)
        try:
            content = await upload.read()
        except Exception as e:  # e.g. the client disconnected mid-upload
            logger.error("upload_read_failed", file=filename, cause=type(e).__name__, exc_info=e)
            raise AppError(400, f"File '{filename}' could not be read.", code="FILE_UPLOAD_ERROR") from e
        if len(content) > max_bytes:
            raise _too_large(filename)
        _check_is_text(filename, content)
        data_files.append((filename, content))
    # Up to max_upload_files x max_upload_mb of CSV, so it runs off the event loop
    return await asyncio.to_thread(_pseudonymize_id_columns, data_files)


def _safe_filename(filename: str) -> str:
    """Just the file name - no folders, so it can't land outside /data/ - with no control characters."""
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or name in (".", "..") or name.startswith("-") or any(ord(c) < 32 for c in name):
        raise AppError(400, f"'{filename[:100]}' is not a usable file name.", code="FILE_UPLOAD_ERROR")
    return name


def _check_is_text(filename: str, content: bytes) -> None:
    """Reject files that are named .csv but are really binary (images, archives, executables)."""
    sample = content[:_SNIFF_BYTES]
    if b"\x00" in sample:  # text never contains NUL bytes (UTF-16 files do - ask for UTF-8)
        raise AppError(
            415, f"File '{filename}' is not a text CSV file. Save it as UTF-8 CSV and try again.", code="INVALID_FILE_TYPE"
        )
    # A truncated sample may end mid-character, so only the whole file must decode completely
    is_whole_file = len(sample) == len(content)
    for encoding in ("utf-8", "cp1252"):
        try:
            codecs.getincrementaldecoder(encoding)().decode(sample, final=is_whole_file)
            return
        except UnicodeDecodeError:
            continue
    raise AppError(
        415,
        f"File '{filename}' is not valid UTF-8 or Windows-1252 text. Save it as UTF-8 CSV and try again.",
        code="INVALID_FILE_TYPE",
    )


def _pseudonymize_id_columns(files: list[tuple[str, bytes]]) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    """Replace every value in personal ID columns with a pseudonym like ANON_000001.

    The model and the sandbox only ever see the pseudonyms. The mapping is returned so the
    server can put the real IDs back before the user sees an answer (see _restore_ids).

    One mapping covers all files of the upload, so equal IDs get equal pseudonyms and joins,
    counts and group-bys still work. Files without personal ID columns are returned byte for byte.

    Returns the files and the mapping pseudonym -> real ID.
    """
    real_to_pseudonym: dict[str, str] = {}
    result = []
    for filename, content in files:
        # surrogateescape round-trips any byte, so Windows-1252 files keep their encoding too
        text = content.decode("utf-8", "surrogateescape")
        first_line = text.split("\n", 1)[0]
        delimiter = max(",;\t|", key=first_line.count)
        try:
            rows = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter)
            header = next(rows, [])
            id_columns = [i for i, name in enumerate(header) if _is_personal_id_column(name)]
            if not id_columns:
                result.append((filename, content))
                continue
            out = io.StringIO(newline="")
            writer = csv.writer(out, delimiter=delimiter, lineterminator="\r\n" if first_line.endswith("\r") else "\n")
            writer.writerow(header)
            for row in rows:
                for i in id_columns:
                    value = row[i].strip() if i < len(row) else ""
                    if value.lower() not in _MISSING_CELLS:
                        new_pseudonym = f"{_PSEUDONYM_PREFIX}{len(real_to_pseudonym) + 1:06d}"
                        row[i] = real_to_pseudonym.setdefault(value, new_pseudonym)
                writer.writerow(row)
        except csv.Error as e:
            # Not uploaded as-is: its ID columns would reach the model unprotected
            raise AppError(400, f"File '{filename}' could not be read as CSV: {e}", code="FILE_UPLOAD_ERROR") from e
        # Only the column names are logged - never the IDs or the mapping
        logger.info("id_columns_pseudonymized", file=filename, columns=[header[i] for i in id_columns])
        result.append((filename, out.getvalue().encode("utf-8", "surrogateescape")))
    # Flip it, because restoring looks up a pseudonym to find the real ID
    pseudonym_to_real = {pseudonym: real for real, pseudonym in real_to_pseudonym.items()}
    return result, pseudonym_to_real


def _is_personal_id_column(name: str) -> bool:
    """Does this column identify a person? Checked word by word on the snake_case name.

    True:  customer_id, CustomerID, "Employee No", user_uuid, pan, pan_card_no, aadhaar_no, customer_pan
    False: product_id, order_id, customer_zip_code, pan_size, passport_country
    """
    words = _snake_case(name).split("_")
    # A person word right before an ID word (so customer_zip_code is not one)
    if len(words) >= 2 and words[-2] in _PERSON_WORDS and words[-1] in _ID_WORDS:
        return True
    # A national ID word, followed only by ID words or "card"
    for i, word in enumerate(words):
        if word in _NATIONAL_ID_WORDS:
            return all(rest in _ID_WORDS or rest == "card" for rest in words[i + 1 :])
    return False


def _restore_ids(text: str, pseudonyms: dict[str, str]) -> str:
    """Put the real IDs back: "Top customer: ANON_000001" -> "Top customer: C9".

    A pseudonym that is not in the mapping is left as it is.
    """
    if not text or not pseudonyms:
        return text
    return _PSEUDONYM_RE.sub(lambda match: pseudonyms.get(match.group(0), match.group(0)), text)


def _restore_event(event: dict, pseudonyms: dict[str, str]) -> dict:
    """The same UI event with real IDs in the text the user reads (notes, answers, tool output).

    Tool call arguments (the scripts the agent wrote) are left as they are.
    """
    restored = dict(event)
    for key in ("text", "output"):
        if isinstance(restored.get(key), str):
            restored[key] = _restore_ids(restored[key], pseudonyms)
    return restored


def _restore_file(name: str, content: bytes, pseudonyms: dict[str, str]) -> bytes:
    """A text file (.csv, .md, ...) with real IDs. Images stay as they are: a chart that shows
    pseudonyms in its labels can't be changed after it was drawn.

    Note: a real ID that contains the CSV delimiter or a quote would need quoting in a .csv file.
    IDs almost never do, so this simple text replacement is used.
    """
    if not pseudonyms or posixpath.splitext(name)[1].lower() not in _TEXT_FILE_TYPES:
        return content
    text = content.decode("utf-8", "surrogateescape")
    return _restore_ids(text, pseudonyms).encode("utf-8", "surrogateescape")


def _snake_case(name: str) -> str:
    """customer_id, CustomerID and "Customer ID" all become customer_id."""
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _too_large(filename: str) -> AppError:
    return AppError(413, f"File '{filename}' is larger than {settings.max_upload_mb} MB.", code="FILE_TOO_LARGE")


def _final_answer(result: dict, thread_id: str) -> str:
    """Return the text of the agent's last message."""
    try:
        last = result["messages"][-1]
        answer = last.text
    except (KeyError, IndexError, TypeError, AttributeError) as e:
        logger.error("agent_result_invalid", thread_id=thread_id, cause=type(e).__name__, exc_info=e)
        raise AppError(500, "The AI agent finished without returning an answer.", code="AGENT_ERROR") from e
    if answer and answer.strip():
        return answer

    # Gemini doesn't raise when it blocks something: a blocked PROMPT comes back as an empty
    # message with no metadata, a blocked ANSWER as an empty message with a safety finish_reason
    finish_reason = (getattr(last, "response_metadata", None) or {}).get("finish_reason")
    if finish_reason is None or finish_reason in _BLOCKED_FINISH_REASONS:
        logger.warning("request_refused", thread_id=thread_id, finish_reason=finish_reason)
        raise AppError(422, "The model refused to answer this request (blocked by its safety filters).", code="REQUEST_REFUSED")
    raise AppError(500, "The AI agent finished without returning an answer.", code="AGENT_ERROR")