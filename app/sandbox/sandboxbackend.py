"""The LangSmith sandbox: a temporary, isolated Linux machine where the agent runs its Python code.

Each analysis (or conversation in the web UI) gets its own sandbox, used in 5 steps:

    backend = await create_sandbox()                # 1. start a fresh sandbox
    await upload_skills(backend)                    # 2. copy app/skills/  ->  /skills/
    await upload_data_files(backend, files)         # 3. copy the CSV files ->  /data/, create /output/
    ...                                             # 4. the agent works, saving files in /output/
    files = await download_output_files(backend)    # 5a. fetch the charts and reports
    await delete_sandbox(backend)                   # 5b. always delete the sandbox at the end

list_tree() and read_file() let the web UI browse the sandbox while the agent works.

The LangSmith SDK is blocking, so its calls run in a worker thread (asyncio.to_thread) and the
server can keep answering other requests meanwhile.
"""

import asyncio
import os
import shlex
import time
import uuid

from deepagents.backends import LangSmithSandbox
from langsmith import _features as langsmith_features
from langsmith import sandbox as langsmith_sandbox

from app.config.settings import PROJECT_ROOT, settings
from app.logger.custom_logger import get_logger

logger = get_logger(__name__)

LOCAL_SKILLS_DIR = PROJECT_ROOT / "app" / "skills"
SANDBOX_SKILLS_PATH = "/skills/"
SANDBOX_DATA_PATH = "/data/"
SANDBOX_OUTPUT_PATH = "/output/"

# Only these files are returned from /output/; anything else the agent writes is ignored
OUTPUT_TYPES = {".png": "image/png", ".svg": "image/svg+xml", ".md": "text/markdown", ".csv": "text/csv"}

# LangSmith deletes a sandbox nobody used for this long - a safety net for sandboxes the server
# could not delete itself (crash, killed process). A conversation idle this long is closed too.
SANDBOX_IDLE_TTL_SECONDS = 30 * 60

# Folders shown in the web UI's file tree
TREE_ROOTS = (SANDBOX_DATA_PATH, SANDBOX_OUTPUT_PATH, SANDBOX_SKILLS_PATH)

# The SDK reads this option from os.environ only (not from our .env), so copy it over
if settings.sandbox_transport == "sse":
    os.environ[langsmith_features.ENV_VAR] = langsmith_features.SANDBOX_SSE_EXEC

# One client for the whole app. Every sandbox keeps using the client that created it.
client = langsmith_sandbox.SandboxClient(api_key=settings.langsmith_api_key or None)

# Sandboxes not deleted yet, so they can all be deleted when the server stops
live_sandboxes: set[str] = set()


async def create_sandbox() -> LangSmithSandbox:
    started = time.monotonic()
    name = f"data-analysis-{uuid.uuid4().hex[:8]}"
    options = {"name": name, "timeout": settings.sandbox_create_timeout, "idle_ttl_seconds": SANDBOX_IDLE_TTL_SECONDS}
    if settings.sandbox_snapshot:  # the data libraries are pre-installed there
        options["snapshot"] = settings.sandbox_snapshot

    # Blocks until the sandbox is ready, so the time below is the whole start-up
    sandbox = await asyncio.to_thread(client.create_sandbox, **options)
    live_sandboxes.add(name)
    logger.info(
        "sandbox_created",
        sandbox=name,
        snapshot=settings.sandbox_snapshot or "default image",
        startup_seconds=round(time.monotonic() - started, 2),
    )
    # deepagents' wrapper gives the agent its file and `execute` tools on top of the sandbox
    return LangSmithSandbox(sandbox=sandbox)


async def delete_sandbox(backend: LangSmithSandbox) -> None:
    live_sandboxes.discard(backend.id)
    await asyncio.to_thread(_delete, backend.id)
    await backend.aclose()


def delete_all_sandboxes() -> None:
    """Called once when the server stops: delete leftover sandboxes and close the client."""
    for name in list(live_sandboxes):
        _delete(name)
    live_sandboxes.clear()
    client.close()


def _delete(name: str) -> None:
    # This runs during cleanup, often while another error is being handled, so it must never
    # raise: a failed delete is only logged (the sandbox may then need deleting by hand).
    try:
        client.delete_sandbox(name)
        logger.info("sandbox_deleted", sandbox=name)
    except Exception:
        logger.exception("sandbox_delete_failed", sandbox=name)


async def upload_skills(backend: LangSmithSandbox) -> None:
    """Copy every file in app/skills/ to /skills/ in the sandbox."""
    files = [
        (SANDBOX_SKILLS_PATH + path.relative_to(LOCAL_SKILLS_DIR).as_posix(), path.read_bytes())
        for path in sorted(LOCAL_SKILLS_DIR.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    ]
    await _upload(backend, files)


async def upload_data_files(backend: LangSmithSandbox, files: list[tuple[str, bytes]]) -> list[str]:
    """Copy (filename, content) pairs to /data/ and create /output/. Returns the paths inside the sandbox."""
    sandbox_files = [(SANDBOX_DATA_PATH + name, content) for name, content in files]
    await _upload(backend, sandbox_files)
    # The image has no /output/, and pandas' to_csv and matplotlib's savefig do not create folders
    result = await backend.aexecute(f"mkdir -p {SANDBOX_OUTPUT_PATH}")
    if result.exit_code != 0:
        raise RuntimeError(f"Could not create {SANDBOX_OUTPUT_PATH}: {result.output}")
    return [path for path, _ in sandbox_files]


async def _upload(backend: LangSmithSandbox, files: list[tuple[str, bytes]]) -> None:
    sandbox = backend._sandbox  # the LangSmith sandbox inside deepagents' wrapper
    for path, content in files:
        # 60 s plus about 1 s per MB, so big files have time to upload
        timeout = 60 + len(content) // 1_000_000
        await asyncio.to_thread(sandbox.write, path, content, timeout=timeout)
    logger.info("files_uploaded", sandbox=backend.id, files=[path for path, _ in files])


async def download_output_files(backend: LangSmithSandbox) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Fetch the files the agent saved in /output/. Returns (name, content) pairs and warnings."""
    listing = await backend.als(SANDBOX_OUTPUT_PATH)
    if listing.error or not listing.entries:  # the agent saved no files
        return [], []

    paths, warnings = [], []
    for entry in listing.entries:
        name = entry["path"].rsplit("/", 1)[-1]
        if entry.get("is_dir"):
            continue
        if output_type(name):
            paths.append(entry["path"])
        else:
            warnings.append(f"Output file '{name}' was ignored: only .png, .svg, .md and .csv are returned.")

    files = []
    for download in await backend.adownload_files(paths):
        name = download.path.rsplit("/", 1)[-1]
        if download.content is None:
            warnings.append(f"Output file '{name}' could not be downloaded.")
        else:
            files.append((name, download.content))
    return files, warnings


def output_type(name: str) -> str | None:
    """The media type of an output file ("image/png", ...), or None if it is not returned."""
    extension = os.path.splitext(name)[1].lower()
    return OUTPUT_TYPES.get(extension)


async def list_tree(backend: LangSmithSandbox) -> list[dict]:
    """Every file and folder under TREE_ROOTS: [{"path", "type": "file" | "dir", "size"}]."""
    # The roots are our own constants, never user input, so they can go into the command as-is
    command = (
        f"find {' '.join(TREE_ROOTS)} -mindepth 1 -not -path '*/__pycache__*' "
        "-printf '%p\\t%y\\t%s\\n' 2>/dev/null | sort"
    )
    result = await backend.aexecute(command)
    entries = []
    for line in result.output.splitlines():
        parts = line.rsplit("\t", 2)  # the path comes first, so a tab in a file name can't break this
        if len(parts) != 3 or not parts[2].isdigit():
            continue
        path, kind, size = parts
        entries.append({"path": path, "type": "dir" if kind == "d" else "file", "size": int(size)})
    return entries


async def read_file(backend: LangSmithSandbox, path: str, max_bytes: int) -> bytes:
    """The content of one file. FileNotFoundError if it is missing, ValueError if it is too big."""
    quoted = shlex.quote(path)
    result = await backend.aexecute(f"test -f {quoted} && stat -c %s -- {quoted}")
    size = result.output.strip()
    if result.exit_code != 0 or not size.isdigit():
        raise FileNotFoundError(path)
    if int(size) > max_bytes:
        raise ValueError(f"{path} is {int(size) / 1_000_000:.1f} MB, more than the {max_bytes // 1_000_000} MB that can be viewed.")
    [download] = await backend.adownload_files([path])
    if download.content is None:
        raise FileNotFoundError(path)
    return download.content
