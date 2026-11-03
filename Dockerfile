# Data Analysis AI - the FastAPI backend and its web UI in one image.
#
#     docker build -t data-analysis-ai .
#     docker run --rm -p 8000:8000 --env-file .env data-analysis-ai
#
# Then open http://localhost:8000/ (web UI) or http://localhost:8000/docs (API docs).
# The settings come from environment variables (--env-file .env): the .env file itself is never
# copied into the image, so no API key ends up in it.

# ---------------------------------------------------------------------------
# 1. Builder: install the locked dependencies into /app/.venv
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.25 /uv /bin/uv

# Compile to .pyc now so the server starts faster; copy (not hardlink) out of the cache mount;
# use the image's Python 3.12 instead of downloading one
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Only the dependency files, so this layer is rebuilt when they change, not on every code edit.
# --frozen: exactly the versions in uv.lock. --no-dev: no pytest.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# 2. Runtime: the venv and the app code, run as a non-root user
# ---------------------------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

RUN useradd --create-home --uid 1000 app

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY app ./app

# The logger writes logs/<start time>.log relative to the working directory
RUN mkdir logs && chown app:app logs

USER app

EXPOSE 8000

# /api/health needs no API key, so it works with API_AUTH_KEY set too
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]

# uvicorn directly instead of main.py: main.py listens on 127.0.0.1 (unreachable from outside the
# container) with auto-reload on. One worker only: conversations, sandboxes and the rate limit are
# kept in memory, per process.
# Exec form, so uvicorn gets SIGTERM from `docker stop`. It then waits up to 20 s for running
# requests (an answer still streaming), cancels the rest and deletes the open sandboxes - so stop
# the container with `docker stop -t 60`: the default 10 s would kill it before that cleanup.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
