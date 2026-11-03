"""Entry point - runs the FastAPI backend, which also serves the web UI.

    uv run main.py     -> http://127.0.0.1:8000/       web UI (app/static/)
                          http://127.0.0.1:8000/docs   API docs

Settings (environment variables):
    API_HOST / API_PORT  - where the server listens (default 127.0.0.1:8000)
"""

import os

import uvicorn

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))


def run_api(reload: bool = True) -> None:
    uvicorn.run("app.api.main:app", host=API_HOST, port=API_PORT, reload=reload)


if __name__ == "__main__":
    run_api()
