"""FastAPI app: routers, error handlers, and sandbox cleanup on shutdown.

Run it with:  uv run main.py   (or: uvicorn app.api.main:app --reload)
Then open http://127.0.0.1:8000/ for the web UI (app/static/), or /docs for the API.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.endpoints import agent_router, health_router, status_router, thread_router
from app.config.settings import settings
from app.exception.custom_exception import register_exception_handlers
from app.logger.custom_logger import get_logger
from app.sandbox.sandboxbackend import delete_all_sandboxes

logger = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("app_started", app=settings.app_name, env=settings.app_env, provider=settings.llm_provider, model=settings.llm_model)
    if not settings.api_auth_key:
        logger.warning("api_auth_disabled", hint="Set API_AUTH_KEY in .env before exposing the API beyond localhost")
    yield
    # Delete any sandboxes still running (open conversations too) and close the LangSmith client
    delete_all_sandboxes()
    logger.info("app_stopped")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

# Also turns missing / wrong form fields into the same JSON shape as our own errors
register_exception_handlers(app)

app.include_router(health_router)
app.include_router(status_router)
app.include_router(thread_router)
app.include_router(agent_router)

# Web UI (Bootstrap + plain JS). It calls the /api routes above from the same origin.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def web_ui() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
