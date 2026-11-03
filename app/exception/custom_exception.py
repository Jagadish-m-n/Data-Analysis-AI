"""Errors and how the API reports them.

There is ONE error class. Raise it with an HTTP status code and a message for the user:

    from app.exception.custom_exception import AppError

    raise AppError(400, "The question is empty.")

The user then gets this JSON with status 400:

    {"error": {"code": "BAD_REQUEST", "message": "The question is empty."}}

The code is the name of the status code (400 -> BAD_REQUEST, 404 -> NOT_FOUND, ...),
unless you pass your own:  AppError(409, "Stopped.", code="ANALYSIS_CANCELLED")

Any other error (a bug, or a failure in Groq / Gemini / LangSmith) is logged with its full
traceback and the user gets a 500 response that names the error type.
"""

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.logger.custom_logger import get_logger

logger = get_logger(__name__)


class AppError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code or HTTPStatus(status_code).name


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"code": code, "message": message}})


def register_exception_handlers(app: FastAPI) -> None:
    """Call once after creating the FastAPI app."""

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError) -> JSONResponse:
        logger.warning("request_failed", path=request.url.path, code=error.code, message=error.message)
        return error_response(error.status_code, error.code, error.message)

    @app.exception_handler(RequestValidationError)
    async def handle_invalid_form(request: Request, error: RequestValidationError) -> JSONResponse:
        # A form field is missing or has the wrong type (FastAPI checks this before our code runs)
        fields = [str(e["loc"][-1]) for e in error.errors()]
        message = f"Missing or invalid field(s): {', '.join(fields)}."
        logger.warning("request_failed", path=request.url.path, message=message)
        return error_response(422, "INVALID_REQUEST", message)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.exception("unexpected_error", path=request.url.path)
        message = f"Something went wrong ({type(error).__name__}). See the server logs for details."
        return error_response(500, "INTERNAL_ERROR", message)
