"""
OpenAI-format error handling.

All exceptions are converted to OpenAI-format error responses with
appropriate HTTP status codes.
"""

import logging
from typing import Optional
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class OpenAIError(HTTPException):
    """Base OpenAI error with structured response format."""

    def __init__(
        self,
        message: str,
        error_type: str = "server_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
        status_code: int = 500,
    ):
        detail = {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        }
        super().__init__(status_code=status_code, detail=detail)
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code


class InvalidRequestError(OpenAIError):
    """Invalid request parameters."""

    def __init__(
        self,
        message: str,
        param: Optional[str] = None,
        code: Optional[str] = "invalid_request",
    ):
        super().__init__(
            message=message,
            error_type="invalid_request_error",
            param=param,
            code=code,
            status_code=400,
        )


class AuthenticationError(OpenAIError):
    """Authentication failed."""

    def __init__(
        self,
        message: str = "Incorrect API key provided.",
        code: str = "invalid_api_key",
    ):
        super().__init__(
            message=message,
            error_type="authentication_error",
            code=code,
            status_code=401,
        )


class NotFoundError(OpenAIError):
    """Resource not found."""

    def __init__(
        self,
        message: str,
        param: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_type="not_found_error",
            param=param,
            code="resource_not_found",
            status_code=404,
        )


class RateLimitError(OpenAIError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str = "Rate limit reached. Please wait before making another request.",
        code: str = "rate_limit_exceeded",
    ):
        super().__init__(
            message=message,
            error_type="rate_limit_error",
            code=code,
            status_code=429,
        )


class NotImplementedOpenAIError(OpenAIError):
    """Backend does not implement an optional capability."""

    def __init__(
        self,
        message: str,
    ):
        super().__init__(
            message=message,
            error_type="not_implemented_error",
            code=None,
            status_code=501,
        )


def _error_json(
    message: str,
    error_type: str,
    param: Optional[str] = None,
    code: Optional[str] = None,
) -> dict:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


async def openai_error_handler(request: Request, exc: OpenAIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_json(exc.message, exc.error_type, exc.param, exc.code),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    errors = exc.errors()
    message = "; ".join(f"{e.get('loc', '')}: {e.get('msg', '')}" for e in errors)
    param = errors[0].get("loc", [None])[-1] if errors else None
    return JSONResponse(
        status_code=400,
        content=_error_json(
            message=f"Invalid request: {message}",
            error_type="invalid_request_error",
            param=str(param) if param else None,
            code="invalid_request",
        ),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)

    error_type_map = {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
        500: "server_error",
    }
    error_type = error_type_map.get(exc.status_code, "server_error")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_json(
            message=str(exc.detail),
            error_type=error_type,
            code=str(exc.status_code),
        ),
    )


async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=_error_json(
            message="Internal server error. Please try again later.",
            error_type="server_error",
            code="internal_error",
        ),
    )


def register_error_handlers(app) -> None:
    """Register all error handlers on the FastAPI app."""
    app.add_exception_handler(OpenAIError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_error_handler)

    async def not_found_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=_error_json(
                message="Resource not found",
                error_type="not_found_error",
                code="404",
            ),
        )

    app.add_exception_handler(404, not_found_handler)
