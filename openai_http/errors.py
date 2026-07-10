"""
Copyright (C) 2026 The OPENAI-HTTP Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

OpenAI-format error handling.

All exceptions are converted to OpenAI-format error responses with
appropriate HTTP status codes.
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class OpenAIError(HTTPException):
    """Base OpenAI error with structured response format.

    All errors produce a JSON body with ``error.message``,
    ``error.type``, ``error.param``, and ``error.code``.

    Args:
        message: Human-readable error description.
        error_type: OpenAI error type string.
        param: Optional parameter name the error relates to.
        code: Optional machine-readable error code.
        status_code: HTTP status code.
    """

    def __init__(
        self,
        message: str,
        error_type: str = "server_error",
        param: Optional[str] = None,
        code: Optional[str] = None,
        status_code: int = 500,
    ):
        super().__init__(status_code=status_code, detail=None)
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code


class InvalidRequestError(OpenAIError):
    """Invalid request parameters (HTTP 400).

    Args:
        message: Human-readable error description.
        param: Optional name of the invalid parameter.
        code: Machine-readable error code.
    """

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
    """Authentication failed (HTTP 401).

    Args:
        message: Human-readable error description.
        code: Machine-readable error code.
    """

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
    """Resource not found (HTTP 404).

    Args:
        message: Human-readable error description.
        param: Optional name of the resource that was not found.
    """

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
    """Rate limit exceeded (HTTP 429).

    Args:
        message: Human-readable error description.
        code: Machine-readable error code.
    """

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
    """Backend does not implement an optional capability (HTTP 501).

    Args:
        message: Human-readable error description.
    """

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
    """Build an OpenAI-format error response body.

    Args:
        message: Human-readable error description.
        error_type: OpenAI error type string.
        param: Optional parameter name the error relates to.
        code: Optional machine-readable error code.

    Returns:
        A dict structured as ``{"error": {message, type, param, code}}``.
    """
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code,
        }
    }


# pylint: disable=unused-argument
async def openai_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle standard OpenAIError exceptions.

    Args:
        request: The incoming HTTP request.
        exc: The raised exception, expected to be an OpenAIError instance.

    Returns:
        A JSONResponse with the error's status code and body.
    """
    if not isinstance(exc, OpenAIError):
        raise TypeError("openai_error_handler received non-OpenAIError")

    return JSONResponse(
        status_code=exc.status_code,
        content=_error_json(exc.message, exc.error_type, exc.param, exc.code),
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle Pydantic / FastAPI request validation errors.

    Args:
        request: The incoming HTTP request.
        exc: The raised exception, expected to be a RequestValidationError instance.

    Returns:
        A JSONResponse with HTTP 400 and OpenAI-format error body.
    """
    if not isinstance(exc, RequestValidationError):
        raise TypeError("validation_error_handler received non-RequestValidationError")

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


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle generic HTTPException errors.

    If the exception already has an OpenAI-format detail dict it is
    passed through as-is. Otherwise a new error body is built based on
    the status code.

    Args:
        request: The incoming HTTP request.
        exc: The raised exception, expected to be an HTTPException instance.

    Returns:
        A JSONResponse with the appropriate status code and error body.
    """
    if not isinstance(exc, HTTPException):
        raise TypeError("http_exception_handler received non-HTTPException")

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
    """Handle all unhandled exceptions as 500 Internal Server Error.

    Args:
        request: The incoming HTTP request.
        exc: The unhandled exception.

    Returns:
        A JSONResponse with HTTP 500 and an internal error body.
    """
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=_error_json(
            message="Internal server error. Please try again later.",
            error_type="server_error",
            code="internal_error",
        ),
    )


def register_error_handlers(app: FastAPI) -> None:
    """Register all error handlers on the FastAPI app.

    Handlers are registered for:
    - OpenAIError (custom hierarchy)
    - RequestValidationError (Pydantic)
    - HTTPException (FastAPI / Starlette)
    - Exception (catch-all)

    Args:
        app: The FastAPI application instance.
    """
    app.add_exception_handler(OpenAIError, openai_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, generic_error_handler)
