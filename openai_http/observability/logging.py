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

Structured JSON logging.

Provides request ID injection and request/response logging.
"""

import json
import logging
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class JSONFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON string representing the log record.
        """
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        request_id = getattr(record, "request_id", None)
        if request_id is not None:
            log_data["request_id"] = request_id
        extra_data = getattr(record, "extra_data", None)
        if isinstance(extra_data, dict):
            log_data.update(extra_data)
        if record.exc_info and record.exc_info[0] is not None:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


def setup_logging(level: str = "info", log_format: str = "json") -> None:
    """Configure structured logging for the openai_http library.

    Idempotent: calling multiple times adds at most one handler.

    Args:
        level: The logging level string (e.g. "info", "debug").
        log_format: The output format, "json" or "text".
    """
    logger = logging.getLogger("openai_http")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler()
        if log_format == "json":
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
        logger.addHandler(handler)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-ID header into all requests."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Inject a unique request ID into the request state and response header.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response with X-Request-ID header set.
        """
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex)
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


logger = logging.getLogger("openai_http.request")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request/response details."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request completion with timing information.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler.

        Returns:
            The HTTP response after logging.
        """
        start = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")

        response = await call_next(request)

        duration = round(time.perf_counter() - start, 4)
        logger.info(
            "request completed",
            extra={
                "extra_data": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_s": duration,
                }
            },
        )
        return response
