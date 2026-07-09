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

Health and metrics endpoints.

GET /health  - health check (200 if ready, 503 if not)
GET /metrics - backend statistics in JSON format
"""

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from openai_http.errors import NotImplementedOpenAIError


logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=None)
async def health_check(request: Request) -> JSONResponse:
    """Check server health status.

    Returns 200 with model list if ready, 503 otherwise. If the backend
    implements ``health()``, its returned dict is overlaid on top of the
    base response.

    Returns:
        JSONResponse: Health status including backend type and model list.
    """
    app = request.app
    backend = getattr(app.state, "backend", None)

    if backend is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "models": [],
                "backend_type": None,
                "uptime_seconds": None,
            },
        )

    try:
        models = await backend.list_models()
    except Exception:
        models = []

    content = {
        "status": "ready" if models else "not_ready",
        "models": [{"id": m["id"], "status": "loaded"} for m in models],
        "backend_type": type(backend).__name__,
        "uptime_seconds": None,
    }

    try:
        backend_health = await backend.health()
        if isinstance(backend_health, dict):
            content.update(backend_health)
    except NotImplementedError:
        pass
    except Exception as exc:
        logger.exception("Backend health check failed")
        content["status"] = "not_ready"
        content["health_error"] = str(exc)

    return JSONResponse(
        status_code=200 if content.get("status") == "ready" else 503,
        content=content,
    )


@router.get("/metrics", response_model=None)
async def metrics(request: Request) -> JSONResponse:
    """Return backend metrics and statistics in JSON format.

    Args:
        request: The incoming HTTP request.

    Returns:
        JSONResponse: Backend metrics wrapped in a status envelope.

    Raises:
        NotImplementedOpenAIError: If the backend does not implement metrics.
    """
    app = request.app
    backend = getattr(app.state, "backend", None)

    if backend is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "metrics": {}},
        )

    try:
        backend_metrics = await backend.metrics()
    except NotImplementedError:
        raise NotImplementedOpenAIError("Metrics are not supported by this backend")

    if not isinstance(backend_metrics, dict):
        backend_metrics = {"data": backend_metrics}

    return JSONResponse(
        content={"status": "ok", "metrics": backend_metrics},
    )
