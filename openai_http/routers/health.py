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

Health endpoint.

GET /health - health check (200 if ready, 503 if not)
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
    """Check server health status.

    Returns 200 with model list if ready, 503 otherwise.

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

    return JSONResponse(
        status_code=200 if models else 503,
        content={
            "status": "ready" if models else "not_ready",
            "models": [{"id": m["id"], "status": "loaded"} for m in models],
            "backend_type": type(backend).__name__,
            "uptime_seconds": None,
        },
    )
