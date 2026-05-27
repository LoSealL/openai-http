"""
Health endpoint.

GET /health - health check (200 if ready, 503 if not)
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check(request: Request) -> JSONResponse:
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
