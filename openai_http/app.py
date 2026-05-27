"""
FastAPI application factory.

Creates and configures the FastAPI application with:
- Lifespan management (model loading)
- CORS middleware
- Configuration injection into app.state
- Error handlers (OpenAI format)
- Observability middleware (logging + metrics)
- Request queue
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai_http.config import Settings
from openai_http.queue import RequestQueue
from openai_http.errors import register_error_handlers
from openai_http.observability.logging import (
    setup_logging,
    RequestIDMiddleware,
    RequestLoggingMiddleware,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    """
    settings = app.state.config
    app.state.queue = RequestQueue(max_depth=settings.queue.depth)

    # Initialize backend
    from openai_http.backends.mock_backend import MockTransformersBackend

    if settings.backend.type == "transformers":
        # TODO: Phase 6 - Implement real transformers backend
        raise NotImplementedError("Transformers backend not yet implemented")
    else:
        app.state.backend = MockTransformersBackend(
            model_name="mock-model",
            device=settings.backend.device,
        )

    yield
    # Shutdown: cleanup
    app.state.backend = None


def create_app(config: Settings | None = None) -> FastAPI:
    """
    Create and configure FastAPI application.
    """
    from openai_http.config import get_settings

    if config is None:
        config = get_settings()

    setup_logging(
        level=config.observability.log_level,
        log_format=config.observability.log_format,
    )

    app = FastAPI(
        title="OpenAI-Compatible API Server",
        description="OpenAI v1 API-compatible HTTP service with pluggable inference backends",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.config = config

    register_error_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    # Register routers
    from openai_http.routers import chat, models, health

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)

    return app
