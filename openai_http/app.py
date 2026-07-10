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

FastAPI application factory for the OpenAI-compatible HTTP server.

Provides ``create_app()`` which wires configuration, middleware, routers,
error handlers, and a pluggable inference backend into a FastAPI
application instance.
"""

from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI

from .backends.mock_backend import MockTransformersBackend
from .config import Settings, get_settings
from .errors import register_error_handlers
from .observability.logging import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    setup_logging,
)
from .queue import RequestQueue
from .routers import (
    audio,
    chat,
    completions,
    embeddings,
    health,
    images,
    models,
)


try:
    APP_VERSION = version("openai-http")
except PackageNotFoundError:
    APP_VERSION = "0.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Initializes the request queue and inference backend on startup,
    and tears down the injected backend on shutdown.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI to begin serving requests.
    """
    app.state.queue = RequestQueue(max_depth=app.state.config.queue.depth)

    if injected := getattr(app.state, "injected", None):
        app.state.backend = injected
    else:
        app.state.backend = MockTransformersBackend(model_name="mock-model")

    await app.state.backend.setup()

    yield

    await app.state.backend.teardown()
    app.state.backend = None


def create_app(
    config: Settings | None = None,
    backend=None,
) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        config: Application settings. If None, settings are loaded from
            the environment and ``config.toml``.
        backend: Optional backend instance. If provided it is used for
            inference; otherwise a ``MockTransformersBackend`` is created
            during startup.

    Returns:
        The fully configured FastAPI application.
    """
    if config is None:
        config = get_settings()

    setup_logging(
        level=config.observability.log_level,
        log_format=config.observability.log_format,
    )

    app = FastAPI(
        title="OpenAI-Compatible API Server",
        description="OpenAI v1 API-compatible HTTP service with pluggable inference backends",
        version=APP_VERSION,
        lifespan=lifespan,
    )

    app.state.config = config
    if backend is not None:
        app.state.injected = backend

    register_error_handlers(app)

    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(completions.router)
    app.include_router(embeddings.router)
    app.include_router(audio.router)
    app.include_router(images.router)

    return app
