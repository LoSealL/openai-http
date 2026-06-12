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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai_http.config import Settings, get_settings
from openai_http.errors import register_error_handlers
from openai_http.observability.logging import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    setup_logging,
)
from openai_http.queue import RequestQueue
from openai_http.routers import (
    audio,
    chat,
    completions,
    embeddings,
    health,
    images,
    models,
)


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
    settings = app.state.config
    app.state.queue = RequestQueue(max_depth=settings.queue.depth)

    injected = getattr(app.state, "_injected_backend", None)
    if injected is not None:
        app.state.backend = injected
        if not getattr(injected, "_openai_http_initialized", False):
            try:
                await injected.setup()
            except Exception as e:
                raise RuntimeError(f"Backend setup failed: {e}") from e
    else:
        from openai_http.backends.mock_backend import MockTransformersBackend

        app.state.backend = MockTransformersBackend(
            model_name="mock-model",
        )

    yield

    backend = getattr(app.state, "backend", None)
    if backend is not None and injected is not None:
        await backend.teardown()
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
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.config = config
    if backend is not None:
        app.state._injected_backend = backend

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

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(completions.router)
    app.include_router(embeddings.router)
    app.include_router(audio.router)
    app.include_router(images.router)

    return app
