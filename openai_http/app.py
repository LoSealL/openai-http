from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from openai_http.config import Settings
from openai_http.errors import register_error_handlers
from openai_http.observability.logging import (
    RequestIDMiddleware,
    RequestLoggingMiddleware,
    setup_logging,
)
from openai_http.queue import RequestQueue


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = app.state.config
    app.state.queue = RequestQueue(max_depth=settings.queue.depth)

    injected = getattr(app.state, "_injected_backend", None)
    if injected is not None:
        app.state.backend = injected
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

    from openai_http.routers import audio, chat, completions, embeddings, health, images, models

    app.include_router(chat.router)
    app.include_router(models.router)
    app.include_router(health.router)
    app.include_router(completions.router)
    app.include_router(embeddings.router)
    app.include_router(audio.router)
    app.include_router(images.router)

    return app
