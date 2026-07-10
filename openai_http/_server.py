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

Server runner for openai_http.

Provides the ``run_server`` entry point that wires a BackendBase
implementation to the FastAPI application and starts uvicorn.
"""

import uvicorn

from .app import create_app
from .backends.base import BackendBase
from .config import (
    AuthSettings,
    ObservabilitySettings,
    QueueSettings,
    ServerSettings,
    Settings,
)


def run_server(
    backend: BackendBase,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    api_keys: list[str] | None = None,
    queue_depth: int = 32,
) -> None:
    """Run the openai_http server with a given backend.

    Builds configuration from parameters, creates the FastAPI app, and
    starts uvicorn.  Backend ``setup()`` / ``teardown()`` are handled by
    the lifespan context manager.

    Args:
        backend: Backend instance to serve inference requests.
        host: Host address to bind to.
        port: Port to listen on.
        log_level: Logging level (e.g. ``"info"``, ``"debug"``).
        api_keys: Optional list of accepted API keys for auth.
        queue_depth: Maximum number of queued requests.

    Raises:
        TypeError: If *backend* is not a BackendBase instance.
    """
    if not isinstance(backend, BackendBase):
        raise TypeError(
            f"backend must be an instance of BackendBase, got {type(backend).__name__}"
        )

    settings = Settings(
        server=ServerSettings(host=host, port=port),
        auth=AuthSettings(enabled=bool(api_keys), api_keys=api_keys or []),
        queue=QueueSettings(depth=queue_depth),
        observability=ObservabilitySettings(log_level=log_level),
    )

    app = create_app(config=settings, backend=backend)

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
