import asyncio
import sys

import uvicorn

from openai_http._validation import BackendValidationError, validate_backend
from openai_http.app import create_app
from openai_http.backends.base import BackendBase
from openai_http.config import (
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
    skip_validation: bool = False,
) -> None:
    if not isinstance(backend, BackendBase):
        raise TypeError(
            f"backend must be an instance of BackendBase, got {type(backend).__name__}"
        )

    if not skip_validation:
        try:
            asyncio.run(validate_backend(backend))
        except BackendValidationError as e:
            print(f"Backend validation failed: {e}", file=sys.stderr)
            sys.exit(1)

    settings = Settings(
        server=ServerSettings(host=host, port=port),
        auth=AuthSettings(enabled=bool(api_keys), api_keys=api_keys or []),
        queue=QueueSettings(depth=queue_depth),
        observability=ObservabilitySettings(log_level=log_level),
    )

    app = create_app(config=settings, backend=backend)

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())
