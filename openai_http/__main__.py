"""
Entry point for running the application.

Usage:
    python -m openai_http

This module loads configuration and starts uvicorn server.
"""

import uvicorn

from openai_http.app import create_app
from openai_http.config import get_settings


def main():
    """Load config, create app, and run uvicorn server."""
    settings = get_settings()
    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.observability.log_level.lower(),
    )


if __name__ == "__main__":
    main()
