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
