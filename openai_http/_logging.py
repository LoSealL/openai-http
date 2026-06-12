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

Logging setup for openai_http.

Configures the ``openai_http`` logger with a JSON stream handler
provided by the observability module.
"""

import logging

from openai_http.observability.logging import JSONFormatter


def setup_logging(level: str = "info") -> None:
    """Configure the openai_http logger with a JSON stream handler.

    If a StreamHandler (not FileHandler) already exists, the logger is
    left untouched to avoid duplicate handlers.

    Args:
        level: Logging level string (e.g. ``"info"``, ``"debug"``,
            ``"warning"``).
    """
    logger = logging.getLogger("openai_http")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_stream_handler:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
