import logging

from openai_http.observability.logging import JSONFormatter


def setup_logging(level: str = "info") -> None:
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
