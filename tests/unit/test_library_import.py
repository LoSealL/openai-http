import logging
import socket

from tests.conftest import _free_port


def test_public_api_names_exposed():
    """Verify that openai_http exposes the expected public API names."""
    import openai_http

    expected = {
        "BackendBase",
        "ParserBase",
        "run_server",
        "setup_logging",
        "__version__",
    }
    assert expected.issubset(set(dir(openai_http)))
    assert set(openai_http.__all__) == expected


def test_import_does_not_open_ports():
    """Verify that importing the library does not open any network ports."""
    free = _free_port()
    assert not _port_in_use(free)


def test_setup_logging_idempotent():
    """Calling setup_logging multiple times adds at most one handler."""
    import openai_http

    logger = logging.getLogger("openai_http")
    logger.handlers.clear()
    before = len(logger.handlers)
    openai_http.setup_logging()
    openai_http.setup_logging()
    openai_http.setup_logging()
    after = len(logger.handlers)
    assert after == before + 1
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def test_setup_logging_sets_level():
    """setup_logging respects the level argument."""
    import openai_http

    logger = logging.getLogger("openai_http")
    logger.handlers.clear()
    openai_http.setup_logging(level="debug")
    assert logger.level == logging.DEBUG
    openai_http.setup_logging(level="warning")
    assert logger.level == logging.WARNING
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def _port_in_use(port: int) -> bool:
    """Check whether a TCP port is already in use on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0