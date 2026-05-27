import logging
import socket


def test_public_api_names_exposed():
    import openai_http
    expected = {"BackendBase", "run_server", "setup_logging", "__version__"}
    assert expected.issubset(set(dir(openai_http)))
    assert set(openai_http.__all__) == expected


def test_import_does_not_open_ports():
    free = _find_free_port()
    assert not _port_in_use(free)


def test_setup_logging_idempotent():
    import openai_http
    logger = logging.getLogger("openai_http")
    before = len(logger.handlers)
    openai_http.setup_logging()
    openai_http.setup_logging()
    openai_http.setup_logging()
    after = len(logger.handlers)
    assert after == before + 1
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def test_setup_logging_sets_level():
    import openai_http
    logger = logging.getLogger("openai_http")
    openai_http.setup_logging(level="debug")
    assert logger.level == logging.DEBUG
    openai_http.setup_logging(level="warning")
    assert logger.level == logging.WARNING
    for handler in list(logger.handlers):
        logger.removeHandler(handler)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0
