import os
import threading

import pytest
import uvicorn
from openai import OpenAI

from openai_http.app import create_app
from openai_http.config import Settings
from tests.conftest import _free_port, _wait_for_server

TEST_HOST = "127.0.0.1"


def _start_server_thread(host: str, port: int):
    """Start uvicorn server in a daemon thread on the given host:port."""
    os.environ["OPENAI_HTTP__AUTH__ENABLED"] = "false"
    os.environ["OPENAI_HTTP__OBSERVABILITY__LOG_LEVEL"] = "warning"
    os.environ["OPENAI_HTTP__OBSERVABILITY__LOG_FORMAT"] = "text"

    settings = Settings()
    app = create_app(settings)

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    return server, thread


@pytest.fixture(scope="session")
def sdk_server():
    """Start the mock server for SDK tests on a random free port."""
    max_retries = 5

    for _ in range(max_retries):
        port = _free_port()
        server, thread = _start_server_thread(TEST_HOST, port)

        try:
            _wait_for_server(f"http://{TEST_HOST}:{port}/health")
            yield {
                "host": TEST_HOST,
                "port": port,
                "base_url": f"http://{TEST_HOST}:{port}/v1",
                "server": server,
            }
            server.should_exit = True
            return
        except RuntimeError:
            server.should_exit = True

    pytest.fail(
        f"Test server failed to start on {TEST_HOST} after {max_retries} attempts"
    )


@pytest.fixture
def client(sdk_server):
    """Synchronous OpenAI client for SDK tests."""
    return OpenAI(
        api_key="test-key",
        base_url=sdk_server["base_url"],
    )
