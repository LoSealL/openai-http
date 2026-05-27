"""
SDK test fixtures — overrides the parent tests/conftest.py fixtures.

The parent conftest.py defines `client` as an async httpx.AsyncClient fixture
used for HTTP-level integration tests.

For SDK tests, we need a sync OpenAI client connected to a running server.
Uses threading-based uvicorn startup (more reliable on Windows than subprocess).
"""

import pytest
import threading
import time
import os
import httpx

# Use default config.toml port (8000) for SDK tests
TEST_HOST = "127.0.0.1"
TEST_PORT = 8000
TEST_BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}/v1"


def _wait_for_server(host: str, port: int, timeout: float = 15.0) -> bool:
    """Wait for server to become available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://{host}:{port}/health", timeout=1.0)
            if r.status_code in (200, 503):
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def _start_server_thread():
    """Start uvicorn server in a daemon thread."""
    import uvicorn
    from openai_http.app import create_app
    from openai_http.config import Settings

    # Ensure mock backend
    os.environ["OPENAI_HTTP__BACKEND__TYPE"] = "mock"
    os.environ["OPENAI_HTTP__AUTH__ENABLED"] = "false"
    os.environ["OPENAI_HTTP__OBSERVABILITY__LOG_LEVEL"] = "warning"
    os.environ["OPENAI_HTTP__OBSERVABILITY__LOG_FORMAT"] = "text"
    os.environ["OPENAI_HTTP__OBSERVABILITY__METRICS_ENABLED"] = "false"

    settings = Settings()
    app = create_app(settings)

    config = uvicorn.Config(
        app,
        host=TEST_HOST,
        port=TEST_PORT,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    return server, thread


@pytest.fixture(scope="session")
def sdk_server():
    """Start the mock server for SDK tests."""
    # Check if port is already in use
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = s.connect_ex((TEST_HOST, TEST_PORT))
        s.close()
        if result == 0:
            # Port already in use - check if it's our server
            if _wait_for_server(TEST_HOST, TEST_PORT, timeout=2.0):
                yield {
                    "host": TEST_HOST,
                    "port": TEST_PORT,
                    "base_url": TEST_BASE_URL,
                    "server": None,
                }
                return
            else:
                pytest.skip(f"Port {TEST_PORT} is occupied by another service")
    except Exception:
        pass

    server, thread = _start_server_thread()

    if not _wait_for_server(TEST_HOST, TEST_PORT):
        server.should_exit = True
        pytest.fail(f"Test server failed to start on {TEST_HOST}:{TEST_PORT}")

    yield {
        "host": TEST_HOST,
        "port": TEST_PORT,
        "base_url": TEST_BASE_URL,
        "server": server,
    }

    # Shutdown
    server.should_exit = True


@pytest.fixture
def client(sdk_server):
    """Synchronous OpenAI client for SDK tests."""
    from openai import OpenAI
    return OpenAI(
        api_key="test-key",
        base_url=sdk_server["base_url"],
    )


@pytest.fixture
def async_client(sdk_server):
    """Asynchronous OpenAI client for SDK tests."""
    from openai import AsyncOpenAI
    return AsyncOpenAI(
        api_key="test-key",
        base_url=sdk_server["base_url"],
    )
