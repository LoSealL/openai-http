import socket
import time

import httpx
import pytest
from httpx import AsyncClient, ASGITransport

from openai_http.app import create_app
from openai_http.config import (
    Settings,
    AuthSettings,
    ServerSettings,
    QueueSettings,
    ObservabilitySettings,
)
from openai_http.backends.mock_backend import MockTransformersBackend


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(url: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            httpx.get(url, timeout=1.0).raise_for_status()
            return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError(f"server at {url} did not start within {timeout}s")


def _wrap_call(call, *, endpoint_skip: str = ""):
    """Call a 501-stub endpoint; return the SDK exception or the response."""
    try:
        return call()
    except Exception as e:
        return e


@pytest.fixture(scope="session")
def mock_config():
    """Create test configuration with mock backend and auth disabled."""
    return Settings(
        server=ServerSettings(host="127.0.0.1", port=8000),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=32),
        observability=ObservabilitySettings(
            log_level="debug", log_format="text"
        ),
    )


@pytest.fixture(scope="session")
def app(mock_config):
    """Create FastAPI app with test configuration."""
    return create_app(mock_config)


@pytest.fixture
async def client(app):
    """Create async test client with lifespan context."""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
def mock_backend():
    """Mock backend instance for testing."""
    return MockTransformersBackend(model_name="mock-model")