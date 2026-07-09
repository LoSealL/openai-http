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

SDK test fixtures — overrides the parent tests/conftest.py fixtures.

The parent conftest.py defines `client` as an async httpx.AsyncClient fixture
used for HTTP-level integration tests.

For SDK tests, we need a sync OpenAI client connected to a running server.
Uses threading-based uvicorn startup (more reliable on Windows than subprocess).
"""

import os
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from openai import AsyncOpenAI, OpenAI

from openai_http.app import create_app
from openai_http.config import Settings

TEST_HOST = "127.0.0.1"


def _free_port(host: str) -> int:
    """Find an available TCP port on the given host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


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


def _start_server_thread(host: str, port: int):
    """Start uvicorn server in a daemon thread on the given host:port."""
    os.environ["OPENAI_HTTP__AUTH__ENABLED"] = "false"
    os.environ["OPENAI_HTTP__OBSERVABILITY__LOG_LEVEL"] = "warning"
    os.environ["OPENAI_HTTP__OBSERVABILITY__LOG_FORMAT"] = "text"
    os.environ["OPENAI_HTTP__OBSERVABILITY__METRICS_ENABLED"] = "false"

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
        port = _free_port(TEST_HOST)
        server, thread = _start_server_thread(TEST_HOST, port)

        if _wait_for_server(TEST_HOST, port):
            yield {
                "host": TEST_HOST,
                "port": port,
                "base_url": f"http://{TEST_HOST}:{port}/v1",
                "server": server,
            }
            server.should_exit = True
            return

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


@pytest.fixture
def async_client(sdk_server):
    """Asynchronous OpenAI client for SDK tests."""
    return AsyncOpenAI(
        api_key="test-key",
        base_url=sdk_server["base_url"],
    )
