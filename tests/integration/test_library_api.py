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

Integration tests for the openai_http library API.
"""

import asyncio
import threading
import time
import socket
import pytest
import httpx

from openai_http.backends.base import BackendBase
from openai_http.backends.mock_backend import MockTransformersBackend
from openai_http._server import run_server


def _free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 10.0) -> bool:
    """Wait for the server to become available on the given port."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code in (200, 503):
                return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


class EchoBackend(BackendBase):
    """A backend that echoes back the user prompt."""

    async def generate(self, prompt, **kwargs):
        """Echo the prompt back as generated text."""
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        return {
            "generated_text": f"ECHO: {text}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def generate_stream(self, prompt, **kwargs):
        """Stream the echo response word by word."""
        result = await self.generate(prompt, **kwargs)
        for word in result["generated_text"].split():
            yield word + " "

    async def list_models(self):
        """Return the list of available models."""
        return [{"id": "echo", "object": "model", "created": 0, "owned_by": "test"}]

    async def get_model(self, model_id):
        """Return model info for the given model_id."""
        if model_id == "echo":
            return (await self.list_models())[0]
        return None


class ErrorBackend(BackendBase):
    """A backend whose generate() always raises RuntimeError."""

    async def generate(self, prompt, **kwargs):
        """Raise a RuntimeError to test error handling."""
        raise RuntimeError("boom")

    async def generate_stream(self, prompt, **kwargs):
        """Yield a single token as stream response."""
        yield "ok"

    async def list_models(self):
        """Return the list of available models."""
        return [{"id": "err", "object": "model", "created": 0, "owned_by": "test"}]

    async def get_model(self, model_id):
        """Return model info for the given model_id."""
        return {"id": "err", "object": "model", "created": 0, "owned_by": "test"}


class _RunResult:
    """Container for capturing exceptions from a server thread."""

    exception = None


def _run_server_thread(backend, port, result):
    """Run the server in a background thread, capturing any exception."""
    try:
        run_server(
            backend,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            skip_validation=True,
        )
    except Exception as e:
        result.exception = e


def _start(backend, port):
    """Start the server on a background thread and wait for it to become available."""
    result = _RunResult()
    t = threading.Thread(
        target=_run_server_thread, args=(backend, port, result), daemon=True
    )
    t.start()
    ok = _wait_for_server(port)
    if not ok:
        if result.exception:
            raise result.exception
        pytest.fail("server did not start")
    return t


def _stop(port):
    """Shut down the server running on the given port."""
    try:
        httpx.post(f"http://127.0.0.1:{port}/shutdown", timeout=1.0)
    except Exception:
        pass


def test_custom_backend_chat():
    """A custom backend serves chat completions via the OpenAI-compatible API."""
    port = _free_port()
    _start(EchoBackend(), port)
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "echo", "messages": [{"role": "user", "content": "hello"}]},
            timeout=5.0,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"].startswith("ECHO:")
    finally:
        _stop(port)


def test_custom_backend_models_list():
    """A custom backend serves the models list via the OpenAI-compatible API."""
    port = _free_port()
    _start(EchoBackend(), port)
    try:
        resp = httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"][0]["id"] == "echo"
    finally:
        _stop(port)


def test_custom_backend_501_embeddings():
    """Embeddings endpoint returns 501 when the backend does not support it."""
    port = _free_port()
    _start(EchoBackend(), port)
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{port}/v1/embeddings",
            json={"input": "hi", "model": "echo"},
            timeout=5.0,
        )
        assert resp.status_code == 501
        body = resp.json()
        assert body["error"]["type"] == "not_implemented_error"
    finally:
        _stop(port)


def test_backend_exception_maps_to_500():
    """A RuntimeError in generate() is caught and returned as a 500 error response."""
    port = _free_port()
    _start(ErrorBackend(), port)
    try:
        resp = httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "err", "messages": [{"role": "user", "content": "hi"}]},
            timeout=5.0,
        )
        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["type"] == "server_error"
        assert "boom" not in body["error"].get("message", "")
        assert "Traceback" not in str(body)
    finally:
        _stop(port)


def test_invalid_backend_raises_type_error():
    """Passing a non-BackendBase to run_server raises TypeError."""
    with pytest.raises(TypeError, match="BackendBase"):
        run_server("not-a-backend")


def test_lifecycle_hooks_called():
    """setup(), generate(), and teardown() lifecycle hooks are invoked in order."""

    class TrackingBackend(EchoBackend):
        setup_called = False
        teardown_called = False
        request_served = False

        async def setup(self):
            self.setup_called = True

        async def teardown(self):
            self.teardown_called = True

        async def generate(self, prompt, **kwargs):
            self.request_served = True
            return await super().generate(prompt, **kwargs)

    backend = TrackingBackend()
    port = _free_port()
    _start(backend, port)
    try:
        httpx.post(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            json={"model": "echo", "messages": [{"role": "user", "content": "hi"}]},
            timeout=5.0,
        )
        assert backend.setup_called
        assert backend.request_served
    finally:
        _stop(port)


def test_setup_failure_aborts_startup():
    """If setup() raises, the server startup is aborted."""

    class FailingBackend(EchoBackend):
        async def setup(self):
            raise RuntimeError("GPU OOM")

    port = _free_port()
    result = _RunResult()
    t = threading.Thread(
        target=_run_server_thread,
        args=(FailingBackend(), port, result),
        daemon=True,
    )
    t.start()
    time.sleep(2.0)
    if result.exception:
        assert "Backend setup failed" in str(result.exception)
    else:
        assert not _wait_for_server(port, timeout=1.0)


def test_mock_backend_lifecycle_noop():
    """MockTransformersBackend.setup() and teardown() are no-ops."""
    b = MockTransformersBackend()

    async def _run():
        await b.setup()
        await b.teardown()

    asyncio.run(_run())
