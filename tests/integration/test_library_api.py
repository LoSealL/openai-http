import threading
import pytest
import httpx

from openai_http.backends.base import BackendBase
from openai_http._server import run_server
from tests.conftest import _free_port, _wait_for_server
from tests._backend_fixtures import _MockBackendBase


class EchoBackend(BackendBase):
    """A backend that echoes back the user prompt."""

    async def generate(self, prompt, **kwargs):
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        return {
            "generated_text": f"ECHO: {text}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def generate_stream(self, prompt, **kwargs):
        result = await self.generate(prompt, **kwargs)
        for word in result["generated_text"].split():
            yield word + " "

    async def list_models(self):
        return [{"id": "echo", "object": "model", "created": 0, "owned_by": "test"}]

    async def get_model(self, model_id):
        if model_id == "echo":
            return (await self.list_models())[0]
        return None


class ErrorBackend(_MockBackendBase):
    async def generate(self, prompt, **kwargs):
        raise RuntimeError("boom")

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "err", "object": "model", "created": 0, "owned_by": "test"}]


def _start(backend, port):
    """Start the server on a background thread and wait for it to become available."""
    t = threading.Thread(
        target=run_server,
        args=(backend,),
        kwargs={"host": "127.0.0.1", "port": port, "log_level": "warning"},
        daemon=True,
    )
    t.start()
    try:
        _wait_for_server(f"http://127.0.0.1:{port}/health")
    except RuntimeError:
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
        request_served = False

        async def setup(self):
            self.setup_called = True

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
    t = threading.Thread(
        target=run_server,
        args=(FailingBackend(),),
        kwargs={"host": "127.0.0.1", "port": port, "log_level": "warning"},
        daemon=True,
    )
    t.start()
    try:
        _wait_for_server(f"http://127.0.0.1:{port}/health", timeout=2.0)
        assert False, "server should not have started"
    except RuntimeError:
        pass
