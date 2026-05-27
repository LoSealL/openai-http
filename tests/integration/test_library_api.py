import threading
import time
import socket
import pytest
import httpx

from openai_http.backends.base import BackendBase
from openai_http._server import run_server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(port: int, timeout: float = 10.0) -> bool:
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


class ErrorBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        raise RuntimeError("boom")

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "err", "object": "model", "created": 0, "owned_by": "test"}]

    async def get_model(self, model_id):
        return {"id": "err", "object": "model", "created": 0, "owned_by": "test"}


class _RunResult:
    exception = None


def _run_server_thread(backend, port, result):
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
    result = _RunResult()
    t = threading.Thread(target=_run_server_thread, args=(backend, port, result), daemon=True)
    t.start()
    ok = _wait_for_server(port)
    if not ok:
        if result.exception:
            raise result.exception
        pytest.fail("server did not start")
    return t


def _stop(port):
    try:
        httpx.post(f"http://127.0.0.1:{port}/shutdown", timeout=1.0)
    except Exception:
        pass


def test_custom_backend_chat():
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
    with pytest.raises(TypeError, match="BackendBase"):
        run_server("not-a-backend")


def test_lifecycle_hooks_called():

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
    import asyncio
    from openai_http.backends.mock_backend import MockTransformersBackend

    b = MockTransformersBackend()

    async def _run():
        await b.setup()
        await b.teardown()

    asyncio.run(_run())
