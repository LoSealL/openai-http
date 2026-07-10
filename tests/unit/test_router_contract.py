import pytest
from httpx import AsyncClient, ASGITransport

from openai_http.app import create_app
from openai_http.config import (
    AuthSettings,
    ObservabilitySettings,
    QueueSettings,
    ServerSettings,
    Settings,
)
from tests._backend_fixtures import _MockBackendBase


_MODEL_INFO = {
    "id": "broken",
    "object": "model",
    "created": 0,
    "owned_by": "test",
}


class _BadFinishBackend(_MockBackendBase):
    async def generate(self, prompt, **kwargs):
        return {
            "generated_text": "x",
            "finish_reason": "bogus",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def list_models(self):
        return [_MODEL_INFO]


class _MissingUsageBackend(_MockBackendBase):
    async def generate(self, prompt, **kwargs):
        return {"generated_text": "x"}

    async def list_models(self):
        return [_MODEL_INFO]


class _BadModelListBackend(_MockBackendBase):
    async def generate(self, prompt, **kwargs):
        return {
            "generated_text": "x",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def list_models(self):
        return [{"id": "broken", "object": "model", "created": 0}]


def _settings() -> Settings:
    return Settings(
        server=ServerSettings(host="127.0.0.1", port=0),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=4),
        observability=ObservabilitySettings(
            log_level="warning",
            log_format="text",
        ),
    )


async def _client_for(backend):
    """Create an ASGI test client wired to a custom backend.

    The lifespan context is entered to install ``app.state.queue``.
    """
    app = create_app(_settings(), backend=backend)

    class _Ctx:
        async def __aenter__(self):
            self._lifespan = app.router.lifespan_context(app)
            await self._lifespan.__aenter__()
            transport = ASGITransport(app=app)
            self._client = AsyncClient(transport=transport, base_url="http://test")
            await self._client.__aenter__()
            return self._client

        async def __aexit__(self, *exc):
            try:
                await self._client.__aexit__(*exc)
            finally:
                await self._lifespan.__aexit__(*exc)

    return _Ctx()


@pytest.mark.asyncio
async def test_chat_completions_bad_finish_reason_returns_500_envelope():
    backend = _BadFinishBackend()
    async with await _client_for(backend) as ac:
        # Lifespan would normally validate; we bypass by using ASGITransport
        # without entering the lifespan context, which is what the standard
        # tests/conftest.py also does outside of the explicit fixture.
        resp = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "broken",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == "backend_contract_error"
    assert "finish_reason" in body["error"]["message"]


@pytest.mark.asyncio
async def test_chat_completions_missing_usage_returns_500_envelope():
    backend = _MissingUsageBackend()
    async with await _client_for(backend) as ac:
        resp = await ac.post(
            "/v1/chat/completions",
            json={
                "model": "broken",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "backend_contract_error"


@pytest.mark.asyncio
async def test_completions_bad_finish_reason_returns_500_envelope():
    backend = _BadFinishBackend()
    async with await _client_for(backend) as ac:
        resp = await ac.post(
            "/v1/completions",
            json={"model": "broken", "prompt": "hi", "max_tokens": 4},
        )
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["code"] == "backend_contract_error"


@pytest.mark.asyncio
async def test_models_list_bad_entry_returns_500_envelope():
    backend = _BadModelListBackend()
    async with await _client_for(backend) as ac:
        resp = await ac.get("/v1/models")
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"]["type"] == "server_error"
    assert body["error"]["code"] == "backend_contract_error"
