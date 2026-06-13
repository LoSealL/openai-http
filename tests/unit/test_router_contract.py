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

Router-level tests that inject malformed backends and assert callers
receive clean OpenAI-format error envelopes instead of leaked
``KeyError`` / ``TypeError`` 500s.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from openai_http.app import create_app
from openai_http.backends.base import BackendBase
from openai_http.config import (
    AuthSettings,
    ObservabilitySettings,
    QueueSettings,
    ServerSettings,
    Settings,
)


_MODEL_INFO = {
    "id": "broken",
    "object": "model",
    "created": 0,
    "owned_by": "test",
}


class _BadFinishBackend(BackendBase):
    """Returns ``finish_reason="bogus"`` which violates the contract."""

    async def generate(self, prompt, **kwargs):
        return {
            "generated_text": "x",
            "finish_reason": "bogus",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [_MODEL_INFO]

    async def get_model(self, model_id):
        return _MODEL_INFO if model_id == _MODEL_INFO["id"] else None


class _MissingUsageBackend(BackendBase):
    """Returns a generate result that lacks ``usage`` entirely."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "x"}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [_MODEL_INFO]

    async def get_model(self, model_id):
        return _MODEL_INFO if model_id == _MODEL_INFO["id"] else None


class _BadModelListBackend(BackendBase):
    """``list_models`` returns entries missing the ``owned_by`` field."""

    async def generate(self, prompt, **kwargs):
        return {
            "generated_text": "x",
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "broken", "object": "model", "created": 0}]

    async def get_model(self, model_id):
        return None


def _settings() -> Settings:
    return Settings(
        server=ServerSettings(host="127.0.0.1", port=0),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=4),
        observability=ObservabilitySettings(
            log_level="warning",
            log_format="text",
            metrics_enabled=False,
        ),
    )


async def _client_for(backend: BackendBase):
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
