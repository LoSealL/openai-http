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
from openai_http.backends.base import BackendBase


class _NoEmbedBackend(BackendBase):
    """A backend that does not implement embeddings."""

    async def generate(self, prompt, **kwargs):
        return {
            "generated_text": "ok",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        if model_id == "m":
            return (await self.list_models())[0]
        return None


@pytest.fixture
async def no_embed_client():
    """Fixture providing an async test client with NoEmbedBackend."""
    settings = Settings(
        server=ServerSettings(host="127.0.0.1", port=8000),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=32),
        observability=ObservabilitySettings(
            log_level="debug", log_format="text"
        ),
    )
    backend = _NoEmbedBackend()
    app = create_app(config=settings, backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_embeddings_returns_501(no_embed_client):
    """POST /v1/embeddings returns 501 when backend lacks embed()."""
    resp = await no_embed_client.post(
        "/v1/embeddings",
        json={"input": "hello", "model": "m"},
    )
    assert resp.status_code == 501
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "not_implemented_error"
    assert "Embeddings" in body["error"]["message"]


@pytest.mark.asyncio
async def test_chat_still_works_with_no_embed_backend(no_embed_client):
    """Chat completions still work even when the backend lacks embed()."""
    resp = await no_embed_client.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200