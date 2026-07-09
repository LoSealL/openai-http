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

Tests for health and metrics endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from openai_http.app import create_app
from openai_http.config import Settings, AuthSettings, ServerSettings, QueueSettings, ObservabilitySettings
from openai_http.backends.base import BackendBase


class _HealthyBackend(BackendBase):
    """A backend that exposes custom health and metrics."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        if model_id == "m":
            return (await self.list_models())[0]
        return None

    async def metrics(self):
        return {"requests_total": 42}

    async def health(self):
        return {"custom_field": "ok", "status": "ready"}


class _NoMetricsBackend(BackendBase):
    """A backend without metrics support."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        if model_id == "m":
            return (await self.list_models())[0]
        return None


class _BadHealthBackend(BackendBase):
    """A backend with misbehaving health()."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        if model_id == "m":
            return (await self.list_models())[0]
        return None

    async def health(self):
        return "not a dict"


class _FailingHealthBackend(BackendBase):
    """A backend whose health check raises unexpectedly."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        if model_id == "m":
            return (await self.list_models())[0]
        return None

    async def health(self):
        raise RuntimeError("health probe failed")


class _UnhealthyBackend(BackendBase):
    """A backend that reports itself as not ready."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        if model_id == "m":
            return (await self.list_models())[0]
        return None

    async def health(self):
        return {"status": "not_ready", "reason": "overload"}


class _NonDictMetricsBackend(BackendBase):
    """A backend that returns non-dict metrics."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        if model_id == "m":
            return (await self.list_models())[0]
        return None

    async def metrics(self):
        return [1, 2, 3]


@pytest.fixture
async def healthy_client():
    """Fixture providing an async test client with HealthyBackend."""
    settings = Settings(
        server=ServerSettings(host="127.0.0.1", port=8000),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=32),
        observability=ObservabilitySettings(log_level="debug", log_format="text", metrics_enabled=False),
    )
    backend = _HealthyBackend()
    app = create_app(config=settings, backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def no_metrics_client():
    """Fixture providing an async test client with NoMetricsBackend."""
    settings = Settings(
        server=ServerSettings(host="127.0.0.1", port=8000),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=32),
        observability=ObservabilitySettings(log_level="debug", log_format="text", metrics_enabled=False),
    )
    backend = _NoMetricsBackend()
    app = create_app(config=settings, backend=backend)
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


def _make_client(backend):
    settings = Settings(
        server=ServerSettings(host="127.0.0.1", port=8000),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=32),
        observability=ObservabilitySettings(log_level="debug", log_format="text", metrics_enabled=False),
    )
    app = create_app(config=settings, backend=backend)
    return app


@pytest.fixture
async def bad_health_client():
    """Fixture with a backend that returns non-dict health."""
    app = _make_client(_BadHealthBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def failing_health_client():
    """Fixture with a backend that raises from health()."""
    app = _make_client(_FailingHealthBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def unhealthy_client():
    """Fixture with a backend that reports status not_ready."""
    app = _make_client(_UnhealthyBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def non_dict_metrics_client():
    """Fixture with a backend that returns non-dict metrics."""
    app = _make_client(_NonDictMetricsBackend())
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health_ignores_non_dict_health(bad_health_client):
    """GET /health ignores non-dict backend health returns."""
    resp = await bad_health_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert "not a dict" not in body


@pytest.mark.asyncio
async def test_health_marks_not_ready_on_health_exception(failing_health_client):
    """GET /health becomes 503 when backend health raises unexpectedly."""
    resp = await failing_health_client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert "health_error" in body


@pytest.mark.asyncio
async def test_health_becomes_503_when_backend_reports_not_ready(unhealthy_client):
    """GET /health respects backend's not_ready status."""
    resp = await unhealthy_client.get("/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "not_ready"
    assert body["reason"] == "overload"


@pytest.mark.asyncio
async def test_metrics_wraps_non_dict_metrics(non_dict_metrics_client):
    """GET /metrics wraps non-dict backend metrics in a data field."""
    resp = await non_dict_metrics_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["metrics"]["data"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_health_overlays_backend_fields(healthy_client):
    """GET /health overlays backend health dict on the base response."""
    resp = await healthy_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["custom_field"] == "ok"
    assert body["backend_type"] == "_HealthyBackend"
    assert any(m["id"] == "m" for m in body["models"])


@pytest.mark.asyncio
async def test_metrics_returns_backend_metrics(healthy_client):
    """GET /metrics returns JSON statistics from the backend."""
    resp = await healthy_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["metrics"]["requests_total"] == 42


@pytest.mark.asyncio
async def test_metrics_returns_501_when_not_implemented(no_metrics_client):
    """GET /metrics returns 501 when the backend lacks metrics()."""
    resp = await no_metrics_client.get("/metrics")
    assert resp.status_code == 501
    body = resp.json()
    assert "error" in body
    assert body["error"]["type"] == "not_implemented_error"
    assert "Metrics" in body["error"]["message"]
