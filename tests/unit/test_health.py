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
from tests._backend_fixtures import _MockBackendBase


class _HealthyBackend(_MockBackendBase):
    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def metrics(self):
        return {"requests_total": 42}

    async def health(self):
        return {"custom_field": "ok", "status": "ready"}


class _NoMetricsBackend(_MockBackendBase):
    pass


class _BadHealthBackend(_MockBackendBase):
    async def health(self):
        return "not a dict"


class _FailingHealthBackend(_MockBackendBase):
    async def health(self):
        raise RuntimeError("health probe failed")


class _UnhealthyBackend(_MockBackendBase):
    async def health(self):
        return {"status": "not_ready", "reason": "overload"}


class _NonDictMetricsBackend(_MockBackendBase):
    async def metrics(self):
        return [1, 2, 3]


def _settings() -> Settings:
    return Settings(
        server=ServerSettings(host="127.0.0.1", port=8000),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=32),
        observability=ObservabilitySettings(
            log_level="debug", log_format="text"
        ),
    )


def _make_app(backend):
    return create_app(config=_settings(), backend=backend)


@pytest.fixture
async def healthy_client():
    app = _make_app(_HealthyBackend())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def no_metrics_client():
    app = _make_app(_NoMetricsBackend())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def bad_health_client():
    app = _make_app(_BadHealthBackend())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def failing_health_client():
    app = _make_app(_FailingHealthBackend())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def unhealthy_client():
    app = _make_app(_UnhealthyBackend())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            yield ac


@pytest.fixture
async def non_dict_metrics_client():
    app = _make_app(_NonDictMetricsBackend())
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
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
async def test_metrics_passes_through_non_dict_metrics(non_dict_metrics_client):
    """GET /metrics passes non-dict backend metrics as-is."""
    resp = await non_dict_metrics_client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["metrics"] == [1, 2, 3]


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