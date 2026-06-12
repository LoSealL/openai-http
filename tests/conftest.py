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

Shared test fixtures.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from openai_http.app import create_app
from openai_http.config import Settings, AuthSettings, ServerSettings, QueueSettings, ObservabilitySettings
from openai_http.backends.mock_backend import MockTransformersBackend


@pytest.fixture(scope="session")
def mock_config():
    """Create test configuration with mock backend and auth disabled."""
    return Settings(
        server=ServerSettings(host="127.0.0.1", port=8000),
        auth=AuthSettings(enabled=False, api_keys=[]),
        queue=QueueSettings(depth=32),
        observability=ObservabilitySettings(log_level="debug", log_format="text", metrics_enabled=False),
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
