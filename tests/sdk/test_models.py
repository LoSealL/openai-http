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

Tests for OpenAI Models API.

Tests: client.models.list(), client.models.retrieve()
"""

import pytest
from openai import OpenAI
from .test_base import OpenAITestBase


class TestModelsAPI(OpenAITestBase):
    """Test suite for Models API endpoints."""

    def test_models_list(self, client):
        """Test listing all available models."""
        response = client.models.list()

        assert hasattr(response, "object")
        assert response.object == "list"
        assert hasattr(response, "data")
        assert isinstance(response.data, list)
        assert len(response.data) > 0

        for model in response.data:
            self.assert_valid_model(model)

    def test_models_retrieve_invalid(self, client):
        """Test retrieving non-existent model returns 404."""
        with pytest.raises(Exception) as exc_info:
            client.models.retrieve("non-existent-model")

        assert (
            "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
        )

    def test_models_list_empty_auth(self, client, sdk_server):
        """Test models list works without authentication (mock backend)."""
        test_client = OpenAI(
            api_key="random-key-12345", base_url=sdk_server["base_url"]
        )

        response = test_client.models.list()
        assert response.object == "list"
