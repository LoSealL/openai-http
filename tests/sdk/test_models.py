"""
Tests for OpenAI Models API.

Tests: client.models.list(), client.models.retrieve()
"""

import pytest
from .test_base import OpenAITestBase


class TestModelsAPI(OpenAITestBase):
    """Test suite for Models API endpoints."""
    
    def test_models_list(self, client):
        """Test listing all available models."""
        response = client.models.list()
        
        assert hasattr(response, 'object')
        assert response.object == 'list'
        assert hasattr(response, 'data')
        assert isinstance(response.data, list)
        assert len(response.data) > 0
        
        # Validate each model
        for model in response.data:
            self.assert_valid_model(model)
    
    def test_models_retrieve(self, client):
        """Test retrieving a specific model."""
        # First get a valid model ID
        models = client.models.list()
        assert len(models.data) > 0
        model_id = models.data[0].id
        
        # Retrieve specific model
        model = client.models.retrieve(model_id)
        self.assert_valid_model(model)
        assert model.id == model_id
    
    def test_models_retrieve_invalid(self, client):
        """Test retrieving non-existent model returns 404."""
        with pytest.raises(Exception) as exc_info:
            client.models.retrieve("non-existent-model")
        
        # OpenAI client raises NotFound error
        assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
    
    def test_models_list_empty_auth(self, client, sdk_server):
        """Test models list works without authentication (mock backend)."""
        from openai import OpenAI
        
        test_client = OpenAI(
            api_key="random-key-12345",
            base_url=sdk_server["base_url"]
        )
        
        response = test_client.models.list()
        assert response.object == 'list'
