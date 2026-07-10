import pytest
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
