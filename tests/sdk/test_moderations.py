import pytest
from .test_base import OpenAITestBase


def _call_or_skip(client, **kwargs):
    """Call moderations.create, skip test if 404 (endpoint not implemented)."""
    try:
        return client.moderations.create(**kwargs)
    except Exception as e:
        if "404" in str(e):
            pytest.skip("/v1/moderations endpoint not yet implemented (Phase 12)")
        raise


class TestModerationsAPI(OpenAITestBase):
    """Test suite for Moderations API."""

    def test_moderation_basic(self, client):
        """Test basic moderation check."""
        response = _call_or_skip(client, input="Hello, how are you?")

        assert hasattr(response, "id")
        assert hasattr(response, "results")
        assert len(response.results) > 0
        assert hasattr(response.results[0], "flagged")

    def test_moderation_batch(self, client):
        """Test moderation for batch of inputs."""
        response = _call_or_skip(
            client, input=["Hello world", "How are you?", "What's the weather?"]
        )
        assert len(response.results) == 3

    def test_moderation_empty_string(self, client):
        """Test moderation with empty string."""
        response = _call_or_skip(client, input="")
        assert len(response.results) == 1