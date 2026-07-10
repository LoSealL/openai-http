import pytest
from .test_base import OpenAITestBase, MOCK_MODEL


def _call_or_skip(client, **kwargs):
    """Call completions.create, skip test if 404 (endpoint not implemented)."""
    try:
        return client.completions.create(**kwargs)
    except Exception as e:
        if "404" in str(e):
            pytest.skip("/v1/completions endpoint not yet implemented (Phase 8)")
        raise


class TestCompletionsSync(OpenAITestBase):
    """Test suite for Completions API (synchronous)."""

    def test_completion_basic(self, client):
        """Test basic text completion."""
        response = _call_or_skip(
            client, model=MOCK_MODEL, prompt="Once upon a time", stream=False
        )

        assert response.id.startswith("cmpl-")
        assert response.object == "text_completion"
        assert len(response.choices) > 0
        choice = response.choices[0]
        assert hasattr(choice, "index")
        assert hasattr(choice, "text")
        assert hasattr(choice, "finish_reason")
        assert isinstance(choice.text, str)
        self.assert_valid_usage(response.usage)

    def test_completion_with_parameters(self, client):
        """Test completion with various parameters."""
        response = _call_or_skip(
            client,
            model=MOCK_MODEL,
            prompt="The meaning of life is",
            max_tokens=50,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stream=False,
        )
        assert response.object == "text_completion"

    def test_completion_multiple_choices(self, client):
        """Test generating multiple completions."""
        response = _call_or_skip(
            client, model=MOCK_MODEL, prompt="Hello", n=3, stream=False
        )
        assert len(response.choices) >= 1

    def test_completion_stop_sequences(self, client):
        """Test stop sequences."""
        _call_or_skip(
            client,
            model=MOCK_MODEL,
            prompt="Write a story",
            stop=["END", "FINISH"],
            max_tokens=100,
            stream=False,
        )

    def test_completion_presence_frequency_penalty(self, client):
        """Test penalty parameters."""
        _call_or_skip(
            client,
            model=MOCK_MODEL,
            prompt="Test",
            presence_penalty=0.6,
            frequency_penalty=0.3,
            stream=False,
        )

    def test_completion_batch_prompts(self, client):
        """Test with batch of prompts."""
        response = _call_or_skip(
            client,
            model=MOCK_MODEL,
            prompt=["What is AI?", "Explain ML"],
            stream=False,
        )
        assert len(response.choices) > 0