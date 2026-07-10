MOCK_MODEL = "mock-gpt"


class OpenAITestBase:
    """Base class for OpenAI SDK tests with common utilities."""

    @staticmethod
    def assert_valid_model(model):
        """Assert model object is valid."""
        assert hasattr(model, "id")
        assert hasattr(model, "object")
        assert model.object == "model"
        assert hasattr(model, "created")
        assert hasattr(model, "owned_by")

    @staticmethod
    def assert_valid_usage(usage):
        """Assert usage object is valid."""
        assert hasattr(usage, "prompt_tokens")
        assert hasattr(usage, "completion_tokens")
        assert hasattr(usage, "total_tokens")
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.prompt_tokens >= 0
        assert usage.completion_tokens >= 0

    @staticmethod
    def expect_not_implemented(error):
        """Check if exception indicates endpoint not yet implemented (404 or 501)."""
        error_str = str(error)
        return (
            "404" in error_str or "501" in error_str or "not found" in error_str.lower()
        )