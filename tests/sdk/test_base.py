"""
Base class and utilities for OpenAI SDK tests.

Provides OpenAITestBase class with common assertion methods.
"""

# Available models in mock backend
MOCK_MODELS = ["mock-gpt", "mock-llama"]
DEFAULT_MODEL = "mock-gpt"


class OpenAITestBase:
    """Base class for OpenAI SDK tests with common utilities."""
    
    @staticmethod
    def assert_valid_model(model):
        """Assert model object is valid."""
        assert hasattr(model, 'id')
        assert hasattr(model, 'object')
        assert model.object == 'model'
        assert hasattr(model, 'created')
        assert hasattr(model, 'owned_by')
    
    @staticmethod
    def assert_valid_usage(usage):
        """Assert usage object is valid."""
        assert hasattr(usage, 'prompt_tokens')
        assert hasattr(usage, 'completion_tokens')
        assert hasattr(usage, 'total_tokens')
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
        assert usage.prompt_tokens >= 0
        assert usage.completion_tokens >= 0
    
    @staticmethod
    def assert_valid_chat_choice(choice):
        """Assert chat completion choice is valid."""
        assert hasattr(choice, 'index')
        assert hasattr(choice, 'message')
        assert hasattr(choice, 'finish_reason')
        assert choice.message.role == 'assistant'
        assert choice.finish_reason in ['stop', 'length', 'content_filter', 'tool_calls', None]
    
    @staticmethod
    def assert_valid_completion_choice(choice):
        """Assert completion choice is valid."""
        assert hasattr(choice, 'index')
        assert hasattr(choice, 'text')
        assert hasattr(choice, 'finish_reason')
        assert isinstance(choice.text, str)
    
    @staticmethod
    def expect_not_implemented(error):
        """Check if exception indicates endpoint not yet implemented (404 or 501)."""
        error_str = str(error)
        return "404" in error_str or "501" in error_str or "not found" in error_str.lower()
