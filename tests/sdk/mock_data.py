"""
Mock data generators for OpenAI SDK tests.

Provides consistent test inputs and expected outputs.
"""

from typing import Dict, List, Any


def simple_chat_messages() -> List[Dict[str, str]]:
    """Generate simple chat messages for testing."""
    return [
        {"role": "user", "content": "Hello, how are you?"}
    ]


def multi_turn_chat_messages() -> List[Dict[str, str]]:
    """Generate multi-turn conversation for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "And what is 3+3?"}
    ]


def long_prompt() -> str:
    """Generate a long text prompt for testing."""
    return "Hello " * 100


def batch_prompts() -> List[str]:
    """Generate batch prompts for testing."""
    return [
        "What is AI?",
        "Explain machine learning",
        "Define neural network"
    ]


def embedding_texts() -> List[str]:
    """Generate texts for embedding tests."""
    return [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is fascinating",
        "OpenAI creates amazing tools"
    ]


def function_definitions() -> List[Dict[str, Any]]:
    """Generate function definitions for tool calling tests."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]


def temperature_params() -> List[float]:
    """Generate temperature values for parameter testing."""
    return [0.0, 0.5, 1.0, 1.5, 2.0]


def max_tokens_params() -> List[int]:
    """Generate max_tokens values for parameter testing."""
    return [10, 50, 100, 500]


def mock_jsonl_content() -> str:
    """Generate mock JSONL content for file upload tests."""
    lines = [
        '{"prompt": "What is AI?", "completion": "AI is artificial intelligence"}',
        '{"prompt": "Define ML", "completion": "ML is machine learning"}',
        '{"prompt": "Explain NN", "completion": "NN is neural network"}'
    ]
    return "\n".join(lines)
