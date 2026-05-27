"""
Backend Protocol defining the inference abstraction layer.

All backends must implement these methods to be used with the service.
The HTTP layer passes raw data through this interface.
"""

from typing import Protocol, AsyncGenerator, Optional, Any


class Backend(Protocol):
    """
    Protocol for inference backends.

    Backends are responsible for:
    - Applying chat templates (LLaMA, Mistral, Qwen, ChatML, etc.)
    - Loading and managing model weights
    - Generating text completions (sync)
    - Streaming text completions (async)
    - Generating embeddings (if supported)
    """

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> dict:
        """
        Generate text completion.

        Args:
            prompt: Raw text or messages list (backend applies chat template)
            **kwargs: Generation parameters (max_tokens, temperature, etc.)

        Returns:
            dict with 'generated_text' and 'usage' keys:
            {
                "generated_text": "response text",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            }
        """
        ...

    async def generate_stream(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text completion token by token.

        Args:
            prompt: Raw text or messages list
            **kwargs: Generation parameters

        Yields:
            str: Individual tokens or text chunks
        """
        ...
        yield ""

    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of input texts
            **kwargs: Embedding parameters (dimensions, encoding_format, etc.)

        Returns:
            List of embedding vectors (list of floats)
        """
        ...

    async def list_models(self) -> list[dict]:
        """
        List available models.

        Returns:
            List of model info dicts with 'id', 'created', 'owned_by', 'object' keys
        """
        ...

    async def get_model(self, model_id: str) -> Optional[dict]:
        """
        Get info for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Model info dict or None if not found
        """
        ...
