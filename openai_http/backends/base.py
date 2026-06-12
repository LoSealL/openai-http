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

Backend abstraction layer.

Defines the BackendBase abstract class that all inference backends
must implement to integrate with the openai_http server.
"""

import abc
from typing import Any, AsyncGenerator, Optional


class BackendBase(abc.ABC):
    """Abstract base class for inference backends.

    All backend implementations must subclass this and implement the
    abstract methods. Optional methods provide default implementations
    that raise NotImplementedError for unsupported functionality.
    """

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> dict:
        """Generate a completion for the given prompt.

        Args:
            prompt: A plain text string or a list of message dicts
                with role/content keys.
            **kwargs: Additional generation parameters (temperature,
                max_tokens, etc.).

        Returns:
            A dict with at minimum a "generated_text" key and
            optionally a "usage" key with token counts.
        """
        ...

    @abc.abstractmethod
    async def generate_stream(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming completion.

        Yields text chunks as they become available.

        Args:
            prompt: A plain text string or a list of message dicts.
            **kwargs: Additional generation parameters.

        Yields:
            Text chunks as strings.
        """
        ...
        yield ""

    @abc.abstractmethod
    async def list_models(self) -> list[dict]:
        """List all available models.

        Returns:
            A list of model dicts with id, object, created,
            and owned_by keys.
        """
        ...

    @abc.abstractmethod
    async def get_model(self, model_id: str) -> Optional[dict]:
        """Get details for a specific model.

        Args:
            model_id: The model identifier string.

        Returns:
            A model dict if found, or None.
        """
        ...

    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Create embeddings for the given texts.

        Args:
            texts: List of text strings to embed.
            **kwargs: Additional embedding parameters (dimensions, etc.).

        Returns:
            A list of float vectors, one per input text.

        Raises:
            NotImplementedError: If embeddings are not supported.
        """
        raise NotImplementedError("Embeddings are not supported by this backend")

    async def generate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate tool/function call responses.

        Args:
            messages: The conversation history.
            tools: The tool definitions available to the model.
            **kwargs: Additional parameters including tool_choice.

        Returns:
            A list of tool call dicts with id, type, and function keys.

        Raises:
            NotImplementedError: If tool calls are not supported.
        """
        raise NotImplementedError("Tool calls are not supported by this backend")

    async def setup(self) -> None:
        """Initialize backend resources.

        Called once at server startup. Override to load models,
        establish connections, etc.
        """
        pass

    async def teardown(self) -> None:
        """Clean up backend resources.

        Called once at server shutdown. Override to release models,
        close connections, etc.
        """
        pass
