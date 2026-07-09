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

The router boundary validates each backend return value against the
typed contracts in :mod:`openai_http.backends.types`. Backends may
return either a plain ``dict`` matching the documented shape or the
corresponding Pydantic model instance. Mismatched output produces an
HTTP 500 ``server_error`` with code ``backend_contract_error``.
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
            A mapping that conforms to
            :class:`openai_http.backends.types.GenerationResult`.
            Required keys::

                {
                    "generated_text": Optional[str],
                    "reasoning_content": Optional[str],
                    "tool_calls": Optional[list[BackendToolCall-shaped]],
                    "finish_reason": "stop" | "length" | "tool_calls"
                                     | "content_filter",
                    "usage": {
                        "prompt_tokens": int,
                        "completion_tokens": int,
                        "total_tokens": int,
                    },
                }

            Returning a ``GenerationResult`` instance directly is also
            supported.
        """

    @abc.abstractmethod
    async def generate_stream(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """Generate a streaming completion.

        Yields chunks as they become available.

        Each yielded item is either:

        * A plain ``str``: treated as a content chunk (backward
          compatible with simple backends).
        * A typed dict matching one of the
          :data:`openai_http.backends.types.StreamChunk` variants::

              {"type": "reasoning", "content": "..."}
              {"type": "content",   "content": "..."}
              {"type": "finish",    "reason": "stop" | "length" | ...}

        * One of the corresponding Pydantic instances
          (``ReasoningChunk``, ``ContentChunk``, ``FinishChunk``).

        Args:
            prompt: A plain text string or a list of message dicts.
            **kwargs: Additional generation parameters.

        Yields:
            Text chunks or typed dicts.
        """

        yield ""

    @abc.abstractmethod
    async def list_models(self) -> list[dict]:
        """List all available models.

        Returns:
            A list of mappings each conforming to
            :class:`openai_http.backends.types.ModelInfo`. Required
            keys per entry: ``id``, ``object``, ``created``,
            ``owned_by``.
        """

    @abc.abstractmethod
    async def get_model(self, model_id: str) -> Optional[dict]:
        """Get details for a specific model.

        Args:
            model_id: The model identifier string.

        Returns:
            A mapping conforming to
            :class:`openai_http.backends.types.ModelInfo` if found, or
            ``None``.
        """

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

    async def setup(self) -> None:
        """Initialize backend resources.

        Called once at server startup. Override to load models,
        establish connections, etc.
        """

    async def teardown(self) -> None:
        """Clean up backend resources.

        Called once at server shutdown. Override to release models,
        close connections, etc.
        """

    async def metrics(self) -> dict:
        """Return backend-specific metrics and statistics.

        Called for ``GET /metrics``. Override to expose counters,
        latency histograms, or other operational statistics.

        Returns:
            A mapping of metric names to values.

        Raises:
            NotImplementedError: If metrics are not supported.
        """
        raise NotImplementedError("Metrics are not supported by this backend")

    async def health(self) -> dict:
        """Return backend-specific health status.

        Called for ``GET /health`` and merged into the base health
        response, allowing backends to overlay custom status fields.

        Returns:
            A mapping of health field names to values.

        Raises:
            NotImplementedError: If health details are not supported.
        """
        raise NotImplementedError("Health details are not supported by this backend")
