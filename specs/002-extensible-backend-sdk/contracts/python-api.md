# Python Public API Contract: openai_http

**Feature**: `002-extensible-backend-sdk`
**Version**: 0.2.0
**Date**: 2026-05-27

## Top-Level Namespace

```python
import openai_http

# Public names (enforced via __all__):
openai_http.BackendBase       # Abstract base class for custom backends
openai_http.run_server        # Blocking server entry point
openai_http.setup_logging     # Convenience function for logging
openai_http.__version__       # Package version string
```

---

## BackendBase

```python
class BackendBase(abc.ABC):
    """
    Abstract base class for inference backends.

    Subclass this and implement the required abstract methods
    to create a custom backend for the openai_http server.

    Required methods (must implement):
        generate, generate_stream, list_models, get_model

    Optional methods (default raises NotImplementedError):
        embed, generate_tool_calls

    Lifecycle hooks (default no-op):
        setup, teardown
    """

    # --- Required Abstract Methods ---

    @abstractmethod
    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> dict:
        """
        Generate text completion.

        Args:
            prompt: Raw text string or list of message dicts
                    (each with "role" and "content" keys).
            **kwargs: Generation parameters — common keys include:
                max_tokens (int): Maximum tokens to generate
                temperature (float): Sampling temperature
                top_p (float): Nucleus sampling probability

        Returns:
            dict with exactly these keys:
                generated_text (str): The generated completion text
                usage (dict): Token usage stats with keys:
                    prompt_tokens (int)
                    completion_tokens (int)
                    total_tokens (int)

        Raises:
            Any exception — will be caught by the server and mapped
            to an OpenAI-format 500 error response.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text completion token by token.

        Args:
            prompt: Same as generate().
            **kwargs: Same as generate().

        Yields:
            str: Individual text chunks to be streamed via SSE.

        Raises:
            Any exception — will be caught and mapped to error.
        """
        ...
        yield ""

    @abstractmethod
    async def list_models(self) -> list[dict]:
        """
        List available models.

        Returns:
            List of dicts, each with:
                id (str): Model identifier
                object (str): Always "model"
                created (int): Unix timestamp
                owned_by (str): Model owner/organization
        """
        ...

    @abstractmethod
    async def get_model(self, model_id: str) -> Optional[dict]:
        """
        Get info for a specific model.

        Args:
            model_id: Model identifier string.

        Returns:
            Model info dict (same shape as list_models entries)
            or None if the model is not found.
        """
        ...

    # --- Optional Methods ---

    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """
        Generate embeddings (optional).

        Default implementation raises NotImplementedError,
        causing POST /v1/embeddings to return HTTP 501.

        Args:
            texts: List of input text strings.
            **kwargs: Optional parameters (e.g., dimensions).

        Returns:
            List of embedding vectors (each a list of floats).
        """
        raise NotImplementedError("Embeddings are not supported by this backend")

    async def generate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate tool calls (optional).

        Default implementation raises NotImplementedError.

        Args:
            messages: Conversation messages list.
            tools: Tool definitions from the request.
            **kwargs: Additional parameters.

        Returns:
            List of tool call dicts, each with:
                id (str): Unique call ID
                type (str): "function"
                function (dict): {"name": str, "arguments": str (JSON)}
        """
        raise NotImplementedError("Tool calls are not supported by this backend")

    # --- Lifecycle Hooks ---

    async def setup(self) -> None:
        """
        Called before the server starts accepting requests.

        Override to perform async initialization such as
        loading model weights, initializing GPU context,
        or establishing external connections.

        Raises:
            Any exception — server startup will be aborted
            and the error surfaced to the operator.
        """
        pass

    async def teardown(self) -> None:
        """
        Called during server shutdown.

        Override to release resources such as GPU memory,
        close file handles, or flush buffers.
        """
        pass
```

---

## run_server

```python
def run_server(
    backend: BackendBase,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    log_level: str = "info",
    api_keys: list[str] | None = None,
    queue_depth: int = 32,
    skip_validation: bool = False,
) -> None:
    """
    Start the OpenAI-compatible HTTP server with the given backend.

    This function blocks the calling thread, running the server
    until interrupted (Ctrl+C / SIGINT), then returns.

    Args:
        backend: An instance of BackendBase (or subclass).
        host: Bind address (default: "0.0.0.0").
        port: Listen port (default: 8000).
        log_level: Uvicorn log level (default: "info").
        api_keys: List of valid API keys for Bearer token auth.
                  None or empty list disables authentication.
        queue_depth: Max pending requests before returning 429
                     (default: 32).
        skip_validation: If True, skip backend contract validation
                        at startup (default: False).

    Raises:
        BackendValidationError: If backend fails contract validation
                               (and skip_validation is False).
        TypeError: If backend is not an instance of BackendBase.

    Example:
        >>> import openai_http
        >>> class MyBackend(openai_http.BackendBase):
        ...     async def generate(self, prompt, **kwargs):
        ...         return {"generated_text": "Hello!", "usage": {...}}
        ...     # ... implement other required methods
        >>> backend = MyBackend()
        >>> openai_http.run_server(backend=backend, port=8000)
    """
```

---

## setup_logging

```python
def setup_logging(level: str = "info") -> None:
    """
    Install a default console log handler on the openai_http logger.

    Optional convenience function. Call before run_server() if you
    want to see server log output without configuring Python's
    logging module yourself.

    Args:
        level: Log level string (default: "info").
               Accepted values: "debug", "info", "warning", "error".
    """
```

---

## Error Behavior Summary

| Scenario | HTTP Status | Error Type | Response Body |
| -------- | ----------- | ---------- | ------------- |
| Backend method raises unhandled exception | 500 | `server_error` | `{"error": {"message": "...", "type": "server_error", "param": null, "code": "internal_error"}}` |
| Optional backend method not implemented | 501 | `not_implemented_error` | `{"error": {"message": "<method> not supported by this backend", "type": "not_implemented_error", "param": null, "code": null}}` |
| Backend setup() raises exception | N/A (server won't start) | RuntimeError | Traceback printed to stderr, process exits non-zero |
