---
name: custom-backend
description: >
  How to implement a custom inference Backend for the openai_http server
  and start the server with it. Use this skill whenever the user asks about
  creating or customizing backends, implementing inference logic, setting up
  run_server(), extending the openai_http server with custom model support,
  or working with BackendBase. Also trigger when the user wants to serve
  custom models via an OpenAI-compatible API endpoint.
---

# Custom Backend Skill for openai_http

## Overview

`openai_http` is an OpenAI v1 API-compatible HTTP server with a pluggable backend system. The core abstraction is `BackendBase` (an ABC in `openai_http.backends.base`). You subclass it, implement the required methods, and pass an instance to `openai_http.run_server()`.

```
┌─────────────────────────────────────────────┐
│                 HTTP Clients                │
├─────────────────────────────────────────────┤
│        FastAPI (routers/chat, models...)    │
├─────────────────────────────────────────────┤
│           Your Backend (BackendBase)        │
│         ┌─────────────────────────────┐     │
│         │  Your model / API / logic   │     │
│         └─────────────────────────────┘     │
└─────────────────────────────────────────────┘
```

## BackendBase Method Reference

### Required Abstract Methods

You **must** implement these four methods:

#### `async def generate(prompt, **kwargs) -> dict`

Called for **non-streaming** chat completions (`POST /v1/chat/completions` with `stream: false`).

| Parameter | Type | Description |
|-----------|------|-------------|
| `prompt` | `str \| list[dict[str, str]]` | If `str`: raw prompt text. If `list`: messages array — `[{"role": "user", "content": "..."}, ...]` |
| `**kwargs` | `Any` | Extra params the client passes: `max_tokens`, `temperature`, `top_p`, `tools` (when the server handles tool calls inline) |

**Return** a dict (or `GenerationResult` instance from `openai_http.backends.types`) with these keys:

```python
{
    "generated_text": str | None,        # The model's response text
    "reasoning_content": str | None,     # Optional <think> reasoning text
    "tool_calls": list[dict] | None,     # Backend-emitted tool calls (BackendToolCall shape)
    "finish_reason": "stop" | "length" | "tool_calls" | "content_filter",
    "usage": {                           # Token accounting
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
    },
}
```

> **Contract validation**: every router validates this dict against
> `openai_http.backends.types.GenerationResult`. Returning extra keys,
> the wrong `finish_reason`, or a missing `usage` block produces an
> HTTP 500 with `error.code == "backend_contract_error"`. Pydantic
> instances from `openai_http.backends.types` are also accepted.

#### `async def generate_stream(prompt, **kwargs) -> AsyncGenerator[str | dict, None]`

Called for **streaming** chat completions (`stream: true`). Yield either plain `str` tokens (treated as content) or typed dicts. Each dict is validated against `openai_http.backends.types.StreamChunk` at the router boundary.

| Yielded value | Type field | Meaning |
|---|---|---|
| `"...text..."` | n/a | Content fragment (legacy form, equivalent to `{"type": "content", ...}`) |
| `{"type": "content", "content": "..."}` | `content` | Answer chunk |
| `{"type": "reasoning", "content": "..."}` | `reasoning` | `<think>` reasoning chunk |
| `{"type": "finish", "reason": "stop" \| "length" \| "tool_calls" \| "content_filter"}` | `finish` | Terminal marker |

```python
async for token in backend.generate_stream(messages, **kwargs):
    yield token  # str OR one of the typed dicts above
```

Anything else — including a chunk with an unknown `type` value or a
non-string `content` field — produces an HTTP 500 mid-stream.

#### `async def list_models() -> list[dict]`

Called for `GET /v1/models`. Return a list of model objects — each entry is validated against `openai_http.backends.types.ModelInfo`:

```python
[
    {
        "id": str,           # e.g. "my-model"
        "object": "model",
        "created": int,      # Unix timestamp
        "owned_by": str,     # e.g. "me"
    },
]
```

A missing field (`id`, `created`, or `owned_by`) raises HTTP 500 `backend_contract_error`.

#### `async def get_model(model_id: str) -> Optional[dict]`

Called for `GET /v1/models/{model_id}`. Return the matching model dict or `None` (which causes the server to return HTTP 404).

```python
async def get_model(self, model_id: str) -> Optional[dict]:
    for model in await self.list_models():
        if model["id"] == model_id:
            return model
    return None
```

### Optional Methods

These have default implementations that raise `NotImplementedError`, causing the server to return HTTP 501.

#### `async def embed(texts, **kwargs) -> list[list[float]]`

Called for `POST /v1/embeddings`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `texts` | `list[str]` | Input texts to embed |
| `**kwargs` | `Any` | `dimensions` (int), `model` (str), etc. |

Return a list of embedding vectors — `list[list[float]]`.

#### `async def generate_tool_calls(messages, tools, **kwargs) -> list[dict]`

Called when the chat request includes `tools` and `tool_choice` is not `"none"`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `messages` | `list[dict[str, Any]]` | Full chat messages array |
| `tools` | `list[dict[str, Any]]` | Tool definitions per OpenAI spec |
| `**kwargs` | `Any` | `tool_choice` (str or dict) |

Return a list of tool call objects:

```python
[
    {
        "id": str,       # Unique call ID, e.g. "call_abc123"
        "type": "function",
        "function": {
            "name": str,
            "arguments": str,  # JSON-encoded args
        },
    },
]
```

### Lifecycle Hooks

#### `async def setup()`

Called **once** when the server starts, before accepting requests. Use for:
- Loading model weights
- Initializing GPU context
- Connecting to external services
- Any expensive one-time initialization

The server crashes with `RuntimeError("Backend setup failed: ...")` if `setup()` raises.

#### `async def teardown()`

Called **once** when the server shuts down. Use for:
- Releasing GPU memory
- Closing connections
- Cleaning up temporary files

The server only calls `teardown()` if a custom backend was injected (not the default mock).

## Server Startup

### Using `run_server()`

```python
import openai_http

class MyBackend(openai_http.BackendBase):
    ...

openai_http.setup_logging()
openai_http.run_server(backend=MyBackend())
```

#### `run_server()` signature:

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
) -> None
```

### Validation at Startup

Unless `skip_validation=True`, `run_server()` performs a **fail-fast contract check** at startup:

1. Calls `backend.setup()` to let the backend initialize
2. Calls each abstract method with sample inputs to verify they don't crash
3. Checks return types are dict/list where expected
4. Exits with a clear error message if validation fails

This catches common issues (missing methods, wrong return types) before the server starts listening.

### Configuration Options

| `run_server` parameter | Default | Description |
|---|---|---|
| `host` | `"0.0.0.0"` | Bind address |
| `port` | `8000` | Listen port |
| `log_level` | `"info"` | Uvicorn log level |
| `api_keys` | `None` | List of API keys for Bearer auth (no auth if empty/None) |
| `queue_depth` | `32` | Max pending requests before HTTP 429 |
| `skip_validation` | `False` | Skip startup contract checks |

### Using config.toml

When running via `python -m openai_http` (CLI mode), configuration comes from `config.toml` and the built-in mock backend is used. This path does NOT use custom backends — point the user to `run_server()` instead.

## Examples

### Minimal Backend

```python
import openai_http
from typing import AsyncGenerator, Optional


class EchoBackend(openai_http.BackendBase):

    async def generate(self, prompt, **kwargs):
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        return {
            "generated_text": f"Echo: {text}",
            "usage": {
                "prompt_tokens": len(text.split()),
                "completion_tokens": len(text.split()) + 1,
                "total_tokens": len(text.split()) * 2 + 1,
            },
        }

    async def generate_stream(self, prompt, **kwargs):
        result = await self.generate(prompt, **kwargs)
        for word in result["generated_text"].split():
            yield word + " "

    async def list_models(self):
        return [{"id": "echo-model", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        models = await self.list_models()
        return next((m for m in models if m["id"] == model_id), None)


if __name__ == "__main__":
    openai_http.setup_logging()
    openai_http.run_server(backend=EchoBackend(), port=8000)
```

### Backend with Lifecycle Hooks

Use when the backend needs to load model weights, initialize GPU context, or manage external connections.

```python
class GPUBackend(openai_http.BackendBase):

    async def setup(self):
        import torch
        # Load model weights before accepting requests
        self.model = torch.load("model.pt", map_location="cuda")
        self.tokenizer = torch.load("tokenizer.pt")

    async def teardown(self):
        # Clean up on shutdown
        import torch
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

    async def generate(self, prompt, **kwargs):
        # Use self.model for inference
        ...

    async def generate_stream(self, prompt, **kwargs):
        ...

    async def list_models(self):
        return [{"id": "gpu-model", ...}]

    async def get_model(self, model_id):
        ...
```

### Backend with All Optional Methods

```python
class FullBackend(openai_http.BackendBase):

    async def setup(self):
        self.client = OpenAI(...)  # wrap another API

    async def teardown(self):
        self.client.close()

    # Required
    async def generate(self, prompt, **kwargs): ...
    async def generate_stream(self, prompt, **kwargs): ...
    async def list_models(self): ...
    async def get_model(self, model_id): ...

    # Optional: Embeddings
    async def embed(self, texts, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]

    # Optional: Tool calls
    async def generate_tool_calls(self, messages, tools, **kwargs):
        return [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": tools[0]["function"]["name"],
                    "arguments": '{"key": "value"}',
                },
            }
        ]
```

### Backend with API Key Auth

```python
openai_http.run_server(
    backend=MyBackend(),
    api_keys=["sk-my-secret-key"],
    port=8000,
)
```

Clients must send `Authorization: Bearer sk-my-secret-key`.

## How the Server Maps Requests to Backend Methods

| HTTP Endpoint | Backend Method | Streaming? |
|---|---|---|
| `GET /v1/models` | `list_models()` | No |
| `GET /v1/models/{id}` | `get_model(id)` | No |
| `POST /v1/chat/completions` (non-stream) | `generate()` | No |
| `POST /v1/chat/completions` (stream) | `generate_stream()` | Yes |
| `POST /v1/chat/completions` (with tools) | `generate_tool_calls()` | No (tool prefix) |
| `POST /v1/embeddings` | `embed()` | No |
| Other endpoints | → HTTP 501 `NotImplemented` | — |

## Error Handling Convention

All errors in `openai_http` follow this JSON format:

```json
{
    "error": {
        "message": "Human-readable description",
        "type": "server_error",
        "param": null,
        "code": "error_code"
    }
}
```

If your backend encounters an error, raise `NotImplementedError` for optional features — the server converts it to HTTP 501. For other errors, let the exception propagate — the global error handler formats it correctly.

## Testing

The project ships with a `MockTransformersBackend` in `openai_http.backends.mock_backend` — use it as a reference implementation. The test suite uses `pytest` with an async `httpx.AsyncClient` for endpoint tests and the OpenAI SDK for compatibility tests. See the project's `AGENTS.md` for test commands.

## Quick Reference Card

```
BackendBase
├── Required
│   ├── generate(prompt, **kwargs) -> dict
│   ├── generate_stream(prompt, **kwargs) -> AsyncGenerator[str]
│   ├── list_models() -> list[dict]
│   └── get_model(model_id) -> Optional[dict]
├── Optional (→ 501 if not implemented)
│   ├── embed(texts, **kwargs) -> list[list[float]]
│   └── generate_tool_calls(messages, tools, **kwargs) -> list[dict]
└── Lifecycle
    ├── setup()  (before serving)
    └── teardown()  (on shutdown)

run_server(backend, host="0.0.0.0", port=8000, ...)
```
