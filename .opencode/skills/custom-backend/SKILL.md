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

## BackendBase Method Reference

### Required Abstract Methods

You **must** implement these four methods:

- `async def generate(prompt, **kwargs) -> dict` — non-streaming completion
- `async def generate_stream(prompt, **kwargs) -> AsyncGenerator[str | dict, None]` — streaming completion
- `async def list_models() -> list[dict]` — list available models
- `async def get_model(model_id: str) -> Optional[dict]` — get model by ID

See `specs/002-extensible-backend-sdk/contracts/python-api.md` for full signatures and return shapes.

### Optional Methods

These have default implementations that raise `NotImplementedError`, causing the server to return HTTP 501.

- `async def embed(texts, **kwargs) -> list[list[float]]` — embeddings
- `async def generate_tool_calls(messages, tools, **kwargs) -> list[dict]` — tool calling

### Lifecycle Hooks

- `async def setup()` — called before accepting requests (load model weights, init GPU)
- `async def teardown()` — called on shutdown (release resources)

## Server Startup

### Using `run_server()`

```python
import openai_http

class MyBackend(openai_http.BackendBase):
    async def generate(self, prompt, **kwargs):
        text = prompt if isinstance(prompt, str) else prompt[-1]["content"]
        return {
            "generated_text": f"Echo: {text}",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def generate_stream(self, prompt, **kwargs):
        result = await self.generate(prompt, **kwargs)
        for word in result["generated_text"].split():
            yield word + " "

    async def list_models(self):
        return [{"id": "my-model", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        models = await self.list_models()
        return next((m for m in models if m["id"] == model_id), None)

if __name__ == "__main__":
    openai_http.setup_logging()
    openai_http.run_server(backend=MyBackend(), port=8000)
```

### Configuration Options

| `run_server` parameter | Default | Description |
|---|---|---|
| `host` | `"0.0.0.0"` | Bind address |
| `port` | `8000` | Listen port |
| `log_level` | `"info"` | Uvicorn log level |
| `api_keys` | `None` | List of API keys for Bearer auth (no auth if empty/None) |
| `queue_depth` | `32` | Max pending requests before HTTP 429 |

## Error Handling

All errors follow OpenAI JSON format. Raise `NotImplementedError` for optional features — the server converts it to HTTP 501. Other exceptions propagate to the global error handler (HTTP 500).

## Testing

The project ships with a `MockTransformersBackend` in `openai_http.backends.mock_backend` — use it as a reference implementation. See `AGENTS.md` for test commands.
