# Quickstart: Extensible Backend SDK for openai_http

**Feature**: `002-extensible-backend-sdk`

## Prerequisites

- Python >= 3.12
- A custom inference model or API to wrap in a backend

## Installation

```bash
pip install openai_http
```

Or from source:

```bash
git clone <repo> && cd openai-http
uv pip install -e ".[dev]"
```

## Minimal Example (5 minutes)

Create a single file `myapp.py`:

```python
import openai_http


class MyBackend(openai_http.BackendBase):

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
        return [
            {"id": "my-model", "object": "model", "created": 0, "owned_by": "me"}
        ]

    async def get_model(self, model_id):
        if model_id == "my-model":
            models = await self.list_models()
            return models[0]
        return None


if __name__ == "__main__":
    openai_http.setup_logging()
    openai_http.run_server(backend=MyBackend())
```

Run it:

```bash
python myapp.py
```

The server starts at `http://localhost:8000`. Test it:

```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "my-model", "messages": [{"role": "user", "content": "Hello"}]}'
```

Use the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

response = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Adding Lifecycle Hooks

For backends that need to load model weights or initialize GPU context:

```python
class GPUBackend(openai_http.BackendBase):

    async def setup(self):
        import torch
        self.model = torch.load("model.pt", map_location="cuda")

    async def teardown(self):
        del self.model
        import torch
        torch.cuda.empty_cache()

    async def generate(self, prompt, **kwargs):
        # use self.model for inference
        ...

    # ... implement other required methods
```

The server calls `setup()` before accepting requests and `teardown()` on shutdown.

## Adding Authentication

```python
openai_http.run_server(
    backend=MyBackend(),
    api_keys=["sk-my-secret-key"],
    port=8000,
)
```

Clients must send `Authorization: Bearer sk-my-secret-key` in request headers.

## Configuration Options

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `host` | `"0.0.0.0"` | Bind address |
| `port` | `8000` | Listen port |
| `log_level` | `"info"` | Uvicorn log level |
| `api_keys` | `None` | API keys for Bearer auth (disables auth if empty) |
| `queue_depth` | `32` | Max pending requests before 429 |
| `skip_validation` | `False` | Skip startup contract checks |

## Implementing Optional Methods

### Embeddings

```python
class EmbeddingBackend(openai_http.BackendBase):
    # ... required methods ...

    async def embed(self, texts, **kwargs):
        return [[0.1, 0.2, 0.3] for _ in texts]  # your embedding logic
```

If you don't implement `embed`, `POST /v1/embeddings` returns HTTP 501.

### Tool Calls

```python
class ToolBackend(openai_http.BackendBase):
    # ... required methods ...

    async def generate_tool_calls(self, messages, tools, **kwargs):
        return [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": tools[0]["function"]["name"],
                    "arguments": '{"key": "value"}',
                },
            }
        ]
```

## Backward Compatibility

The existing CLI entry point continues to work:

```bash
python -m openai_http
```

This uses `config.toml` for configuration and the built-in mock backend — no changes required for existing deployments.

## Project Layout

```
openai_http/                    # Library package
├── __init__.py                 # Public API: BackendBase, run_server, setup_logging
├── backends/
│   ├── base.py                 # BackendBase ABC definition
│   └── mock_backend.py         # Built-in mock backend
├── app.py                      # FastAPI application factory
├── config.py                   # Settings (TOML + env vars)
├── routers/                    # API endpoint handlers
├── schemas/                    # Pydantic request/response models
├── errors.py                   # OpenAI-format error handling
├── queue.py                    # Request concurrency control
└── observability/              # Logging and metrics

specs/002-extensible-backend-sdk/  # Feature documentation
```
