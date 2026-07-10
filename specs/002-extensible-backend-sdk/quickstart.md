# Quickstart: Extensible Backend SDK for openai_http

**Feature**: `002-extensible-backend-sdk`

## Prerequisites

- Python >= 3.12

## Installation

```bash
pip install openai_http
```

Or from source:

```bash
git clone <repo> && cd openai-http
uv pip install -e ".[dev]"
```

## Minimal Example

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

## Adding Authentication

```python
openai_http.run_server(
    backend=MyBackend(),
    api_keys=["sk-my-secret-key"],
    port=8000,
)
```

## Configuration Options

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `host` | `"0.0.0.0"` | Bind address |
| `port` | `8000` | Listen port |
| `log_level` | `"info"` | Uvicorn log level |
| `api_keys` | `None` | API keys for Bearer auth (disables auth if empty) |
| `queue_depth` | `32` | Max pending requests before 429 |
| `skip_validation` | `False` | Skip startup contract checks |

## Backward Compatibility

The existing CLI entry point continues to work:

```bash
python -m openai_http
```
