# OpenAI HTTP API Service

OpenAI v1 API compatible HTTP service with pluggable inference backends. Built with FastAPI and supports streaming responses, request queueing, and OpenTelemetry metrics.

## Features

- ✅ OpenAI v1 API compatibility (Models, Chat Completions, Completions, Embeddings)
- ✅ Server-Sent Events (SSE) streaming support
- ✅ Request queue with GPU serialization (prevents concurrent GPU access)
- ✅ TOML + environment variable configuration
- ✅ Structured JSON logging with request IDs
- ✅ OpenTelemetry metrics (Prometheus endpoint)
- ✅ Mock backend for testing (1536-dim embeddings, streaming chat, tool calls)
- ✅ Transformers backend example (`examples/transformers-backend/` — uses Qwen2.5-0.5B)
- ✅ Custom backend SDK (`BackendBase` ABC, `run_server()` entry point)
- 🚧 Audio (speech, transcriptions, translations) endpoints
- 🚧 Image generation / editing endpoints
- ✅ Tool calling (function definitions in chat) — supported in transformers backend

## Quick Start

### Prerequisites

- Python 3.12+ (see `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone and enter directory
git clone <repo>
cd openai-http

# Create venv and install dependencies
uv sync --all-extras
```

### Run the Server

```bash
# Start with mock backend (no GPU required)
uv run -m openai_http

# Server runs on http://0.0.0.0:8000
```

### Test with curl

```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion (non-streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-gpt",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Chat completion (streaming)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-gpt",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'

# Health check
curl http://localhost:8000/health
```

### Run with a real Transformers backend

```bash
# Install heavy deps (not included in default install)
uv pip install torch transformers accelerate

# Start with Qwen2.5-0.5B-Instruct (downloads ~1 GB on first run)
uv run python examples/transformers-backend/transformers_backend.py
```

See [`examples/transformers-backend/README.md`](examples/transformers-backend/README.md) for details.

## Configuration

Configuration uses `config.toml` with environment variable overrides.

### config.toml Example

```toml
[server]
host = "0.0.0.0"
port = 8000

[auth]
enabled = false
api_keys = []  # ["sk-..."] when enabled

[queue]
depth = 32  # max queued requests

[observability]
log_level = "info"
log_format = "json"  # "json" or "text"
metrics_enabled = true
metrics_port = 9464
```

### Environment Variables

Override any config with `OPENAI_HTTP__` prefix (double underscore for nesting):

```bash
OPENAI_HTTP__SERVER__PORT=9000
OPENAI_HTTP__AUTH__ENABLED=true
OPENAI_HTTP__AUTH__API_KEYS=["sk-test-key"]
```

## API Endpoints

### Implemented

- `GET /v1/models` — List available models
- `GET /v1/models/{model_id}` — Get model info
- `POST /v1/chat/completions` — Chat completion (streaming + non-streaming)
- `POST /v1/completions` — Text completion
- `POST /v1/embeddings` — Text embeddings
- `GET /health` — Health check

### Stub endpoints (registered but return 501)

- `POST /v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`
- `POST /v1/images/generations`, `/v1/images/edits`, `/v1/images/variations`

### Planned

- Audio / image generation (currently return 501)
- `POST /v1/moderations`
- `POST /v1/files`, file management
- `POST /v1/fine_tuning/jobs`
- Batch API

## Development

### Run Tests

```bash
# All tests
uv run pytest tests/ -v

# SDK tests only (auto-starts server)
uv run pytest tests/sdk/ -v

# Unit tests only
uv run pytest tests/unit/ -v

# Single test
uv run pytest tests/sdk/test_models.py::TestModelsAPI::test_models_list -v
```

**Note**: SDK tests (`tests/sdk/`) automatically start a mock server on port 8000 in a background thread. Don't start the server manually before running tests.

### Linting & Type Checking

```bash
uv run ruff check openai_http/ tests/
uv run mypy openai_http/
```

### Project Structure

```
openai_http/
├── __init__.py         # Public API: BackendBase, run_server, setup_logging
├── _server.py          # run_server() entry point (library mode)
├── app.py              # FastAPI factory, lifespan, middleware
├── config.py           # pydantic-settings configuration
├── errors.py           # OpenAI-format error handlers
├── queue.py            # Request queue (asyncio.Semaphore)
├── routers/            # API endpoints
│   ├── chat.py        # /v1/chat/completions
│   ├── completions.py # /v1/completions
│   ├── embeddings.py  # /v1/embeddings
│   ├── models.py      # /v1/models
│   ├── audio.py       # /v1/audio/* (stub)
│   ├── images.py      # /v1/images/* (stub)
│   └── health.py      # /health
├── schemas/            # Pydantic v2 models
├── backends/           # Inference backends
│   ├── base.py        # BackendBase ABC
│   └── mock_backend.py # Mock implementation
└── observability/      # Logging + metrics

examples/
└── transformers-backend/  # Real Transformers backend using Qwen2.5-0.5B

tests/
├── conftest.py         # Async httpx fixtures
├── unit/              # Unit tests
└── sdk/               # OpenAI SDK compatibility tests
    └── conftest.py    # Sync OpenAI client fixtures

config.toml            # Configuration
specs/                 # Feature specs & plans
```

## Architecture

### Request Flow

1. Request arrives at FastAPI router
2. `RequestIDMiddleware` assigns unique request ID
3. `RequestLoggingMiddleware` logs request metadata
4. Router calls `queue.acquire()` (waits if GPU busy)
5. Backend executes inference (mock or transformers)
6. Response formatted to OpenAI v1 schema
7. Error handlers catch exceptions and format `{"error": {...}}`

### Custom backends

Create your own backend by subclassing `BackendBase` and plugging it into `run_server()`:

```python
import openai_http

class MyBackend(openai_http.BackendBase):
    async def generate(self, prompt, **kwargs):
        return {"generated_text": "...", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}

    async def generate_stream(self, prompt, **kwargs):
        yield "Hello"
        yield " world"

    async def list_models(self):
        return [{"id": "my-model", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        return {"id": model_id, "object": "model", "created": 0, "owned_by": "me"} if model_id == "my-model" else None

openai_http.run_server(MyBackend(), port=8000)
```

See `examples/transformers-backend/` for a full working implementation.

### Built-in Backend Protocol

All backends subclass `BackendBase` (defined in `backends/base.py`):

| Method | Required | Purpose |
|--------|----------|---------|
| `generate()` | ✅ | Non-streaming completion |
| `generate_stream()` | ✅ | Streaming completion (yields `str` chunks) |
| `list_models()` | ✅ | List available models |
| `get_model()` | ✅ | Get a model by ID |
| `embed()` | ❌ | Embeddings (default → HTTP 501) |
| `generate_tool_calls()` | ❌ | Tool/function calling (default → HTTP 501) |
| `setup()` / `teardown()` | ❌ | Lifecycle hooks (default no-op) |

### Request Queue

GPU inference is serialized using `asyncio.Semaphore(1)` to prevent out-of-memory errors. When the queue reaches `queue.depth`, new requests receive HTTP 429 (Too Many Requests).

## Testing Strategy

- **Unit tests** (`tests/unit/`): Test individual functions/classes in isolation
- **Integration tests** (`tests/integration/`): Test endpoint behavior with httpx
- **Contract tests** (`tests/contract/`): Verify OpenAI API contract compliance
- **SDK tests** (`tests/sdk/`): Use official `openai` Python SDK against running server

Unimplemented endpoints use `pytest.skip()` via `_call_or_skip()` helper and auto-activate when routes are added.

## License

MIT

## Contributing

See `specs/001-openai-http-api/plan.md` for implementation roadmap and `AGENTS.md` for development guidelines.
