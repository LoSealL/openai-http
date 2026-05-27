# OpenAI HTTP API Service

OpenAI v1 API compatible HTTP service with pluggable inference backends. Built with FastAPI and supports streaming responses, request queueing, and OpenTelemetry metrics.

## Features

- ✅ OpenAI v1 API compatibility (Models, Chat Completions)
- ✅ Server-Sent Events (SSE) streaming support
- ✅ Request queue with GPU serialization (prevents concurrent GPU access)
- ✅ TOML + environment variable configuration
- ✅ Structured JSON logging with request IDs
- ✅ OpenTelemetry metrics (Prometheus endpoint)
- ✅ Mock backend for testing (1536-dim embeddings, streaming chat)
- 🚧 Transformers backend (Phase 6 - in progress)
- 🚧 Completions, Embeddings endpoints (Phase 7-8)
- 🚧 Tool calling, Files API (Phase 9-10)

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

## Configuration

Configuration uses `config.toml` with environment variable overrides.

### config.toml Example

```toml
[server]
host = "0.0.0.0"
port = 8000

[backend]
type = "mock"  # "mock" or "transformers"
device = "auto"  # "cuda", "cpu", "auto"
# model_path = "Qwen/Qwen2.5-0.5B-Instruct"  # for transformers backend

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
OPENAI_HTTP__BACKEND__TYPE=transformers
OPENAI_HTTP__BACKEND__MODEL_PATH=Qwen/Qwen2.5-0.5B-Instruct
OPENAI_HTTP__AUTH__ENABLED=true
OPENAI_HTTP__AUTH__API_KEYS=["sk-test-key"]
```

## API Endpoints

### Implemented

- `GET /v1/models` — List available models
- `GET /v1/models/{model_id}` — Get model info
- `POST /v1/chat/completions` — Chat completion (streaming + non-streaming)
- `GET /health` — Health check

### Planned (by phase)

- **Phase 7**: `POST /v1/embeddings`
- **Phase 8**: `POST /v1/completions`, `POST /v1/moderations`
- **Phase 9**: Tool calling (function definitions in chat)
- **Phase 10**: `POST /v1/files`, file management
- **Phase 11**: `POST /v1/fine_tuning/jobs`
- **Phase 12**: Batch API

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
├── app.py              # FastAPI factory, lifespan, middleware
├── config.py           # pydantic-settings configuration
├── errors.py           # OpenAI-format error handlers
├── queue.py            # Request queue (asyncio.Semaphore)
├── routers/            # API endpoints
│   ├── chat.py        # /v1/chat/completions
│   ├── models.py      # /v1/models
│   └── health.py      # /health
├── schemas/            # Pydantic v2 models
├── backends/           # Inference backends
│   ├── base.py        # Backend Protocol
│   └── mock_backend.py # Mock implementation
├── services/           # Business logic (planned)
└── observability/      # Logging + metrics

tests/
├── conftest.py         # Async httpx fixtures
├── unit/              # Unit tests
├── integration/       # Integration tests
├── contract/          # Contract tests
└── sdk/               # OpenAI SDK compatibility tests
    └── conftest.py    # Sync OpenAI client fixtures

config.toml            # Configuration
specs/001-openai-http-api/  # Specification & planning
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

### Backend Protocol

All backends implement the `Backend` Protocol in `backends/base.py`:

```python
class Backend(Protocol):
    async def generate(self, prompt: str | list[dict], **kwargs) -> dict
    async def generate_stream(self, prompt: str | list[dict], **kwargs) -> AsyncGenerator[str, None]
    async def embed(self, texts: list[str], **kwargs) -> list[list[float]]
    async def list_models(self) -> list[dict]
    async def get_model(self, model_id: str) -> Optional[dict]
```

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
