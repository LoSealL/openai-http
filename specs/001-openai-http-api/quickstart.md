# Quickstart: OpenAI-Compatible HTTP API Service

**Feature**: `001-openai-http-api`

## Prerequisites

- Python >= 3.12
- CUDA-compatible GPU (for real model inference; CPU works for mock mode)
- ~8GB disk for a small model (e.g., Qwen2.5-0.5B)

## Installation

```bash
# Clone and enter project
git clone <repo> && cd openai-http && git checkout 001-openai-http-api

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"
# or
pip install -r requirements.txt
```

## Configuration

Create a `config.toml` in the project root:

```toml
[server]
host = "0.0.0.0"
port = 8000

[auth]
enabled = false
api_keys = ["sk-test-key-1"]

[backend]
type = "transformers"
model_path = "Qwen/Qwen2.5-0.5B-Instruct"
device = "auto"

[queue]
depth = 32

[observability]
log_level = "info"
metrics_enabled = true
metrics_port = 9464
```

Environment variables override TOML values with prefix `OPENAI_HTTP__`:

```bash
OPENAI_HTTP__SERVER__PORT=9000
OPENAI_HTTP__AUTH__ENABLED=true
OPENAI_HTTP__AUTH__API_KEYS='["sk-prod-1","sk-prod-2"]'
OPENAI_HTTP__BACKEND__MODEL_PATH="meta-llama/Llama-3.2-1B-Instruct"
```

## Running the Service

```bash
# Production mode
python -m openai_http

# Development mode (auto-reload)
uvicorn openai_http.app:app --host 0.0.0.0 --port 8000 --reload

# Mock mode (no GPU/model required)
OPENAI_HTTP__BACKEND__TYPE=mock python -m openai_http
```

## Verifying the Service

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Using with OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-test-key-1",  # or any string if auth disabled
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Explain transformers in one sentence."}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

## Running Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests (requires running service)
pytest tests/integration/ -v

# SDK compatibility tests
pytest tests/contract/ -v

# All tests
pytest -v
```

## Project Layout

```
openai_http/              # Main package
├── app.py                # FastAPI app factory
├── config.py             # Config (TOML + env vars)
├── routers/              # API endpoint handlers
├── schemas/              # Pydantic request/response models
├── backends/             # Inference backends (transformers, mock)
├── services/             # Business logic (files, fine-tuning, batch)
└── observability/        # Logging and metrics

tests/                    # Test suite
├── unit/                 # Unit tests
├── integration/          # Endpoint integration tests
└── contract/             # OpenAI SDK compatibility tests

config.toml               # Configuration file
specs/001-openai-http-api/ # Feature documentation
```

## Common Tasks

- **Swap model**: Edit `backend.model_path` in `config.toml` and restart
- **Add API key**: Add to `auth.api_keys` array in config
- **Enable auth**: Set `auth.enabled = true` in config
- **Change queue depth**: Set `queue.depth` (default: 32)
- **Check metrics**: Visit `http://localhost:9464/metrics` (Prometheus format)
