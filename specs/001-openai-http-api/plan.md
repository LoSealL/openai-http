# Implementation Plan: OpenAI-Compatible HTTP API Service

**Branch**: `001-openai-http-api` | **Date**: 2026-05-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-openai-http-api/spec.md`

## Summary

Implement a production-ready OpenAI v1 API-compatible HTTP service in Python using FastAPI, backed by HuggingFace Transformers for inference. The service abstracts model inference behind a pluggable backend interface, supporting all OpenAI v1 endpoints (chat/completions, completions, embeddings, models, moderations, files, fine-tuning, audio, images, batches) with full streaming, authentication, observability, and queue-based concurrency control. The implementation extends the existing FastAPI codebase (`main.py`, `mock_backend.py`) with real transformers inference, proper request/response schemas, configuration via TOML, and OpenTelemetry metrics.

## Technical Context

**Language/Version**: Python >=3.12 (per `pyproject.toml`)

**Primary Dependencies**: FastAPI, uvicorn[standard], pydantic>=2.5, pydantic-settings, transformers, torch, tomllib (builtin), opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-prometheus

**Storage**: Local filesystem (files API, batch files) + in-memory state (fine-tuning jobs, batch jobs) + SQLite for file metadata index

**Testing**: pytest, httpx (for async endpoint tests), openai SDK (for integration/compat tests)

**Target Platform**: Linux server (primary), macOS/Windows (development). CUDA GPU required for transformers backend.

**Project Type**: Web service (HTTP API server)

**Performance Goals**: First token < 2s (SC-002), service startup < 60s (SC-008), 50 concurrent non-streaming requests without errors (SC-003)

**Constraints**: Single GPU inference (one request at a time, bounded queue depth 32 default). All responses must pass OpenAI v1 schema validation (SC-004).

**Scale/Scope**: Self-hosted single model / small cluster. ~30 HTTP endpoints covering full OpenAI v1 API surface.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is a template with no enforced principles. No gates to evaluate. **PASSED**.

## Project Structure

### Documentation (this feature)

```text
specs/001-openai-http-api/
├── plan.md              # This file
├── research.md          # Research decisions (R-001 through R-013)
├── data-model.md        # Entity definitions and relationships
├── quickstart.md        # Developer setup guide
├── contracts/           # OpenAPI v3 specification
│   └── openai-v1-api.yaml
└── tasks.md             # Tasks (created by /speckit.tasks)
```

### Source Code (repository root)

```text
openai_http/
├── __init__.py
├── config.py                  # TOML + env var configuration (pydantic-settings)
├── app.py                     # FastAPI application factory, lifespan, middleware
├── auth.py                    # Bearer token auth dependency
├── errors.py                  # OpenAI-format error responses, exception handlers
├── queue.py                   # Bounded FIFO request queue (asyncio.Semaphore + Queue)
├── routers/
│   ├── __init__.py
│   ├── chat.py                # POST /v1/chat/completions (stream + non-stream)
│   ├── completions.py         # POST /v1/completions (stream + non-stream)
│   ├── embeddings.py          # POST /v1/embeddings
│   ├── models.py              # GET /v1/models, GET /v1/models/{model}
│   ├── moderations.py         # POST /v1/moderations
│   ├── files.py               # Files API (upload/list/retrieve/delete/content)
│   ├── fine_tuning.py         # Fine-tuning API (create/list/retrieve/cancel/events)
│   ├── audio.py               # POST /v1/audio/transcriptions, /v1/audio/translations
│   ├── images.py              # POST /v1/images/generations, edits, variations
│   ├── batches.py             # Batch API (create/retrieve/cancel/list)
│   └── health.py              # GET /health
├── schemas/
│   ├── __init__.py
│   ├── chat.py                # ChatCompletionRequest/Response, ChatMessage, ToolCall
│   ├── completions.py         # CompletionRequest/Response
│   ├── embeddings.py          # EmbeddingRequest/Response
│   ├── files.py               # FileObject, file upload models
│   ├── fine_tuning.py         # FineTuningJob, events
│   ├── audio.py               # Transcription/Translation request/response
│   ├── images.py              # ImageGenerationRequest/Response
│   ├── batches.py             # BatchRequest/Response
│   ├── models.py              # ModelObject, ModelList
│   ├── common.py              # UsageInfo, ErrorResponse, shared types
│   └── moderation.py          # ModerationRequest/Response
├── backends/
│   ├── __init__.py            # BackendFactory, backend registry
│   ├── base.py                # Backend Protocol / ABC definition
│   ├── transformers_backend.py # HuggingFace Transformers backend (real inference)
│   └── mock_backend.py        # Mock backend (testing, development)
├── services/
│   ├── __init__.py
│   ├── file_store.py          # File persistence (local FS + SQLite metadata)
│   ├── fine_tuning_service.py # Fine-tuning job lifecycle management
│   └── batch_service.py       # Batch job processing
└── observability/
    ├── __init__.py
    ├── logging.py             # Structured JSON logging setup
    └── metrics.py             # OpenTelemetry metric instruments

tests/
├── __init__.py
├── conftest.py                # Shared fixtures (test client, mock backend)
├── unit/
│   ├── __init__.py
│   ├── test_schemas.py        # Pydantic model validation tests
│   ├── test_config.py         # Config merging tests
│   ├── test_auth.py           # Auth middleware tests
│   ├── test_queue.py          # Queue behavior tests
│   └── test_errors.py         # Error format tests
├── integration/
│   ├── __init__.py
│   ├── test_chat_completions.py
│   ├── test_completions.py
│   ├── test_embeddings.py
│   ├── test_models.py
│   ├── test_files.py
│   ├── test_fine_tuning.py
│   ├── test_health.py
│   └── test_batch.py
└── contract/
    ├── __init__.py
    └── test_openai_sdk.py     # End-to-end with openai Python SDK

config.toml                    # Default TOML config file
requirements.txt               # Production dependencies (updated)
pyproject.toml                 # Project metadata (updated)
```

**Structure Decision**: Single Python package `openai_http/` with clean separation: routers (HTTP layer), schemas (data models), backends (inference abstraction), services (business logic), observability (logging/metrics). The existing `main.py` becomes `app.py` (FastAPI factory), and `mock_backend.py` moves to `backends/mock_backend.py`. Existing `test_client.py` is preserved as a manual integration test; automated tests use pytest.

## Complexity Tracking

> No constitution violations. Table intentionally omitted.
