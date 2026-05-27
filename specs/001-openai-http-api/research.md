# Research: OpenAI-Compatible HTTP API Service

**Feature**: `001-openai-http-api`
**Date**: 2026-05-27

## R-001: HTTP Framework

**Decision**: FastAPI (existing dependency, extend current implementation)

**Rationale**: The project already uses FastAPI with uvicorn and pydantic. FastAPI provides built-in async support, automatic OpenAPI docs generation, Pydantic v2 integration for request/response validation, and `StreamingResponse`/`EventSourceResponse` for SSE streaming. No framework migration justified.

**Alternatives considered**:
- Flask + Flask-SSE: Sync-first, less ideal for streaming, no auto docs
- Litestar (formerly Starlite): More batteries-included but less ecosystem maturity
- aiohttp: Lower-level, no built-in Pydantic integration or auto docs

## R-002: SSE Streaming Implementation

**Decision**: Use FastAPI's `StreamingResponse` with manual SSE formatting (current approach in codebase), not `EventSourceResponse`

**Rationale**: The current implementation in `main.py:194-230` already uses `StreamingResponse` with correct SSE headers (`Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`). This is the proven pattern for OpenAI-compatible servers (vLLM and litellm both use this approach). `EventSourceResponse` from `fastapi-sse` adds dependency complexity for no significant benefit.

**Alternatives considered**:
- `fastapi.sse.EventSourceResponse` (added in FastAPI 0.135+): Handles keep-alive automatically but adds version constraint and is newer/less battle-tested for this use case
- `sse-starlette`: Third-party package, works but adds another dependency

## R-003: Transformers Backend Integration with Async Event Loop

**Decision**: Use `threading.Thread` + `TextIteratorStreamer` + `asyncio.Queue` bridge for streaming; `asyncio.to_thread()` for non-streaming calls

**Rationale**: HuggingFace Transformers `model.generate()` is synchronous and GPU-blocking. The `TextIteratorStreamer` class is specifically designed for the pattern of running `generate()` in a background thread while consuming tokens in another thread. Bridging to async via `asyncio.Queue` is the standard pattern. `asyncio.to_thread()` wraps synchronous generate for non-streaming mode.

**Alternatives considered**:
- `concurrent.futures.ThreadPoolExecutor`: Works but adds pool management complexity without benefit for single-GPU (only one request active at a time per FR-031)
- `multiprocessing`: Unnecessary overhead; GIL is released during CUDA operations
- `run_in_executor` with ProcessPoolExecutor: Overkill for GPU inference

## R-004: Pydantic v2 Schema Strategy

**Decision**: Define explicit Pydantic v2 request/response models for all OpenAI v1 endpoints, using `model_config = ConfigDict(extra="allow")` for request models

**Rationale**: Explicit models provide request validation, automatic documentation, and clean response serialization. `extra="allow"` on request models matches OpenAI's behavior of silently accepting unknown parameters (user story acceptance scenario 3). Response models use `model_dump(exclude_none=True)` for clean output.

**Alternatives considered**:
- Raw dict request/response: Simpler but no validation, no docs, error-prone
- `extra="forbid"` on requests: Would reject common OpenAI SDK parameters

## R-005: Configuration Strategy

**Decision**: Built-in `tomllib` (Python 3.12+) for TOML reading + `pydantic-settings` for env var merging

**Rationale**: Python 3.12+ includes `tomllib` for TOML parsing. `pydantic-settings` provides automatic environment variable overriding (matches FR-026: env vars take precedence). The `Settings` class integrates naturally with the rest of the Pydantic-based codebase.

**Alternatives considered**:
- `tomli` (third-party backport): Unnecessary since Python 3.12+ has built-in `tomllib`
- `dynaconf`: Feature-rich but heavy for this use case
- Manual env var merging: Error-prone, doesn't scale with nested config

## R-006: Observability Stack

**Decision**: OpenTelemetry API/SDK + Prometheus exporter for metrics + Python `structlog` or `logging` with JSON formatter for structured logs

**Rationale**: OpenTelemetry is the industry standard for metrics instrumentation. The Python SDK provides Counter, Histogram, and UpDownCounter instruments. Prometheus exporter exposes metrics at `/metrics` on a separate port. Structured JSON logging via standard library `logging` with `json` formatter is lightweight and sufficient.

**Alternatives considered**:
- Raw Prometheus client: Tighter coupling, less portable
- `structlog`: Feature-rich structured logging, adds dependency but provides better context binding
- `loguru`: Popular but non-standard API, harder for team adoption

## R-007: Concurrency / Request Queue

**Decision**: `asyncio.Semaphore(1)` for single-GPU serialization + `asyncio.Queue` with configurable max size for bounded FIFO waiting

**Rationale**: FR-031 requires one request at a time with a bounded queue (default depth: 32). A semaphore ensures only one generation runs at a time (preventing GPU OOM). When the queue is full, return 429 with `Retry-After` header.

**Alternatives considered**:
- Per-model semaphores: Needed when multi-model support (FR-016) is added; will refactor queue per-model
- Priority queues: Over-engineering for v1
- `asyncio.LifoQueue`: Wrong ordering semantics

## R-008: Backend Abstraction Design

**Decision**: Define a `Backend` protocol/ABC with methods `generate()`, `generate_stream()`, `embed()`, and model lifecycle methods. The default implementation is `TransformersBackend` wrapping HuggingFace's `AutoModelForCausalLM` and `AutoTokenizer`.

**Rationale**: Protocol-based design (Python `typing.Protocol` or `abc.ABC`) enables pluggable backends without inheritance. The transformers backend implements the protocol by loading models with `AutoModelForCausalLM.from_pretrained()` and using `generate()` for inference and `TextIteratorStreamer` for streaming.

**Alternatives considered**:
- Abstract base class with method stubs: Works but less flexible than Protocol for duck-typing
- Plugin system with entry points: Too complex for v1, may revisit when adding vLLM/TGI backends
- Strategy pattern with registry: Good fit but implemented via simpler module-level factory function

## R-009: Files & Fine-tuning Storage

**Decision**: Local filesystem storage with JSON metadata index for Files API; in-memory job state for fine-tuning API

**Rationale**: Files are stored on local disk with a SQLite or JSON metadata index (file ID, filename, bytes, purpose, status, created_at). Fine-tuning jobs are tracked in-memory with a state machine (queued → running → completed/failed/cancelled). This matches the "self-hosted/development use" assumption.

**Alternatives considered**:
- Object storage (S3/minio): Overkill for dev/self-hosted; may add as storage backend option later
- Database for job persistence: Adds complexity; in-memory is acceptable since fine-tuning is P3 and jobs can restart
- Redis for job state: Adds dependency without clear benefit at current scale

## R-010: Error Handling Strategy

**Decision**: Global FastAPI exception handlers that convert all exceptions to OpenAI-format error responses

**Rationale**: FR-022 requires all errors to match `{"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}`. Global exception handlers ensure consistency without per-endpoint error wrapping. Custom exception classes map to HTTP status codes.

**Alternatives considered**:
- Per-endpoint try/except: Verbose, error-prone, inconsistent
- Middleware error handling: Already done via exception handlers, cleaner separation

## R-011: Audio & Image Endpoint Strategy

**Decision**: Implement endpoints that return 501 Not Implemented when no compatible backend is configured

**Rationale**: FR-008 and FR-009 specify audio and image endpoints, but the assumption states "if no compatible backend is configured, those endpoints return a 501". This allows the endpoints to exist for v1 compatibility while not requiring Whisper or DALL-E model support.

**Alternatives considered**:
- Mock audio/image responses: Would break real client expectations
- Exclude endpoints entirely: Breaks SDK compatibility checks
- 501 with "coming soon" message: Matches assumption, clean implementation

## R-012: Auth Middleware

**Decision**: FastAPI dependency injection with `Depends()` for Bearer token validation

**Rationale**: FR-018-020 require Bearer token auth. A FastAPI `Security` dependency extracts and validates the token from the `Authorization: Bearer <token>` header. A list of valid keys is stored in the config. When auth is disabled (open mode), the dependency is a no-op.

**Alternatives considered**:
- ASGI middleware: Works globally but harder to compose with route-specific logic
- Custom header extraction: Less clean than dependency injection

## R-013: Batch API Implementation

**Decision**: Asynchronous file-based batch processing with background worker

**Rationale**: FR-010 requires batch API endpoints. Batches are stored as JSONL files (uploaded via Files API). A background task processes batch requests sequentially through the existing backend. Job state transitions: validating → in_progress → completed/failed.

**Alternatives considered**:
- Celery task queue: Overkill for single-worker batch processing
- Thread pool workers: Sufficient for sequential through single GPU
