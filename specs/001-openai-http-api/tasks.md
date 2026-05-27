# Tasks: OpenAI-Compatible HTTP API Service

**Feature**: `001-openai-http-api`
**Date**: 2026-05-27
**Spec**: [spec.md](spec.md)
**Plan**: [plan.md](plan.md)

**Tests**: Not explicitly required by spec — tasks include tests for completeness
**Organization**: Organized by user story (US1-US9) with shared phases first

## Format: `TNNN [P] [US#] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[US#]**: User story (US1-US9). Setup/Foundational/Polish tasks have NO story label.

---

## Phase 1: Project Setup & Structure

- [ ] T001 Create `openai_http/` package directory with `__init__.py` exposing version and package metadata
- [ ] T002 Update `pyproject.toml` with all dependencies: fastapi, uvicorn[standard], pydantic-settings, pydantic>=2.5, transformers, torch, opentelemetry-api, opentelemetry-sdk, opentelemetry-exporter-prometheus (dev: pytest, httpx, openai, ruff, mypy)
- [ ] T003 [P] Create `openai_http/config.py` with `pydantic-settings` `Settings` class merging TOML (`config.toml`) and environment variables (model path, api_keys, host, port, backend, queue_depth, auth_enabled)
- [ ] T004 [P] Create `config.toml` with sensible defaults (host 0.0.0.0, port 8000, backend "mock", queue_depth 32, auth_enabled false, empty api_keys list)
- [ ] T005 Create `openai_http/app.py` with FastAPI factory `create_app()` using lifespan, CORS middleware, and config injection into `app.state`
- [ ] T006 [P] Create `openai_http/__main__.py` entry point using `create_app()` + `uvicorn.run`
- [ ] T007 [P] Create `tests/` directory tree: `conftest.py`, `unit/`, `integration/`, `contract/` with `__init__.py` in each
- [ ] T008 Create `tests/conftest.py` with shared fixtures: `test_client` (httpx.AsyncClient), `mock_backend`, `app` (create_app with mock config)
- [ ] T009 [P] Create `openai_http/schemas/__init__.py` for centralized schema exports
- [ ] T010 [P] Create `openai_http/routers/__init__.py` for centralized router registration
- [ ] T011 [P] Create `openai_http/backends/__init__.py` with `BackendFactory` and backend registry function

---

## Phase 2: Foundational Infrastructure

> **MUST complete before any user story task begins.**

- [ ] T012 Create `openai_http/errors.py` with `OpenAIError` base exception class carrying `message`, `type`, `param`, `code` fields and mapping to HTTP status codes (400, 401, 404, 429, 500)
- [ ] T013 Add global FastAPI exception handlers in `openai_http/errors.py` that convert `OpenAIError`, `RequestValidationError`, `HTTPException`, and unhandled `Exception` into OpenAI-format `{"error": {"message", "type", "param", "code"}}` JSON responses
- [ ] T014 Catch 404 for undefined routes and return structured OpenAI error response (not HTML) via `app.add_exception_handler(404, ...)`
- [ ] T015 [P] Create `openai_http/schemas/common.py` with `UsageInfo` model (`prompt_tokens`, `completion_tokens`, `total_tokens`) and `ErrorResponse` model
- [ ] T016 [P] Create `openai_http/queue.py` with `RequestQueue` class: `asyncio.Semaphore(1)` for GPU serialization + `asyncio.Queue(maxsize=queue_depth)` for FIFO bound, with `acquire()` context manager (raises 429 `TooManyRequestsError` with `Retry-After` header on overflow)
- [ ] T017 Inject `RequestQueue` into `app.state` during lifespan startup
- [ ] T018 [P] Create `openai_http/backends/base.py` with `Backend` Protocol class defining: `generate()` -> dict, `generate_stream()` -> AsyncGenerator, `embed()` -> list, `list_models()` -> list, `get_model()` -> Model
- [ ] T019 [P] Create `openai_http/backends/mock_backend.py` migrating existing `MockTransformersBackend` and `MockTokenizer` from root `mock_backend.py` to implement the Protocol; include `embed()` method
- [ ] T020 [P] Create `openai_http/observability/logging.py` with structured JSON logging formatter, request ID middleware (generate `X-Request-ID`), and request/response logging (model, latency, status, token counts)
- [ ] T021 [P] Create `openai_http/observability/metrics.py` with OpenTelemetry instruments: `request_counter`, `request_duration_histogram`, `tokens_generated_counter`, `error_counter` (by type), `active_requests_gauge`
- [ ] T022 Register observability middleware and request ID injection in `openai_http/app.py` factory
- [ ] T023 [P] Write unit tests `tests/unit/test_errors.py` verifying all error format conversions (400, 401, 404, 429, 500)
- [ ] T024 [P] Write unit tests `tests/unit/test_queue.py` verifying FIFO order, semaphore enforcement, 429 on overflow
- [ ] T025 [P] Write unit tests `tests/unit/test_config.py` verifying TOML loading, env var override precedence

**Checkpoint**: Foundation ready — user story implementation can now begin in parallel

---

## Phase 3: US1 — Chat Completions with Streaming (P1)

**Goal**: Chat completion endpoint supporting streaming and non-streaming modes with full OpenAI v1 protocol compliance

**Independent Test**: `POST /v1/chat/completions` (stream=true and stream=false) returns correct response shapes and SSE format

### Implementation

- [ ] T026 [P] [US1] Create `openai_http/schemas/chat.py` with Pydantic request models: `ChatMessage` (role, content, name, tool_calls, tool_call_id), `ChatCompletionRequest` (all FR-001 fields: model, messages, temperature, top_p, n, stream, stop, max_tokens, presence_penalty, frequency_penalty, logit_bias, logprobs, top_logprobs, response_format, tools, tool_choice, user, seed, stream_options) using `ConfigDict(extra="allow")`
- [ ] T027 [US1] Add response models in `openai_http/schemas/chat.py`: `ChatCompletionResponse` (id `chatcmpl-*`, object, created, model, choices, usage, system_fingerprint, service_tier), `ChatCompletionChunk` (object `chat.completion.chunk`), `Choice`, `ChunkChoice` (with delta containing role/content/tool_calls)
- [ ] T028 [US1] Create `openai_http/routers/chat.py` with `POST /v1/chat/completions` router handling non-stream mode: acquire queue, call `backend.generate()`, return JSONResponse matching `ChatCompletionResponse` schema with `chatcmpl-*` id and usage
- [ ] T029 [US1] Implement streaming mode in `openai_http/routers/chat.py`: acquire queue, call `backend.generate_stream()`, yield SSE chunks via `StreamingResponse` with `text/event-stream` media type and correct headers (`Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no`)
- [ ] T030 [US1] Ensure SSE output emits first chunk with `{"role": "assistant"}` delta, content chunks via backend stream, final chunk with `finish_reason` and usage (when `stream_options.include_usage = true`), terminated by `data: [DONE]\n\n`
- [ ] T031 [US1] Add model validation in `openai_http/routers/chat.py`: return 404 with `invalid_request_error` if requested model not in `backend.list_models()`
- [ ] T032 [US1] Add request parameter validation: reject `max_tokens <= 0` with 400, silently ignore unknown parameters via `extra="allow"`, handle `stop` as string or array
- [ ] T033 [US1] Handle streaming connection drops: ensure backend resources are cleaned up (cancel backend thread/generation) via `try/finally` in stream generator
- [ ] T034 [US1] Wrap backend errors (OOM, CUDA, timeout) in stream generator to emit formatted error SSE event and close stream, not leave orphaned connections
- [ ] T035 [P] [US1] Write integration tests `tests/integration/test_chat_completions.py`: non-stream response schema validation, stream chunk format validation, `data: [DONE]` termination, invalid model 404
- [ ] T036 [P] [US1] Write contract test `tests/contract/test_openai_sdk.py` verifying `openai` Python SDK `client.chat.completions.create()` (both stream=false and stream=true) works against the service using `base_url` parameter

**Checkpoint**: US1 complete — chat completions fully functional with streaming

---

## Phase 4: US3 — Models API (P1)

**Goal**: Model listing and information endpoints for client auto-discovery

**Independent Test**: `GET /v1/models` and `GET /v1/models/{model_id}` return correct schema

### Implementation

- [ ] T037 [P] [US3] Create `openai_http/schemas/models.py` with `ModelObject` (id, object literal `"model"`, created, owned_by) and `ModelList` (object literal `"list"`, data array) Pydantic models
- [ ] T038 [US3] Create `openai_http/routers/models.py` with `GET /v1/models` returning `ModelList` from `backend.list_models()`, and `GET /v1/models/{model_id}` returning `ModelObject` or 404 with OpenAI error format
- [ ] T039 [US3] Register models router in `openai_http/app.py` factory
- [ ] T040 [P] [US3] Write integration tests `tests/integration/test_models.py`: list models, retrieve specific model, 404 for nonexistent model

**Checkpoint**: US3 complete — model auto-discovery works

---

## Phase 5: US9 — API Key Authentication (P1)

**Goal**: Bearer token authentication compatible with all OpenAI client libraries

**Independent Test**: Requests with valid key (200), invalid key (401), no key (401), and open mode (200)

### Implementation

- [ ] T041 [US9] Create `openai_http/auth.py` with `verify_api_key` FastAPI `Security` dependency: extract `Authorization: Bearer <token>` header, validate against `config.api_keys` list, return 401 `authentication_error` on invalid/missing key
- [ ] T042 [US9] Implement open mode: when `config.auth_enabled` is false, `verify_api_key` becomes a no-op dependency passing any request through
- [ ] T043 [US9] Inject `verify_api_key` dependency into all `/v1/*` routers (exclude `/health` and `/docs`), returning 401 with `invalid_api_key` code when Authorization header missing, 401 with `authentication_error` when key invalid
- [ ] T044 [US9] Add multi-key support: `config.api_keys` accepts list, validate token against any entry (FR-020)
- [ ] T045 [P] [US9] Write unit tests `tests/unit/test_auth.py`: valid key passes, invalid key returns 401, missing header returns 401, open mode accepts all

**Checkpoint**: US9 complete — authentication works with OpenAI SDK

---

## Phase 6: US5 — Backend Abstraction (P1)

**Goal**: Pluggable backend interface with real HuggingFace Transformers inference as default

**Independent Test**: Swap model in config, restart service, verify same API serves new backend's responses

### Implementation

- [ ] T046 [US5] Create `openai_http/backends/transformers_backend.py` with `TransformersBackend` class implementing the `Backend` Protocol: load model via `AutoModelForCausalLM.from_pretrained()` and `AutoTokenizer.from_pretrained()` with configurable `model_path`, `device` (cuda/cpu), `torch_dtype`
- [ ] T047 [US5] Implement `TransformersBackend.generate()` using `asyncio.to_thread()` wrapping `tokenizer.apply_chat_template()` + `model.generate()` + `tokenizer.decode()`, returning dict with `generated_text` and `usage` (token counts from tokenizer)
- [ ] T048 [US5] Implement `TransformersBackend.generate_stream()` using `threading.Thread` + `TextIteratorStreamer` + `asyncio.Queue` bridge pattern (R-003): spawn generate in thread, poll streamer for tokens, yield via async generator
- [ ] T049 [P] [US5] Implement `TransformersBackend.embed()` using a separate embedding model loaded via `SentenceTransformer` or `AutoModel`, returning list of float arrays with configurable dimensions
- [ ] T050 [US5] Implement `TransformersBackend.list_models()` and `get_model()` returning model metadata (id from model_path basename or config, created timestamp, owned_by)
- [ ] T051 [US5] Update `BackendFactory` in `openai_http/backends/__init__.py` to instantiate `TransformersBackend` or `MockBackend` based on `config.backend` setting
- [ ] T052 [US5] Handle backend load failure gracefully: catch exceptions during model loading, log error, expose failure reason via `/health` endpoint (SC-007)
- [ ] T053 [US5] Implement FR-014 chat template delegation: backend applies `tokenizer.apply_chat_template(messages, ...)` with model-specific format (LLaMA, Mistral, Qwen, ChatML handled by tokenizer config), HTTP layer passes raw messages through
- [ ] T054 [US5] Register loaded models in `app.state.backend` during lifespan startup, expose via Models API
- [ ] T055 [US5] Catch CUDA OOM, timeout, and runtime errors from `generate()` calls and convert to `OpenAIError(type="server_error", status=500)` with descriptive message (FR-017)
- [ ] T056 [P] [US5] Write unit tests for `TransformersBackend` Protocol compliance (using mock tokenizer); write integration test loading a tiny model from HF Hub

**Checkpoint**: US5 complete — real model inference works through backend abstraction

---

## Phase 7: US4 — Embeddings Generation (P2)

**Goal**: Vector embeddings endpoint compatible with OpenAI v1 Embeddings API

**Independent Test**: `POST /v1/embeddings` with single string and batch input returns correct schema

### Implementation

- [ ] T057 [P] [US4] Create `openai_http/schemas/embeddings.py` with `EmbeddingRequest` (input: string/string[]/int[]/int[][], model, encoding_format, dimensions, user) and `EmbeddingResponse` (object `"list"`, data array, model, usage) Pydantic models
- [ ] T058 [P] [US4] Create response model `EmbeddingObject` (object `"embedding"`, index, embedding as float[]|string) in `openai_http/schemas/embeddings.py`
- [ ] T059 [US4] Create `openai_http/routers/embeddings.py` with `POST /v1/embeddings`: normalize input (single string -> list), validate input length vs model context, call `backend.embed()`, return `EmbeddingResponse` with consistent dimensions and `usage.prompt_tokens`/`usage.total_tokens`
- [ ] T060 [US4] Implement `encoding_format: "base64"` support: convert float embedding arrays to base64-encoded strings when requested
- [ ] T061 [US4] Implement input length validation: return 400 `invalid_request_error` when token count exceeds model context length
- [ ] T062 [P] [US4] Write integration tests `tests/integration/test_embeddings.py`: single string input, batch input, base64 encoding, consistent dimensions, correct index ordering

**Checkpoint**: US4 complete — embeddings work for RAG/search pipelines

---

## Phase 8: US2 — Text Completions Legacy (P2)

**Goal**: Legacy `/v1/completions` endpoint for older tooling compatibility

**Independent Test**: `POST /v1/completions` (stream=true and stream=false) returns `text_completion` schema

### Implementation

- [ ] T063 [P] [US2] Create `openai_http/schemas/completions.py` with `CompletionRequest` (model, prompt: string/string[]/int[], suffix, max_tokens default 16, temperature, top_p, n, stream, logprobs, echo, stop, presence_penalty, frequency_penalty, best_of, logit_bias, user) using `ConfigDict(extra="allow")`
- [ ] T064 [US2] Add response models in `openai_http/schemas/completions.py`: `CompletionResponse` (id `cmpl-*`, object `"text_completion"`, created, model, choices, usage, system_fingerprint), `TextChoice` (text, index, logprobs, finish_reason)
- [ ] T065 [US2] Create `openai_http/routers/completions.py` with `POST /v1/completions` non-stream mode: acquire queue, convert prompt to messages for backend, return `CompletionResponse` with `cmpl-*` id and usage
- [ ] T066 [US2] Implement streaming mode in `openai_http/routers/completions.py`: yield SSE chunks with `choices[].text` deltas (not delta.content), terminate with `data: [DONE]\n\n`
- [ ] T067 [US2] Handle `n > 1` / `best_of > 1`: return multiple completions in choices array or clear error if backend unsupported
- [ ] T068 [P] [US2] Write integration tests `tests/integration/test_completions.py`: non-stream, stream, `n>1`, `echo`, response schema validation with `object: "text_completion"`

**Checkpoint**: US2 complete — legacy completions work

---

## Phase 9: US6 — Tool Calling / Function Calling (P2)

**Goal**: Function calling support for agentic AI workflows (LangChain, AutoGen)

**Independent Test**: Chat completion with `tools` array returns `tool_calls` in response when model triggers function call

### Implementation

- [ ] T069 [P] [US6] Extend `openai_http/schemas/chat.py` with tool schemas: `FunctionDefinition` (name, description, parameters JSON schema, strict), `Tool` (type `"function"`, function), `ToolChoice` (string literal "auto"/"none"/"required" or object with type+function), `ToolCall` (id, type `"function"`, function: {name, arguments})
- [ ] T070 [US6] Extend `ChatMessage` schema in `openai_http/schemas/chat.py` to support assistant messages with `tool_calls` array and tool messages with `tool_call_id` field
- [ ] T071 [US6] Update `openai_http/routers/chat.py` to pass `tools` and `tool_choice` parameters to backend `generate()`, include `tool_calls` in response message when present
- [ ] T072 [US6] Implement `finish_reason: "tool_calls"` in response when backend produces tool calls instead of content
- [ ] T073 [US6] Implement `tool_choice: "none"` handling: strip tools from backend input, ensure no `tool_calls` in response
- [ ] T074 [US6] Implement `tool_choice` with specific function name: constrain backend tool call to target the specified function
- [ ] T075 [US6] Implement tool call streaming: stream `tool_calls` deltas in `ChatCompletionChunk` choices (function name and arguments incrementally)
- [ ] T076 [P] [US6] Write tests for tool calling: auto tool call generation, tool_choice none, specific function choice, tool call in streaming chunks

**Checkpoint**: US6 complete — function calling works for agentic workflows

---

## Phase 10: US7 — Files API (P3)

**Goal**: File upload and management for fine-tuning and batch workflows

**Independent Test**: Upload JSONL file, list files, retrieve metadata, download content

### Implementation

- [ ] T077 [P] [US7] Create `openai_http/schemas/files.py` with `FileObject` (id `file-*`, object `"file"`, bytes, created_at, filename, purpose, status, status_details), `FileList`, `FileDeleteResponse` Pydantic models
- [ ] T078 [US7] Create `openai_http/services/file_store.py` with `FileStore` class: SQLite metadata table (id, filename, bytes, purpose, status, created_at, storage_path), local filesystem storage under configurable directory, `upload()`, `list()`, `get()`, `delete()`, `get_content()` methods
- [ ] T079 [US7] Create `openai_http/routers/files.py` with `POST /v1/files` (multipart upload: file + purpose field), `GET /v1/files` (list with optional purpose filter), `GET /v1/files/{file_id}` (metadata), `DELETE /v1/files/{file_id}`, `GET /v1/files/{file_id}/content` (binary download)
- [ ] T080 [US7] Implement file validation on upload: validate JSONL format for `purpose: "fine-tune"`, update status from `uploaded` to `processed` or `error`
- [ ] T081 [US7] Initialize SQLite database in app lifespan startup, inject `FileStore` into `app.state`
- [ ] T082 [P] [US7] Write integration tests `tests/integration/test_files.py`: upload JSONL, list files, retrieve metadata, download content, delete file, 404 for missing file

**Checkpoint**: US7 complete — file management works for fine-tuning pipelines

---

## Phase 11: US8 — Fine-tuning API (P3)

**Goal**: Fine-tuning job creation, monitoring, and management

**Independent Test**: Create fine-tuning job with training file, retrieve job, cancel job

### Implementation

- [ ] T083 [P] [US8] Create `openai_http/schemas/fine_tuning.py` with `FineTuningJob` (id `ftjob-*`, object `"fine_tuning.job"`, created_at, finished_at, model, fine_tuned_model, organization_id, status, hyperparameters, training_file, validation_file, result_files, trained_tokens, error, seed, method), `FineTuningJobList`, `FineTuningEvent` Pydantic models
- [ ] T084 [US8] Create `openai_http/services/fine_tuning_service.py` with `FineTuningService` class: in-memory job registry, state machine (`queued` -> `running` -> `succeeded`/`failed`/`cancelled`), `create_job()`, `list_jobs()`, `get_job()`, `cancel_job()`, `get_events()`
- [ ] T085 [US8] Create `openai_http/routers/fine_tuning.py` with `POST /v1/fine_tuning/jobs` (create: validate training_file exists, create job with `status: "queued"`), `GET /v1/fine_tuning/jobs` (list), `GET /v1/fine_tuning/jobs/{id}` (retrieve), `POST /v1/fine_tuning/jobs/{id}/cancel`, `GET /v1/fine_tuning/jobs/{id}/events`
- [ ] T086 [US8] Implement cancel job endpoint in `openai_http/routers/fine_tuning.py`: transition job from `queued`/`running` to `cancelled`, return updated job object
- [ ] T087 [US8] Implement events endpoint: return chronological list of `FineTuningEvent` objects (created, started, completed, cancelled) for a job
- [ ] T088 [P] [US8] Write integration tests `tests/integration/test_fine_tuning.py`: create job with valid training file, list jobs, retrieve job, cancel running job, events list

**Checkpoint**: US8 complete — fine-tuning lifecycle works end-to-end

---

## Phase 12: Polish & Cross-Cutting Concerns

**Purpose**: Additional endpoints, health checks, cleanup, and end-to-end validation

### Additional Endpoints

- [ ] T089 [P] Create `openai_http/routers/health.py` with `GET /health` returning 200 `{status: "ready", models, backend_type, uptime_seconds}` when backend available, 503 when not (FR-028)
- [ ] T090 [P] Create `openai_http/routers/moderations.py` with `POST /v1/moderations` returning permissive no-violation result (FR-005) using schema `openai_http/schemas/moderation.py`
- [ ] T091 [P] Create `openai_http/routers/audio.py` with `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` returning 501 Not Implemented with OpenAI error format when no audio backend configured (R-011)
- [ ] T092 [P] Create `openai_http/routers/images.py` with `POST /v1/images/generations`, `/edits`, `/variations` returning 501 Not Implemented when no image backend configured (R-011)
- [ ] T093 Create `openai_http/schemas/batches.py` and `openai_http/services/batch_service.py` with `BatchJob` model and in-memory batch processor; create `openai_http/routers/batches.py` with `POST /v1/batches`, `GET /v1/batches`, `GET /v1/batches/{id}`, `POST /v1/batches/{id}/cancel` (FR-010)

### Cleanup & Validation

- [ ] T094 Write `GET /health` integration test in `tests/integration/test_health.py`: 200 with model status when ready, 503 when not
- [ ] T095 Add `system_fingerprint` field to chat completion and text completion responses (FR-024), generated from config hash
- [ ] T096 Migrate `main.py` entry point and delete root-level `mock_backend.py`; ensure `python -m openai_http` runs correctly
- [ ] T097 Verify `openai>=1.0` SDK compatibility: full end-to-end test in `tests/contract/test_openai_sdk.py` for chat.completions (stream + non-stream), models.list, models.retrieve (SC-001)
- [ ] T098 Run full test suite (`pytest tests/ -v`), ensure all integration and contract tests pass with mock backend
- [ ] T099 Update `README.md` with usage instructions, configuration reference, supported endpoints table, and quickstart guide

---

## Dependencies & Execution Order

```
Phase 1 (Setup)
  └──> Phase 2 (Foundational)
         ├──> Phase 3 (US1: Chat Completions)
         ├──> Phase 4 (US3: Models API)
         ├──> Phase 5 (US9: Auth)
         └──> Phase 6 (US5: Backend)
                ├──> Phase 7 (US4: Embeddings)
                ├──> Phase 8 (US2: Completions)
                ├──> Phase 9 (US6: Tool Calling)
                ├──> Phase 10 (US7: Files API) ──> Phase 11 (US8: Fine-tuning)
                └──> Phase 12 (Polish)
```

**Critical path**: Phase 1 → Phase 2 → Phase 3 (US1) + Phase 6 (US5) → Phase 9 (US6)

**Independent (can start after Phase 2)**: Phase 3, Phase 4, Phase 5

**Depends on Phase 6 (Backend)**: Phase 7, Phase 8, Phase 9

**Phase 11 depends on Phase 10**: Fine-tuning requires Files API for training file reference

### Within Each User Story

1. Schemas (Pydantic models) first
2. Router/endpoint logic second
3. Integration with existing components third
4. Tests last (or in parallel where marked [P])

### MVP Scope

**Minimum Viable Product**: Phase 1 + Phase 2 + Phase 3 (US1) + Phase 4 (US3) + Phase 5 (US9) + Phase 6 (US5)

This delivers: working chat completions with streaming, model discovery, Bearer auth, and real transformers inference — enough to connect any standard OpenAI client.

---

## Parallel Execution Examples

### After Phase 1 Completes

```
Agent A: T012-T025 (Phase 2: Foundational)
```

### After Phase 2 Completes

```
Agent A: T026-T036 (Phase 3: US1 Chat Completions)
Agent B: T037-T040 (Phase 4: US3 Models API)
Agent C: T041-T045 (Phase 5: US9 Auth)
```

### After Phase 3 + Phase 5 Complete

```
Agent A: T046-T056 (Phase 6: US5 Backend)
Agent B: T089-T092 (Phase 12: Health/Audio/Images stubs)
Agent C: T077-T082 (Phase 10: US7 Files API)
Agent D: T090 (Phase 12: Moderations)
```

### After Phase 6 (Backend) Completes

```
Agent A: T063-T068 (Phase 8: US2 Completions)
Agent B: T057-T062 (Phase 7: US4 Embeddings)
Agent C: T069-T076 (Phase 9: US6 Tool Calling)
Agent D: T083-T088 (Phase 11: US8 Fine-tuning)
Agent E: T093 (Phase 12: Batch API)
```

---

## Summary

| Phase | User Story | Tasks | Priority |
|-------|-----------|-------|----------|
| 1: Setup | All | 11 tasks (T001-T011) | P1 |
| 2: Foundational | All | 14 tasks (T012-T025) | P1 |
| 3: Chat Completions | US1 | 11 tasks (T026-T036) | P1 |
| 4: Models API | US3 | 4 tasks (T037-T040) | P1 |
| 5: Auth | US9 | 5 tasks (T041-T045) | P1 |
| 6: Backend | US5 | 11 tasks (T046-T056) | P1 |
| 7: Embeddings | US4 | 6 tasks (T057-T062) | P2 |
| 8: Completions | US2 | 6 tasks (T063-T068) | P2 |
| 9: Tool Calling | US6 | 8 tasks (T069-T076) | P2 |
| 10: Files API | US7 | 6 tasks (T077-T082) | P3 |
| 11: Fine-tuning | US8 | 6 tasks (T083-T088) | P3 |
| 12: Polish | All | 11 tasks (T089-T099) | Mixed |
| **Total** | | **99 tasks** | |

**MVP (Phase 1-6)**: 56 tasks — delivers working chat completions with real model inference
**Full v1**: 99 tasks — complete OpenAI API compatibility
