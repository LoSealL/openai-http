# Feature Specification: OpenAI-Compatible HTTP API Service

**Feature Branch**: `001-openai-http-api`

**Created**: 2026-05-27

**Status**: Draft

**Input**: User description: "实现针对任意后端的openai http服务，使用python，提供兼容openai所有v1 api的逻辑处理。后端可以假定用transformers generate。"

## User Scenarios & Testing *(mandatory)*

---

### User Story 1 - Chat Completions with Streaming (Priority: P1)

A developer deploys the service and makes chat completion requests using any standard OpenAI client (Python, Node.js, or raw HTTP), receiving responses in both streaming and non-streaming modes. The service accepts all standard parameters (messages, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop sequences, response_format, tools/tool_choice, and others defined in the OpenAI v1 Chat Completions specification) and translates them to backend-specific calls.

**Why this priority**: Chat Completions is the most-used OpenAI endpoint. Streaming is critical for user experience; without it, users wait for full generation before seeing any output. This is the MVP.

**Independent Test**: Can be fully tested by sending `POST /v1/chat/completions` (with `stream: true` and `stream: false`) to the running service and verifying the response shape, token counts, and streaming chunk format match the OpenAI v1 specification exactly.

**Acceptance Scenarios**:

1. **Given** the service is running with a transformers backend model loaded, **When** a user sends `POST /v1/chat/completions` with `stream: false`, **Then** the response matches the OpenAI v1 ChatCompletion schema, including `choices[].message.content`, `choices[].finish_reason`, `usage.prompt_tokens`, `usage.completion_tokens`, `usage.total_tokens`, and a valid `id` in `chatcmpl-*` format.
2. **Given** the service is running, **When** a user sends `POST /v1/chat/completions` with `stream: true`, **Then** the service returns a Server-Sent Events (SSE) stream where each chunk is a valid ChatCompletionChunk with `choices[].delta`, and the final chunk contains `choices[].finish_reason` and `data: [DONE]` terminates the stream.
3. **Given** a request with an unrecognized parameter (e.g., `user`, `logit_bias`), **When** the service receives it, **Then** it either honors the parameter or silently ignores it without returning an error, matching OpenAI's behavior of accepting but not necessarily enforcing all parameters.
4. **Given** a request with an invalid `model` name, **When** the service receives it, **Then** it returns a proper OpenAI-format error response with `error.type = "invalid_request_error"`, `error.message`, and appropriate HTTP status (404 or 400).

---

### User Story 2 - Text Completions (Legacy Endpoint) (Priority: P2)

A developer uses the service to make text completion requests via `POST /v1/completions`, supporting both streaming and non-streaming modes. This is the legacy completions API used by older tooling and applications.

**Why this priority**: Many existing applications still use the `/v1/completions` endpoint. Full compatibility requires this endpoint to function correctly.

**Independent Test**: Can be fully tested by sending `POST /v1/completions` and verifying the response matches the OpenAI v1 Completion schema (not ChatCompletion), including `prompt_tokens`, `completion_tokens`, and the `text` field in choices.

**Acceptance Scenarios**:

1. **Given** the service is running, **When** a user sends `POST /v1/completions` with a `prompt` string and `stream: false`, **Then** the response includes `choices[].text`, `choices[].finish_reason`, `usage` fields, and `object: "text_completion"`.
2. **Given** the service is running, **When** a user sends `POST /v1/completions` with `stream: true`, **Then** the service returns SSE chunks with `choices[].text` deltas and terminates with `data: [DONE]`.
3. **Given** a request with `best_of > 1` or `n > 1`, **When** the service receives it, **Then** it returns the requested number of completions in the `choices` array (or returns a clear error if the backend does not support it).

---

### User Story 3 - Model Listing and Information (Priority: P1)

A developer calls `GET /v1/models` or `GET /v1/models/{model}` to discover which models the service has loaded. The service returns model metadata in the standard OpenAI v1 Models format.

**Why this priority**: This endpoint is nearly always the first call made by any OpenAI client library to validate that the service is reachable and the requested model exists. It is critical for compatibility auto-discovery.

**Independent Test**: Can be fully tested by calling `GET /v1/models` and `GET /v1/models/<model_id>` and verifying the response schema matches OpenAI's v1 Models API, with correct `id`, `object`, `owned_by`, and `created` fields.

**Acceptance Scenarios**:

1. **Given** the service is running with a model loaded, **When** a user calls `GET /v1/models`, **Then** the response includes a `data` array with an entry for each loaded model, each having `id`, `object: "model"`, `owned_by`, and `created` fields.
2. **Given** a model "my-model" is loaded, **When** a user calls `GET /v1/models/my-model`, **Then** the response contains the single model object with the correct `id`.
3. **Given** a non-existent model name, **When** a user calls `GET /v1/models/nonexistent-model`, **Then** the service returns a 404 with an OpenAI-format error body.

---

### User Story 4 - Embeddings Generation (Priority: P2)

A developer calls `POST /v1/embeddings` to generate vector representations of one or more text inputs. The service accepts input as a single string, an array of strings, or an array of token arrays (matching OpenAI's input flexibility), and returns embeddings in the standard v1 format.

**Why this priority**: Embeddings are essential for RAG (Retrieval Augmented Generation), semantic search, and clustering pipelines. Many OpenAI SDK-based applications use this endpoint heavily.

**Independent Test**: Can be fully tested by sending `POST /v1/embeddings` with a single string and an array of strings, and verifying that each response contains a `data` array with `embedding` arrays of consistent dimension, along with correct `usage` counts.

**Acceptance Scenarios**:

1. **Given** the service is running with an embedding-capable model, **When** a user sends `POST /v1/embeddings` with a single string input, **Then** the response contains `data[0].embedding` (a numeric array), `data[0].index: 0`, `object: "embedding"`, and `usage.prompt_tokens` / `usage.total_tokens`.
2. **Given** the service is running, **When** a user sends `POST /v1/embeddings` with a batch of 5 strings, **Then** the response contains 5 items in the `data` array, each with the correct `index` and `embedding` of consistent dimension.
3. **Given** a request specifying `encoding_format: "base64"`, **When** the service receives it, **Then** the `embedding` field is returned as a base64-encoded string instead of a numeric array.

---

### User Story 5 - Backend Abstraction and Configuration (Priority: P1)

An administrator configures which backend model or inference engine the service uses, without changing application code. The service supports plugging in different backends (initially a transformers-based backend) behind a common interface, allowing the same HTTP API to serve different models or inference engines.

**Why this priority**: The ability to swap backends is the core value proposition of this service. Without backend abstraction, the service is no different from running a model directly.

**Independent Test**: Can be fully tested by deploying the service with Backend A, making a request, swapping to Backend B (e.g., different model), and verifying the new backend's responses are served through the same API endpoints without restarting the HTTP layer.

**Acceptance Scenarios**:

1. **Given** a backend is configured (e.g., a transformers model path), **When** the service starts, **Then** it loads the backend and registers the model(s) under a model ID accessible via the Models API.
2. **Given** the service supports multiple backend types, **When** an administrator changes the backend configuration, **Then** the service exposes the new backend's models through all API endpoints without requiring code changes.
3. **Given** a backend fails to load, **When** the service starts, **Then** it returns a clear error message indicating the failure reason rather than silently starting with no available models.

---

### User Story 6 - Completions with Tools / Function Calling (Priority: P2)

A developer sends a chat completion request that includes a `tools` array with function definitions and specifies a `tool_choice`. The service returns structured `tool_calls` in the response message, matching OpenAI's v1 tool calling format.

**Why this priority**: Function calling is widely used in agentic AI workflows (LangChain, AutoGen, etc.). Compatibility with tools is a high-signal requirement for real-world SDK compatibility.

**Independent Test**: Can be fully tested by sending a request with `tools` and `tool_choice: "auto"`, and verifying the response contains a `tool_calls` array with properly structured objects (`id`, `type: "function"`, `function: {name, arguments}`).

**Acceptance Scenarios**:

1. **Given** a request includes a `tools` array and the model produces a function call, **When** the service processes the request, **Then** the response message contains `tool_calls` with entries matching the OpenAI tool calling schema, and `finish_reason: "tool_calls"`.
2. **Given** `tool_choice: "none"`, **When** the service processes the request, **Then** no `tool_calls` appear in the response regardless of the tool definitions provided.
3. **Given** `tool_choice.tool.type = "function"` with a specific function name, **When** the service processes it, **Then** the returned tool call targets the specified function.

---

### User Story 7 - Files API (Priority: P3)

A developer uses `POST /v1/files` to upload a file, `GET /v1/files` to list files, and `GET /v1/files/{file_id}` to retrieve file metadata. The service persists uploaded files and returns them in the standard OpenAI v1 Files API format.

**Why this priority**: Files API is required for fine-tuning workflows. It is less frequently used than chat/completions but necessary for full v1 compatibility.

**Independent Test**: Can be fully tested by uploading a JSONL file via `POST /v1/files` (purpose: `fine-tune`), then listing files and retrieving the uploaded file metadata.

**Acceptance Scenarios**:

1. **Given** a valid JSONL file upload via `POST /v1/files` with `purpose: "fine-tune"`, **When** the request is processed, **Then** the response contains `id`, `object: "file"`, `filename`, `bytes`, `created_at`, `purpose`, and `status`.
2. **Given** files have been uploaded, **When** `GET /v1/files` is called, **Then** all files appear in the `data` array with correct metadata.
3. **Given** a file has been uploaded, **When** `DELETE /v1/files/{file_id}` is called, **Then** the file is removed and the response confirms deletion.

---

### User Story 8 - Fine-tuning API (Priority: P3)

A developer uses the fine-tuning endpoints (`POST /v1/fine_tuning/jobs`, `GET /v1/fine_tuning/jobs`, `GET /v1/fine_tuning/jobs/{id}`, `POST /v1/fine_tuning/jobs/{id}/cancel`) to create, monitor, and manage fine-tuning jobs.

**Why this priority**: Fine-tuning is a complete OpenAI v1 feature area. Full compatibility requires these endpoints, though they are lower priority than inference endpoints.

**Independent Test**: Can be fully tested by creating a fine-tuning job with a valid training file reference and verifying the job object is returned with `status: "queued"` or equivalent.

**Acceptance Scenarios**:

1. **Given** a valid training file, **When** `POST /v1/fine_tuning/jobs` is called, **Then** a job object is returned with `id`, `object: "fine_tuning.job"`, `model`, `status`, `created_at`, and `trained_tokens` (initially null).
2. **Given** a running fine-tuning job, **When** `POST /v1/fine_tuning/jobs/{id}/cancel` is called, **Then** the job transitions to `cancelled` status.
3. **Given** completed fine-tuning jobs, **When** `GET /v1/fine_tuning/jobs` is called, **Then** all jobs are listed with their current statuses.

---

### User Story 9 - API Key Authentication (Priority: P1)

A developer authenticates requests using a Bearer token in the `Authorization` header (or via the `api_key` parameter for library compatibility). The service validates the token and rejects requests with invalid or missing authentication.

**Why this priority**: All OpenAI client libraries send API key authentication by default. Without compatible auth handling, no standard client can connect. This is foundational.

**Independent Test**: Can be fully tested by making requests with a valid key, an invalid key, and no key, and verifying the correct HTTP status codes (200, 401) and error responses are returned.

**Acceptance Scenarios**:

1. **Given** a valid API key is configured, **When** a request includes `Authorization: Bearer <valid-key>`, **Then** the request is processed normally.
2. **Given** an API key requirement is configured, **When** a request is sent without an Authorization header, **Then** the service returns 401 with `error.type: "invalid_request_error"` and `error.code: "invalid_api_key"`.
3. **Given** an API key requirement is configured, **When** a request includes an incorrect key, **Then** the service returns 401 with `error.type: "authentication_error"`.
4. **Given** API key authentication is disabled (open mode), **When** any request is sent, **Then** it is processed regardless of the Authorization header value.

---

### Edge Cases

- What happens when the backend model runs out of memory during generation? The service MUST return a 500 with a structured OpenAI-format error body (`error.type: "server_error"`, `error.message`), not a raw traceback.
- What happens when a streaming connection is dropped mid-generation? The service MUST clean up backend resources (stop generation, free GPU memory) and not leave orphaned processes.
- What happens when concurrent requests exceed the backend's capacity? The service MUST queue requests in the bounded FIFO queue (one active, up to 32 queued by default) and reject excess requests with a 429 Too Many Requests response and `Retry-After` header.
- What happens when a request body is malformed or unparsable JSON? The service MUST return 400 with `error.type: "invalid_request_error"` and a descriptive message, not a generic 500.
- What happens when a token array (for embeddings) exceeds the model's context length? The service MUST return 400 with `error.type: "invalid_request_error"` indicating the input is too long.
- What happens when the service receives a request to an undefined route (e.g., `GET /v1/nonexistent`)? The service MUST return 404 with a structured error response, not an HTML error page.
- What happens when `max_tokens` in a request is 0 or negative? The service MUST return a 400 error with a descriptive message.

## Requirements *(mandatory)*

### Functional Requirements

#### Core API Endpoints

- **FR-001**: System MUST implement `POST /v1/chat/completions` with full parameter support (messages, model, temperature, top_p, n, stream, stop, max_tokens, presence_penalty, frequency_penalty, logit_bias, response_format, tools, tool_choice, user, seed) and return responses matching the OpenAI v1 ChatCompletion schema.
- **FR-002**: System MUST implement `POST /v1/completions` with full parameter support (model, prompt, suffix, max_tokens, temperature, top_p, n, stream, logprobs, echo, stop, presence_penalty, frequency_penalty, best_of, logit_bias, user) and return responses matching the OpenAI v1 Completion schema.
- **FR-003**: System MUST implement `POST /v1/embeddings` accepting input as string, array of strings, or array of token arrays, with `encoding_format` support, and return responses matching the OpenAI v1 Embedding schema.
- **FR-004**: System MUST implement `GET /v1/models` and `GET /v1/models/{model}` returning currently loaded model(s) in OpenAI v1 Models schema format.
- **FR-005**: System MUST implement `POST /v1/moderations` accepting the standard input and returning the OpenAI v1 Moderation schema (may return a permissive "no violation" result if no moderation model is loaded).
- **FR-006**: System MUST implement the Files API: `POST /v1/files` (upload), `GET /v1/files` (list), `GET /v1/files/{file_id}` (retrieve), `DELETE /v1/files/{file_id}` (delete), and `GET /v1/files/{file_id}/content` (download content).
- **FR-007**: System MUST implement the Fine-tuning API: `POST /v1/fine_tuning/jobs` (create), `GET /v1/fine_tuning/jobs` (list), `GET /v1/fine_tuning/jobs/{id}` (retrieve), `POST /v1/fine_tuning/jobs/{id}/cancel` (cancel), and `GET /v1/fine_tuning/jobs/{id}/events` (events list).
- **FR-008**: System MUST implement `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` (Whisper-compatible audio endpoints), returning text transcription/translation in the standard format when an audio-capable backend model is available.
- **FR-009**: System MUST implement `POST /v1/images/generations`, `POST /v1/images/edits`, and `POST /v1/images/variations` in OpenAI's v1 Images API format, returning image data (URL or base64) when an image generation backend is configured.
- **FR-010**: System MUST implement the Batch API: `POST /v1/batches` (create), `GET /v1/batches/{id}` (retrieve), `POST /v1/batches/{id}/cancel` (cancel), and `GET /v1/batches` (list), for asynchronous batch processing.
- **FR-028**: System MUST implement `GET /health` returning HTTP 200 when the service is ready to accept requests, and HTTP 503 when not ready. The response body MUST include model loading status and backend availability details.

#### Observability

- **FR-029**: System MUST emit structured JSON logs for all requests and errors, including: request ID, model identifier, latency, token counts (prompt/completion), streaming flag, HTTP status code, and error details when applicable.
- **FR-030**: System MUST expose metrics via an OpenTelemetry-compatible exporter, including: request count, request duration histogram, tokens generated counter, error count by type, and concurrent active requests gauge.
- **FR-031**: System MUST process inference requests with a bounded FIFO queue model: one request executes at a time, additional requests wait in a configurable-depth queue (default: 32), and requests that exceed queue capacity MUST receive 429 Too Many Requests with `Retry-After` header.

#### Streaming & Protocols

- **FR-011**: System MUST support Server-Sent Events (SSE) streaming for chat completions and text completions, emitting `data: {chunk}\n\n` events and terminating with `data: [DONE]\n\n`.
- **FR-012**: Streaming chunks for chat completions MUST follow the ChatCompletionChunk schema, with deltas for content, tool_calls, and role, and the final chunk MUST include `finish_reason`.
- **FR-013**: System MUST handle HTTP/2 and HTTP/1.1 connections, and support `keep-alive` for long-running streaming connections.

#### Backend Abstraction

- **FR-014**: System MUST provide a common backend interface (abstraction layer) so that any inference engine or model serving framework can be plugged in without modifying the HTTP route handlers. The backend interface MUST receive raw `messages` arrays from the HTTP API and be responsible for applying the appropriate chat template formatting (LLaMA, Mistral, Qwen, ChatML, etc.) based on the model's tokenizer configuration before generating text.
- **FR-015**: System MUST include a default transformers-based backend that loads a model from a local path or Hugging Face Hub identifier and uses HuggingFace Transformers' `generate` method for inference.
- **FR-016**: System MUST support registering multiple models simultaneously, each identified by a unique model ID that appears in the Models API.
- **FR-017**: System MUST gracefully handle backend errors (OOM, CUDA errors, timeouts) by returning structured OpenAI-format error responses rather than raw exceptions.

#### Authentication & Authorization

- **FR-018**: System MUST support Bearer token authentication via the `Authorization: Bearer <token>` header, matching OpenAI client library behavior.
- **FR-019**: System MUST allow authentication to be disabled (open mode), bypassing token validation for all requests.
- **FR-020**: System MUST support validating against a list of multiple valid API keys.

#### Response Format Compliance

- **FR-021**: All successful responses MUST include `object` field with the correct value (e.g., `"chat.completion"`, `"text_completion"`, `"list"`, etc.) as defined in the OpenAI v1 spec.
- **FR-022**: All error responses MUST use the OpenAI error format: `{"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}` with appropriate HTTP status codes (400, 401, 404, 429, 500).
- **FR-023**: System MUST generate `id` fields in the standard format: `chatcmpl-<random>` for chat completions, `cmpl-<random>` for completions, and similar for other resource types.
- **FR-024**: System MUST include a `system_fingerprint` field in chat completion and completion responses.
- **FR-025**: Token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`) MUST be calculated and included in all completion-type responses (non-streaming). For streaming responses, usage MUST be included in the final chunk when `stream_options.include_usage` is set to `true`.

#### Configuration

- **FR-026**: System MUST support configuration via both a TOML config file and environment variables. Environment variables MUST take precedence over config file values when both are provided. Configuration MUST cover: model path(s), API key(s), host/port, authentication mode, backend selection, and queue depth.
- **FR-027**: System MUST allow specifying the host and port the HTTP server binds to, with sensible defaults (e.g., `0.0.0.0:8000`).

### Key Entities

- **Model**: Represents a loaded model with `id` (string identifier), `created` (unix timestamp), `owned_by` (string, e.g., "local"), and `object` (fixed to "model").
- **Completion Response**: Represents the result of a chat or text completion request, containing `id`, `object`, `created`, `model`, `choices` array (each with `index`, message/text content, `finish_reason`), `usage` (token counts), and `system_fingerprint`.
- **Completion Request**: Represents an incoming inference request with parameters: `model`, `messages` or `prompt`, sampling parameters, `stream`, `tools`, and other OpenAI-defined fields.
- **Streaming Chunk**: A partial response emitted via SSE, containing `id`, `object`, `created`, `model`, and `choices` array with deltas.
- **Embedding Object**: Represents a single embedding result with `object: "embedding"`, `index`, and `embedding` (numeric array or base64 string).
- **File Object**: Represents a stored file with `id`, `object: "file"`, `bytes`, `created_at`, `filename`, `purpose`, and `status`.
- **Fine-tuning Job**: Represents a fine-tuning job with `id`, `object: "fine_tuning.job"`, `model`, `created_at`, `finished_at`, `fine_tuned_model`, `status`, `trained_tokens`, and related metadata.
- **API Key**: An authentication credential (string token) used to authorize incoming requests.
- **Backend**: The abstraction layer responsible for loading models and executing inference, with a defined interface for `generate`, `stream`, `embed`, and model lifecycle operations.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Any standard OpenAI Python SDK (openai >= 1.0) can connect to the service using the `base_url` parameter and successfully complete a chat completion request without modification.
- **SC-002**: Streaming chat completions produce the first output token (first SSE chunk) within 2 seconds of request initiation on a local GPU under single-user load.
- **SC-003**: The service handles at least 50 concurrent non-streaming chat completion requests without returning errors, assuming sufficient backend capacity.
- **SC-004**: All responses from all v1 endpoints pass schema validation against the OpenAI v1 OpenAPI specification for their respective resource types.
- **SC-005**: A user can swap the underlying model by changing only the configuration (model path) and restarting the service, without modifying application code or client-side logic.
- **SC-006**: Token usage counts (`prompt_tokens`, `completion_tokens`, `total_tokens`) reported in responses are accurate to within 5% of actual token counts for a representative test set.
- **SC-007**: An error from any backend (model load failure, OOM, timeout) is translated into a well-formed OpenAI-format error response within 5 seconds, with no leaked stack traces.
- **SC-008**: The service starts from a fresh process to accepting requests in under 60 seconds, assuming models are already downloaded locally.

## Clarifications

### Session 2026-05-27

- Q: Should the service expose health/readiness probe endpoints? → A: Add `GET /health` returning 200 when service is ready, 503 when not (with model status details)
- Q: What level of operational observability (logging & metrics) is required? → A: Structured JSON logs + OpenTelemetry-compatible metrics exporter (request counts, latencies, token usage, errors)
- Q: How should the service handle concurrent requests exceeding GPU capacity? → A: Process one request at a time with a bounded FIFO queue (configurable depth, reject when full with 429)
- Q: How should the service apply chat templates (LLaMA, Mistral, Qwen, ChatML formats)? → A: Delegate chat template formatting entirely to the backend; the HTTP layer passes raw `messages` arrays through without template application.
- Q: What configuration mechanism should the service use? → A: Support both TOML config file and environment variables, with environment variables taking precedence over file values when both are provided.

## Assumptions

- Users will have a GPU (CUDA-compatible) available on the host machine for local model inference, as transformers `generate` is prohibitively slow on CPU for large models.
- The initial implementation targets HuggingFace Transformers as the default backend; other backends (vLLM, TGI, llama.cpp, ExLlamaV2) are not in scope for this iteration but the architecture must accommodate them.
- The service will run on Linux (primary) or macOS/Windows (secondary); no Windows-specific assumptions are made.
- Model files are already downloaded or accessible via HuggingFace Hub; the service does not implement model downloading from scratch.
- The OpenAI API specification version targeted is v1 as of early 2026; deprecated v1 sub-endpoints (e.g., engines) are not required.
- The embeddings model and the generative model may be different; the service supports loading multiple models of different types.
- Fine-tuning jobs for a transformers backend use the native HuggingFace training APIs or are delegated to a configurable training backend.
- Audio (Whisper) and image generation backends are supported as optional; if no compatible backend is configured, those endpoints return a 501 Not Implemented or similar error.
- The service is intended for self-hosted/development use; production hardening (rate limiting beyond 429, load balancing, HTTPS termination) is out of scope for v1 of the service logic itself.
- API key storage is in-memory or in the configuration file; no external secrets management is required for v1.
