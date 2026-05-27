# Feature Specification: Extensible Backend SDK for openai_http

**Feature Branch**: `002-extensible-backend-sdk`

**Created**: 2026-05-27

**Status**: Draft

**Input**: User description: "将openai_http重新封装，提供一个可以供开发者扩展的接口，来实现自己的后端。比如：pip install openai_http，然后 import openai_http，定义 MyBackend(openai_http.BackendBase)，运行 python src/myapp.py 即可启动服务。"

## User Scenarios & Testing *(mandatory)*

---

### User Story 1 - Install and Import openai_http as a Library (Priority: P1)

A developer installs `openai_http` via pip and imports it in their own application code. The package exposes a stable, top-level public API (`openai_http.BackendBase`, `openai_http.serve`, etc.) that is suitable for use as a library dependency, not just as a standalone CLI application.

**Why this priority**: Without a clean importable interface, the entire extensibility story falls apart. This is the foundational requirement that unlocks all other user stories.

**Independent Test**: Can be fully tested by installing `openai_http` into a fresh environment, writing a Python script that performs `import openai_http` and instantiates the public API objects, and confirming that no internal modules need to be reached into.

**Acceptance Scenarios**:

1. **Given** a Python environment where `openai_http` is installed, **When** a developer runs `import openai_http` in a Python script, **Then** the import succeeds with no errors and `openai_http.BackendBase` is accessible as a base class.
2. **Given** `openai_http` is installed, **When** a developer inspects `dir(openai_http)` or reads the package's API reference, **Then** all public entry points (`BackendBase`, `run_server`, version info) are discoverable at the top-level namespace.
3. **Given** `openai_http` is installed as a library dependency, **When** the developer's application imports it alongside other packages, **Then** no side-effects (server auto-start, global state mutation, log handler conflicts) occur at import time.
4. **Given** a developer uses `openai_http` as a library and has no logging configured, **When** they call `openai_http.setup_logging()` (or equivalent convenience function) before starting the server, **Then** a default console handler is installed on the library's logger namespace, producing visible server log output without the developer needing to configure Python's `logging` module manually.
5. **Given** a developer uses `openai_http` as a library and neither calls `setup_logging()` nor configures their own logging, **When** the server runs, **Then** the server still functions correctly; log records emitted by the library are suppressed (as expected for a quiet-by-default library following Python logging best practices).

---

### User Story 2 - Implement a Custom Backend by Subclassing BackendBase (Priority: P1)

A developer creates a custom inference backend by subclassing `openai_http.BackendBase` and implementing the required abstract methods (text generation, streaming generation, model listing, and optionally embeddings). The base class provides clear method signatures, type hints, and documentation so that the developer knows exactly what each method must accept and return.

**Why this priority**: This is the core extensibility mechanism. If subclassing is unclear or brittle, developers cannot build their own backends reliably.

**Independent Test**: Can be fully tested by writing a standalone Python file that defines `class MyBackend(openai_http.BackendBase)`, implements all abstract methods, and confirming that the class can be instantiated without errors and satisfies the contract defined by `BackendBase`.

**Acceptance Scenarios**:

1. **Given** a developer subclasses `BackendBase`, **When** they implement all required abstract methods (`generate`, `generate_stream`, `list_models`, `get_model`), **Then** the custom class can be instantiated without raising `TypeError` for unimplemented abstract methods.
2. **Given** a developer subclasses `BackendBase` but omits a required abstract method, **When** they attempt to instantiate the class, **Then** a clear `TypeError` is raised naming the missing method(s).
3. **Given** a developer inspects `BackendBase` (via `help()`, IDE autocomplete, or documentation), **When** they read method signatures, **Then** each method includes a docstring describing its expected input types, return types, and return value structure (e.g., the dict shape for `generate`).
4. **Given** a developer's custom backend implements the optional `embed` method, **When** the service receives a request to the embeddings endpoint, **Then** the custom `embed` implementation is called and its output is served correctly in the OpenAI-compatible response format.
5. **Given** a developer's custom backend does **not** implement the optional `embed` method, **When** a client calls `POST /v1/embeddings`, **Then** the server returns an HTTP 501 response with an OpenAI-format error body `{"error": {"message": "Embeddings are not supported by this backend", "type": "not_implemented_error"}}`. The same 501 behavior applies to any other optional backend method (e.g., audio transcription, image generation) that the backend does not implement.

---

### User Story 3 - Start the Server with a Custom Backend Programmatically (Priority: P1)

A developer wires their custom backend into the openai_http server by passing it (or a factory that produces it) to a blocking server entry point such as `openai_http.run_server(backend=my_backend_instance)` or equivalent. The call blocks the calling thread, runs the server until interrupted (Ctrl+C / SIGINT), then shuts down cleanly. The server then routes all incoming OpenAI-compatible API requests through the developer's custom backend, and the developer does not need to modify any code inside the `openai_http` package itself.

**Why this priority**: Programmatic startup is the primary integration pattern shown in the user's example. Developers must be able to launch the server from their own entry-point script without editing openai_http internals.

**Independent Test**: Can be fully tested by writing `src/myapp.py` that instantiates `MyBackend`, calls `openai_http.run_server(backend=backend)`, making an HTTP request to the running server, and verifying that the response body was produced by the custom backend logic (not the built-in mock backend).

**Acceptance Scenarios**:

1. **Given** a developer has a valid `MyBackend` instance, **When** they call `openai_http.run_server(backend=my_backend)` in a script, **Then** an HTTP server starts on the configured host and port, serving OpenAI-compatible endpoints backed by the custom backend.
2. **Given** the server is running with a custom backend, **When** a client calls `POST /v1/chat/completions`, **Then** the request is routed to the custom backend's `generate` (or `generate_stream`) method, and the response reflects the custom backend's output — not the built-in mock.
3. **Given** the server is running with a custom backend, **When** a client calls `GET /v1/models`, **Then** the response contains the model list returned by the custom backend's `list_models` method.
4. **Given** the developer wants to pass configuration options (host, port, authentication settings, request queue depth), **When** they call `run_server` with config arguments, **Then** the server applies those settings to the running instance.
5. **Given** the server is running with a custom backend and the backend's `generate` or `generate_stream` method raises an unhandled exception during a live request, **When** the error propagates, **Then** the server catches it and returns a 500 response body in the standard OpenAI error shape (`{"error": {"message": "<description>", "type": "server_error"}}`) — the raw Python traceback is not exposed to the client but is written to server logs.

---

### User Story 4 - Custom Backend Lifecycle Hooks (Priority: P2)

A developer needs to perform setup and teardown work in their custom backend — for example, loading model weights into GPU memory before the server starts accepting requests, and releasing resources after the server shuts down. `BackendBase` provides overridable lifecycle hooks (`setup` and `teardown`, or equivalent) that the server calls at the appropriate points.

**Why this priority**: Without lifecycle hooks, developers are forced to put initialization logic into `__init__`, which may block the event loop or make error handling awkward. Proper lifecycle support is important for real-world GPU/model backends but is secondary to the core subclass-and-run story.

**Independent Test**: Can be fully tested by implementing `setup` and `teardown` in a custom backend, starting and stopping the server, and verifying (via log messages or side-effects) that `setup` ran before any request was served and `teardown` ran after the server stopped.

**Acceptance Scenarios**:

1. **Given** a custom backend overrides the `setup` lifecycle hook, **When** the server starts, **Then** `setup` is called before the server begins accepting requests, and any errors raised in `setup` are surfaced clearly to the operator (the server does not silently start with an unready backend).
2. **Given** a custom backend overrides the `teardown` lifecycle hook, **When** the server is shut down gracefully, **Then** `teardown` is called before the process exits, allowing cleanup of resources such as GPU memory or file handles.
3. **Given** a developer does not override the lifecycle hooks, **When** the server starts and stops, **Then** no errors occur — the default (no-op) implementations in `BackendBase` are safe to inherit.

---

### User Story 5 - Backend Validation at Startup (Priority: P2)

When the server starts with a custom backend, it verifies that the backend satisfies the required contract — that all required methods exist, are awaitable (where required), and return values of the expected shape. The developer receives a clear, actionable error message if the backend does not meet expectations, rather than a cryptic failure during the first live request.

**Why this priority**: Fail-fast validation prevents developers from deploying a broken backend and only discovering the issue when the first production request fails. It is important for developer experience but is secondary to the core extensibility mechanism.

**Independent Test**: Can be fully tested by defining a backend whose `generate` method returns a string (instead of the required dict), starting the server, and verifying that the server fails fast at startup with a clear error message describing the shape mismatch.

**Acceptance Scenarios**:

1. **Given** a custom backend's `generate` method does not return the required dict shape, **When** the server performs startup validation, **Then** a clear error is raised describing the expected return shape and the actual return value received.
2. **Given** a custom backend's `generate_stream` method is not an async generator, **When** the server performs startup validation, **Then** a clear error is raised identifying the method as incorrectly typed.
3. **Given** a fully conformant custom backend, **When** the server performs startup validation, **Then** validation passes silently and the server starts normally.
4. **Given** a developer wants to skip validation (e.g., for rapid prototyping), **When** they pass a `skip_validation=True` flag to the startup entry point, **Then** the server starts without performing contract checks.

## Success Criteria *(mandatory)*

1. Developers can define a working custom backend in a single Python file of under 80 lines (excluding model-loading logic) and start serving OpenAI-compatible requests within 10 minutes of reading the quick-start documentation.
2. The public API surface exposed at `import openai_http` includes at most 5 top-level names, keeping the library's interface small and easy to learn.
3. 100% of the methods defined in `BackendBase` are covered by type hints, with documented return shapes, so that IDE autocompletion and static analysis tools assist developers during implementation.
4. A custom backend that implements only `generate`, `generate_stream`, `list_models`, and `get_model` (the minimum required set) is sufficient to serve chat-completions and model-listing endpoints correctly — no additional internal methods need to be overridden.
5. Startup validation catches at least the following contract violations before any request is served: missing required methods, non-async methods where async is required, and return values that do not match the documented shape.
6. The existing mock backend continues to work unchanged when the server is run via the CLI (`python -m openai_http`), preserving backward compatibility with the current `config.toml`-based workflow.

## Key Entities *(optional)*

### BackendBase (abstract base class)
The primary contract that all custom backends must implement. Defines abstract async methods for text generation, streaming generation, model listing, and model retrieval; optional methods for embeddings; and overridable lifecycle hooks for setup and teardown.

### ServerConfig (configuration object)
An object (or plain keyword arguments) that developers pass to `run_server` to configure host, port, authentication, request queue depth, and other server-wide settings. Maps to the existing `Settings` structure but is exposed as a library-friendly API rather than through TOML files.

### RequestQueue (concurrency control)
The existing semaphore-based request queue remains internal. Developers do not interact with it directly; it is configured via `ServerConfig`.

## Scope & Constraints *(mandatory)*

### In Scope
- Refactoring `openai_http` into a library with a clean top-level public API (`openai_http.BackendBase`, `openai_http.run_server`, etc.).
- Defining `BackendBase` as a base class with documented method contracts, replacing or wrapping the existing `Backend` Protocol.
- Providing a programmatic server entry point (`run_server` or equivalent) that accepts a backend instance and optional configuration.
- Providing lifecycle hooks (`setup`, `teardown`) in `BackendBase`.
- Providing fail-fast backend validation at startup.
- Packaging and distributing the project so that standard installation (`pip install openai_http`) works.
- Documenting a quick-start guide showing a complete `myapp.py` example from install to running server.

### Out of Scope
- Implementing any real inference backend (transformers, vLLM, ONNX, etc.) — only the extension interface is delivered.
- Plugin auto-discovery mechanisms (e.g., entry points, `load_backend_from_string`). Developers explicitly instantiate and pass their backend.
- Changes to the OpenAI-compatible API contract itself (endpoints, request/response schemas). Those remain as defined in feature 001.
- Multi-backend routing (serving multiple backends behind one server instance).
- Hot-reloading or swapping backends at runtime without server restart.

## Dependencies & Assumptions *(mandatory)*

### Dependencies
- The existing OpenAI-compatible routing and response-formatting layer defined in feature 001.
- The existing request queue and observability infrastructure.
- A packaging configuration that correctly declares `openai_http` as an installable distribution.

### Assumptions
- Developers will instantiate their custom backend in their own code and pass the instance to `run_server`; the library will not attempt to import user code by path or name.
- The target audience is developers who are comfortable writing async Python and can implement a `generate` method for their model framework of choice.
- Backward compatibility with the current `python -m openai_http` CLI entry point and `config.toml` configuration flow is maintained; the library API is an additional usage path, not a replacement.
- All methods on `BackendBase` are asynchronous. Synchronous backends are out of scope (developers are expected to wrap synchronous code appropriately).
- The library follows semantic versioning: breaking changes to `BackendBase`'s abstract method signatures, required return shapes, or lifecycle semantics occur only in major version bumps; minor releases introduce new optional methods with deprecation warnings for any future removals.

## Clarifications

### Session 2026-05-27
- Q: How should backend exceptions during live requests be surfaced to API clients? → A: Map to OpenAI error format: catch exceptions and wrap as `{"error": {"message": "...", "type": "server_error", ...}}`.
- Q: Should `run_server` block or return a handle? → A: Blocking — `run_server` runs the event loop until interrupted (Ctrl+C / SIGINT), then returns.
- Q: How should the library handle its own log output when imported? → A: Quiet library by default (no handlers installed); expose a convenience function (e.g., `openai_http.setup_logging()`) that users can call to install a default console handler if they have no logging configured.
- Q: What should the server do when a client calls an optional endpoint (e.g., embeddings) whose backend method was not implemented? → A: Return HTTP 501 with an OpenAI-format error body `{"error": {"message": "...", "type": "not_implemented_error"}}`.
- Q: What stability promise should the library make about `BackendBase`'s method signatures across releases? → A: Semantic versioning — breaking changes only in major version bumps; deprecation warnings in minor releases.

## Open Questions

None.
