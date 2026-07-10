# Feature Specification: Extensible Backend SDK for openai_http

**Feature Branch**: `002-extensible-backend-sdk`

**Created**: 2026-05-27

**Status**: Complete

**Input**: User description: "将openai_http重新封装，提供一个可以供开发者扩展的接口，来实现自己的后端。"

## User Scenarios & Testing *(mandatory)*

---

### User Story 1 - Install and Import openai_http as a Library (Priority: P1)

A developer installs `openai_http` via pip and imports it in their own application code. The package exposes a stable, top-level public API (`openai_http.BackendBase`, `openai_http.run_server`, etc.) that is suitable for use as a library dependency, not just as a standalone CLI application.

**Acceptance Scenarios**:

1. **Given** a Python environment where `openai_http` is installed, **When** a developer runs `import openai_http`, **Then** the import succeeds and `openai_http.BackendBase` is accessible.
2. **Given** `openai_http` is installed, **When** a developer inspects `dir(openai_http)`, **Then** all public entry points are discoverable at the top-level namespace.
3. **Given** `openai_http` is installed as a library dependency, **When** the developer's application imports it, **Then** no side-effects occur at import time.

---

### User Story 2 - Implement a Custom Backend by Subclassing BackendBase (Priority: P1)

A developer creates a custom inference backend by subclassing `openai_http.BackendBase` and implementing the required abstract methods.

**Acceptance Scenarios**:

1. **Given** a developer subclasses `BackendBase` and implements all required methods, **When** they instantiate it, **Then** no `TypeError` is raised.
2. **Given** a developer omits a required method, **When** they attempt to instantiate, **Then** a `TypeError` is raised naming the missing method(s).
3. **Given** a developer's custom backend does **not** implement an optional method, **When** a client calls that endpoint, **Then** the server returns HTTP 501 with an OpenAI-format error body.

---

### User Story 3 - Start the Server with a Custom Backend Programmatically (Priority: P1)

A developer wires their custom backend into the openai_http server by passing it to `openai_http.run_server(backend=my_backend_instance)`.

**Acceptance Scenarios**:

1. **Given** a valid `MyBackend` instance, **When** `run_server(backend=my_backend)` is called, **Then** an HTTP server starts serving OpenAI-compatible endpoints backed by the custom backend.
2. **Given** the server is running, **When** a client calls `POST /v1/chat/completions`, **Then** the request is routed to the custom backend.
3. **Given** the backend raises an unhandled exception, **When** the error propagates, **Then** the server returns a 500 response in OpenAI error format without exposing the traceback.

---

### User Story 4 - Custom Backend Lifecycle Hooks (Priority: P2)

`BackendBase` provides overridable lifecycle hooks (`setup` and `teardown`) that the server calls at the appropriate points.

---

### User Story 5 - Backend Validation at Startup (Priority: P2)

`run_server` performs a dry-run validation of the backend contract before starting uvicorn; `skip_validation=True` bypasses it.

---

## Success Criteria *(mandatory)*

1. Developers can define a working custom backend in a single Python file of under 80 lines and start serving OpenAI-compatible requests within 10 minutes of reading the quick-start documentation.
2. The public API surface exposed at `import openai_http` includes at most 5 top-level names.
3. 100% of the methods defined in `BackendBase` are covered by type hints.
4. A custom backend that implements only the minimum required set is sufficient to serve chat-completions and model-listing endpoints correctly.
5. Startup validation catches contract violations before any request is served.
6. The existing mock backend continues to work unchanged when run via the CLI.

## Key Entities *(optional)*

### BackendBase (abstract base class)
The primary contract that all custom backends must implement. Defines abstract async methods for text generation, streaming generation, model listing, and model retrieval; optional methods for embeddings; and overridable lifecycle hooks for setup and teardown.

### RequestQueue (concurrency control)
The existing semaphore-based request queue remains internal. Developers do not interact with it directly.

## Scope & Constraints *(mandatory)*

### In Scope
- Refactoring `openai_http` into a library with a clean top-level public API.
- Defining `BackendBase` as a base class with documented method contracts.
- Providing a programmatic server entry point (`run_server`).
- Providing lifecycle hooks and fail-fast backend validation.
- Documenting a quick-start guide.

### Out of Scope
- Implementing any real inference backend.
- Plugin auto-discovery mechanisms.
- Changes to the OpenAI-compatible API contract itself.
- Multi-backend routing.
- Hot-reloading or swapping backends at runtime.

## Dependencies & Assumptions *(mandatory)*

### Dependencies
- The existing OpenAI-compatible routing and response-formatting layer.
- The existing request queue and observability infrastructure.
- A packaging configuration that correctly declares `openai_http` as an installable distribution.

### Assumptions
- Developers will instantiate their custom backend and pass the instance to `run_server`.
- The target audience is developers comfortable writing async Python.
- Backward compatibility with `python -m openai_http` is maintained.
- All methods on `BackendBase` are asynchronous.
- The library follows semantic versioning.

## Open Questions

None.
