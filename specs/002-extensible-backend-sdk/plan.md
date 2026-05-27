# Implementation Plan: Extensible Backend SDK for openai_http

**Branch**: `002-extensible-backend-sdk` | **Date**: 2026-05-27 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-extensible-backend-sdk/spec.md`

## Summary

Refactor `openai_http` from a standalone CLI application into an installable Python library with a clean public API. The core deliverable is `BackendBase` — an abstract base class that developers subclass to implement custom inference backends — and `run_server()` — a blocking entry point that starts the OpenAI-compatible HTTP server backed by the custom backend. The library also provides optional lifecycle hooks (`setup`/`teardown`), fail-fast backend validation at startup, a convenience logging function, and 501 responses for unimplemented optional endpoints. All changes preserve backward compatibility with the existing `python -m openai_http` CLI workflow.

## Technical Context

**Language/Version**: Python >=3.12 (per `pyproject.toml`)

**Primary Dependencies**: FastAPI, uvicorn[standard], pydantic>=2.5, pydantic-settings (all existing)

**Storage**: N/A (no new storage requirements; backend-specific storage is developer responsibility)

**Testing**: pytest, httpx (async client fixture for endpoint tests), openai SDK (compatibility tests)

**Target Platform**: Any platform Python 3.12+ supports (Linux server primary, macOS/Windows for development)

**Project Type**: Library (pip-installable package with optional CLI entry point)

**Performance Goals**: N/A (performance is backend-dependent; the library adds negligible overhead beyond HTTP routing)

**Constraints**: Max 5 top-level public names (SC-002). No side-effects at import time (US-1 scenario 3).

**Scale/Scope**: ~5 files modified, 1 new file (`backends/base.py` rewritten, `__init__.py` rewritten, `app.py` modified, `errors.py` extended, optional router changes for 501 handling)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

The project constitution (`.specify/memory/constitution.md`) is a template with no enforced principles. No gates to evaluate. **PASSED**.

## Project Structure

### Documentation (this feature)

```text
specs/002-extensible-backend-sdk/
├── plan.md              # This file
├── research.md          # Research decisions (R-001 through R-008)
├── data-model.md        # Entity definitions (BackendBase, ServerConfig)
├── quickstart.md        # Developer quick-start guide
├── contracts/
│   └── python-api.md    # Public API contract (BackendBase, run_server, setup_logging)
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Tasks (created by /speckit.tasks)
```

### Source Code (repository root)

```text
openai_http/
├── __init__.py                  # REWRITTEN: expose BackendBase, run_server, setup_logging, __version__
├── __main__.py                  # UNCHANGED: CLI entry point (backward compat)
├── app.py                       # MODIFIED: lifespan() supports backend injection; create_app() accepts optional backend
├── errors.py                    # MODIFIED: add NotImplementedError handling for optional endpoints
├── config.py                    # UNCHANGED: existing Settings class
├── queue.py                     # UNCHANGED: existing RequestQueue
├── backends/
│   ├── __init__.py              # MODIFIED: re-export BackendBase from base.py
│   ├── base.py                  # REWRITTEN: Backend Protocol → BackendBase ABC with lifecycle hooks
│   └── mock_backend.py          # MODIFIED: inherit from BackendBase (was standalone)
├── routers/
│   ├── __init__.py              # UNCHANGED
│   ├── chat.py                  # MODIFIED: wrap optional method calls with NotImplementedError → 501
│   ├── completions.py           # MODIFIED: same pattern
│   ├── embeddings.py            # MODIFIED: same pattern
│   ├── audio.py                 # MODIFIED: same pattern
│   ├── images.py                # MODIFIED: same pattern
│   ├── models.py                # UNCHANGED (get_model and list_models are required methods)
│   └── health.py                # UNCHANGED
├── schemas/                     # UNCHANGED
├── services/                    # UNCHANGED
└── observability/
    └── logging.py               # MODIFIED: extract setup_logging() as a reusable public function

tests/
├── unit/
│   └── test_backend_base.py     # NEW: ABC enforcement, lifecycle hooks, validation
├── integration/
│   └── test_library_api.py      # NEW: end-to-end test of run_server() with a inline custom backend
└── sdk/                          # EXISTING: should continue to pass (mock backend still works)

pyproject.toml                   # POSSIBLY MODIFIED: add package metadata for PyPI publishing
```

**Structure Decision**: Single Python package `openai_http/` with minimal modifications. The primary change is rewriting `backends/base.py` (Protocol → ABC), updating `__init__.py` to expose the public API, and modifying `app.py` to accept an externally-injected backend. All other modules remain largely unchanged.

## Implementation Tasks (by phase)

### Phase 3: BackendBase ABC (P1)

**Files**: `openai_http/backends/base.py`, `openai_http/backends/__init__.py`

1. Rewrite `backends/base.py`:
   - Replace `class Backend(Protocol)` with `class BackendBase(abc.ABC)`
   - Mark `generate`, `generate_stream`, `list_models`, `get_model` as `@abstractmethod`
   - Add optional methods with default `raise NotImplementedError`: `embed`, `generate_tool_calls`
   - Add lifecycle hooks with default no-op: `async def setup(self) -> None`, `async def teardown(self) -> None`
   - Add `from abc import ABC, abstractmethod` import
   - Include comprehensive docstrings per the contract in `contracts/python-api.md`

2. Update `backends/__init__.py`:
   - `from openai_http.backends.base import BackendBase`
   - `__all__ = ["BackendBase"]`

### Phase 4: Update MockBackend to inherit BackendBase (P1)

**Files**: `openai_http/backends/mock_backend.py`

1. Change `class MockTransformersBackend:` to `class MockTransformersBackend(BackendBase):`
2. Add `from openai_http.backends.base import BackendBase` import
3. Ensure all method signatures match `BackendBase` abstract definitions
4. Verify no changes needed to method bodies (they already match the contract)

### Phase 5: Public API in __init__.py (P1)

**Files**: `openai_http/__init__.py`

1. Add imports:
   ```python
   from openai_http.backends.base import BackendBase
   from openai_http._server import run_server
   from openai_http._logging import setup_logging
   ```
2. Define `__all__ = ["BackendBase", "run_server", "setup_logging", "__version__"]`
3. Ensure no side-effects at import time (no server auto-start, no global state mutation)

### Phase 6: run_server entry point (P1)

**Files**: NEW `openai_http/_server.py`, MODIFIED `openai_http/app.py`

1. Create `_server.py`:
   - Define `run_server(backend, *, host="0.0.0.0", port=8000, log_level="info", api_keys=None, queue_depth=32, skip_validation=False)`
   - Validate `isinstance(backend, BackendBase)` (raise `TypeError` if not)
   - Build a `Settings` object from keyword arguments
   - Run backend validation (unless `skip_validation=True`)
   - Call `create_app(config=settings, backend=backend)`
   - Call `uvicorn.run(app, host=..., port=..., log_level=...)`

2. Modify `app.py`:
   - Add `backend` parameter to `create_app(config=None, backend=None)`
   - Modify `lifespan()`: if `app.state.backend` is already set (library path), call `setup()`/`teardown()` lifecycle hooks; otherwise use existing config-driven backend creation (CLI path)
   - Inject `backend` into `app.state.backend` before lifespan runs

### Phase 7: Backend Validation (P2)

**Files**: NEW `openai_http/_validation.py`

1. Define `BackendValidationError(Exception)`
2. Define `validate_backend(backend: BackendBase) -> None`:
   - Inspect `generate`: verify it's a coroutine function
   - Inspect `generate_stream`: verify it's an async generator function
   - Dry-run `list_models()`: verify returns list of dicts with `id`, `object`, `created`, `owned_by`
   - Dry-run `get_model("__test__")`: verify returns dict or None
   - Dry-run `generate("test")`: verify returns dict with `generated_text` and `usage` keys
   - Raise `BackendValidationError` with descriptive message on failure

### Phase 8: setup_logging convenience (P1)

**Files**: NEW `openai_http/_logging.py`, MODIFIED `openai_http/observability/logging.py`

1. Create `_logging.py`:
   - Define `setup_logging(level: str = "info") -> None`
   - Get the `openai_http` logger
   - Add a `StreamHandler` with the existing `JSONFormatter`
   - Set the log level

2. Modify `observability/logging.py`:
   - Extract `setup_logging` as a callable function (may already exist as `setup_logging`)
   - Ensure it's importable from `openai_http._logging`

### Phase 9: 501 handling for unimplemented optional methods (P2)

**Files**: `openai_http/errors.py`, routers (`chat.py`, `completions.py`, `embeddings.py`, `audio.py`, `images.py`), `openai_http/app.py`

1. In `errors.py`:
   - Add `NotImplementedOpenAIError` subclass (or reuse `OpenAIError` with `status_code=501, error_type="not_implemented_error"`)

2. In each router that calls optional backend methods:
   - Wrap calls to `backend.embed()`, `backend.generate_tool_calls()`, etc. in try/except `NotImplementedError`
   - On catch, raise `NotImplementedOpenAIError` (or equivalent)

3. In `app.py` lifespan:
   - Check after setup whether optional methods are implemented; log which endpoints will return 501

### Phase 10: Error mapping for backend exceptions (P1)

**Files**: `openai_http/routers/chat.py`, `openai_http/routers/completions.py`, `openai_http/errors.py`

1. The existing `generic_error_handler` in `errors.py` already catches `Exception` and returns a 500 with `type: "server_error"`. Verify this covers backend exceptions during request processing.
2. If routers have their own try/except that doesn't re-raise through the error handler, add a catch-all that maps to the OpenAI error format.
3. Ensure raw Python tracebacks are logged to the server logger but NOT included in the response body.

### Phase 11: Tests (P1)

**Files**: `tests/unit/test_backend_base.py`, `tests/integration/test_library_api.py`

1. **Unit tests** (`test_backend_base.py`):
   - Test that subclassing `BackendBase` without implementing all abstract methods raises `TypeError` at instantiation
   - Test that a complete subclass instantiates without error
   - Test that default `setup()` and `teardown()` are no-ops
   - Test that default `embed()` raises `NotImplementedError`
   - Test that default `generate_tool_calls()` raises `NotImplementedError`
   - Test `BackendValidationError` for bad return shapes

2. **Integration tests** (`test_library_api.py`):
   - Define an inline custom backend in the test
   - Start server via `run_server()` in a background thread
   - Make HTTP requests to verify routing to custom backend
   - Test 501 response for unimplemented optional endpoints
   - Test that `setup()` was called before requests and `teardown()` after shutdown

### Phase 12: Quickstart and Documentation (P1)

**Files**: `specs/002-extensible-backend-sdk/quickstart.md`

1. Already written in Phase 1 design.
2. Verify the example code works by creating a test script matching the quickstart and running it.

## Complexity Tracking

> No constitution violations. Table intentionally omitted.
