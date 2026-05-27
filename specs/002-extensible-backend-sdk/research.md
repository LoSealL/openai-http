# Research: Extensible Backend SDK for openai_http

**Feature**: `002-extensible-backend-sdk`
**Date**: 2026-05-27

## R-001: BackendBase as ABC vs Protocol

**Decision**: Abstract Base Class (ABC) using `abc.ABCMeta` + `@abstractmethod`, replacing the existing `Backend` Protocol in `backends/base.py`

**Rationale**: The spec requires that omitting a required method raises `TypeError` at instantiation time (US-2 acceptance scenario 2). Python Protocols only enforce structural subtyping at type-check time (mypy), not at runtime. ABC with `@abstractmethod` provides immediate runtime feedback and is the idiomatic Python approach for defining extension points. ABC also supports concrete default implementations for optional methods and lifecycle hooks — something protocols cannot express cleanly.

**Alternatives considered**:
- `typing.Protocol`: No runtime enforcement of missing methods; errors surface only via mypy or at first request, violating fail-fast requirement
- Hybrid (Protocol for typing + ABC for runtime): Adds complexity with two parallel definitions for no real benefit
- `zope.interface`: Non-standard, adds dependency

## R-002: Public API Surface Design

**Decision**: Expose exactly 4 names at the `openai_http` top level: `BackendBase`, `run_server`, `setup_logging`, `__version__`

**Rationale**: The spec constrains the public API to at most 5 top-level names (SC-002). Four names cover all user-facing functionality: the base class for implementing backends, the blocking server entry point, the logging convenience, and the version string. Developers access everything they need via `import openai_http` without reaching into subpackages. The `__all__` list in `__init__.py` enforces this boundary.

**Alternatives considered**:
- More names (e.g., `ServerConfig`, `RequestQueue`): Exceeds minimalist goal; `ServerConfig` maps to keyword arguments on `run_server`, and `RequestQueue` is internal
- Single `serve()` function name instead of `run_server()`: Too generic; `run_server` is more explicit about blocking behavior

## R-003: run_server Implementation

**Decision**: `run_server(backend, **config_kwargs)` creates the FastAPI app via `create_app()`, injects the backend, then calls `uvicorn.run()` synchronously (blocking). Accepts individual keyword arguments (host, port, etc.) rather than a separate `ServerConfig` object.

**Rationale**: Blocking matches the user's expected UX (`python src/myapp.py` → server starts and runs until Ctrl+C). Using `uvicorn.run()` directly is the standard pattern — it manages the event loop, signal handling, and graceful shutdown. Keyword arguments (rather than a config object) reduce boilerplate for the common case while remaining compatible with `**settings.model_dump()` for advanced users. The backend is injected into `app.state.backend` in a modified lifespan, bypassing the existing config-driven backend creation.

**Alternatives considered**:
- Non-blocking return value: Violates spec clarification (blocking behavior)
- ServerConfig dataclass: Adds an extra object for little benefit; kwargs are more Pythonic for 4-5 settings
- `asyncio.run(uvicorn.Server.serve())`: Lower-level, but `uvicorn.run()` is simpler and well-tested

## R-004: Backend Lifecycle Hook Design

**Decision**: `BackendBase` defines two async methods with default no-op implementations: `async def setup(self) -> None` and `async def teardown(self) -> None`. The server's lifespan calls `backend.setup()` before yielding (before accepting requests) and `backend.teardown()` after yielding (during shutdown).

**Rationale**: Default no-op implementations mean developers who don't need lifecycle hooks inherit working defaults (US-4 acceptance scenario 3). Async methods allow GPU operations (model loading, CUDA context cleanup) without blocking the event loop. The lifespan pattern in FastAPI naturally maps to before-yield/after-yield, so no additional framework machinery is needed.

**Alternatives considered**:
- Synchronous setup/teardown: Would block event loop during model loading; unacceptable for GPU operations
- Context manager protocol (`__aenter__`/`__aexit__`): Less discoverable than named hooks; harder to override individually
- Event-emitter pattern with callbacks: Over-engineered for two lifecycle points

## R-005: Backend Validation Strategy

**Decision**: Validation runs at startup (inside `run_server`, before calling `uvicorn.run()`). It checks: (1) all `@abstractmethod` are implemented (guaranteed by ABC but explicit for clarity), (2) `generate` returns a dict with required keys, (3) `generate_stream` is an `AsyncGenerator`. Can be skipped via `skip_validation=True`. Validation performs a single dry-run call with minimal input.

**Rationale**: Fail-fast validation (US-5) requires checking return shapes before any real request. A dry-run call to each method with dummy input is the most reliable way to verify return shape without complex type introspection. `skip_validation=True` supports rapid prototyping where dry-run calls may be expensive (e.g., real model backends that take 30s to respond).

**Alternatives considered**:
- Static type introspection via `inspect.signature` + `typing.get_type_hints`: Cannot verify runtime dict shapes, only declared return annotations
- Runtime monitoring (first-request validation): Defeats fail-fast goal; errors discovered too late
- Schema-based validation (e.g., JSON Schema for return dicts): Adds complexity; a simple key-check on dry-run output is sufficient

## R-006: 501 Handling for Unimplemented Optional Methods

**Decision**: Each router endpoint that calls an optional backend method wraps the call in a try/except for `NotImplementedError`. `BackendBase` provides default implementations for optional methods (`embed`, `generate_tool_calls`, etc.) that raise `NotImplementedError`. The router catches this, raises an `OpenAIError` subclass with status 501 and type `"not_implemented_error"`.

**Rationale**: `NotImplementedError` is the standard Python idiom for unimplemented methods. Wrapping it in the existing `OpenAIError` hierarchy (which already has `OpenAIError` base → `JSONResponse` handler) keeps error handling consistent. Each router adds one try/except block — minimal code change. New `NotImplementedOpenAIError` subclass (or reusing with status_code=501) provides the correct OpenAI-format response.

**Alternatives considered**:
- Checking `isinstance` or `hasattr` at request time: Fragile; ABC always has the methods (inherited defaults)
- Router middleware/decorator: Over-engineered; a simple try/except per endpoint is clearer and more maintainable
- Conditional route registration: More complex; routes should always exist for API discoverability

## R-007: Library Logging Strategy

**Decision**: The library uses Python's standard `logging` module with logger names under the `openai_http.*` namespace (already the case). No `NullHandler` is explicitly added — Python 3.2+ automatically handles unconfigured loggers. `setup_logging(level="info")` is a convenience function that adds a `StreamHandler` with the existing `JSONFormatter` to the `openai_http` root logger.

**Rationale**: Python's logging documentation explicitly recommends that libraries add no handlers and let applications configure them. The `openai_http.*` namespace already isolates library logs. `setup_logging()` is a one-liner convenience that makes the common case (wanting to see logs) trivial. The existing `JSONFormatter` from `observability/logging.py` is reused to maintain consistent log format.

**Alternatives considered**:
- `NullHandler` on `openai_http` logger: Defensive but unnecessary in Python 3.12+; adds a line for no benefit
- Auto-detect if handlers exist before adding: Over-engineered; `setup_logging()` is explicitly opt-in

## R-008: Backward Compatibility with CLI Entry Point

**Decision**: Keep `python -m openai_http` (i.e., `__main__.py`) working as-is. The `lifespan()` function in `app.py` retains its current config-driven backend creation path as a fallback. `run_server()` uses a different code path (backend-injected) that does not interfere with the existing CLI flow.

**Rationale**: The spec explicitly requires backward compatibility (SC-006). The existing `__main__.py` → `get_settings()` → `create_app()` → `lifespan()` chain creates a `MockTransformersBackend` from config. This path remains untouched. Library users go through `run_server()` which injects a backend before `create_app()` is called, bypassing the config-driven path.

**Alternatives considered**:
- Deprecating CLI entry point: Violates backward compatibility requirement
- Merging both paths into one: Adds conditional complexity for no benefit
