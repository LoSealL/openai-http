# Tasks: Extensible Backend SDK for openai_http

**Feature**: `002-extensible-backend-sdk`
**Date**: 2026-05-27
**Plan**: [plan.md](plan.md)
**Spec**: [spec.md](spec.md)

## Implementation Strategy

- **MVP scope**: US1 + US2 + US3 (Phases 3-5 below) — install `openai_http` as a library, subclass `BackendBase`, and start a server with `run_server()`. These three stories are P1 and deliver the full developer extension loop end-to-end.
- **Incremental delivery**: US4 (lifecycle hooks) and US5 (startup validation) are P2 — they enhance DX but are not required for a working extension SDK. Implement after MVP verification.
- **Backward compatibility checkpoint**: After every phase, verify `python -m openai_http` (CLI mode with mock backend + config.toml) still works and the existing SDK/unit test suite passes. Do not ship any phase that breaks the existing CLI path.

## Dependencies (story completion order)

```
Setup (Phase 1)
    │
    ▼
Foundational (Phase 2)  ── blocking prerequisite for all user stories
    │
    ├──▶ US1: Import as library (Phase 3)            [independent of US2/US3]
    │
    ├──▶ US2: Subclass BackendBase (Phase 4)         [depends on Foundational]
    │
    ├──▶ US3: Start server programmatically (Phase 5) [depends on US1 + US2]
    │
    ├──▶ US4: Lifecycle hooks (Phase 6)              [depends on US3]
    │
    └──▶ US5: Backend validation (Phase 7)           [depends on US3]

Polish & Cross-cutting (Phase 8) ── after all stories complete
```

## Parallel Execution Examples

Within each story phase, tasks marked `[P]` can run concurrently:

- **Phase 2 (Foundational)**: `T003 [P]` (mock_backend update) and `T004 [P]` (NotImplementedOpenAIError) are independent after `T001`+`T002` complete.
- **Phase 3 (US1)**: `T007 [P]` (_server.py) and `T008 [P]` (_logging.py) can be created in parallel after `T005` rewrites `__init__.py`.
- **Phase 4 (US2)**: Router 501-handling tasks `T011`-`T015` are all `[P]` — each router file is independent.
- **Phase 5 (US3)**: `T022 [P]` (error-mapping verification in routers) and `T020` (app.py lifespan) are independent.

---

## Phase 1 — Setup

No project-level setup is required; the package, dev venv (`uv`), and test infrastructure already exist.

- [ ] T001 Verify Python 3.12 venv is active and `uv run pytest tests/ -v` baseline passes on current code before any modifications
- [ ] T002 Verify `python -m openai_http` starts the mock CLI server on port 8000 (baseline backward-compat check)

---

## Phase 2 — Foundational (blocking prerequisites)

These tasks define the core `BackendBase` contract and error types that every subsequent user story depends on. Must complete before US1/US2/US3 work can begin.

- [ ] T003 Rewrite `BackendBase` as ABC in openai_http/backends/base.py: replace `class Backend(Protocol)` with `class BackendBase(abc.ABC)`; add `@abstractmethod` decorators on `generate`, `generate_stream`, `list_models`, `get_model`; add optional methods `embed` and `generate_tool_calls` with default `raise NotImplementedError`; add lifecycle hooks `async def setup(self) -> None` and `async def teardown(self) -> None` with default no-op bodies; include full docstrings matching the contract in specs/002-extensible-backend-sdk/contracts/python-api.md
- [ ] T004 Update openai_http/backends/__init__.py to re-export `BackendBase` from `base.py` and set `__all__ = ["BackendBase"]` (drop the legacy `Backend` Protocol export)
- [ ] T005 [P] Update openai_http/backends/mock_backend.py so `MockTransformersBackend` inherits from `BackendBase`; add `from openai_http.backends.base import BackendBase`; verify class still instantiates and that method bodies/signatures still match the ABC contract (no method-body changes required beyond adding `raise NotImplementedError` for any new optional methods inherited from the base)
- [ ] T006 [P] Add `NotImplementedOpenAIError` to openai_http/errors.py: subclass `OpenAIError` with `status_code=501`, `error_type="not_implemented_error"`, default message `"<feature> is not supported by this backend"`. Also add a `register_error_handlers()` registration for it (or rely on the existing `OpenAIError` handler since it already handles all subclasses)

### Independent test
- `uv run pytest tests/unit/ -v` and `uv run pytest tests/sdk/ -v` both still pass (mock backend continues to work).

---

## Phase 3 — US1: Install and Import openai_http as a Library (P1)

**Story goal**: Developers can `import openai_http` and see `BackendBase`, `run_server`, `setup_logging`, `__version__` at the top-level namespace with no side effects at import time (spec US-1).

- [ ] T007 [P] [US1] Rewrite openai_http/__init__.py to: (1) keep `__version__`; (2) import `BackendBase` from `openai_http.backends.base`; (3) import `run_server` from `openai_http._server`; (4) import `setup_logging` from `openai_http._logging`; (5) set `__all__ = ["BackendBase", "run_server", "setup_logging", "__version__"]`. Imports must be lazy/deferred-safe — `import openai_http` must not trigger `uvicorn`, `fastapi`, or server construction at module-load time.
- [ ] T008 [P] [US1] Create openai_http/_server.py with a `run_server(backend, *, host="0.0.0.0", port=8000, log_level="info", api_keys=None, queue_depth=32, skip_validation=False) -> None` stub. Initial body: validate `isinstance(backend, BackendBase)`, build a `Settings` object from the kwargs, call `create_app(config=settings, backend=backend)` (function signature in app.py will be updated in Phase 5), and call `uvicorn.run(app, host=..., port=..., log_level=...)`. Leave `skip_validation` and lifecycle hook wiring as TODO comments for later phases.
- [ ] T009 [P] [US1] Create openai_http/_logging.py with `setup_logging(level: str = "info") -> None` that gets the `openai_http` root logger, attaches a `logging.StreamHandler()` using the existing `JSONFormatter` from `openai_http.observability.logging`, and sets the logger level. Must be idempotent (calling twice does not add duplicate handlers).
- [ ] T010 [US1] Refactor openai_http/observability/logging.py so its existing internal `setup_logging(level, log_format)` function (called by `app.py`) is a thin wrapper around the public one, OR rename the existing internal function (e.g., `_configure_app_logging`) to avoid clashing with the new public `setup_logging`. Verify existing CLI-mode server still produces the same JSON request logs.
- [ ] T011 [US1] Add tests/unit/test_library_import.py: (a) import `openai_http` and assert `set(dir(openai_http)) & {"BackendBase","run_server","setup_logging","__version__"} == {"BackendBase","run_server","setup_logging","__version__"}`; (b) assert `import openai_http` does not raise and does not auto-start any server (no ports open, no background threads spawned); (c) assert `openai_http.setup_logging()` installs exactly one handler on the `openai_http` logger and is idempotent on repeat calls.

### Independent test
- Run `python -c "import openai_http; print(openai_http.__version__); print(openai_http.BackendBase)"` — no exceptions, version string prints.
- `uv run pytest tests/unit/test_library_import.py -v`.

---

## Phase 4 — US2: Implement a Custom Backend by Subclassing BackendBase (P1)

**Story goal**: Subclassing `BackendBase` and omitting a required method raises `TypeError` at instantiation; optional methods raise `NotImplementedError` when called; all unimplemented optional endpoints return HTTP 501 with OpenAI error format (spec US-2).

- [ ] T012 [P] [US2] Update openai_http/routers/embeddings.py to wrap the call to `backend.embed(...)` in `try/except NotImplementedError`. On catch, raise `NotImplementedOpenAIError("Embeddings are not supported by this backend")`.
- [ ] T013 [P] [US2] Update openai_http/routers/chat.py to wrap the call to `backend.generate_tool_calls(...)` (used when `body.tools` is present) in `try/except NotImplementedError`. On catch, raise `NotImplementedOpenAIError("Tool calls are not supported by this backend")`.
- [ ] T014 [P] [US2] Update openai_http/routers/completions.py to follow the same `try/except NotImplementedError → NotImplementedOpenAIError` pattern for any optional backend call used by that router.
- [ ] T015 [P] [US2] Update openai_http/routers/audio.py with the same `try/except NotImplementedError → NotImplementedOpenAIError` pattern for any optional backend call.
- [ ] T016 [P] [US2] Update openai_http/routers/images.py with the same `try/except NotImplementedError → NotImplementedOpenAIError` pattern for any optional backend call.
- [ ] T017 [US2] Add tests/unit/test_backend_base.py: (a) define a subclass that omits one required method and assert `TypeError` is raised on instantiation with the missing method name in the message; (b) define a complete subclass with all four required methods and assert it instantiates cleanly; (c) instantiate a backend that inherits the default `embed`/`generate_tool_calls` and assert each raises `NotImplementedError` when awaited.
- [ ] T018 [US2] Add tests/unit/test_501_responses.py: build a FastAPI test client against an app whose backend has NOT overridden `embed`; send `POST /v1/embeddings`; assert response status is 501, body shape is `{"error": {"message": "...", "type": "not_implemented_error", ...}}`, and the message mentions "Embeddings".

### Independent test
- `uv run pytest tests/unit/test_backend_base.py tests/unit/test_501_responses.py -v`.
- Write a throwaway `class MinimalBackend(BackendBase)` in a temp file with only the four required methods; `BackendBase` enforcement confirmed via the unit tests above.

---

## Phase 5 — US3: Start the Server with a Custom Backend Programmatically (P1)

**Story goal**: `openai_http.run_server(backend=my_backend)` starts a blocking server that routes all OpenAI-compatible endpoints through the custom backend; backend exceptions during a request are mapped to OpenAI-format 500 errors (spec US-3, including scenario 5).

- [ ] T019 [US3] Modify openai_http/app.py: add `backend: BackendBase | None = None` parameter to `create_app(config=None, backend=None)`. In `lifespan()`: if `backend` was passed, assign `app.state.backend = backend` (skip the existing config-driven `MockTransformersBackend`/`transformers` branch); if `backend` is None, keep the existing config-driven creation path to preserve backward compatibility.
- [ ] T020 [US3] Wire `_server.run_server()` (created in Phase 3) to pass the `backend` argument through to `create_app(config=..., backend=backend)`. Verify `run_server` blocks until Ctrl+C by manual test (run in a terminal, Ctrl+C).
- [ ] T021 [US3] Verify error mapping: audit openai_http/errors.py — the existing `generic_error_handler` for `Exception` already returns 500 with `type: "server_error"` in OpenAI format and calls `logger.exception(...)`. Confirm by tracing that an uncaught exception inside a router's `backend.generate()` body is caught by this handler and that the raw traceback is NOT included in the response body (only `"Internal server error. Please try again later."` or the exception's safe message).
- [ ] T022 [P] [US3] Audit all routers (openai_http/routers/*.py) to ensure no router has a broad `except Exception` that swallows backend errors without re-raising through the OpenAI error handler; any such handler must either re-raise or call the `_error_json(...)` helper with `error_type="server_error"`. Raw tracebacks must go to `logger.exception(...)`, never to the response body.
- [ ] T023 [US3] Add tests/integration/test_library_api.py: define an inline `class EchoBackend(BackendBase)` inside the test file that returns distinctive values (e.g., `"ECHO: <input>"`), start `run_server(backend=EchoBackend(), port=<free_port>)` in a background thread (using the same `threading`+`uvicorn.Server` pattern already used in tests/sdk/conftest.py per the Windows quirks note in AGENTS.md), issue `POST /v1/chat/completions` and `GET /v1/models` via `httpx`, assert the responses reflect the custom backend (not the mock). Tear down the thread cleanly.
- [ ] T024 [US3] Add an integration test (in the same tests/integration/test_library_api.py) that uses a backend whose `generate` raises `RuntimeError("boom")`, calls `POST /v1/chat/completions`, and asserts: HTTP 500, OpenAI error shape `{"error": {"message": "...", "type": "server_error", ...}}`, and that the body does NOT contain the string "boom" or a Python traceback.
- [ ] T025 [US3] Add an integration test asserting that `openai_http.run_server("not-a-backend")` raises `TypeError` immediately without starting any server sockets.

### Independent test
- `uv run pytest tests/integration/test_library_api.py -v`.
- Manual: write `myapp.py` matching the example in specs/002-extensible-backend-sdk/quickstart.md, run it, `curl http://localhost:8000/v1/models` and `POST /v1/chat/completions` — responses must come from the custom backend.

---

## Phase 6 — US4: Custom Backend Lifecycle Hooks (P2)

**Story goal**: The server calls `await backend.setup()` before accepting requests and `await backend.teardown()` on graceful shutdown; errors in `setup()` abort startup clearly; default no-op hooks are safe to inherit (spec US-4).

- [ ] T026 [US4] Update openai_http/app.py `lifespan()`: in the library path (backend injected via `create_app(backend=...)`), `await backend.setup()` inside the lifespan *before* the `yield` and inside a `try/except` that re-raises with a clear `RuntimeError("Backend setup failed: <original exception>")` message so that startup aborted by setup is visible on stderr. After the `yield` (shutdown), `await backend.teardown()`.
- [ ] T027 [US4] Update tests/integration/test_library_api.py: define a backend that tracks `setup_called`, `teardown_called`, `requests_served` flags; assert after a test sequence that `setup_called` became True before any request, and `teardown_called` became True after shutdown.
- [ ] T028 [US4] Add a negative test: backend whose `setup()` raises `RuntimeError("GPU OOM")`; call `create_app(backend=...)` and assert the lifespan raises a `RuntimeError` whose message mentions "Backend setup failed".
- [ ] T029 [US4] Regression test: instantiate `MockTransformersBackend` (inherits no-op hooks) and `await backend.setup()` / `await backend.teardown()` — assert no exceptions.

### Independent test
- `uv run pytest tests/integration/test_library_api.py -k "lifecycle" -v`.

---

## Phase 7 — US5: Backend Validation at Startup (P2)

**Story goal**: `run_server` performs a dry-run validation of the backend contract before starting uvicorn; `skip_validation=True` bypasses it; errors are clear and actionable (spec US-5).

- [ ] T030 [US5] Create openai_http/_validation.py with `class BackendValidationError(Exception)` and `async def validate_backend(backend: BackendBase) -> None`. Checks: (1) `inspect.iscoroutinefunction(backend.generate)`; (2) `inspect.isasyncgenfunction(backend.generate_stream)`; (3) `await backend.list_models()` returns `list[dict]` each with `id`, `object`, `created`, `owned_by`; (4) `await backend.get_model("__validation_test__")` returns `dict | None`; (5) `await backend.generate("validation probe")` returns `dict` with `generated_text: str` and `usage: dict` containing `prompt_tokens`, `completion_tokens`, `total_tokens`. Raise `BackendValidationError` with a message naming the failing check and the actual value seen.
- [ ] T031 [US5] Wire validation into openai_http/_server.py `run_server`: call `asyncio.run(validate_backend(backend))` before `uvicorn.run(...)`, unless `skip_validation=True`. Translate `BackendValidationError` into a non-zero process exit with the error printed to stderr (do NOT start the server if validation fails).
- [ ] T032 [US5] Add tests/unit/test_validation.py: (a) complete valid backend → `validate_backend` returns without error; (b) backend whose `generate` returns a string → raises `BackendValidationError` mentioning `generate` and `generated_text`; (c) backend whose `generate_stream` is sync → raises `BackendValidationError` mentioning `generate_stream`; (d) backend whose `list_models` returns dicts missing `owned_by` → raises `BackendValidationError` mentioning `owned_by`.
- [ ] T033 [US5] Add an integration test calling `run_server(backend=<invalid>, skip_validation=True)` and asserting the server starts (validation bypassed).

### Independent test
- `uv run pytest tests/unit/test_validation.py -v`.

---

## Phase 8 — Polish & Cross-Cutting Concerns

- [ ] T034 Verify `python -m openai_http` still starts the mock CLI server with `config.toml` exactly as before (backward compatibility, spec success criterion 6).
- [ ] T035 Run the full existing suite: `uv run pytest tests/ -v`. All pre-existing tests in `tests/sdk/` and `tests/unit/` must pass. Fix any regressions introduced by this feature.
- [ ] T036 Run `uv run ruff check openai_http/` and `uv run mypy openai_http/`; fix any new lint/type errors. Ensure `mypy` accepts the new `BackendBase` ABC usages and that `Any` / `**kwargs` in method signatures are typed consistently with the rest of the codebase.
- [ ] T037 Verify the quickstart example works end-to-end: create `specs/002-extensible-backend-sdk/examples/myapp.py` matching specs/002-extensible-backend-sdk/quickstart.md, run it with `python specs/002-extensible-backend-sdk/examples/myapp.py`, and verify `GET /v1/models` and `POST /v1/chat/completions` both succeed with the example's `MyBackend`.
- [ ] T038 Bump `version` in pyproject.toml and `__version__` in openai_http/__init__.py to `0.2.0` per semantic-versioning assumption (new public API = minor bump). Update README.md (if it exists) with a brief "Library usage" section pointing to the quickstart.
- [ ] T039 Final review against spec success criteria: (1) custom backend fits in a single ~80-line file — confirm via the quickstart example; (2) top-level namespace ≤ 5 names — verify by `python -c "import openai_http; print([n for n in dir(openai_http) if not n.startswith('_')])"`; (3) 100% of `BackendBase` methods have type hints — `uv run mypy openai_http/backends/base.py` clean; (4) minimum required set (`generate`, `generate_stream`, `list_models`, `get_model`) serves chat + models endpoints — confirm via `T023`/`T024`; (5) validation catches contract violations — confirm via `T032`; (6) mock backend CLI unchanged — confirm via `T034`.

---

## Task Summary

| Phase          | Tasks     | Count | Parallel opportunities                  |
| -------------- | --------- | ----- | --------------------------------------- |
| 1. Setup       | T001-T002 | 2     | —                                       |
| 2. Foundational | T003-T006 | 4     | T005 ‖ T006 (after T003+T004)           |
| 3. US1 (P1)    | T007-T011 | 5     | T008 ‖ T009 (after T007)                |
| 4. US2 (P1)    | T012-T018 | 7     | T012 ‖ T013 ‖ T014 ‖ T015 ‖ T016       |
| 5. US3 (P1)    | T019-T025 | 7     | T022 (parallel audit)                   |
| 6. US4 (P2)    | T026-T029 | 4     | —                                       |
| 7. US5 (P2)    | T030-T033 | 4     | —                                       |
| 8. Polish      | T034-T039 | 6     | T035 ‖ T036 ‖ T037                      |
| **Total**      |           | **39** |                                        |

**Suggested MVP scope**: Phases 1-5 (T001-T025) — delivers the complete library import → subclass → run_server loop that satisfies the user's original example. Phases 6 and 7 are incremental improvements.
