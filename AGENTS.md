# AGENTS.md

## Toolchain

- **Package manager**: `uv` — use for all dependency, venv, and dev tool operations
- **Python**: 3.12 (see `.python-version`)
- **Venv**: Always run commands inside the `uv`-managed venv at `.venv/`

```bash
uv venv                       # create venv (only once)
uv pip install -e ".[dev]"    # install project + dev extras
uv run python -m openai_http  # run server
uv run pytest                 # run tests
uv run ruff check .           # lint
uv run mypy openai_http/      # typecheck
```

## Commands

| Task | Command |
|------|---------|
| Run server (mock) | `uv run python -m openai_http` |
| All tests | `uv run pytest tests/ -v` |
| SDK tests only | `uv run pytest tests/sdk/ -v` |
| Unit tests | `uv run pytest tests/unit/ -v` |
| Single test | `uv run pytest tests/sdk/test_models.py::TestModelsAPI::test_models_list -v` |
| Lint | `uv run ruff check openai_http/ tests/` |
| Typecheck | `uv run mypy openai_http/` |

**Test prerequisites**: SDK tests (`tests/sdk/`) auto-start a server on port 8000 in a background thread, or reuse one already running. Do not start the server manually before running tests.

## Architecture

Single-package service `openai_http/` with layered modules:

- **`app.py`** — FastAPI factory `create_app()`, lifespan, middleware. **Not a global `app` object**; import `create_app`.
- **`config.py`** — `pydantic-settings` `Settings` class. TOML (`config.toml`) + env vars with `OPENAI_HTTP__` prefix (e.g. `OPENAI_HTTP__SERVER__PORT=9000`). Env vars override TOML.
- **`routers/`** — One file per API domain (`chat.py`, `models.py`, `health.py`). Register via `app.include_router()` in `app.py`.
- **`schemas/`** — Pydantic v2 models. Request models use `ConfigDict(extra="allow")` to accept unknown OpenAI params.
- **`backends/`** — `base.py` defines `Backend` Protocol. `mock_backend.py` for testing. Real transformers backend is Phase 6 (not yet implemented — `backend.type == "transformers"` raises `NotImplementedError`).
- **`errors.py`** — `OpenAIError` exception hierarchy + global exception handlers returning `{"error": {message, type, param, code}}`. All errors must use this format.
- **`queue.py`** — `RequestQueue` with `asyncio.Semaphore(1)` for GPU serialization. Use `async with queue.acquire()` in inference endpoints.
- **`observability/`** — JSON logging + OpenTelemetry metrics.

## Testing Gotchas

- Root `tests/conftest.py` provides an **async** `httpx.AsyncClient` fixture. `tests/sdk/conftest.py` overrides it with a **sync** `openai.OpenAI` client for SDK tests.
- SDK tests use model ID `"mock-gpt"` or `MOCK_MODELS[0]` from `test_base.py`. Never use `"test-model"` — it returns 404.
- Tests for unimplemented endpoints (completions, embeddings, files, fine-tuning, batches, moderations) use `pytest.skip()` via `_call_or_skip()`. They auto-activate when routes are added.
- `pydantic.Field()` does not accept both positional default and `default=` kwarg. Use `Field(default=None, ge=0.0)`.
- FastAPI routes returning `JSONResponse | StreamingResponse` require `response_model=None` in the decorator.

## Windows Quirks

- `subprocess.DEVNULL` fails with `OSError [WinError 6]` in some pytest session-scoped fixture contexts. The SDK test conftest uses `threading` + `uvicorn.Server` instead.
- Use `Select-Object -Last N` instead of `tail -n N` for PowerShell output piping.

<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan:

**Plan**: `specs/001-openai-http-api/plan.md`
**Spec**: `specs/001-openai-http-api/spec.md`
**Research**: `specs/001-openai-http-api/research.md`
**Data Model**: `specs/001-openai-http-api/data-model.md`
**API Contract**: `specs/001-openai-http-api/contracts/openai-v1-api.yaml`
**Quickstart**: `specs/001-openai-http-api/quickstart.md`
<!-- SPECKIT END -->
