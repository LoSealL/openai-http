# Data Model: Extensible Backend SDK for openai_http

**Feature**: `002-extensible-backend-sdk`
**Date**: 2026-05-27

## Entity: BackendBase (abstract base class)

The core interface contract that all custom backends must implement.

### Required Abstract Methods

| Method | Signature | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `generate` | `async (prompt: str \| list[dict[str, str]], **kwargs: Any) -> dict` | `dict` with keys `generated_text: str`, `usage: dict` (containing `prompt_tokens`, `completion_tokens`, `total_tokens`) | Generate text completion for a prompt or messages list |
| `generate_stream` | `async (prompt: str \| list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]` | Async generator yielding text chunks | Stream text completion token-by-token |
| `list_models` | `async () -> list[dict]` | List of dicts with keys `id: str`, `object: "model"`, `created: int`, `owned_by: str` | List available models |
| `get_model` | `async (model_id: str) -> Optional[dict]` | Model info dict or `None` | Get info for a specific model |

### Optional Methods (default: raise NotImplementedError)

| Method | Signature | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `embed` | `async (texts: list[str], **kwargs: Any) -> list[list[float]]` | List of embedding vectors | Generate embeddings for input texts |
| `generate_tool_calls` | `async (messages: list[dict], tools: list[dict], **kwargs: Any) -> list[dict]` | List of tool call dicts | Generate tool call responses |

### Lifecycle Hooks (default: no-op)

| Method | Signature | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `setup` | `async () -> None` | None | Called before server starts accepting requests. Override to load model weights, initialize GPU context, etc. |
| `teardown` | `async () -> None` | None | Called during server shutdown. Override to release resources, close file handles, free GPU memory. |

### Validation Rules
- `generate` must return a dict containing at minimum `generated_text` (str) and `usage` (dict with `prompt_tokens`, `completion_tokens`, `total_tokens` integers)
- `generate_stream` must be an async generator (yield str chunks)
- `list_models` must return a list of dicts, each with `id`, `object`, `created`, `owned_by` keys
- `get_model` must return either a model info dict or `None`

### State Transitions
1. **Instantiated**: Developer calls `MyBackend(...)` — constructor runs, no async setup
2. **Setup**: Server calls `await backend.setup()` — model weights loaded, GPU context initialized
3. **Serving**: Server routes requests to `generate`, `generate_stream`, etc.
4. **Teardown**: Server calls `await backend.teardown()` — resources released
5. **Destroyed**: Backend is garbage-collected

---

## Entity: ServerConfig (keyword arguments to run_server)

Not a separate class; configuration is passed as keyword arguments to `run_server()`.

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `host` | `str` | `"0.0.0.0"` | Bind address |
| `port` | `int` | `8000` | Listen port |
| `log_level` | `str` | `"info"` | Uvicorn log level |
| `api_keys` | `list[str]` | `[]` | Bearer token API keys (empty = auth disabled) |
| `queue_depth` | `int` | `32` | Max pending requests before 429 |
| `skip_validation` | `bool` | `False` | Skip backend contract validation at startup |

**Internal mapping**: These kwargs are converted into the existing `Settings` pydantic-settings object (`ServerSettings`, `AuthSettings`, `QueueSettings`) for compatibility with the app internals.

---

## Entity: BackendValidationResult

Not a public entity; used internally by the startup validation routine.

### Validation Checks
1. **Method existence**: Verified by ABC mechanism at instantiation time
2. **Async conformance**: Inspect each method to confirm it's a coroutine function (or async generator)
3. **Return shape**: Dry-run call with minimal input, verify return dict contains required keys

### Failure Modes
- `BackendValidationError`: Raised at startup if any check fails; message identifies the failing method and expected vs actual shape

---

## Relationship Diagram

```
User Code
    │
    ├─ import openai_http
    │   └─ BackendBase (subclass this)
    │
    ├─ class MyBackend(BackendBase):
    │   └─ implements: generate, generate_stream, list_models, get_model
    │                  [optional: embed, generate_tool_calls]
    │                  [optional: setup, teardown]
    │
    └─ openai_http.run_server(backend=MyBackend(), host=..., port=...)
        │
        ├─ Validates backend contract
        ├─ Creates FastAPI app via create_app()
        │   └─ Routers access request.app.state.backend
        ├─ Calls backend.setup()
        ├─ Starts uvicorn (blocking)
        ├─ Routes requests to backend methods
        └─ Calls backend.teardown() on shutdown
```
