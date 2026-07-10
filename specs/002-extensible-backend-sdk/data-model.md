# Data Model: Extensible Backend SDK for openai_http

**Feature**: `002-extensible-backend-sdk`
**Date**: 2026-05-27

## Entity: BackendBase (abstract base class)

The core interface contract that all custom backends must implement.

### Required Abstract Methods

| Method | Signature | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `generate` | `async (prompt: str \| list[dict[str, str]], **kwargs: Any) -> dict` | `dict` with keys `generated_text: str`, `usage: dict` | Generate text completion |
| `generate_stream` | `async (prompt: str \| list[dict[str, str]], **kwargs: Any) -> AsyncGenerator[str, None]` | Async generator yielding text chunks | Stream text completion |
| `list_models` | `async () -> list[dict]` | List of dicts with `id`, `object`, `created`, `owned_by` | List available models |
| `get_model` | `async (model_id: str) -> Optional[dict]` | Model info dict or `None` | Get info for a specific model |

### Optional Methods (default: raise NotImplementedError)

| Method | Signature | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `embed` | `async (texts: list[str], **kwargs: Any) -> list[list[float]]` | List of embedding vectors | Generate embeddings |
| `generate_tool_calls` | `async (messages: list[dict], tools: list[dict], **kwargs: Any) -> list[dict]` | List of tool call dicts | Generate tool call responses |

### Lifecycle Hooks (default: no-op)

| Method | Signature | Returns | Description |
| ------ | --------- | ------- | ----------- |
| `setup` | `async () -> None` | None | Called before server starts accepting requests |
| `teardown` | `async () -> None` | None | Called during server shutdown |

### Validation Rules
- `generate` must return a dict containing `generated_text` (str) and `usage` (dict with `prompt_tokens`, `completion_tokens`, `total_tokens`)
- `generate_stream` must be an async generator
- `list_models` must return a list of dicts with `id`, `object`, `created`, `owned_by`
- `get_model` must return a dict or `None`
