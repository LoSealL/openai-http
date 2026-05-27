import inspect

from openai_http.backends.base import BackendBase


class BackendValidationError(Exception):
    pass


async def validate_backend(backend: BackendBase) -> None:
    if not inspect.iscoroutinefunction(backend.generate):
        raise BackendValidationError(
            f"generate must be an async method (coroutine function), "
            f"got {type(backend.generate).__name__}"
        )

    if not inspect.isasyncgenfunction(backend.generate_stream):
        raise BackendValidationError(
            f"generate_stream must be an async generator function, "
            f"got {type(backend.generate_stream).__name__}"
        )

    models = await backend.list_models()
    if not isinstance(models, list):
        raise BackendValidationError(
            f"list_models must return a list, got {type(models).__name__}"
        )
    for i, m in enumerate(models):
        if not isinstance(m, dict):
            raise BackendValidationError(
                f"list_models[{i}] must be a dict, got {type(m).__name__}"
            )
        for key in ("id", "object", "created", "owned_by"):
            if key not in m:
                raise BackendValidationError(
                    f"list_models[{i}] missing required key '{key}'"
                )

    model_info = await backend.get_model("__validation_test__")
    if model_info is not None and not isinstance(model_info, dict):
        raise BackendValidationError(
            f"get_model must return dict or None, got {type(model_info).__name__}"
        )

    result = await backend.generate("validation probe")
    if not isinstance(result, dict):
        raise BackendValidationError(
            f"generate must return a dict, got {type(result).__name__}"
        )
    if "generated_text" not in result:
        raise BackendValidationError(
            "generate return dict must contain 'generated_text' key"
        )
    usage = result.get("usage")
    if not isinstance(usage, dict):
        raise BackendValidationError(
            "generate return dict must contain 'usage' dict, "
            f"got {type(usage).__name__}"
        )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if key not in usage:
            raise BackendValidationError(
                f"generate usage dict must contain '{key}'"
            )
