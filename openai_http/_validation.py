"""
Copyright (C) 2026 The OPENAI-HTTP Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Backend validation utilities.

Validates that a BackendBase implementation conforms to the expected
interface contract by checking method signatures, return types, and
required response fields.
"""

import inspect

from openai_http.backends.base import BackendBase


class BackendValidationError(Exception):
    """Raised when a backend fails validation checks."""

    pass


async def validate_backend(backend: BackendBase) -> None:
    """Validate a backend implementation against the interface contract.

    Checks that:
    - ``generate`` is an async coroutine function
    - ``generate_stream`` is an async generator function
    - ``list_models`` returns a list of dicts with required keys
    - ``get_model`` returns dict or None
    - ``generate`` returns a dict with ``generated_text`` and ``usage``

    Args:
        backend: The backend instance to validate.

    Raises:
        BackendValidationError: If any validation check fails.
    """
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

    result = await backend.generate(
        [{"role": "user", "content": "validation probe"}]
    )
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
