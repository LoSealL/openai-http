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

Router-side helpers that coerce backend output into the typed contract.

Backends are documented to return dicts (or Pydantic instances) matching
:mod:`openai_http.backends.types`. These helpers validate that contract
at the router boundary and surface contract violations as clean
``server_error`` 500 responses instead of leaking raw ``KeyError`` /
``TypeError`` to the client.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from openai_http.backends.types import (
    BackendToolCall,
    ContentChunk,
    FinishChunk,
    GenerationResult,
    ModelInfo,
    ReasoningChunk,
    StreamChunk,
)
from openai_http.errors import OpenAIError

logger = logging.getLogger(__name__)


def _format_validation_error(exc: ValidationError) -> str:
    """Render a ``ValidationError`` as a single-line summary."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", []))
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts)


def _contract_error(stage: str, exc: ValidationError) -> OpenAIError:
    """Build an ``OpenAIError`` describing a backend contract violation.

    Args:
        stage: A short label identifying which call produced the bad
            output, e.g. ``"generate"`` or ``"stream chunk"``.
        exc: The underlying validation error.

    Returns:
        An ``OpenAIError`` with HTTP 500 and ``server_error`` type.
    """
    detail = _format_validation_error(exc)
    logger.error("Backend contract violation in %s: %s", stage, detail)
    return OpenAIError(
        message=(
            f"Backend returned a malformed {stage} payload "
            f"(contract violation): {detail}"
        ),
        error_type="server_error",
        code="backend_contract_error",
        status_code=500,
    )


def validate_generation(raw: Any) -> GenerationResult:
    """Coerce a backend ``generate`` return value into ``GenerationResult``.

    Accepts either a dict matching the schema or an existing
    ``GenerationResult`` instance.

    Args:
        raw: The backend return value.

    Returns:
        A validated ``GenerationResult``.

    Raises:
        OpenAIError: If validation fails, surfaced as HTTP 500 with
            ``server_error`` type.
    """
    if isinstance(raw, GenerationResult):
        return raw
    try:
        return GenerationResult.model_validate(raw)
    except ValidationError as exc:
        raise _contract_error("generate", exc) from exc


def validate_stream_chunk(raw: Any) -> StreamChunk | str:
    """Coerce a streaming chunk into ``StreamChunk`` or pass plain str through.

    Backends are allowed to yield either a plain ``str`` (treated as
    content) or a typed dict. ``None`` and unknown types raise.

    Args:
        raw: The yielded chunk.

    Returns:
        Either the original ``str`` or a validated ``StreamChunk``.

    Raises:
        OpenAIError: If the dict cannot be validated.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, (ReasoningChunk, ContentChunk, FinishChunk)):
        return raw
    if isinstance(raw, dict):
        try:
            return _validate_chunk_dict(raw)
        except ValidationError as exc:
            raise _contract_error("stream chunk", exc) from exc
    raise OpenAIError(
        message=(
            f"Backend yielded unsupported chunk type {type(raw).__name__}; "
            "expected str or dict"
        ),
        error_type="server_error",
        code="backend_contract_error",
        status_code=500,
    )


def _validate_chunk_dict(raw: dict) -> StreamChunk:
    """Validate a stream chunk dict against the discriminator union."""
    chunk_type = raw.get("type")
    if chunk_type == "reasoning":
        return ReasoningChunk.model_validate(raw)
    if chunk_type == "content":
        return ContentChunk.model_validate(raw)
    if chunk_type == "finish":
        return FinishChunk.model_validate(raw)
    return ContentChunk.model_validate(raw)


def validate_model_info(raw: Any) -> ModelInfo:
    """Coerce a backend model entry into ``ModelInfo``."""
    if isinstance(raw, ModelInfo):
        return raw
    try:
        return ModelInfo.model_validate(raw)
    except ValidationError as exc:
        raise _contract_error("model info", exc) from exc


def validate_model_list(raw: Any) -> list[ModelInfo]:
    """Coerce a backend ``list_models`` return value into a list of ``ModelInfo``."""
    if not isinstance(raw, list):
        raise OpenAIError(
            message=(
                f"Backend list_models must return a list, "
                f"got {type(raw).__name__}"
            ),
            error_type="server_error",
            code="backend_contract_error",
            status_code=500,
        )
    return [validate_model_info(m) for m in raw]


def validate_tool_calls(raw: Any) -> list[BackendToolCall]:
    """Coerce a backend ``generate_tool_calls`` return value to typed calls."""
    if not isinstance(raw, list):
        raise OpenAIError(
            message=(
                f"Backend generate_tool_calls must return a list, "
                f"got {type(raw).__name__}"
            ),
            error_type="server_error",
            code="backend_contract_error",
            status_code=500,
        )
    out: list[BackendToolCall] = []
    for item in raw:
        if isinstance(item, BackendToolCall):
            out.append(item)
            continue
        try:
            out.append(BackendToolCall.model_validate(item))
        except ValidationError as exc:
            raise _contract_error("tool call", exc) from exc
    return out
