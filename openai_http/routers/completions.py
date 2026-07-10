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

Text Completions (legacy) endpoint.

POST /v1/completions - text completion (streaming + non-streaming)
"""

import json
import time
import uuid
from typing import Any, AsyncGenerator, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import verify_api_key
from ..backends.contract import (
    validate_generation,
    validate_stream_chunk,
)
from ..backends.types import ContentChunk, FinishChunk, ReasoningChunk
from ..errors import InvalidRequestError, NotFoundError
from ..schemas.common import UsageInfo
from ..schemas.completions import (
    CompletionChunk,
    CompletionChunkChoice,
    CompletionRequest,
    CompletionResponse,
    TextChoice,
)

router = APIRouter(tags=["Completions"], dependencies=[Depends(verify_api_key)])


# Legacy completions only allow "stop"/"length" as finish reason.
_LegacyFinish = Literal["stop", "length"]


def _coerce_legacy_finish(reason: str) -> _LegacyFinish:
    """Coerce a backend finish reason into the legacy completion subset.

    Tool calls / content filters do not exist on the legacy endpoint;
    fall back to ``"stop"`` rather than producing an invalid response.

    Args:
        reason: The backend finish reason.

    Returns:
        Either ``"stop"`` or ``"length"``.
    """
    if reason == "length":
        return "length"
    return "stop"


@router.post("/v1/completions", response_model=None)
async def create_completion(
    body: CompletionRequest,
    request: Request,
):
    """Create a text completion (legacy endpoint).

    Supports both streaming and non-streaming modes.

    Args:
        body: The completion request parameters.
        request: The HTTP request object.

    Returns:
        JSONResponse or StreamingResponse: The completion result.

    Raises:
        NotFoundError: If the requested model does not exist.
        InvalidRequestError: If max_tokens is invalid.
    """
    backend = request.app.state.backend

    model_info = await backend.get_model(body.model)
    if model_info is None:
        raise NotFoundError(message=f"The model '{body.model}' does not exist")

    if body.max_tokens is not None and body.max_tokens <= 0:
        raise InvalidRequestError(
            message="max_tokens must be greater than 0",
            param="max_tokens",
        )

    n = body.n or 1
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    prompt_str = _normalize_prompt(body.prompt)

    gen_kwargs = body.model_dump(
        exclude_none=True,
        exclude={
            "prompt",
            "stream",
            "model",
            "suffix",
            "logprobs",
            "echo",
            "stop",
            "best_of",
            "logit_bias",
            "user",
            "n",
            "presence_penalty",
            "frequency_penalty",
        },
    )

    if body.stream:
        return await _completion_stream(
            body, request, prompt_str, gen_kwargs, n, request_id, created
        )
    return await _completion_non_stream(
        body, request, prompt_str, gen_kwargs, n, request_id, created
    )


async def _completion_stream(
    body: CompletionRequest,
    request: Request,
    prompt_str: str,
    gen_kwargs: dict[str, Any],
    n: int,
    request_id: str,
    created: int,
) -> StreamingResponse:
    backend = request.app.state.backend
    queue = request.app.state.queue

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            async with queue.acquire():
                for idx in range(n):
                    yield _make_chunk(
                        request_id,
                        body.model,
                        created,
                        idx,
                        text="",
                    )

                    stream_finish_reason: _LegacyFinish = "stop"
                    async for token in backend.generate_stream(
                        prompt_str, **gen_kwargs
                    ):
                        chunk = validate_stream_chunk(token)
                        if isinstance(chunk, FinishChunk):
                            stream_finish_reason = _coerce_legacy_finish(chunk.reason)
                            continue
                        if isinstance(chunk, ReasoningChunk):
                            continue
                        if isinstance(chunk, ContentChunk):
                            chunk_text = chunk.content
                        else:
                            chunk_text = chunk
                        yield _make_chunk(
                            request_id,
                            body.model,
                            created,
                            idx,
                            text=chunk_text,
                        )

                    yield _make_chunk(
                        request_id,
                        body.model,
                        created,
                        idx,
                        text="",
                        finish_reason=stream_finish_reason,
                    )

                yield "data: [DONE]\n\n"

        except Exception:  # pylint: disable=broad-exception-caught
            error_msg = json.dumps(
                {
                    "error": {
                        "message": "Internal server error. Please try again later.",
                        "type": "server_error",
                        "param": None,
                        "code": "generation_error",
                    }
                }
            )
            yield f"data: {error_msg}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _completion_non_stream(
    body: CompletionRequest,
    request: Request,
    prompt_str: str,
    gen_kwargs: dict[str, Any],
    n: int,
    request_id: str,
    created: int,
) -> JSONResponse:
    backend = request.app.state.backend
    queue = request.app.state.queue

    async with queue.acquire():
        choices: list[TextChoice] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for idx in range(n):
            raw_result = await backend.generate(prompt_str, **gen_kwargs)
            result = validate_generation(raw_result)
            generated = result.generated_text or ""

            if body.echo:
                generated = prompt_str + generated

            choices.append(
                TextChoice(
                    text=generated,
                    index=idx,
                    finish_reason=_coerce_legacy_finish(result.finish_reason),
                )
            )
            total_prompt_tokens += result.usage.prompt_tokens
            total_completion_tokens += result.usage.completion_tokens

    response = CompletionResponse(
        id=request_id,
        created=created,
        model=body.model,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
        ),
        system_fingerprint="fp_default",
    )

    return JSONResponse(content=response.model_dump(exclude_none=True))


def _normalize_prompt(prompt) -> str:
    """Normalize a prompt into a single string.

    Args:
        prompt: The raw prompt input.

    Returns:
        str: The normalized prompt string.

    Raises:
        InvalidRequestError: If prompt is a list of token arrays.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        if not prompt:
            return ""
        if isinstance(prompt[0], str):
            return "\n".join(prompt)
        if isinstance(prompt[0], list):
            raise InvalidRequestError(
                message="Tokenized input (list of int arrays) is not supported",
                param="prompt",
            )
        return str(prompt)
    return str(prompt)


def _make_chunk(
    request_id: str,
    model: str,
    created: int,
    index: int,
    text: str,
    finish_reason: _LegacyFinish | None = None,
) -> str:
    """Build an SSE-formatted completion chunk.

    Args:
        request_id: The unique request identifier.
        model: The model name.
        created: Unix timestamp of creation time.
        index: The choice index.
        text: The generated text for this chunk.
        finish_reason: Optional reason for finishing (e.g. "stop").

    Returns:
        str: An SSE data message containing the chunk JSON.
    """
    chunk = CompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            CompletionChunkChoice(
                text=text,
                index=index,
                finish_reason=finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
