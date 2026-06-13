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
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from openai_http.auth import verify_api_key
from openai_http.schemas.completions import CompletionRequest
from openai_http.errors import NotFoundError, InvalidRequestError


router = APIRouter(tags=["Completions"], dependencies=[Depends(verify_api_key)])


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
    queue = request.app.state.queue

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

    kwargs: dict[str, Any] = {}
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p

    if body.stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            """Generate streaming completion chunks.

            Yields:
                str: SSE-formatted completion chunks, ending with a [DONE] signal.
            """
            try:
                async with queue.acquire():
                    for idx in range(n):
                        first_chunk = _make_chunk(
                            request_id, body.model, created, idx, text="",
                        )
                        yield first_chunk

                        stream_finish_reason = "stop"
                        async for token in backend.generate_stream(prompt_str, **kwargs):
                            if isinstance(token, dict):
                                if token.get("type") == "finish":
                                    stream_finish_reason = token.get("reason", "stop")
                                    continue
                                chunk_text = token.get("content", "")
                            else:
                                chunk_text = token
                            chunk = _make_chunk(
                                request_id, body.model, created, idx, text=chunk_text,
                            )
                            yield chunk

                        final = _make_chunk(
                            request_id, body.model, created, idx,
                            text="", finish_reason=stream_finish_reason,
                        )
                        yield final

                    yield "data: [DONE]\n\n"

            except Exception as e:
                error_msg = json.dumps({
                    "error": {
                        "message": str(e),
                        "type": "server_error",
                        "param": None,
                        "code": "generation_error",
                    }
                })
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

    async with queue.acquire():
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for idx in range(n):
            result = await backend.generate(prompt_str, **kwargs)
            generated = result["generated_text"]

            if body.echo:
                generated = prompt_str + generated

            choices.append({
                "text": generated,
                "index": idx,
                "logprobs": None,
                "finish_reason": result.get("finish_reason", "stop"),
            })
            total_prompt_tokens += result["usage"]["prompt_tokens"]
            total_completion_tokens += result["usage"]["completion_tokens"]

    response = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": body.model,
        "choices": choices,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
        },
        "system_fingerprint": "fp_default",
    }

    return JSONResponse(content=response)


def _normalize_prompt(prompt) -> str:
    """Normalize a prompt into a single string.

    Handles string, list-of-strings, and list-of-token-array formats.

    Args:
        prompt: The raw prompt input.

    Returns:
        str: The normalized prompt string.
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        if not prompt:
            return ""
        if isinstance(prompt[0], str):
            return "\n".join(prompt)
        if isinstance(prompt[0], list):
            return ""
        return str(prompt)
    return str(prompt)


def _make_chunk(
    request_id: str,
    model: str,
    created: int,
    index: int,
    text: str,
    finish_reason: str | None = None,
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
    chunk = {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}\n\n"
