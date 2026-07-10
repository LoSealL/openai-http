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

Chat Completions endpoint.

POST /v1/chat/completions - chat completion (streaming + non-streaming)
"""

import json
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import verify_api_key
from ..backends.contract import (
    validate_generation,
    validate_stream_chunk,
)
from ..backends.types import (
    BackendToolCall,
    ContentChunk,
    FinishChunk,
    GenerationUsage,
    ReasoningChunk,
)
from ..errors import (
    InvalidRequestError,
    NotFoundError,
)
from ..schemas.chat import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ChunkChoice,
    DeltaMessage,
    FinishReason,
    FunctionCall,
    ToolCall,
)
from ..schemas.common import UsageInfo

router = APIRouter(tags=["Chat"], dependencies=[Depends(verify_api_key)])


def _to_response_tool_calls(calls: list[BackendToolCall]) -> list[ToolCall]:
    """Convert backend tool calls to wire ``ToolCall`` objects."""
    return [
        ToolCall(
            id=tc.id,
            type=tc.type,
            function=FunctionCall(
                name=tc.function.name,
                arguments=tc.function.arguments,
            ),
        )
        for tc in calls
    ]


def _usage_to_wire(usage: GenerationUsage) -> UsageInfo:
    """Convert a backend ``GenerationUsage`` to the wire ``UsageInfo``."""
    return UsageInfo(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
    """Create a chat completion.

    Supports both streaming and non-streaming modes, as well as tool calls.

    Args:
        body: The chat completion request parameters.
        request: The HTTP request object.

    Returns:
        JSONResponse or StreamingResponse: The chat completion result.

    Raises:
        NotFoundError: If the requested model does not exist.
        InvalidRequestError: If max_tokens is invalid.
        NotImplementedOpenAIError: If tool calls are requested but not supported.
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

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    messages = [
        m.model_dump(exclude_none=True) | {"content": m.content or ""}
        for m in body.messages
    ]

    gen_kwargs = body.model_dump(
        exclude_none=True,
        exclude={"messages", "stream", "model", "tools", "tool_choice"},
    )
    if body.tools and body.tool_choice != "none":
        gen_kwargs["tools"] = [t.model_dump() for t in body.tools]
    if body.tool_choice is not None:
        gen_kwargs["tool_choice"] = body.tool_choice

    if body.stream:
        return await _chat_stream(
            body, request, messages, gen_kwargs, request_id, created
        )
    return await _chat_non_stream(
        body, request, messages, gen_kwargs, request_id, created
    )


async def _chat_stream(
    body: ChatCompletionRequest,
    request: Request,
    messages: list[dict],
    gen_kwargs: dict[str, Any],
    request_id: str,
    created: int,
) -> StreamingResponse:
    backend = request.app.state.backend
    queue = request.app.state.queue

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            async with queue.acquire():
                yield _make_chunk(
                    request_id,
                    body.model,
                    created,
                    delta=DeltaMessage(role="assistant"),
                )

                stream_finish_reason: FinishReason = "stop"
                async for token in backend.generate_stream(messages, **gen_kwargs):
                    chunk = validate_stream_chunk(token)
                    if isinstance(chunk, FinishChunk):
                        stream_finish_reason = chunk.reason
                        continue
                    if isinstance(chunk, ReasoningChunk):
                        delta = DeltaMessage(reasoning_content=chunk.content)
                    elif isinstance(chunk, ContentChunk):
                        delta = DeltaMessage(content=chunk.content)
                    else:
                        delta = DeltaMessage(content=chunk)
                    yield _make_chunk(
                        request_id,
                        body.model,
                        created,
                        delta=delta,
                    )

                final_usage: UsageInfo | None = None
                if body.stream_options and body.stream_options.get("include_usage"):
                    final_usage = UsageInfo(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    )
                yield _make_chunk(
                    request_id,
                    body.model,
                    created,
                    delta=DeltaMessage(),
                    finish_reason=stream_finish_reason,
                    usage=final_usage,
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


async def _chat_non_stream(
    body: ChatCompletionRequest,
    request: Request,
    messages: list[dict],
    gen_kwargs: dict[str, Any],
    request_id: str,
    created: int,
) -> JSONResponse:
    backend = request.app.state.backend
    queue = request.app.state.queue

    async with queue.acquire():
        raw_result = await backend.generate(messages, **gen_kwargs)
        result = validate_generation(raw_result)

    message = ChoiceMessage(
        content=result.generated_text,
        reasoning_content=result.reasoning_content,
        tool_calls=(
            _to_response_tool_calls(result.tool_calls) if result.tool_calls else None
        ),
    )

    response = ChatCompletionResponse(
        id=request_id,
        created=created,
        model=body.model,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason=result.finish_reason,
            )
        ],
        usage=_usage_to_wire(result.usage),
        system_fingerprint="fp_default",
    )

    return JSONResponse(content=response.model_dump(exclude_none=True))


def _make_chunk(
    request_id: str,
    model: str,
    created: int,
    delta: DeltaMessage,
    finish_reason: FinishReason | None = None,
    usage: UsageInfo | None = None,
) -> str:
    """Build an SSE-formatted chat completion chunk.

    Args:
        request_id: The unique request identifier.
        model: The model name.
        created: Unix timestamp of creation time.
        delta: The message delta as a ``DeltaMessage``.
        finish_reason: Optional reason for finishing (e.g. "stop", "tool_calls").
        usage: Optional usage statistics for the final chunk.

    Returns:
        str: An SSE data message containing the chunk JSON.
    """
    chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        choices=[
            ChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"
