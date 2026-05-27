"""
Chat Completions endpoint.

POST /v1/chat/completions - chat completion (streaming + non-streaming)
"""

import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from openai_http.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ChatCompletionChunk,
    ChunkChoice,
    DeltaMessage,
)
from openai_http.schemas.common import UsageInfo
from openai_http.errors import NotFoundError, InvalidRequestError, OpenAIError


router = APIRouter(tags=["Chat"])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
):
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

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    messages = [{"role": m.role, "content": m.content or ""} for m in body.messages]

    kwargs = {}
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p

    if body.stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                async with queue.acquire():
                    first_chunk = _make_chunk(
                        request_id, body.model, created,
                        delta={"role": "assistant", "content": None},
                    )
                    yield first_chunk

                    async for token in backend.generate_stream(messages, **kwargs):
                        chunk = _make_chunk(
                            request_id, body.model, created,
                            delta={"content": token},
                        )
                        yield chunk

                    final_usage = None
                    if body.stream_options and body.stream_options.get("include_usage"):
                        final_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    final = _make_chunk(
                        request_id, body.model, created,
                        delta={},
                        finish_reason="stop",
                        usage=final_usage,
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
        result = await backend.generate(messages, **kwargs)

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["generated_text"],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": result["usage"],
        "system_fingerprint": "fp_default",
    }

    return JSONResponse(content=response)


def _make_chunk(
    request_id: str,
    model: str,
    created: int,
    delta: dict,
    finish_reason: str | None = None,
    usage: dict | None = None,
) -> str:
    chunk = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return f"data: {json.dumps(chunk)}\n\n"
