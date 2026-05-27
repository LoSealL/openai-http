"""
Chat Completions endpoint.

POST /v1/chat/completions - chat completion (streaming + non-streaming)
"""

import json
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from openai_http.auth import verify_api_key
from openai_http.schemas.chat import (
    ChatCompletionRequest,
)
from openai_http.errors import NotFoundError, InvalidRequestError, NotImplementedOpenAIError


router = APIRouter(tags=["Chat"], dependencies=[Depends(verify_api_key)])


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
    messages = [
        _serialize_message(m) for m in body.messages
    ]

    kwargs: dict[str, Any] = {}
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.top_p is not None:
        kwargs["top_p"] = body.top_p

    use_tools = (
        body.tools is not None
        and len(body.tools) > 0
        and body.tool_choice != "none"
    )
    tool_calls_result: list[dict[str, Any]] | None = None
    tool_finish_reason: str | None = None

    if use_tools and body.tools is not None:
        tc_kwargs: dict[str, Any] = {}
        if body.tool_choice is not None:
            tc_kwargs["tool_choice"] = body.tool_choice
        if hasattr(backend, "generate_tool_calls"):
            try:
                tool_calls_result = await backend.generate_tool_calls(
                    messages,
                    [t.model_dump() for t in body.tools],
                    **tc_kwargs,
                )
            except NotImplementedError:
                raise NotImplementedOpenAIError("Tool calls are not supported by this backend")

        if tool_calls_result:
            tool_finish_reason = "tool_calls"
            kwargs["tools"] = [t.model_dump() for t in body.tools]
            if body.tool_choice is not None:
                kwargs["tool_choice"] = body.tool_choice

    if body.stream:

        async def stream_generator() -> AsyncGenerator[str, None]:
            try:
                async with queue.acquire():
                    first_chunk = _make_chunk(
                        request_id, body.model, created,
                        delta={"role": "assistant", "content": None},
                    )
                    yield first_chunk

                    if tool_finish_reason == "tool_calls" and tool_calls_result:
                        delta_tool_calls = [
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["function"]["name"],
                                    "arguments": tc["function"]["arguments"],
                                },
                            }
                            for i, tc in enumerate(tool_calls_result)
                        ]
                        chunk = _make_chunk(
                            request_id, body.model, created,
                            delta={"tool_calls": delta_tool_calls},
                            finish_reason="tool_calls",
                        )
                        yield chunk
                    else:
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
        if tool_finish_reason == "tool_calls" and tool_calls_result:
            prompt_tok = sum(
                max(1, int(len(str(m.get("content", ""))) * 0.25))
                for m in messages
            )
            comp_tok = sum(
                max(1, int(len(tc.get("function", {}).get("arguments", "")) * 0.25))
                for tc in tool_calls_result
            )
            result: dict[str, Any] = {
                "generated_text": None,
                "tool_calls": tool_calls_result,
                "usage": {
                    "prompt_tokens": prompt_tok,
                    "completion_tokens": comp_tok,
                    "total_tokens": prompt_tok + comp_tok,
                },
            }
        else:
            result = await backend.generate(messages, **kwargs)

    message = {"role": "assistant", "content": result.get("generated_text")}
    if result.get("tool_calls"):
        message["tool_calls"] = result["tool_calls"]

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": body.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": tool_finish_reason or "stop",
            }
        ],
        "usage": result["usage"],
        "system_fingerprint": "fp_default",
    }

    return JSONResponse(content=response)


def _serialize_message(m) -> dict:
    msg = {"role": m.role, "content": m.content or ""}
    if m.tool_calls:
        msg["tool_calls"] = [tc.model_dump() for tc in m.tool_calls]
    if m.tool_call_id:
        msg["tool_call_id"] = m.tool_call_id
    return msg


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
