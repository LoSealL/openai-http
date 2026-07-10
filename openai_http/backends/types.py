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

Backend-facing typed contracts.

These Pydantic models are the canonical structure routers expect from
backend implementations. Backends may return ``dict`` instances with
the same shape; routers validate them at the boundary.

These types are deliberately kept separate from the HTTP wire schemas
in ``openai_http.schemas`` so that backends do not have to know the
OpenAI response format.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field

FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class GenerationUsage(BaseModel):
    """Token usage for a single generation call.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens generated.
        total_tokens: Sum of prompt and completion tokens.
    """

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class BackendToolCallFunction(BaseModel):
    """Function payload of a backend-emitted tool call.

    Attributes:
        name: The function name.
        arguments: A JSON string with the function arguments.
    """

    name: str
    arguments: str


class BackendToolCall(BaseModel):
    """A tool call emitted by the backend.

    Attributes:
        id: The tool call identifier.
        type: Always ``"function"``.
        function: The function name and JSON-encoded arguments.
    """

    id: str
    type: Literal["function"] = "function"
    function: BackendToolCallFunction


class GenerationResult(BaseModel):
    """Result returned by ``BackendBase.generate``.

    Attributes:
        generated_text: The model's plain answer text. May be ``None``
            when ``tool_calls`` is set.
        reasoning_content: Optional reasoning/thinking content extracted
            from ``<think>`` tags or equivalent.
        tool_calls: Optional list of backend-emitted tool calls.
        finish_reason: Why generation stopped.
        usage: Token usage statistics.
    """

    generated_text: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[BackendToolCall]] = None
    finish_reason: FinishReason = "stop"
    usage: GenerationUsage


class ReasoningChunk(BaseModel):
    """Streaming chunk of reasoning/thinking content.

    Attributes:
        type: Always ``"reasoning"``.
        content: The reasoning text fragment.
    """

    type: Literal["reasoning"]
    content: str


class ContentChunk(BaseModel):
    """Streaming chunk of normal content.

    Attributes:
        type: Always ``"content"``.
        content: The content text fragment.
    """

    type: Literal["content"]
    content: str


class FinishChunk(BaseModel):
    """Terminal streaming chunk indicating why generation ended.

    Attributes:
        type: Always ``"finish"``.
        reason: The finish reason.
    """

    type: Literal["finish"]
    reason: FinishReason


StreamChunk = Annotated[
    Union[ReasoningChunk, ContentChunk, FinishChunk],
    Field(discriminator="type"),
]
"""Discriminated union of the dict shapes a backend may yield.

Backends are also permitted to yield plain ``str`` instances, which
routers treat as content chunks. This type only describes the dict
form.
"""


class ModelInfo(BaseModel):
    """A model entry returned by ``list_models``/``get_model``.

    Attributes:
        id: The model identifier.
        object: Always ``"model"``.
        created: Unix timestamp of model creation.
        owned_by: The organization that owns the model.
    """

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
