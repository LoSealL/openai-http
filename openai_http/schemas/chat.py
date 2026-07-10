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

Chat completion schema definitions.

Compatible with OpenAI v1 /v1/chat/completions endpoint.
"""

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from .common import UsageInfo

FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class ChatMessage(BaseModel):
    """A single chat message.

    Attributes:
        role: The message role (system, user, assistant, tool).
        content: The message content.
        name: An optional participant name.
        tool_calls: Tool calls made by the assistant, if any.
        tool_call_id: The ID of a tool call this message responds to.
    """

    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Any] = None
    name: Optional[str] = None
    tool_calls: Optional[list["ToolCall"]] = None
    tool_call_id: Optional[str] = None


class FunctionCall(BaseModel):
    """Function call details.

    Attributes:
        name: The function name.
        arguments: The function arguments as a JSON string.
    """

    name: str
    arguments: str


class ToolCall(BaseModel):
    """A tool/function call made by the model.

    Attributes:
        id: The tool call ID.
        type: The tool type, always "function".
        function: The function call details.
    """

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    """Tool function definition.

    Attributes:
        name: The function name.
        description: An optional description.
        parameters: An optional JSON Schema of parameters.
        strict: Whether to use strict schema adherence.
    """

    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    strict: Optional[bool] = None


class Tool(BaseModel):
    """Tool definition for function calling.

    Attributes:
        type: The tool type, always "function".
        function: The function definition.
    """

    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI v1 ChatCompletion request.

    Attributes:
        model: The model ID to use.
        messages: List of chat messages (minimum 1).
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling probability mass.
        n: Number of completions to generate (1-128).
        stream: Whether to stream the response.
        stop: Stop sequence(s).
        max_tokens: Maximum tokens to generate.
        presence_penalty: Penalty for token presence (-2 to 2).
        frequency_penalty: Penalty for token frequency (-2 to 2).
        logit_bias: Token bias dictionary.
        logprobs: Whether to return log probabilities.
        top_logprobs: Number of top logprobs to return (0-20).
        response_format: Structured output format specification.
        seed: A seed for deterministic sampling.
        tools: Available tool definitions.
        tool_choice: Tool selection strategy ("none", "auto", etc.).
        user: An optional end-user identifier.
        stream_options: Options for streaming responses.
    """

    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=None, ge=1, le=128)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, list[str]]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    logit_bias: Optional[dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(None, ge=0, le=20)
    response_format: Optional[dict[str, Any]] = None
    seed: Optional[int] = None
    tools: Optional[list[Tool]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None
    user: Optional[str] = None
    stream_options: Optional[dict[str, Any]] = None


class ChoiceMessage(BaseModel):
    """Response message.

    Attributes:
        role: The message role, always "assistant".
        content: The response content.
        reasoning_content: The reasoning/thinking content, if any.
        tool_calls: Tool calls made by the model, if any.
    """

    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    """A single completion choice.

    Attributes:
        index: The choice index.
        message: The response message.
        logprobs: Optional log probability data.
        finish_reason: The reason generation stopped.
    """

    index: int
    message: ChoiceMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[FinishReason] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI v1 ChatCompletion response.

    Attributes:
        id: The completion ID.
        object: The object type, always "chat.completion".
        created: Unix timestamp of creation.
        model: The model used.
        choices: List of Choice objects.
        usage: Token usage information.
        system_fingerprint: System fingerprint string.
        service_tier: The service tier used.
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None


class DeltaMessage(BaseModel):
    """Streaming delta message.

    Attributes:
        role: The message role (only "assistant" for deltas).
        content: The content delta.
        reasoning_content: The reasoning/thinking content delta.
        tool_calls: Tool call deltas, if any.
    """

    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChunkChoice(BaseModel):
    """A single streaming chunk choice.

    Attributes:
        index: The choice index.
        delta: The message delta.
        logprobs: Optional log probability data.
        finish_reason: The reason generation stopped.
    """

    index: int
    delta: DeltaMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[FinishReason] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI v1 ChatCompletion streaming chunk.

    Attributes:
        id: The chunk ID.
        object: The object type, always "chat.completion.chunk".
        created: Unix timestamp of creation.
        model: The model used.
        choices: List of ChunkChoice objects.
        system_fingerprint: System fingerprint string.
        usage: Optional token usage information.
    """

    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[UsageInfo] = None
