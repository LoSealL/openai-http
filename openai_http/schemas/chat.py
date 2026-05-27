"""
Chat completion schema definitions.

Compatible with OpenAI v1 /v1/chat/completions endpoint.
"""

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from openai_http.schemas.common import UsageInfo


class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[Any] = None
    name: Optional[str] = None
    tool_calls: Optional[list["ToolCall"]] = None
    tool_call_id: Optional[str] = None


class FunctionCall(BaseModel):
    """Function call details."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """A tool/function call made by the model."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class FunctionDefinition(BaseModel):
    """Tool function definition."""
    name: str
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    strict: Optional[bool] = None


class Tool(BaseModel):
    """Tool definition for function calling."""
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionRequest(BaseModel):
    """OpenAI v1 ChatCompletion request."""
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
    """Response message."""
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None


class Choice(BaseModel):
    """A single completion choice."""
    index: int
    message: ChoiceMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI v1 ChatCompletion response."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None
    service_tier: Optional[str] = None


class DeltaMessage(BaseModel):
    """Streaming delta message."""
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None


class ChunkChoice(BaseModel):
    """A single streaming chunk choice."""
    index: int
    delta: DeltaMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter"]] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI v1 ChatCompletion streaming chunk."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    system_fingerprint: Optional[str] = None
    usage: Optional[UsageInfo] = None
