"""
Text completions (legacy) API schema definitions.

Compatible with OpenAI v1 /v1/completions endpoint.
"""

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from openai_http.schemas.common import UsageInfo


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: Union[str, list[str], list[int], list[list[int]]] = ""
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=None, ge=1, le=128)
    stream: bool = Field(default=False)
    logprobs: Optional[int] = Field(default=None, ge=0, le=5)
    echo: bool = Field(default=False)
    stop: Optional[Union[str, list[str]]] = None
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(default=None, ge=1, le=20)
    logit_bias: Optional[dict[str, float]] = None
    user: Optional[str] = None


class TextChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[TextChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None


class CompletionChunkChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]
    system_fingerprint: Optional[str] = None
