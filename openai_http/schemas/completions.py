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

Text completions (legacy) API schema definitions.

Compatible with OpenAI v1 /v1/completions endpoint.
"""

from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from openai_http.schemas.common import UsageInfo


class CompletionRequest(BaseModel):
    """OpenAI v1 Completion request.

    Attributes:
        model: The model ID to use.
        prompt: The prompt string(s) to complete.
        suffix: A suffix to append after the completion.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0-2).
        top_p: Nucleus sampling probability mass.
        n: Number of completions to generate.
        stream: Whether to stream the response.
        logprobs: Include log probabilities for top K tokens.
        echo: Echo back the prompt in the response.
        stop: Stop sequence(s).
        presence_penalty: Penalty for token presence (-2 to 2).
        frequency_penalty: Penalty for token frequency (-2 to 2).
        best_of: Generates this many completions server-side and
            returns the best.
        logit_bias: Token bias dictionary.
        user: An optional end-user identifier.
    """

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
    """A single completion text choice.

    Attributes:
        text: The generated text.
        index: The choice index.
        logprobs: Optional log probability data.
        finish_reason: The reason generation stopped.
    """

    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    """OpenAI v1 Completion response.

    Attributes:
        id: The completion ID.
        object: The object type, always "text_completion".
        created: Unix timestamp of creation.
        model: The model used.
        choices: List of TextChoice objects.
        usage: Token usage information.
        system_fingerprint: System fingerprint string.
    """

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[TextChoice]
    usage: UsageInfo
    system_fingerprint: Optional[str] = None


class CompletionChunkChoice(BaseModel):
    """A single streaming completion chunk choice.

    Attributes:
        text: The text delta.
        index: The choice index.
        logprobs: Optional log probability data.
        finish_reason: The reason generation stopped.
    """

    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionChunk(BaseModel):
    """OpenAI v1 Completion streaming chunk.

    Attributes:
        id: The completion ID.
        object: The object type, always "text_completion".
        created: Unix timestamp of creation.
        model: The model used.
        choices: List of CompletionChunkChoice objects.
        system_fingerprint: System fingerprint string.
    """

    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[CompletionChunkChoice]
    system_fingerprint: Optional[str] = None
