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

Embeddings API schema definitions.

Compatible with OpenAI v1 /v1/embeddings endpoint.
"""

import base64
import struct
from typing import Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .common import UsageInfo


class EmbeddingRequest(BaseModel):
    """OpenAI v1 Embeddings request.

    Attributes:
        input: The input text(s) to embed. Accepts a string, a list of
            strings, or a list of token integer lists.
        model: The model ID to use for embedding.
        encoding_format: The format for the embedding vector, "float"
            or "base64".
        dimensions: The number of dimensions for the output embedding.
        user: An optional end-user identifier.
    """

    model_config = ConfigDict(extra="allow")

    input: Union[str, list[Union[str, list[int]]]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = Field(default="float")
    dimensions: Optional[int] = Field(default=None, ge=1)
    user: Optional[str] = None

    @field_validator("input", mode="before")
    @classmethod
    def _normalize(cls, v):
        if isinstance(v, str):
            return [v]
        if isinstance(v, list) and v and isinstance(v[0], int):
            raise ValueError("input must be a string or list of strings, not token IDs")
        return v


class EmbeddingObject(BaseModel):
    """A single embedding object.

    Attributes:
        object: The object type, always "embedding".
        index: The index of this embedding in the response.
        embedding: The embedding vector as a list of floats or a
            base64-encoded string.
    """

    object: Literal["embedding"] = "embedding"
    index: int
    embedding: Union[list[float], str]


class EmbeddingResponse(BaseModel):
    """OpenAI v1 Embeddings response.

    Attributes:
        object: The object type, always "list".
        data: List of EmbeddingObject results.
        model: The model used for embedding.
        usage: Token usage information.
    """

    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: UsageInfo


def floats_to_base64(vec: list[float]) -> str:
    """Encode a list of floats as a base64 string.

    Uses IEEE 754 32-bit little-endian packing.

    Args:
        vec: The list of float values to encode.

    Returns:
        A base64-encoded ASCII string.
    """
    raw = struct.pack(f"{len(vec)}f", *vec)
    return base64.b64encode(raw).decode("ascii")
