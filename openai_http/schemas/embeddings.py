"""
Embeddings API schema definitions.

Compatible with OpenAI v1 /v1/embeddings endpoint.
"""

import base64
import struct
from typing import Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator

from openai_http.schemas.common import UsageInfo


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: Union[str, list[Union[str, list[int]]]]
    model: str
    encoding_format: Optional[Literal["float", "base64"]] = Field(default="float")
    dimensions: Optional[int] = Field(default=None, ge=1)
    user: Optional[str] = None

    @field_validator("input", mode="before")
    @classmethod
    def normalize_input(cls, v):
        if isinstance(v, str):
            return [v]
        return v


class EmbeddingObject(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: Union[list[float], str]


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingObject]
    model: str
    usage: UsageInfo


def floats_to_base64(vec: list[float]) -> str:
    raw = struct.pack(f"{len(vec)}f", *vec)
    return base64.b64encode(raw).decode("ascii")
