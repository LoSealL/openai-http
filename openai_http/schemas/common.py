"""
Common schema types shared across all endpoints.

Provides UsageInfo for token counts and ErrorResponse for OpenAI-format errors.
"""

from typing import Optional, Literal
from pydantic import BaseModel


class UsageInfo(BaseModel):
    """Token usage information."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_counts(cls, prompt: int, completion: int) -> "UsageInfo":
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


class ErrorDetail(BaseModel):
    """OpenAI error detail format."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI error response format."""
    error: ErrorDetail


class ListResponse(BaseModel):
    """Generic list response."""
    object: Literal["list"] = "list"
    data: list


class DeletionResponse(BaseModel):
    """Deletion confirmation response."""
    id: str
    object: str = "deleted"
    deleted: bool = True
