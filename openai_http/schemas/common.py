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

Common schema types shared across all endpoints.

Provides UsageInfo for token counts and ErrorResponse for OpenAI-format errors.
"""

from typing import Optional, Literal
from pydantic import BaseModel


class UsageInfo(BaseModel):
    """Token usage information.

    Attributes:
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens generated.
        total_tokens: Total number of tokens (prompt + completion).
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_counts(cls, prompt: int, completion: int) -> "UsageInfo":
        """Create a UsageInfo from prompt and completion token counts.

        Args:
            prompt: Prompt token count.
            completion: Completion token count.

        Returns:
            A UsageInfo instance with computed total_tokens.
        """
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


class ErrorDetail(BaseModel):
    """OpenAI error detail format.

    Attributes:
        message: The error message.
        type: The error type string.
        param: The parameter that caused the error, if any.
        code: The error code, if any.
    """

    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI error response format.

    Attributes:
        error: An ErrorDetail object describing the error.
    """

    error: ErrorDetail


class ListResponse(BaseModel):
    """Generic list response.

    Attributes:
        object: The object type, always "list".
        data: The list of items.
    """

    object: Literal["list"] = "list"
    data: list


class DeletionResponse(BaseModel):
    """Deletion confirmation response.

    Attributes:
        id: The ID of the deleted resource.
        object: The object type, always "deleted".
        deleted: Whether the deletion was successful.
    """

    id: str
    object: str = "deleted"
    deleted: bool = True
