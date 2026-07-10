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

Provides UsageInfo for token counts.
"""

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
