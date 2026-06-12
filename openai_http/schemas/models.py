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

Model schema definitions.
"""

from typing import Literal
from pydantic import BaseModel


class Model(BaseModel):
    """OpenAI Model object.

    Attributes:
        id: The model identifier.
        object: The object type, always "model".
        created: Unix timestamp of model creation.
        owned_by: The organization that owns the model.
    """

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI model list response.

    Attributes:
        object: The object type, always "list".
        data: List of Model objects.
    """

    object: Literal["list"] = "list"
    data: list[Model]
