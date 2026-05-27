"""
Model schema definitions.
"""

from typing import Literal
from pydantic import BaseModel


class Model(BaseModel):
    """OpenAI Model object."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelListResponse(BaseModel):
    """OpenAI model list response."""
    object: Literal["list"] = "list"
    data: list[Model]
