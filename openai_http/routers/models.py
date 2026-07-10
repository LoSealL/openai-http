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

Models API endpoint.

GET /v1/models         - list all available models
GET /v1/models/{id}    - retrieve a specific model
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..auth import verify_api_key
from ..backends.contract import _contract_error
from ..errors import NotFoundError
from ..schemas.models import Model, ModelListResponse


router = APIRouter(tags=["Models"], dependencies=[Depends(verify_api_key)])


@router.get("/v1/models", response_model=None)
async def list_models(request: Request) -> JSONResponse:
    """List all available models.

    Returns:
        JSONResponse: A list of model objects.
    """
    backend = request.app.state.backend
    raw_models = await backend.list_models()
    try:
        response = ModelListResponse(
            data=[Model.model_validate(m) for m in raw_models],
        )
    except ValidationError as exc:
        raise _contract_error("model list", exc) from exc
    return JSONResponse(content=response.model_dump(exclude_none=True))


@router.get("/v1/models/{model_id}", response_model=None)
async def retrieve_model(model_id: str, request: Request) -> JSONResponse:
    """Retrieve a specific model by ID.

    Args:
        model_id: The model identifier.

    Returns:
        JSONResponse: The model object.

    Raises:
        NotFoundError: If the model does not exist.
    """
    backend = request.app.state.backend
    raw = await backend.get_model(model_id)
    if raw is None:
        raise NotFoundError(message=f"The model '{model_id}' does not exist")
    return JSONResponse(content=Model.model_validate(raw).model_dump(exclude_none=True))
