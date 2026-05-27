"""
Models API endpoint.

GET /v1/models         - list all available models
GET /v1/models/{id}    - retrieve a specific model
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from openai_http.schemas.models import Model, ModelListResponse
from openai_http.errors import NotFoundError


router = APIRouter(tags=["Models"])


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    backend = request.app.state.backend
    models = await backend.list_models()
    return JSONResponse(
        content={"object": "list", "data": models},
    )


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, request: Request) -> JSONResponse:
    backend = request.app.state.backend
    model = await backend.get_model(model_id)
    if model is None:
        raise NotFoundError(message=f"The model '{model_id}' does not exist")
    return JSONResponse(content=model)
