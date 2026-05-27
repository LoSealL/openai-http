"""
Images API endpoints (stub).

POST /v1/images/generations  - generate images (not implemented)
POST /v1/images/edits        - edit images (not implemented)
POST /v1/images/variations   - create image variations (not implemented)
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["Images"])


def _not_implemented(message: str) -> JSONResponse:
    return JSONResponse(
        status_code=501,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": "not_implemented",
            }
        },
    )


@router.post("/v1/images/generations")
async def generate_images(request: Request) -> JSONResponse:
    return _not_implemented("Image generation is not implemented. No image backend configured.")


@router.post("/v1/images/edits")
async def edit_images(request: Request) -> JSONResponse:
    return _not_implemented("Image editing is not implemented. No image backend configured.")


@router.post("/v1/images/variations")
async def create_image_variations(request: Request) -> JSONResponse:
    return _not_implemented("Image variations are not implemented. No image backend configured.")
