from fastapi import APIRouter

from openai_http.errors import NotImplementedOpenAIError

router = APIRouter(tags=["Images"])


@router.post("/v1/images/generations")
async def generate_images() -> None:
    raise NotImplementedOpenAIError("Image generation is not supported by this backend")


@router.post("/v1/images/edits")
async def edit_images() -> None:
    raise NotImplementedOpenAIError("Image editing is not supported by this backend")


@router.post("/v1/images/variations")
async def create_image_variations() -> None:
    raise NotImplementedOpenAIError("Image variations are not supported by this backend")
