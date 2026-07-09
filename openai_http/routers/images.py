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

Image generation endpoints.

POST /v1/images/generations  - generate images
POST /v1/images/edits        - edit images
POST /v1/images/variations   - create image variations
"""

from fastapi import APIRouter

from openai_http.errors import NotImplementedOpenAIError

router = APIRouter(tags=["Images"])


@router.post("/v1/images/generations")
async def generate_images() -> None:
    """Generate images from a text prompt.

    Raises:
        NotImplementedOpenAIError: Always, as this backend does not support image generation.
    """
    raise NotImplementedOpenAIError("Image generation is not supported by this backend")


@router.post("/v1/images/edits")
async def edit_images() -> None:
    """Edit an existing image based on a prompt.

    Raises:
        NotImplementedOpenAIError: Always, as this backend does not support image editing.
    """
    raise NotImplementedOpenAIError("Image editing is not supported by this backend")


@router.post("/v1/images/variations")
async def create_image_variations() -> None:
    """Create variations of an input image.

    Raises:
        NotImplementedOpenAIError: Always, as this backend does not support image variations.
    """
    raise NotImplementedOpenAIError(
        "Image variations are not supported by this backend"
    )
