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

Embeddings endpoint.

POST /v1/embeddings - generate vector embeddings
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from openai_http.auth import verify_api_key
from openai_http.errors import (
    InvalidRequestError,
    NotFoundError,
    NotImplementedOpenAIError,
)
from openai_http.schemas.common import UsageInfo
from openai_http.schemas.embeddings import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    floats_to_base64,
)


router = APIRouter(tags=["Embeddings"], dependencies=[Depends(verify_api_key)])


@router.post("/v1/embeddings", response_model=None)
async def create_embeddings(
    body: EmbeddingRequest,
    request: Request,
):
    """Create vector embeddings for the given input text.

    Args:
        body: The embedding request containing input text, model, and encoding options.
        request: The HTTP request object.

    Returns:
        JSONResponse: A list of embedding vectors with usage information.

    Raises:
        NotFoundError: If the requested model does not exist.
        InvalidRequestError: If the input is empty or contains invalid types.
        NotImplementedOpenAIError: If the backend does not support embeddings.
    """
    backend = request.app.state.backend

    model_info = await backend.get_model(body.model)
    if model_info is None:
        raise NotFoundError(message=f"The model '{body.model}' does not exist")

    texts: list[str] = []
    raw_inputs = body.input if isinstance(body.input, list) else [body.input]
    for item in raw_inputs:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, list):
            raise InvalidRequestError(
                message="Tokenized input (int arrays) is not supported in mock mode",
                param="input",
            )
        else:
            texts.append(str(item))

    if not texts or all(t.strip() == "" for t in texts):
        raise InvalidRequestError(
            message="Input must be a non-empty string or array of strings",
            param="input",
        )

    kwargs = {}
    if body.dimensions is not None:
        kwargs["dimensions"] = body.dimensions

    try:
        embeddings = await backend.embed(texts, **kwargs)
    except NotImplementedError:
        raise NotImplementedOpenAIError("Embeddings are not supported by this backend")

    data: list[EmbeddingObject] = []
    for i, vec in enumerate(embeddings):
        if body.encoding_format == "base64":
            data.append(EmbeddingObject(index=i, embedding=floats_to_base64(vec)))
        else:
            data.append(EmbeddingObject(index=i, embedding=vec))

    total_tokens = sum(max(1, int(len(t) * 0.25)) for t in texts)

    response = EmbeddingResponse(
        data=data,
        model=body.model,
        usage=UsageInfo(
            prompt_tokens=total_tokens,
            completion_tokens=0,
            total_tokens=total_tokens,
        ),
    )

    return JSONResponse(content=response.model_dump(exclude_none=True))
