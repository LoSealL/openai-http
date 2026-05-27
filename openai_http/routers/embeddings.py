"""
Embeddings endpoint.

POST /v1/embeddings - generate vector embeddings
"""


from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from openai_http.auth import verify_api_key
from openai_http.schemas.embeddings import (
    EmbeddingRequest,
    EmbeddingObject,
    floats_to_base64,
)
from openai_http.errors import NotFoundError, InvalidRequestError, NotImplementedOpenAIError


router = APIRouter(tags=["Embeddings"], dependencies=[Depends(verify_api_key)])


@router.post("/v1/embeddings")
async def create_embeddings(
    body: EmbeddingRequest,
    request: Request,
):
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

    data = []
    for i, vec in enumerate(embeddings):
        if body.encoding_format == "base64":
            data.append(EmbeddingObject(index=i, embedding=floats_to_base64(vec)))
        else:
            data.append(EmbeddingObject(index=i, embedding=vec))

    total_tokens = sum(
        max(1, int(len(t) * 0.25)) for t in texts
    )

    response = {
        "object": "list",
        "data": [d.model_dump() for d in data],
        "model": body.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "completion_tokens": 0,
            "total_tokens": total_tokens,
        },
    }

    return JSONResponse(content=response)
