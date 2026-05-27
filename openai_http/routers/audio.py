"""
Audio API endpoints (stub).

POST /v1/audio/transcriptions - transcribe audio (not implemented)
POST /v1/audio/translations   - translate audio (not implemented)
POST /v1/audio/speech         - text-to-speech (not implemented)
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["Audio"])


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


@router.post("/v1/audio/transcriptions")
async def transcribe_audio(request: Request) -> JSONResponse:
    return _not_implemented("Audio transcription is not implemented. No audio backend configured.")


@router.post("/v1/audio/translations")
async def translate_audio(request: Request) -> JSONResponse:
    return _not_implemented("Audio translation is not implemented. No audio backend configured.")


@router.post("/v1/audio/speech")
async def create_speech(request: Request) -> JSONResponse:
    return _not_implemented("Text-to-speech is not implemented. No audio backend configured.")
