from fastapi import APIRouter

from openai_http.errors import NotImplementedOpenAIError

router = APIRouter(tags=["Audio"])


@router.post("/v1/audio/transcriptions")
async def transcribe_audio() -> None:
    raise NotImplementedOpenAIError("Audio transcription is not supported by this backend")


@router.post("/v1/audio/translations")
async def translate_audio() -> None:
    raise NotImplementedOpenAIError("Audio translation is not supported by this backend")


@router.post("/v1/audio/speech")
async def create_speech() -> None:
    raise NotImplementedOpenAIError("Text-to-speech is not supported by this backend")
