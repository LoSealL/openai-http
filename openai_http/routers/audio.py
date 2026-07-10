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

Audio processing endpoints.

POST /v1/audio/transcriptions  - transcribe audio to text
POST /v1/audio/translations    - translate audio to English text
POST /v1/audio/speech          - generate speech from text
"""

from fastapi import APIRouter

from ..errors import NotImplementedOpenAIError

router = APIRouter(tags=["Audio"])


@router.post("/v1/audio/transcriptions")
async def transcribe_audio() -> None:
    """Transcribe audio to text.

    Raises:
        NotImplementedOpenAIError: Always, as this backend does not support transcription.
    """
    raise NotImplementedOpenAIError(
        "Audio transcription is not supported by this backend"
    )


@router.post("/v1/audio/translations")
async def translate_audio() -> None:
    """Translate audio to English text.

    Raises:
        NotImplementedOpenAIError: Always, as this backend does not support translation.
    """
    raise NotImplementedOpenAIError(
        "Audio translation is not supported by this backend"
    )


@router.post("/v1/audio/speech")
async def create_speech() -> None:
    """Generate speech from text input.

    Raises:
        NotImplementedOpenAIError: Always, as this backend does not support text-to-speech.
    """
    raise NotImplementedOpenAIError("Text-to-speech is not supported by this backend")
