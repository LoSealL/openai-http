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

Tests for OpenAI Audio and Images APIs (501 verification).

These endpoints are not yet implemented — tests verify that they return
an appropriate error (404 or 501) when called.
"""

import pytest
import io
from .test_base import OpenAITestBase


class TestAudioAPI(OpenAITestBase):
    """Test suite for Audio API (expect 404/501)."""

    def test_audio_transcription_not_implemented(self, client):
        """Test that audio transcription returns 404 or 501."""
        audio_file = io.BytesIO(b"fake audio")
        audio_file.name = "test.mp3"

        try:
            client.audio.transcriptions.create(model="whisper-1", file=audio_file)
            pytest.fail("Expected error for unimplemented endpoint")
        except Exception as e:
            assert self.expect_not_implemented(e), f"Expected 404/501, got: {e}"

    def test_audio_translation_not_implemented(self, client):
        """Test that audio translation returns 404 or 501."""
        audio_file = io.BytesIO(b"fake audio")
        audio_file.name = "test.mp3"

        try:
            client.audio.translations.create(model="whisper-1", file=audio_file)
            pytest.fail("Expected error for unimplemented endpoint")
        except Exception as e:
            assert self.expect_not_implemented(e), f"Expected 404/501, got: {e}"

    def test_audio_speech_not_implemented(self, client):
        """Test that audio speech (TTS) returns 404 or 501."""
        try:
            client.audio.speech.create(model="tts-1", voice="alloy", input="Hello")
            pytest.fail("Expected error for unimplemented endpoint")
        except Exception as e:
            assert self.expect_not_implemented(e), f"Expected 404/501, got: {e}"


class TestImagesAPI(OpenAITestBase):
    """Test suite for Images API (expect 404/501)."""

    def test_image_generation_not_implemented(self, client):
        """Test that image generation returns 404 or 501."""
        try:
            client.images.generate(
                model="dall-e-3", prompt="A cute cat", n=1, size="1024x1024"
            )
            pytest.fail("Expected error for unimplemented endpoint")
        except Exception as e:
            assert self.expect_not_implemented(e), f"Expected 404/501, got: {e}"

    def test_image_edit_not_implemented(self, client):
        """Test that image edit returns 404 or 501."""
        image_file = io.BytesIO(b"fake image")
        image_file.name = "test.png"

        try:
            client.images.edit(
                model="dall-e-2", image=image_file, prompt="Make it cuter"
            )
            pytest.fail("Expected error for unimplemented endpoint")
        except Exception as e:
            assert self.expect_not_implemented(e), f"Expected 404/501, got: {e}"

    def test_image_variation_not_implemented(self, client):
        """Test that image variation returns 404 or 501."""
        image_file = io.BytesIO(b"fake image")
        image_file.name = "test.png"

        try:
            client.images.create_variation(image=image_file, n=1, size="1024x1024")
            pytest.fail("Expected error for unimplemented endpoint")
        except Exception as e:
            assert self.expect_not_implemented(e), f"Expected 404/501, got: {e}"
