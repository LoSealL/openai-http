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

Tests for OpenAI Completions API (legacy).

Tests: client.completions.create()

NOTE: /v1/completions is P2 — skipped until implemented. Tests verify expected behavior.
"""

import pytest
from .test_base import OpenAITestBase, MOCK_MODELS


def _call_or_skip(client, **kwargs):
    """Call completions.create, skip test if 404 (endpoint not implemented)."""
    try:
        return client.completions.create(**kwargs)
    except Exception as e:
        if "404" in str(e):
            pytest.skip("/v1/completions endpoint not yet implemented (Phase 8)")
        raise


class TestCompletionsSync(OpenAITestBase):
    """Test suite for Completions API (synchronous)."""
    
    def test_completion_basic(self, client):
        """Test basic text completion."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], prompt="Once upon a time", stream=False)
        
        assert response.id.startswith('cmpl-')
        assert response.object == 'text_completion'
        assert len(response.choices) > 0
        self.assert_valid_completion_choice(response.choices[0])
        self.assert_valid_usage(response.usage)
    
    def test_completion_with_parameters(self, client):
        """Test completion with various parameters."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], prompt="The meaning of life is",
                                  max_tokens=50, temperature=0.7, top_p=0.9, n=1, stream=False)
        assert response.object == 'text_completion'
    
    def test_completion_multiple_choices(self, client):
        """Test generating multiple completions."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], prompt="Hello", n=3, stream=False)
        assert response.object == 'text_completion'
        assert len(response.choices) >= 1
    
    def test_completion_stop_sequences(self, client):
        """Test stop sequences."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], prompt="Write a story",
                                  stop=["END", "FINISH"], max_tokens=100, stream=False)
        assert response.object == 'text_completion'
    
    def test_completion_presence_frequency_penalty(self, client):
        """Test penalty parameters."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], prompt="Test",
                                  presence_penalty=0.6, frequency_penalty=0.3, stream=False)
        assert response.object == 'text_completion'
    
    def test_completion_batch_prompts(self, client):
        """Test with batch of prompts."""
        response = _call_or_skip(client, model=MOCK_MODELS[0],
                                  prompt=["What is AI?", "Explain ML"], stream=False)
        assert response.object == 'text_completion'
        assert len(response.choices) > 0


class TestCompletionsStreaming(OpenAITestBase):
    """Test suite for Completions API (streaming)."""
    
    def test_completion_stream(self, client):
        """Test streaming completion."""
        try:
            stream = client.completions.create(model=MOCK_MODELS[0], prompt="Once upon a time", stream=True)
        except Exception as e:
            if "404" in str(e):
                pytest.skip("/v1/completions endpoint not yet implemented (Phase 8)")
            raise
        
        chunks = list(stream)
        assert len(chunks) > 0
        assert chunks[0].object == 'text_completion'
    
    def test_completion_stream_text_accumulation(self, client):
        """Test accumulating streamed text."""
        try:
            stream = client.completions.create(model=MOCK_MODELS[0], prompt="Hello", stream=True)
        except Exception as e:
            if "404" in str(e):
                pytest.skip("/v1/completions endpoint not yet implemented (Phase 8)")
            raise
        
        full_text = [c.choices[0].text for c in stream if c.choices[0].text]
        assert len("".join(full_text)) > 0
