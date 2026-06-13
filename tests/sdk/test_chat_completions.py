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

Tests for OpenAI Chat Completions API.

Tests: client.chat.completions.create() with various parameters
"""

import pytest
from .test_base import OpenAITestBase, MOCK_MODELS
from .mock_data import simple_chat_messages, multi_turn_chat_messages


class TestChatCompletionsSync(OpenAITestBase):
    """Test suite for Chat Completions API (synchronous)."""
    
    def test_chat_completion_basic(self, client):
        """Test basic chat completion without streaming."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=False
        )
        
        assert hasattr(response, 'id')
        assert response.id.startswith('chatcmpl-')
        assert hasattr(response, 'object')
        assert response.object == 'chat.completion'
        assert hasattr(response, 'created')
        assert isinstance(response.created, int)
        assert hasattr(response, 'model')
        assert hasattr(response, 'choices')
        assert len(response.choices) > 0
        assert hasattr(response, 'usage')
        
        choice = response.choices[0]
        self.assert_valid_chat_choice(choice)
        assert choice.message.content is not None
        assert isinstance(choice.message.content, str)
        
        self.assert_valid_usage(response.usage)
    
    def test_chat_completion_multi_turn(self, client):
        """Test multi-turn conversation."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=multi_turn_chat_messages(),
            stream=False
        )
        
        assert response.object == 'chat.completion'
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
    
    def test_chat_completion_temperature(self, client):
        """Test different temperature values."""
        for temp in [0.0, 0.5, 1.0]:
            response = client.chat.completions.create(
                model=MOCK_MODELS[0],
                messages=simple_chat_messages(),
                temperature=temp,
                stream=False
            )
            assert response.object == 'chat.completion'
            assert len(response.choices) > 0
    
    def test_chat_completion_max_tokens(self, client):
        """Test max_tokens parameter."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            max_tokens=50,
            stream=False
        )
        
        assert response.object == 'chat.completion'
        assert len(response.choices) > 0
    
    def test_chat_completion_n_choicess(self, client):
        """Test generating multiple completions."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            n=3,
            stream=False
        )
        
        assert response.object == 'chat.completion'
        assert len(response.choices) >= 1
    
    def test_chat_completion_stop_sequences(self, client):
        """Test stop sequences."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stop=["END", "STOP"],
            stream=False
        )
        
        assert response.object == 'chat.completion'
    
    def test_chat_completion_top_p(self, client):
        """Test top_p (nucleus sampling) parameter."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            top_p=0.9,
            stream=False
        )
        
        assert response.object == 'chat.completion'
    
    def test_chat_completion_presence_frequency_penalty(self, client):
        """Test presence and frequency penalty."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            presence_penalty=0.5,
            frequency_penalty=0.5,
            stream=False
        )
        
        assert response.object == 'chat.completion'
    
    def test_chat_completion_seed(self, client):
        """Test seed for reproducibility."""
        response1 = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            seed=42,
            stream=False
        )
        
        response2 = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            seed=42,
            stream=False
        )
        
        assert response1.object == 'chat.completion'
        assert response2.object == 'chat.completion'
    
    def test_chat_completion_user_metadata(self, client):
        """Test user metadata field."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            user="test-user-123",
            stream=False
        )
        
        assert response.object == 'chat.completion'
    
    def test_chat_completion_invalid_model(self, client):
        """Test request with invalid model."""
        with pytest.raises(Exception) as exc_info:
            client.chat.completions.create(
                model="non-existent-model",
                messages=simple_chat_messages(),
                stream=False
            )
        
        assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()
    
    def test_chat_completion_empty_messages(self, client):
        """Test request with empty messages."""
        with pytest.raises(Exception) as exc_info:
            client.chat.completions.create(
                model=MOCK_MODELS[0],
                messages=[],
                stream=False
            )
        
        assert "400" in str(exc_info.value) or "validation" in str(exc_info.value).lower()


class TestChatCompletionsStreaming(OpenAITestBase):
    """Test suite for Chat Completions API (streaming)."""
    
    def test_chat_completion_stream(self, client):
        """Test streaming chat completion."""
        stream = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=True
        )
        
        chunks = []
        for chunk in stream:
            assert hasattr(chunk, 'id')
            assert chunk.id.startswith('chatcmpl-')
            assert hasattr(chunk, 'object')
            assert chunk.object == 'chat.completion.chunk'
            assert hasattr(chunk, 'created')
            assert hasattr(chunk, 'model')
            assert hasattr(chunk, 'choices')
            
            chunks.append(chunk)
        
        assert len(chunks) > 0
        
        first_chunk = chunks[0]
        assert len(first_chunk.choices) > 0
        
        last_chunk = chunks[-1]
        assert len(last_chunk.choices) > 0
    
    def test_chat_completion_stream_content_accumulation(self, client):
        """Test that streaming content can be accumulated."""
        stream = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=True
        )
        
        full_content = []
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_content.append(chunk.choices[0].delta.content)
        
        # Should have some content
        combined = "".join(full_content)
        assert len(combined) > 0
    
    def test_chat_completion_stream_vs_non_stream(self, client):
        """Test that stream=True and stream=False both work."""
        non_stream_response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=False
        )
        assert non_stream_response.object == 'chat.completion'
        
        stream = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=True
        )
        chunks = list(stream)
        assert len(chunks) > 0
        assert chunks[0].object == 'chat.completion.chunk'


class TestChatCompletionsReasoning(OpenAITestBase):
    """Test suite for reasoning_content in chat completions."""

    def test_chat_completion_reasoning_non_streaming(self, client):
        """Test that non-streaming responses include reasoning_content."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=False,
        )

        assert response.object == 'chat.completion'
        assert len(response.choices) > 0

        choice = response.choices[0]
        assert choice.message.content is not None
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0

        assert hasattr(choice.message, 'reasoning_content')
        reasoning = choice.message.reasoning_content
        assert reasoning is not None
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0

    def test_chat_completion_reasoning_streaming(self, client):
        """Test that streaming responses include reasoning_content in deltas."""
        stream = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=True,
        )

        reasoning_chunks = []
        content_chunks = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning_chunks.append(delta.reasoning_content)
            if delta.content:
                content_chunks.append(delta.content)

        assert len(reasoning_chunks) > 0, "Should have received reasoning chunks"
        assert len(content_chunks) > 0, "Should have received content chunks"

        reasoning_text = "".join(reasoning_chunks)
        content_text = "".join(content_chunks)
        assert len(reasoning_text) > 0
        assert len(content_text) > 0


class TestChatCompletionsFinishReason(OpenAITestBase):
    """Test suite for finish_reason in chat completions."""

    def test_finish_reason_stop(self, client):
        """Test that normal completion returns finish_reason='stop'."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=False,
        )

        assert response.object == 'chat.completion'
        assert len(response.choices) > 0
        assert response.choices[0].finish_reason == 'stop'

    def test_finish_reason_length(self, client):
        """Test that hitting max_tokens returns finish_reason='length'."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            max_tokens=5,
            stream=False,
        )

        assert response.object == 'chat.completion'
        assert len(response.choices) > 0
        assert response.choices[0].finish_reason == 'length'

    def test_finish_reason_stop_streaming(self, client):
        """Test that normal streaming returns finish_reason='stop'."""
        stream = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=True,
        )

        last_chunk = None
        for chunk in stream:
            if chunk.choices:
                last_chunk = chunk

        assert last_chunk is not None
        assert last_chunk.choices[0].finish_reason == 'stop'

    def test_finish_reason_length_streaming(self, client):
        """Test that streaming with max_tokens returns finish_reason='length'."""
        stream = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            max_tokens=5,
            stream=True,
        )

        last_chunk = None
        for chunk in stream:
            if chunk.choices:
                last_chunk = chunk

        assert last_chunk is not None
        assert last_chunk.choices[0].finish_reason == 'length'
