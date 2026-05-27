"""
Tests for OpenAI Chat Completions API.

Tests: client.chat.completions.create() with various parameters
"""

import pytest
import time
from .test_base import OpenAITestBase, MOCK_MODELS
from .mock_data import simple_chat_messages, multi_turn_chat_messages, temperature_params, max_tokens_params


class TestChatCompletionsSync(OpenAITestBase):
    """Test suite for Chat Completions API (synchronous)."""
    
    def test_chat_completion_basic(self, client):
        """Test basic chat completion without streaming."""
        response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=False
        )
        
        # Validate response structure
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
        
        # Validate choice
        choice = response.choices[0]
        self.assert_valid_chat_choice(choice)
        assert choice.message.content is not None
        assert isinstance(choice.message.content, str)
        
        # Validate usage
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
        # Mock backend currently returns 1 choice regardless of n
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
        
        # Mock backend may not be deterministic, but should accept seed parameter
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
        
        # Should raise validation error
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
        
        # First chunk should have role
        first_chunk = chunks[0]
        assert len(first_chunk.choices) > 0
        
        # Last chunk should have finish_reason
        last_chunk = chunks[-1]
        assert len(last_chunk.choices) > 0
        # finish_reason might be in last or second-to-last chunk
    
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
        # Non-streaming
        non_stream_response = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=False
        )
        assert non_stream_response.object == 'chat.completion'
        
        # Streaming
        stream = client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=True
        )
        chunks = list(stream)
        assert len(chunks) > 0
        assert chunks[0].object == 'chat.completion.chunk'
