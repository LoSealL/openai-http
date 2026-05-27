"""
Tests for OpenAI Chat Completions API (async).

Tests: async_client.chat.completions.create()
"""

import pytest
import pytest_asyncio
from .test_base import OpenAITestBase, MOCK_MODELS
from .mock_data import simple_chat_messages


class TestChatCompletionsAsync(OpenAITestBase):
    """Test suite for Chat Completions API (asynchronous)."""
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_basic(self, async_client):
        """Test basic async chat completion."""
        response = await async_client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=False
        )
        
        assert hasattr(response, 'id')
        assert response.id.startswith('chatcmpl-')
        assert response.object == 'chat.completion'
        assert len(response.choices) > 0
        self.assert_valid_chat_choice(response.choices[0])
        self.assert_valid_usage(response.usage)
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_stream(self, async_client):
        """Test async streaming chat completion."""
        stream = await async_client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            stream=True
        )
        
        chunks = []
        async for chunk in stream:
            assert chunk.object == 'chat.completion.chunk'
            chunks.append(chunk)
        
        assert len(chunks) > 0
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_parameters(self, async_client):
        """Test async chat completion with various parameters."""
        response = await async_client.chat.completions.create(
            model=MOCK_MODELS[0],
            messages=simple_chat_messages(),
            temperature=0.7,
            max_tokens=100,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.1,
            stream=False
        )
        
        assert response.object == 'chat.completion'
        assert len(response.choices) > 0
    
    @pytest.mark.asyncio
    async def test_chat_completion_async_concurrent(self, async_client):
        """Test multiple concurrent async requests."""
        import asyncio
        
        async def make_request():
            return await async_client.chat.completions.create(
                model=MOCK_MODELS[0],
                messages=simple_chat_messages(),
                stream=False
            )
        
        # Make 5 concurrent requests
        tasks = [make_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        
        assert len(responses) == 5
        for response in responses:
            assert response.object == 'chat.completion'
            assert len(response.choices) > 0
