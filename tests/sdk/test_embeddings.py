"""
Tests for OpenAI Embeddings API.

Tests: client.embeddings.create()

NOTE: /v1/embeddings is P2 — skipped until implemented. Tests verify expected behavior.
"""

import pytest
from .test_base import OpenAITestBase, MOCK_MODELS
from .mock_data import embedding_texts


def _call_or_skip(client, **kwargs):
    """Call embeddings.create, skip test if 404 (endpoint not implemented)."""
    try:
        return client.embeddings.create(**kwargs)
    except Exception as e:
        if "404" in str(e):
            pytest.skip("/v1/embeddings endpoint not yet implemented (Phase 7)")
        raise


class TestEmbeddingsAPI(OpenAITestBase):
    """Test suite for Embeddings API."""
    
    def test_embedding_single_text(self, client):
        """Test embedding for single text."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], input="The quick brown fox")
        
        assert response.object == 'list'
        assert len(response.data) == 1
        assert response.data[0].object == 'embedding'
        assert response.data[0].index == 0
        assert len(response.data[0].embedding) > 0
        self.assert_valid_usage(response.usage)
    
    def test_embedding_batch_texts(self, client):
        """Test embedding for batch of texts."""
        texts = embedding_texts()
        response = _call_or_skip(client, model=MOCK_MODELS[0], input=texts)
        
        assert response.object == 'list'
        assert len(response.data) == len(texts)
        for i, emb in enumerate(response.data):
            assert emb.index == i
    
    def test_embedding_dimensions(self, client):
        """Test that all embeddings have consistent dimensions."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], input=embedding_texts())
        
        if len(response.data) > 1:
            dim = len(response.data[0].embedding)
            for emb in response.data:
                assert len(emb.embedding) == dim
    
    def test_embedding_empty_input(self, client):
        """Test with empty input array."""
        try:
            response = _call_or_skip(client, model=MOCK_MODELS[0], input=[])
            assert response.object == 'list'
        except Exception as e:
            if "400" in str(e) or "404" in str(e) or "validation" in str(e).lower():
                pytest.skip("/v1/embeddings not yet implemented or empty input rejected")
            raise
    
    def test_embedding_long_text(self, client):
        """Test with long text input."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], input="word " * 1000)
        assert response.object == 'list'
        assert len(response.data[0].embedding) > 0
    
    def test_embedding_special_characters(self, client):
        """Test with special characters."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], input="Hello 世界! 🌍 \n\t")
        assert response.object == 'list'
        assert len(response.data) > 0
    
    def test_embedding_unicode(self, client):
        """Test with unicode characters."""
        response = _call_or_skip(client, model=MOCK_MODELS[0], input="Unicode: αβγ δεζ ηθι")
        assert response.object == 'list'
