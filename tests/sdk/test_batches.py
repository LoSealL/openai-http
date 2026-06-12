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

Tests for OpenAI Batches API.

Tests: client.batches.*

NOTE: /v1/batches is P3 — skipped until implemented.
Tests verify expected behavior once implemented.
"""

import pytest
import io
from .test_base import OpenAITestBase, MOCK_MODELS


def _wrap_call(func):
    """Wrap a call, skip test if 404 (endpoint not implemented)."""
    try:
        return func()
    except Exception as e:
        if "404" in str(e):
            pytest.skip("Batches API not yet implemented (Phase 12)")
        raise


class TestBatchesAPI(OpenAITestBase):
    """Test suite for Batches API."""
    
    def _create_batch_file(self, client):
        """Helper to create a batch input file."""
        batch_lines = [
            '{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", '
            '"body": {"model": "' + MOCK_MODELS[0] + '", "messages": [{"role": "user", "content": "Hello"}]}}',
            '{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", '
            '"body": {"model": "' + MOCK_MODELS[0] + '", "messages": [{"role": "user", "content": "Hi"}]}}',
        ]
        file_content = "\n".join(batch_lines)
        file_obj = io.BytesIO(file_content.encode('utf-8'))
        file_obj.name = "batch_input.jsonl"
        
        try:
            return client.files.create(file=file_obj, purpose="batch")
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Files API not yet implemented (required by Batches, Phase 10)")
            raise
    
    def test_batch_create(self, client):
        """Test creating a batch request."""
        file_response = self._create_batch_file(client)
        
        def do_create():
            return client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
        
        response = _wrap_call(do_create)
        assert response.id.startswith('batch_')
        assert response.object == 'batch'
        assert response.endpoint == '/v1/chat/completions'
        assert response.input_file_id == file_response.id
    
    def test_batch_list(self, client):
        """Test listing batches."""
        response = _wrap_call(lambda: client.batches.list())
        assert response.object == 'list'
        assert isinstance(response.data, list)
    
    def test_batch_retrieve(self, client):
        """Test retrieving a specific batch."""
        file_response = self._create_batch_file(client)
        
        try:
            created = client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Batches API not yet implemented")
            raise
        
        response = client.batches.retrieve(created.id)
        assert response.id == created.id
        assert response.object == 'batch'
    
    def test_batch_cancel(self, client):
        """Test cancelling a batch."""
        file_response = self._create_batch_file(client)
        
        try:
            created = client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Batches API not yet implemented")
            raise
        
        response = client.batches.cancel(created.id)
        assert response.id == created.id
        assert hasattr(response, 'status')
