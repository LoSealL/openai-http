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

Tests for OpenAI Files API.

Tests: client.files.create(), list(), retrieve(), delete()

NOTE: /v1/files is P3 — skipped until implemented.
"""

import pytest
import io
from .test_base import OpenAITestBase
from .mock_data import mock_jsonl_content


def _wrap_call(func):
    """Wrap a call, skip test if 404 (endpoint not implemented)."""
    try:
        return func()
    except Exception as e:
        if "404" in str(e):
            pytest.skip("Files API not yet implemented (Phase 10)")
        raise


class TestFilesAPI(OpenAITestBase):
    """Test suite for Files API."""
    
    def test_file_upload(self, client):
        """Test uploading a file."""
        def do_upload():
            file_content = mock_jsonl_content()
            file_obj = io.BytesIO(file_content.encode('utf-8'))
            file_obj.name = "test_data.jsonl"
            return client.files.create(file=file_obj, purpose="fine-tune")
        
        response = _wrap_call(do_upload)
        assert response.id.startswith('file-')
        assert response.object == 'file'
        assert response.bytes > 0
        assert response.purpose == "fine-tune"
    
    def test_file_list(self, client):
        """Test listing files."""
        def do_list():
            return client.files.list()
        
        response = _wrap_call(do_list)
        assert response.object == 'list'
        assert isinstance(response.data, list)
    
    def test_file_retrieve(self, client):
        """Test retrieving a specific file."""
        try:
            file_content = mock_jsonl_content()
            file_obj = io.BytesIO(file_content.encode('utf-8'))
            file_obj.name = "retrieve_test.jsonl"
            uploaded = client.files.create(file=file_obj, purpose="fine-tune")
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Files API not yet implemented (Phase 10)")
            raise
        
        response = client.files.retrieve(uploaded.id)
        assert response.id == uploaded.id
        assert response.object == 'file'
    
    def test_file_delete(self, client):
        """Test deleting a file."""
        try:
            file_content = mock_jsonl_content()
            file_obj = io.BytesIO(file_content.encode('utf-8'))
            file_obj.name = "delete_test.jsonl"
            uploaded = client.files.create(file=file_obj, purpose="fine-tune")
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Files API not yet implemented (Phase 10)")
            raise
        
        response = client.files.delete(uploaded.id)
        assert response.id == uploaded.id
        assert response.deleted
    
    def test_file_retrieve_invalid(self, client):
        """Test retrieving non-existent file."""
        try:
            client.files.retrieve("file-non-existent")
            assert False, "Should have raised"
        except Exception as e:
            if "404" in str(e):
                return
            raise
    
    def test_file_delete_invalid(self, client):
        """Test deleting non-existent file."""
        try:
            client.files.delete("file-non-existent")
        except Exception as e:
            if "404" in str(e):
                return
            raise
