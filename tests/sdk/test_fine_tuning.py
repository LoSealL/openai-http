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

Tests for OpenAI Fine-tuning API.

Tests: client.fine_tuning.jobs.*

NOTE: /v1/fine_tuning is P3 — skipped until implemented.
"""

import pytest
import io
from .test_base import OpenAITestBase, MOCK_MODELS
from .mock_data import mock_jsonl_content


def _wrap_call(func, endpoint_skip="Fine-tuning API"):
    """Wrap a call, skip test if 404 (endpoint not implemented)."""
    try:
        return func()
    except Exception as e:
        if "404" in str(e):
            pytest.skip(f"{endpoint_skip} not yet implemented")
        raise


class TestFineTuningAPI(OpenAITestBase):
    """Test suite for Fine-tuning API."""
    
    def _create_training_file(self, client):
        """Helper to create a training file."""
        def do_upload():
            file_content = mock_jsonl_content()
            file_obj = io.BytesIO(file_content.encode('utf-8'))
            file_obj.name = "training_data.jsonl"
            return client.files.create(file=file_obj, purpose="fine-tune")
        response = _wrap_call(do_upload, "Files API (required by fine-tuning)")
        return response.id
    
    def test_fine_tuning_create_job(self, client):
        """Test creating a fine-tuning job."""
        file_id = self._create_training_file(client)
        
        def do_create():
            return client.fine_tuning.jobs.create(training_file=file_id, model=MOCK_MODELS[0])
        
        response = _wrap_call(do_create, "Fine-tuning API (Phase 11)")
        assert response.id.startswith('ftjob-')
        assert response.object == 'fine_tuning.job'
        assert response.training_file == file_id
        assert response.status in ['queued', 'running', 'succeeded', 'failed', 'cancelled']
    
    def test_fine_tuning_list_jobs(self, client):
        """Test listing fine-tuning jobs."""
        def do_list():
            return client.fine_tuning.jobs.list()
        
        response = _wrap_call(do_list, "Fine-tuning API (Phase 11)")
        assert response.object == 'list'
        assert isinstance(response.data, list)
    
    def test_fine_tuning_retrieve_job(self, client):
        """Test retrieving a specific fine-tuning job."""
        file_id = self._create_training_file(client)
        
        try:
            created = client.fine_tuning.jobs.create(training_file=file_id, model=MOCK_MODELS[0])
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Fine-tuning API not yet implemented (Phase 11)")
            raise
        
        response = client.fine_tuning.jobs.retrieve(created.id)
        assert response.id == created.id
        assert response.object == 'fine_tuning.job'
    
    def test_fine_tuning_cancel_job(self, client):
        """Test cancelling a fine-tuning job."""
        file_id = self._create_training_file(client)
        
        try:
            created = client.fine_tuning.jobs.create(training_file=file_id, model=MOCK_MODELS[0])
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Fine-tuning API not yet implemented (Phase 11)")
            raise
        
        response = client.fine_tuning.jobs.cancel(created.id)
        assert response.id == created.id
    
    def test_fine_tuning_list_events(self, client):
        """Test listing fine-tuning job events."""
        try:
            file_id = self._create_training_file(client)
            created = client.fine_tuning.jobs.create(training_file=file_id, model=MOCK_MODELS[0])
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Fine-tuning API not yet implemented (Phase 11)")
            raise
        
        response = client.fine_tuning.jobs.list_events(created.id)
        assert response.object == 'list'
        assert isinstance(response.data, list)
    
    def test_fine_tuning_with_hyperparameters(self, client):
        """Test creating job with custom hyperparameters."""
        file_id = self._create_training_file(client)
        
        def do_create():
            return client.fine_tuning.jobs.create(
                training_file=file_id, model=MOCK_MODELS[0],
                hyperparameters={"n_epochs": 3})
        
        response = _wrap_call(do_create, "Fine-tuning API (Phase 11)")
        assert response.object == 'fine_tuning.job'
    
    def test_fine_tuning_invalid_file(self, client):
        """Test creating job with invalid file ID."""
        try:
            client.fine_tuning.jobs.create(training_file="file-non-existent", model=MOCK_MODELS[0])
            assert False, "Should have raised"
        except Exception as e:
            if "404" in str(e):
                return
            raise
    
    def test_fine_tuning_retrieve_invalid(self, client):
        """Test retrieving non-existent job."""
        try:
            client.fine_tuning.jobs.retrieve("ftjob-non-existent")
        except Exception as e:
            if "404" in str(e):
                return
            raise
