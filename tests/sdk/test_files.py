import pytest
import io
from .test_base import OpenAITestBase

_MOCK_JSONL = "\n".join(
    [
        '{"prompt": "What is AI?", "completion": "AI is artificial intelligence"}',
        '{"prompt": "Define ML", "completion": "ML is machine learning"}',
        '{"prompt": "Explain NN", "completion": "NN is neural network"}',
    ]
)


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
            file_content = _MOCK_JSONL
            file_obj = io.BytesIO(file_content.encode("utf-8"))
            file_obj.name = "test_data.jsonl"
            return client.files.create(file=file_obj, purpose="fine-tune")

        response = _wrap_call(do_upload)
        assert response.id.startswith("file-")
        assert response.object == "file"
        assert response.bytes > 0
        assert response.purpose == "fine-tune"

    def test_file_list(self, client):
        """Test listing files."""

        def do_list():
            return client.files.list()

        response = _wrap_call(do_list)
        assert response.object == "list"
        assert isinstance(response.data, list)

    def test_file_retrieve(self, client):
        """Test retrieving a specific file."""
        try:
            file_content = _MOCK_JSONL
            file_obj = io.BytesIO(file_content.encode("utf-8"))
            file_obj.name = "retrieve_test.jsonl"
            uploaded = client.files.create(file=file_obj, purpose="fine-tune")
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Files API not yet implemented (Phase 10)")
            raise

        response = client.files.retrieve(uploaded.id)
        assert response.id == uploaded.id
        assert response.object == "file"

    def test_file_delete(self, client):
        """Test deleting a file."""
        try:
            file_content = _MOCK_JSONL
            file_obj = io.BytesIO(file_content.encode("utf-8"))
            file_obj.name = "delete_test.jsonl"
            uploaded = client.files.create(file=file_obj, purpose="fine-tune")
        except Exception as e:
            if "404" in str(e):
                pytest.skip("Files API not yet implemented (Phase 10)")
            raise

        response = client.files.delete(uploaded.id)
        assert response.id == uploaded.id
        assert response.deleted
