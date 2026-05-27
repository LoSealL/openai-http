import pytest

from openai_http.backends.base import BackendBase


class _CompleteBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        return None


class _PartialBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        return {}


def test_incomplete_subclass_raises_type_error():
    with pytest.raises(TypeError) as exc_info:
        _PartialBackend()
    msg = str(exc_info.value)
    assert "generate_stream" in msg or "list_models" in msg or "get_model" in msg


def test_complete_subclass_instantiates():
    b = _CompleteBackend()
    assert isinstance(b, BackendBase)


@pytest.mark.asyncio
async def test_default_embed_raises_not_implemented():
    b = _CompleteBackend()
    with pytest.raises(NotImplementedError):
        await b.embed(["hello"])


@pytest.mark.asyncio
async def test_default_generate_tool_calls_raises_not_implemented():
    b = _CompleteBackend()
    with pytest.raises(NotImplementedError):
        await b.generate_tool_calls([], [])


@pytest.mark.asyncio
async def test_setup_noop():
    b = _CompleteBackend()
    await b.setup()


@pytest.mark.asyncio
async def test_teardown_noop():
    b = _CompleteBackend()
    await b.teardown()
