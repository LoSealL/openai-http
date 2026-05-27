import pytest

from openai_http.backends.base import BackendBase
from openai_http._validation import validate_backend, BackendValidationError


class _ValidBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        return None


@pytest.mark.asyncio
async def test_valid_backend_passes():
    await validate_backend(_ValidBackend())


class _StringReturnsBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        return "not a dict"

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        return None


@pytest.mark.asyncio
async def test_generate_returns_string_fails():
    with pytest.raises(BackendValidationError, match="generate"):
        await validate_backend(_StringReturnsBackend())


class _SyncStreamBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        return None


@pytest.mark.asyncio
async def test_sync_stream_fails():
    with pytest.raises(BackendValidationError, match="generate_stream"):
        await validate_backend(_SyncStreamBackend())


class _MissingKeyModelsBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0}]

    async def get_model(self, model_id):
        return None


@pytest.mark.asyncio
async def test_list_models_missing_key_fails():
    with pytest.raises(BackendValidationError, match="owned_by"):
        await validate_backend(_MissingKeyModelsBackend())


class _MissingUsageBackend(BackendBase):
    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok"}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        return None


@pytest.mark.asyncio
async def test_missing_usage_fails():
    with pytest.raises(BackendValidationError, match="usage"):
        await validate_backend(_MissingUsageBackend())
