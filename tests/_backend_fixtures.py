from openai_http.backends.base import BackendBase


class _MockBackendBase(BackendBase):
    """Minimal backend for unit tests: override only what the test needs."""

    async def generate(self, prompt, **kwargs):
        raise NotImplementedError

    async def generate_stream(self, prompt, **kwargs):
        raise NotImplementedError
        yield  # make it an async generator

    async def list_models(self):
        return [{"id": "test", "object": "model", "owned_by": "test", "created": 0}]

    async def get_model(self, model_id):
        for m in await self.list_models():
            if m["id"] == model_id:
                return m
        return None
