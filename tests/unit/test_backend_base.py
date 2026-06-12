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

Tests for BackendBase abstract interface.
"""

import pytest

from openai_http.backends.base import BackendBase


class _CompleteBackend(BackendBase):
    """A backend implementing all required abstract methods."""

    async def generate(self, prompt, **kwargs):
        return {"generated_text": "ok", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

    async def generate_stream(self, prompt, **kwargs):
        yield "ok"

    async def list_models(self):
        return [{"id": "m", "object": "model", "created": 0, "owned_by": "me"}]

    async def get_model(self, model_id):
        return None


class _PartialBackend(BackendBase):
    """A backend that only implements generate, missing other abstract methods."""

    async def generate(self, prompt, **kwargs):
        return {}


def test_incomplete_subclass_raises_type_error():
    """Instantiating a partial BackendBase subclass raises TypeError."""
    with pytest.raises(TypeError) as exc_info:
        _PartialBackend()
    msg = str(exc_info.value)
    assert "generate_stream" in msg or "list_models" in msg or "get_model" in msg


def test_complete_subclass_instantiates():
    """A backend implementing all abstract methods instantiates without error."""
    b = _CompleteBackend()
    assert isinstance(b, BackendBase)


@pytest.mark.asyncio
async def test_default_embed_raises_not_implemented():
    """The default embed() raises NotImplementedError."""
    b = _CompleteBackend()
    with pytest.raises(NotImplementedError):
        await b.embed(["hello"])


@pytest.mark.asyncio
async def test_default_generate_tool_calls_raises_not_implemented():
    """The default generate_tool_calls() raises NotImplementedError."""
    b = _CompleteBackend()
    with pytest.raises(NotImplementedError):
        await b.generate_tool_calls([], [])


@pytest.mark.asyncio
async def test_setup_noop():
    """The default setup() is a no-op and does not raise."""
    b = _CompleteBackend()
    await b.setup()


@pytest.mark.asyncio
async def test_teardown_noop():
    """The default teardown() is a no-op and does not raise."""
    b = _CompleteBackend()
    await b.teardown()
