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

Tests for backend validation logic.
"""

import pytest

from openai_http.backends.base import BackendBase
from openai_http._validation import validate_backend, BackendValidationError


class _ValidBackend(BackendBase):
    """A backend that conforms to the full BackendBase protocol."""

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
    """A conforming backend passes validation without error."""
    await validate_backend(_ValidBackend())


class _StringReturnsBackend(BackendBase):
    """A backend whose generate() returns a string instead of a dict."""

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
    """generate() must return a dict; returning a string raises BackendValidationError."""
    with pytest.raises(BackendValidationError, match="generate"):
        await validate_backend(_StringReturnsBackend())


class _SyncStreamBackend(BackendBase):
    """A backend with a sync generate_stream instead of async."""

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
    """generate_stream must be async; a sync generator raises BackendValidationError."""
    with pytest.raises(BackendValidationError, match="generate_stream"):
        await validate_backend(_SyncStreamBackend())


class _MissingKeyModelsBackend(BackendBase):
    """A backend returning model dicts missing required key 'owned_by'."""

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
    """list_models entries missing required keys raises BackendValidationError."""
    with pytest.raises(BackendValidationError, match="owned_by"):
        await validate_backend(_MissingKeyModelsBackend())


class _MissingUsageBackend(BackendBase):
    """A backend whose generate() response lacks a usage key."""

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
    """A generate() response missing usage raises BackendValidationError."""
    with pytest.raises(BackendValidationError, match="usage"):
        await validate_backend(_MissingUsageBackend())
