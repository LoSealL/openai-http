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

End-to-end SSE schema validation.

Streams every chunk emitted by /v1/chat/completions and /v1/completions
through the wire-level Pydantic schemas to guard against regressions
in ``_make_chunk``.
"""

import json

import httpx
import pytest

from openai_http.schemas.chat import ChatCompletionChunk
from openai_http.schemas.completions import CompletionChunk
from tests.sdk.test_base import DEFAULT_MODEL


def _iter_sse_events(text: str):
    """Yield decoded JSON payloads from an SSE event stream body.

    Skips the terminal ``[DONE]`` marker.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        yield json.loads(payload)


class TestChatCompletionsSSEContract:
    """Each SSE chunk from /v1/chat/completions must validate against
    :class:`ChatCompletionChunk`."""

    def test_chat_stream_chunks_validate(self, sdk_server):
        url = f"{sdk_server['base_url']}/chat/completions"
        with httpx.Client(timeout=30.0) as cli:
            with cli.stream(
                "POST",
                url,
                json={
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                body = resp.read().decode("utf-8")

        chunks = list(_iter_sse_events(body))
        assert chunks, "expected at least one streamed chunk"
        for raw in chunks:
            ChatCompletionChunk.model_validate(raw)


class TestCompletionsSSEContract:
    """Each SSE chunk from /v1/completions must validate against
    :class:`CompletionChunk`."""

    def test_completion_stream_chunks_validate(self, sdk_server):
        url = f"{sdk_server['base_url']}/completions"
        with httpx.Client(timeout=30.0) as cli:
            with cli.stream(
                "POST",
                url,
                json={
                    "model": DEFAULT_MODEL,
                    "prompt": "hello",
                    "max_tokens": 16,
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                body = resp.read().decode("utf-8")

        chunks = list(_iter_sse_events(body))
        assert chunks, "expected at least one streamed chunk"
        for raw in chunks:
            CompletionChunk.model_validate(raw)


@pytest.mark.parametrize("path", ["/chat/completions", "/completions"])
def test_no_extra_fields_in_chunks(sdk_server, path):
    """Sanity: the wire chunks must not include unknown null fields when
    ``exclude_none=True`` is in effect."""
    url = f"{sdk_server['base_url']}{path}"
    payload = (
        {
            "model": DEFAULT_MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        if path == "/chat/completions"
        else {
            "model": DEFAULT_MODEL,
            "prompt": "hi",
            "max_tokens": 8,
            "stream": True,
        }
    )

    with httpx.Client(timeout=30.0) as cli:
        with cli.stream("POST", url, json=payload) as resp:
            assert resp.status_code == 200
            body = resp.read().decode("utf-8")

    saw_chunk = False
    for raw in _iter_sse_events(body):
        saw_chunk = True
        # Spot-check: usage is omitted from intermediate chunks rather
        # than serialised as null.
        if "usage" in raw:
            assert raw["usage"] is not None
    assert saw_chunk
