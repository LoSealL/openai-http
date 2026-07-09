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

Mock backend output must conform to the typed contracts.
"""

import pytest

from openai_http.backends.contract import (
    validate_generation,
    validate_model_info,
    validate_model_list,
    validate_stream_chunk,
)
from openai_http.backends.mock_backend import MockTransformersBackend
from openai_http.backends.types import FinishChunk


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [True, False])
async def test_generate_output_passes_schema(thinking):
    backend = MockTransformersBackend(thinking=thinking)
    raw = await backend.generate("hello", max_tokens=128, temperature=0.0)
    result = validate_generation(raw)
    assert result.usage.total_tokens == (
        result.usage.prompt_tokens + result.usage.completion_tokens
    )


@pytest.mark.asyncio
async def test_generate_truncated_output_passes_schema():
    backend = MockTransformersBackend(thinking=True)
    raw = await backend.generate("a" * 5, max_tokens=2, temperature=0.0)
    result = validate_generation(raw)
    assert result.finish_reason in {"stop", "length"}
    # Re-estimated completion tokens must not exceed max_tokens.
    assert result.usage.completion_tokens <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking", [True, False])
async def test_generate_stream_chunks_pass_schema(thinking):
    backend = MockTransformersBackend(thinking=thinking)
    saw_content = False
    async for raw in backend.generate_stream("hello", max_tokens=64, temperature=0.0):
        chunk = validate_stream_chunk(raw)
        if not isinstance(chunk, FinishChunk):
            saw_content = True
    assert saw_content


@pytest.mark.asyncio
async def test_list_models_passes_schema():
    backend = MockTransformersBackend()
    models = validate_model_list(await backend.list_models())
    assert len(models) >= 1


@pytest.mark.asyncio
async def test_get_model_passes_schema():
    backend = MockTransformersBackend()
    raw = await backend.get_model("mock-gpt")
    assert raw is not None
    info = validate_model_info(raw)
    assert info.id == "mock-gpt"


@pytest.mark.asyncio
async def test_generate_tool_calls_passes_schema():
    backend = MockTransformersBackend()
    raw = await backend.generate(
        prompt=[{"role": "user", "content": "go"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "do_thing",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string"}},
                    },
                },
            }
        ],
    )
    calls = validate_generation(raw).tool_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].function.name == "do_thing"
