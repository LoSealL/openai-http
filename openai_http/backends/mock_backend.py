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

Mock backend for testing and development.

Implements the Backend Protocol using mock data and simple token estimation.
No real model weights are loaded.
"""

import asyncio
import json
import random
from typing import Any, AsyncGenerator, Optional

from .base import BackendBase


AVAILABLE_MODELS = [
    {
        "id": "mock-gpt",
        "object": "model",
        "created": 1677610602,
        "owned_by": "mock-org",
    },
    {
        "id": "mock-llama",
        "object": "model",
        "created": 1677610602,
        "owned_by": "mock-org",
    },
]


class MockTransformersBackend(BackendBase):
    """Mock backend that simulates model responses.

    Generates deterministic or random responses without loading any
    real model weights. Useful for testing and development.
    """

    def __init__(self, model_name: str = "mock-model", thinking: bool = True):
        """Initialize the mock backend.

        Args:
            model_name: The model identifier string.
            thinking: Whether to simulate reasoning/thinking output.
        """
        self.model_name = model_name
        self.thinking = thinking

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> dict:
        """Generate a mock completion response.

        Args:
            prompt: A plain text string or a list of message dicts.
            **kwargs: Generation parameters (max_tokens, temperature, etc.).
                When ``tools`` is present, a mock tool-call response is returned.

        Returns:
            A dict with generated_text (or tool_calls) and usage token counts.
        """
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice", "auto")
        if tools and tool_choice != "none":
            return self._make_mock_tool_calls(messages, tools, tool_choice)

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)

        prompt_tokens = len(str(messages)) // 4

        generation_time = 0.001 * min(max_tokens, 100)
        await asyncio.sleep(generation_time)

        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        if temperature > 0.8:
            templates = [
                f"This is a mock response. Your input was: {last_user_msg[:50]}",
                f'[Random Mode] You said "{last_user_msg[:20]}"... interesting question!',
                f"Mock model response: input length {len(last_user_msg)} chars, "
                f"temperature={temperature:.2f}.",
            ]
            generated_text = random.choice(templates)
        else:
            generated_text = (
                f"This is a simulated response. Your input was: {last_user_msg[:50]}"
            )

        reasoning_content: str | None = None
        if self.thinking:
            reasoning = (
                f"Let me think about this... The user said: {last_user_msg[:30]}. "
                "I should provide a helpful mock response."
            )
            reasoning_content = reasoning
            full_text = f"<think>{reasoning}</think>\n{generated_text}"
        else:
            full_text = generated_text

        completion_tokens = len(full_text) // 4
        finish_reason: str = "stop"
        if completion_tokens > max_tokens:
            finish_reason = "length"

        if reasoning_content:
            think_start = full_text.find("<think>")
            think_end = full_text.rfind("</think>")
            if think_start != -1 and think_end != -1 and think_end > think_start:
                generated_text = full_text[think_end + 8 :].lstrip("\n")
            if finish_reason == "length":
                char_budget = max_tokens * 4
                reasoning_budget = min(len(reasoning_content), char_budget // 2)
                content_budget = char_budget - reasoning_budget
                reasoning_content = reasoning_content[:reasoning_budget]
                generated_text = generated_text[:content_budget]
        else:
            if finish_reason == "length":
                generated_text = full_text[: max_tokens * 4]
            else:
                generated_text = full_text

        emitted = (reasoning_content or "") + (generated_text or "")
        if emitted:
            completion_tokens = min(len(emitted) // 4, max_tokens)
        else:
            completion_tokens = 0

        return {
            "generated_text": generated_text,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def _make_mock_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: Any,
    ) -> dict[str, Any]:
        """Produce a mock tool-call response."""
        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            target_name = tool_choice.get("function", {}).get("name")
            candidates = [
                t for t in tools if t.get("function", {}).get("name") == target_name
            ]
        elif tool_choice == "required":
            candidates = tools
        else:
            candidates = tools

        if not candidates:
            return {
                "generated_text": None,
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 0,
                    "total_tokens": 10,
                },
            }

        selected = candidates[0]
        func_def = selected.get("function", {})
        func_name = func_def.get("name", "unknown_function")
        params = func_def.get("parameters", {})
        props = params.get("properties", {})
        fake_args = {k: "mock_value" for k in props}

        prompt_tok = sum(
            max(1, int(len(str(m.get("content", ""))) * 0.25)) for m in messages
        )
        comp_tok = max(1, int(len(json.dumps(fake_args)) * 0.25))
        return {
            "generated_text": None,
            "tool_calls": [
                {
                    "id": f"call_{random.randbytes(6).hex()}",
                    "type": "function",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(fake_args),
                    },
                }
            ],
            "finish_reason": "tool_calls",
            "usage": {
                "prompt_tokens": prompt_tok,
                "completion_tokens": comp_tok,
                "total_tokens": prompt_tok + comp_tok,
            },
        }

    async def generate_stream(  # type: ignore[override]
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """Generate a streaming mock completion.

        Yields the full response in small random-sized chunks with
        simulated delays. When thinking is enabled, yields reasoning
        chunks first as typed dicts, then content chunks.

        Args:
            prompt: A plain text string or a list of message dicts.
            **kwargs: Generation parameters forwarded to generate().

        Yields:
            Text chunks or typed dicts with reasoning/content.
        """
        result = await self.generate(prompt, **kwargs)
        full_text = result["generated_text"]
        reasoning = result.get("reasoning_content")

        if reasoning:
            idx = 0
            while idx < len(reasoning):
                chunk_size = random.randint(2, 6)
                chunk = reasoning[idx : idx + chunk_size]
                yield {"type": "reasoning", "content": chunk}
                idx += chunk_size
                await asyncio.sleep(random.uniform(0.01, 0.03))

        idx = 0
        while idx < len(full_text):
            chunk_size = random.randint(1, 4)
            chunk = full_text[idx : idx + chunk_size]
            yield {"type": "content", "content": chunk}
            idx += chunk_size
            await asyncio.sleep(random.uniform(0.01, 0.03))

        if result.get("finish_reason") == "length":
            yield {"type": "finish", "reason": "length"}

    async def embed(
        self,
        texts: list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate mock embeddings for the given texts.

        Creates deterministic random vectors based on text hash.

        Args:
            texts: List of text strings to embed.
            **kwargs: Additional parameters including dimensions.

        Returns:
            A list of float vectors, one per input text.
        """
        dims = kwargs.get("dimensions", 1536)
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = random.Random(seed)
            embedding = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
            embeddings.append(embedding)
        return embeddings

    async def list_models(self) -> list[dict]:
        """List all available mock models.

        Returns:
            A list of model dicts defined in AVAILABLE_MODELS.
        """
        return AVAILABLE_MODELS

    async def get_model(self, model_id: str) -> Optional[dict]:
        """Get details for a specific mock model.

        Args:
            model_id: The model identifier string.

        Returns:
            A model dict if found, or None.
        """
        for model in AVAILABLE_MODELS:
            if model["id"] == model_id:
                return model
        return None

    async def metrics(self) -> dict:
        """Return mock metrics.

        Returns:
            A small set of fake operational statistics.
        """
        return {
            "requests_total": 0,
            "generations_total": 0,
            "embeddings_total": 0,
        }

    async def health(self) -> dict:
        """Return mock backend health details.

        Returns:
            A status dict overlaid on the base health response.
        """
        return {"backend": "mock", "healthy": True}
