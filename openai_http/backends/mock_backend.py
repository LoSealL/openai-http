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
import base64
import io
import json
import random
import re
from typing import Any, AsyncGenerator, Optional

from PIL import Image

from openai_http.backends.base import BackendBase

_DATA_URI_RE = re.compile(r"^data:(image/\w+);base64,(.+)$")


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

    def __init__(self, model_name: str = "mock-model"):
        """Initialize the mock backend.

        Args:
            model_name: The model identifier string.
        """
        self.model_name = model_name

    @staticmethod
    def _decode_image(b64_data: str) -> dict[str, Any]:
        try:
            decoded = base64.b64decode(b64_data, validate=True)
        except Exception:
            return {"_type": "image", "format": "unknown", "size_bytes": len(b64_data)}
        info: dict[str, Any] = {"_type": "image", "size_bytes": len(decoded)}
        try:
            img = Image.open(io.BytesIO(decoded))
            info["format"] = img.format.lower() if img.format else "unknown"
            info["mode"] = img.mode
            info["width"] = img.width
            info["height"] = img.height
        except ImportError:
            pass
        except Exception:
            info["format"] = "unknown"
        return info

    @staticmethod
    def _summarize_value(v: Any) -> Any:
        if isinstance(v, str):
            m = _DATA_URI_RE.match(v)
            if m:
                return MockTransformersBackend._decode_image(m.group(2))
            if len(v) > 1000 and re.fullmatch(r"^[A-Za-z0-9+/=]+$", v):
                return MockTransformersBackend._decode_image(v)
            return v
        if isinstance(v, dict):
            return {
                k: MockTransformersBackend._summarize_value(v) for k, v in v.items()
            }
        if isinstance(v, list):
            return [MockTransformersBackend._summarize_value(item) for item in v]
        return v

    def _log_payload(self, method: str, **kwargs) -> None:
        summary: dict[str, Any] = {"method": method}
        for k, v in kwargs.items():
            summary[k] = self._summarize_value(v)
        print(f"[MockBackend] Payload:\n{json.dumps(summary, indent=2, default=str)}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Roughly estimate token count for a text string.

        Uses a simple heuristic: CJK characters count as 1.5 tokens,
        others as 0.25 tokens.

        Args:
            text: The input text string.

        Returns:
            Estimated token count, minimum 1.
        """
        count: float = 0
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                count += 1.5
            else:
                count += 0.25
        return int(max(1, count))

    @staticmethod
    def _build_prompt(messages: list[dict[str, str]]) -> str:
        """Build a plain text prompt from a list of message dicts.

        Args:
            messages: List of message dicts with role and content keys.

        Returns:
            A formatted prompt string suitable for text generation.
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> dict:
        """Generate a mock completion response.

        Args:
            prompt: A plain text string or a list of message dicts.
            **kwargs: Generation parameters (max_tokens, temperature, etc.).

        Returns:
            A dict with generated_text and usage token counts.
        """
        self._log_payload("generate", prompt=prompt, **kwargs)

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)

        built_prompt = self._build_prompt(messages)
        prompt_tokens = self._estimate_tokens(built_prompt)

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
                f"Mock model response: input length {len(last_user_msg)} chars, temperature={temperature:.2f}.",
            ]
            generated_text = random.choice(templates)
        else:
            generated_text = (
                f"This is a simulated response. Your input was: {last_user_msg[:50]}"
            )

        max_chars = int(max_tokens * 3)
        if len(generated_text) > max_chars:
            generated_text = generated_text[:max_chars] + "..."

        completion_tokens = self._estimate_tokens(generated_text)

        return {
            "generated_text": generated_text,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    async def generate_stream(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming mock completion.

        Yields the full response in small random-sized chunks with
        simulated delays.

        Args:
            prompt: A plain text string or a list of message dicts.
            **kwargs: Generation parameters forwarded to generate().

        Yields:
            Text chunks as strings.
        """
        result = await self.generate(prompt, **kwargs)
        full_text = result["generated_text"]

        idx = 0
        while idx < len(full_text):
            chunk_size = random.randint(1, 4)
            chunk = full_text[idx : idx + chunk_size]
            yield chunk
            idx += chunk_size
            await asyncio.sleep(random.uniform(0.01, 0.03))

    async def generate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate mock tool/function call responses.

        Args:
            messages: The conversation history.
            tools: The tool definitions available to the model.
            **kwargs: Additional parameters including tool_choice.

        Returns:
            A list of tool call dicts with mock argument values.
        """
        self._log_payload(
            "generate_tool_calls", messages=messages, tools=tools, **kwargs
        )

        if not tools:
            return []
        tool_choice = kwargs.get("tool_choice", "auto")
        if tool_choice == "none":
            return []

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
            return []

        selected = candidates[0]
        func_def = selected.get("function", {})
        func_name = func_def.get("name", "unknown_function")
        params = func_def.get("parameters", {})
        props = params.get("properties", {})
        fake_args = {k: "mock_value" for k in props}

        return [
            {
                "id": f"call_{random.randbytes(6).hex()}",
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(fake_args),
                },
            }
        ]

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
        self._log_payload("embed", texts=texts, **kwargs)

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
