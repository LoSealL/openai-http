"""
Mock backend for testing and development.

Implements the Backend Protocol using mock data and simple token estimation.
No real model weights are loaded.
"""

import json
import random
import asyncio
from typing import AsyncGenerator, Optional, Any



class MockTokenizer:
    """
    Mock tokenizer using Unicode code point mapping.
    - encode: each char -> ord(char) + OFFSET
    - decode: token ID -> chr(token_id - OFFSET)
    Fully reversible, no vocabulary table needed.
    """

    OFFSET = 100

    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = 3
        self.vocab_size = 0x110000 + self.OFFSET + 10

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for char in text:
            tokens.append(ord(char) + self.OFFSET)
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        special = {self.bos_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id}
        chars = []
        for tid in token_ids:
            if skip_special_tokens and tid in special:
                continue
            if tid >= self.OFFSET:
                chars.append(chr(tid - self.OFFSET))
            else:
                chars.append("\ufffd")
        return "".join(chars)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        special_names = {
            self.bos_token_id: "<s>",
            self.eos_token_id: "</s>",
            self.pad_token_id: "<pad>",
            self.unk_token_id: "<unk>",
        }
        result = []
        for tid in token_ids:
            if tid in special_names:
                result.append(special_names[tid])
            elif tid >= self.OFFSET:
                char = chr(tid - self.OFFSET)
                if char.isprintable():
                    result.append(char)
                else:
                    result.append(f"<0x{tid - self.OFFSET:04X}>")
            else:
                result.append(f"<0x{tid:04X}>")
        return result


AVAILABLE_MODELS = [
    {"id": "mock-gpt", "object": "model", "created": 1677610602, "owned_by": "mock-org"},
    {"id": "mock-llama", "object": "model", "created": 1677610602, "owned_by": "mock-org"},
]


class MockTransformersBackend:
    """
    Mock backend implementing the Backend Protocol.
    Used for testing and development without real model weights.
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.tokenizer = MockTokenizer()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        count = 0
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                count += 1.5
            else:
                count += 0.25
        return int(max(1, count))

    @staticmethod
    def _build_prompt(messages: list[dict[str, str]]) -> str:
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
                f"[Random Mode] You said \"{last_user_msg[:20]}\"... interesting question!",
                f"Mock model response: input length {len(last_user_msg)} chars, temperature={temperature:.2f}.",
            ]
            generated_text = random.choice(templates)
        else:
            generated_text = f"This is a simulated response. Your input was: {last_user_msg[:50]}"

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
        if not tools:
            return []
        tool_choice = kwargs.get("tool_choice", "auto")
        if tool_choice == "none":
            return []

        if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            target_name = tool_choice.get("function", {}).get("name")
            candidates = [t for t in tools if t.get("function", {}).get("name") == target_name]
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
        dims = kwargs.get("dimensions", 1536)
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**31)
            rng = random.Random(seed)
            embedding = [rng.uniform(-1.0, 1.0) for _ in range(dims)]
            embeddings.append(embedding)
        return embeddings

    async def list_models(self) -> list[dict]:
        return AVAILABLE_MODELS

    async def get_model(self, model_id: str) -> Optional[dict]:
        for model in AVAILABLE_MODELS:
            if model["id"] == model_id:
                return model
        return None
