"""
A real HuggingFace Transformers backend for openai_http.

Loads a causal LM (default: Qwen/Qwen2.5-0.5B-Instruct) and serves
chat completions with proper chat template handling and streaming.

Run:
    uv run python examples/transformers-backend/transformers_backend.py

Dependencies:
    uv pip install torch transformers accelerate
"""

import argparse
import asyncio
import json
import re
import time
import uuid
from threading import Thread
from typing import Any, AsyncGenerator, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

import openai_http
from openai_http.backends.base import BackendBase


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


class TransformersBackend(BackendBase):
    def __init__(self, model_id: str = DEFAULT_MODEL, temperature: float = 0.0, thinking: bool = False, max_tokens: int = 1024):
        self.model_id = model_id
        self.temperature = temperature
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.tokenizer = None
        self.model = None
        self.device = None
        self._loaded_at: int | None = None

    async def setup(self) -> None:
        def _load():
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
            dtype = torch.bfloat16 if device != "cpu" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                device_map=device if device != "mps" else None,
                trust_remote_code=True,
            )
            if device == "mps":
                model = model.to(device)
            model.eval()
            return tokenizer, model, device

        self.tokenizer, self.model, self.device = await asyncio.to_thread(_load)
        self._loaded_at = int(time.time())

    async def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc

        gc.collect()

    def _prepare_inputs(self, prompt: str | list[dict[str, str]], max_new_tokens: int):
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[1]
        return inputs, prompt_len, messages

    def _decode_output(self, input_ids_len: int, output_ids) -> tuple[str, int]:
        new_ids = output_ids[0, input_ids_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return text, len(new_ids)

    @staticmethod
    def _parse_reasoning(text: str) -> tuple[str | None, str]:
        """Parse reasoning from generated text.

        The opening <think> tag is part of the chat template prompt, so the
        generated text only contains the reasoning followed by </think>
        and the actual answer. We split on </think> only.

        Args:
            text: Raw generated text (reasoning + </think> + answer).

        Returns:
            A tuple of (reasoning_content, content). reasoning_content
            is None if no </think> tag is found.
        """
        end_tag = "</think>"
        idx = text.find(end_tag)
        if idx == -1:
            return None, text
        reasoning = text[:idx]
        content = text[idx + len(end_tag):].lstrip("\n")
        return reasoning if reasoning else None, content

    async def generate(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> dict:
        max_new_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))

        def _sync_generate():
            inputs, prompt_len, _ = self._prepare_inputs(prompt, max_new_tokens)
            gen_kwargs = {
                **inputs,
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                if "top_p" in kwargs:
                    gen_kwargs["top_p"] = float(kwargs["top_p"])
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)
            return inputs, prompt_len, output_ids

        inputs, prompt_len, output_ids = await asyncio.to_thread(_sync_generate)
        generated_text, completion_tokens = self._decode_output(prompt_len, output_ids)

        reasoning_content: str | None = None
        if self.thinking:
            reasoning_content, generated_text = self._parse_reasoning(generated_text)

        finish_reason = "length" if completion_tokens >= max_new_tokens else "stop"

        return {
            "generated_text": generated_text,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_len,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_len + completion_tokens,
            },
        }

    async def generate_stream(
        self,
        prompt: str | list[dict[str, str]],
        **kwargs: Any,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """Generate a streaming completion with reasoning support.

        When thinking is enabled, yields reasoning chunks as typed
        dicts before content chunks. Otherwise yields content dicts.

        Args:
            prompt: A plain text string or a list of message dicts.
            **kwargs: Generation parameters.

        Yields:
            Typed dicts: {"type": "reasoning", "content": ...} or
            {"type": "content", "content": ...}.
        """
        max_new_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))

        inputs, _, _ = self._prepare_inputs(prompt, max_new_tokens)
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            if "top_p" in kwargs:
                gen_kwargs["top_p"] = float(kwargs["top_p"])

        chunk_queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _producer():
            gen_thread = Thread(target=self.model.generate, kwargs=dict(gen_kwargs))
            gen_thread.start()
            try:
                for text in streamer:
                    if text:
                        loop.call_soon_threadsafe(chunk_queue.put_nowait, text)
            finally:
                gen_thread.join()
                loop.call_soon_threadsafe(chunk_queue.put_nowait, None)

        loop.run_in_executor(None, _producer)

        state = "thinking" if self.thinking else "answering"
        buf = ""
        end_think_tag = "</think>"
        token_count = 0

        while True:
            chunk = await chunk_queue.get()
            if chunk is None:
                break
            token_count += 1
            buf += chunk

            while buf:
                if state == "thinking":
                    idx = buf.find(end_think_tag)
                    if idx != -1:
                        reasoning = buf[:idx]
                        if reasoning:
                            yield {"type": "reasoning", "content": reasoning}
                        buf = buf[idx + len(end_think_tag):]
                        state = "answering"
                    else:
                        safe = len(buf) - (len(end_think_tag) - 1)
                        if safe > 0:
                            yield {"type": "reasoning", "content": buf[:safe]}
                            buf = buf[safe:]
                        break

                else:
                    yield {"type": "content", "content": buf}
                    buf = ""

        if buf:
            if state == "thinking":
                yield {"type": "reasoning", "content": buf}
            else:
                yield {"type": "content", "content": buf}

        if token_count >= max_new_tokens:
            yield {"type": "finish", "reason": "length"}

    async def generate_tool_calls(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        tool_choice = kwargs.get("tool_choice", "auto")
        if tool_choice == "none":
            return []

        tc_for_template = tool_choice
        if isinstance(tool_choice, dict):
            func_name = tool_choice.get("function", {}).get("name")
            if func_name:
                tc_for_template = func_name

        def _sync_generate():
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tool_choice=tc_for_template if tc_for_template != "auto" else None,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.thinking,
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

            gen_kwargs = {
                **inputs,
                "max_new_tokens": 512,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
            with torch.no_grad():
                output_ids = self.model.generate(**gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            generated_text = self.tokenizer.decode(
                output_ids[0, prompt_len:],
                skip_special_tokens=True,
            )
            return generated_text

        generated_text = await asyncio.to_thread(_sync_generate)
        return self._parse_tool_calls(generated_text)

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return []

        tool_calls = []
        for match in matches:
            try:
                call = json.loads(match.strip())
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(call.get("arguments", {})),
                        },
                    }
                )
            except json.JSONDecodeError:
                continue

        return tool_calls

    async def list_models(self) -> list[dict]:
        return [
            {
                "id": self.model_id,
                "object": "model",
                "created": self._loaded_at or int(time.time()),
                "owned_by": "huggingface",
            }
        ]

    async def get_model(self, model_id: str) -> Optional[dict]:
        if model_id != self.model_id:
            return None
        return {
            "id": self.model_id,
            "object": "model",
            "created": self._loaded_at or int(time.time()),
            "owned_by": "huggingface",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve a HuggingFace Transformers model via the OpenAI-compatible API."
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable reasoning tokens via <think> tags (default: off)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens per response (default: 1024)",
    )
    args = parser.parse_args()

    backend = TransformersBackend(
        model_id=args.model,
        temperature=args.temperature,
        thinking=args.thinking,
        max_tokens=args.max_tokens,
    )
    openai_http.run_server(backend=backend, host=args.host, port=args.port)
