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
import io
import time
from typing import Any, AsyncGenerator, Optional

import torch
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

import openai_http
from openai_http.backends.base import BackendBase
from openai_http.parser import (
    ParserBase,
    available_parsers,
    get_parser,
    strip_special_tokens,
)


DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


class TransformersBackend(BackendBase):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        name: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        vision: bool = False,
        reasoning_parser: str = "qwen",
        toolcall_parser: str = "qwen",
    ):
        self.model_id = model_id
        self.model_name = name or model_id
        self.temperature = temperature
        self.top_p = top_p
        self.thinking = True
        self.max_tokens = max_tokens
        self.vision = vision
        self.reasoning_parser = reasoning_parser
        self.toolcall_parser = toolcall_parser
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.device = None
        self._loaded_at: int | None = None
        self._reasoning_parser: ParserBase | None = None
        self._tools_parser: ParserBase | None = None

    async def setup(self) -> None:
        def _load():
            if self.vision:
                processor, tokenizer, model = self._load_vlm()
            else:
                tokenizer, model = self._load_text()
                processor = tokenizer
            return tokenizer, processor, model, "cuda"

        (
            self.tokenizer,
            self.processor,
            self.model,
            self.device,
        ) = await asyncio.to_thread(_load)
        self._loaded_at = int(time.time())

        self._reasoning_parser = get_parser(self.reasoning_parser)
        self._tools_parser = get_parser(self.toolcall_parser)

    def _load_text(self) -> tuple[Any, Any]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        return tokenizer, model

    def _load_vlm(self) -> tuple[Any, Any, Any]:
        processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        tokenizer = processor.tokenizer
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model.eval()
        return processor, tokenizer, model

    async def teardown(self) -> None:
        torch.cuda.empty_cache()
        import gc

        gc.collect()

    def _prepare_inputs(
        self,
        prompt: list[dict[str, Any]],
        thinking: bool,
        **extra: Any,
    ):
        messages = prompt

        images: list[Any] = []
        if self.vision:
            messages, images = self._collect_images(messages)
        text = self._apply_template(self.processor, messages, thinking, **extra)
        print(f"[prepare_inputs] {text}")
        proc_kwargs: dict[str, Any] = {"text": text, "return_tensors": "pt"}
        if images:
            proc_kwargs["images"] = images
        inputs = self.processor(**proc_kwargs).to(self.device)

        prompt_len = inputs["input_ids"].shape[1]
        return inputs, prompt_len, messages

    @staticmethod
    def _collect_images(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        """Extract base64 data-URI images from OpenAI multimodal messages.

        Converts OpenAI ``image_url`` content parts into the transformers
        ``{"type": "image"}`` form and materializes each into a PIL image.
        Only inline base64 data URIs are supported; remote URLs are dropped.

        Args:
            messages: OpenAI-style chat messages, possibly with list content.

        Returns:
            A tuple of ``(normalized_messages, images)``.
        """
        images: list[Any] = []
        normalized: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                normalized.append(msg)
                continue
            new_parts: list[dict[str, Any]] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = ""
                    iu = part.get("image_url")
                    if isinstance(iu, dict):
                        url = iu.get("url", "") or ""
                    elif isinstance(iu, str):
                        url = iu
                    img = TransformersBackend._load_data_uri_image(url)
                    if img is not None:
                        images.append(img)
                        new_parts.append({"type": "image"})
                    else:
                        new_parts.append(
                            {
                                "type": "text",
                                "text": "[unsupported image]",
                            }
                        )
                else:
                    new_parts.append(part)
            normalized.append({**msg, "content": new_parts})
        return normalized, images

    @staticmethod
    def _load_data_uri_image(url: str) -> Optional[Any]:
        """Decode a ``data:image/...;base64,...`` URI into a PIL Image.

        Returns None if the URL is not a decodable base64 data URI.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None

        try:
            import base64

            raw = base64.b64decode(url.split(",", 1)[1])
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def _apply_template(
        self,
        template_obj: Any,
        messages: list[dict[str, Any]],
        thinking: bool,
        **extra: Any,
    ) -> str:
        """Apply a chat template, tolerating templates without enable_thinking.

        Args:
            template_obj: A tokenizer or processor exposing ``apply_chat_template``.
            messages: The chat messages.
            thinking: Whether to request ``enable_thinking`` in the template.
            **extra: Additional template kwargs (e.g. ``tools``, ``tool_choice``).

        Returns:
            The rendered prompt string.
        """
        common: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
        try:
            return template_obj.apply_chat_template(
                messages, enable_thinking=thinking, **common, **extra
            )
        except (TypeError, ValueError):
            return template_obj.apply_chat_template(messages, **common, **extra)

    def _sample_flags(
        self, temperature: float, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Build sampling kwargs (do_sample/temperature/top_p)."""
        flags: dict[str, Any] = {"do_sample": temperature > 0}
        if temperature > 0:
            flags["temperature"] = temperature
            if "top_p" in kwargs:
                flags["top_p"] = float(kwargs["top_p"])
        return flags

    def _strip_specials(self, text: str) -> str:
        """Remove the tokenizer's special tokens from free text.

        Wraps :func:`openai_http.parser.strip_special_tokens` with this
        backend's tokenizer; used after parsing when a parser required
        the specials to be kept through decoding.
        """
        return strip_special_tokens(text, self.tokenizer.all_special_tokens)

    def _generate_simple(
        self,
        inputs: dict[str, Any],
        prompt_len: int,
        max_new_tokens: int,
        temperature: float,
        kwargs: dict[str, Any],
        skip_special_tokens: bool,
    ) -> tuple[str, int]:
        """Single-pass generation without a reasoning budget."""
        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            **self._sample_flags(temperature, kwargs),
        }
        with torch.no_grad():
            output_ids = self.model.generate(**gen_kwargs)
        new_ids = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(new_ids, skip_special_tokens=skip_special_tokens)
        return text, int(new_ids.shape[0])

    async def generate(
        self,
        prompt: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict:
        thinking = bool(kwargs.get("enable_thinking", self.thinking))
        max_new_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))

        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        tc_extra: dict[str, Any] = {}
        if tools and tool_choice != "none":
            tc = tool_choice
            if isinstance(tc, dict):
                name = tc.get("function", {}).get("name")
                if name:
                    tc = name
            tc_extra = {
                "tools": tools,
                "tool_choice": tc if tc != "auto" else None,
            }

        inputs, prompt_len, _ = await asyncio.to_thread(
            self._prepare_inputs, prompt, thinking, **tc_extra
        )

        # Decode adaptively: when a parser in use needs special tokens
        # preserved (e.g. CpmParser, whose <function/<param tags are
        # special tokens), decode with skip_special_tokens=False and
        # strip the leftover framing specials from the final free text
        # only after parsing completes.
        assert self._reasoning_parser is not None and self._tools_parser is not None
        keep_specials = (
            self._reasoning_parser.REQUIRES_SPECIAL_TOKENS
            or self._tools_parser.REQUIRES_SPECIAL_TOKENS
        )
        generated_text, completion_tokens = await asyncio.to_thread(
            self._generate_simple,
            inputs,
            prompt_len,
            max_new_tokens,
            temperature,
            kwargs,
            not keep_specials,
        )
        print(f"[generate] {generated_text}")

        if tc_extra:
            tool_result = self._tools_parser.parse_tool_calls(generated_text)
            if tool_result.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in tool_result.tool_calls
                ]
                # Tool-call arguments are already structured; only the
                # residual content text needs special-token cleanup.
                content = tool_result.content
                if keep_specials:
                    content = self._strip_specials(content)
                return {
                    "generated_text": content or None,
                    "tool_calls": tool_calls,
                    "finish_reason": "tool_calls",
                    "usage": {
                        "prompt_tokens": prompt_len,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_len + completion_tokens,
                    },
                }

        reasoning_content = None
        if thinking:
            r = self._reasoning_parser.parse_reasoning(generated_text)
            reasoning_content = r.reasoning
            generated_text = r.content
        if keep_specials:
            if reasoning_content is not None:
                reasoning_content = self._strip_specials(reasoning_content)
            generated_text = self._strip_specials(generated_text)
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
        prompt: list[dict[str, Any]],
        **kwargs: Any,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        raise NotImplementedError("Streaming is not supported by this backend")

    async def list_models(self) -> list[dict]:
        return [
            {
                "id": self.model_name,
                "object": "model",
                "created": self._loaded_at or int(time.time()),
                "owned_by": "huggingface",
            }
        ]

    async def get_model(self, model_id: str) -> Optional[dict]:
        if model_id != self.model_name:
            return None
        return {
            "id": self.model_name,
            "object": "model",
            "created": self._loaded_at or int(time.time()),
            "owned_by": "huggingface",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve a HuggingFace Transformers model via the OpenAI-compatible API."
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        "-p",
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
        "--top-p",
        type=float,
        default=1.0,
        help="Nucleus sampling threshold (default: 1.0 = off)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens per response (default: 1024)",
    )
    parser.add_argument(
        "--reasoning-parser",
        default="qwen",
        choices=available_parsers(),
        help="Reasoning parser name (registered in openai_http.parser). Default: qwen.",
    )
    parser.add_argument(
        "--tool-call-parser",
        default="qwen",
        choices=available_parsers(),
        help="Tool-call parser name (registered in openai_http.parser). Default: qwen.",
    )
    args = parser.parse_args()

    backend = TransformersBackend(
        model_id=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_parser=args.reasoning_parser,
        toolcall_parser=args.tool_call_parser,
    )
    openai_http.run_server(backend=backend, host=args.host, port=args.port)
