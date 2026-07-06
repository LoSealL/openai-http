"""
A backend for openai_http powered by ONNX Runtime.

Loads ONNX exported models (e.g. from onnx-community) and serves
chat completions with a manual token-by-token generation loop.

Supports vision-language inputs when --vision is enabled.

Run:
    python examples/onnxruntime-backend/onnxruntime_backend.py

Dependencies:
    pip install onnxruntime transformers numpy Pillow
    # or for GPU:
    pip install onnxruntime-gpu transformers numpy Pillow
"""

import argparse
import asyncio
import base64
import io
import os
import time
from typing import Any, AsyncGenerator, Optional

import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import AutoProcessor, PreTrainedTokenizerFast

import openai_http
from openai_http.backends.base import BackendBase
from openai_http.parser import (
    ParserBase,
    available_parsers,
    get_parser,
    strip_special_tokens,
)

DEFAULT_MODEL = "onnx-community/Qwen3.5-0.8B-ONNX-OPT"


class Sampler:
    """Greedy and temperature/top-p sampling for logits."""

    def __init__(self, temperature: float = 0.0, top_p: float = 1.0):
        self.temperature = temperature
        self.top_p = top_p

    def sample(self, logits: np.ndarray) -> int:
        """Sample a token from logits (shape: [vocab_size])."""
        if self.temperature <= 0:
            return int(np.argmax(logits))

        # Apply temperature
        logits = logits / self.temperature

        # Convert to probabilities
        probs = self._softmax(logits)

        # Apply top-p
        if self.top_p < 1.0:
            probs = self._top_p_filter(probs, self.top_p)

        # Sample
        return int(np.random.choice(len(probs), p=probs))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    @staticmethod
    def _top_p_filter(probs: np.ndarray, top_p: float) -> np.ndarray:
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        cumulative_probs = np.cumsum(sorted_probs)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        probs = probs / probs.sum()
        return probs


class OnnxRuntimeBackend(BackendBase):
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        name: str | None = None,
        temperature: float = 0.0,
        *,
        top_p: float = 1.0,
        max_tokens: int = 1024,
        provider: str = "CUDAExecutionProvider",
        dtype: str | None = None,
        thinking: bool = True,
        reasoning_parser: str = "qwen",
        toolcall_parser: str = "qwen",
        vision: bool = False,
    ):
        self.model_id = model_id
        self.model_name = name or model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.provider = provider
        self.dtype = dtype
        self.thinking = thinking
        self.reasoning_parser = reasoning_parser
        self.toolcall_parser = toolcall_parser
        self.vision = vision

        self.tokenizer = None
        self.processor = None
        self.embed_session = None
        self.vision_session = None
        self.decoder_session = None
        self._loaded_at: int | None = None
        self.num_layers = 24
        self.hidden_size = 1024
        self._reasoning_parser: ParserBase | None = None
        self._tools_parser: ParserBase | None = None
        self.image_pad_id = 248056

        # EOS tokens from generation_config.json
        self.eos_token_ids = {248046, 248044}

    def _load_onnx_model(self, onnx_dir: str, base_name: str, providers: list[str]) -> ort.InferenceSession:
        """Load an ONNX model, optionally with a dtype suffix.

        Tries ``{base_name}_{dtype}.onnx`` first, then falls back to
        ``{base_name}.onnx`` if the dtype-specific file does not exist.
        """
        if self.dtype:
            dtype_path = os.path.join(onnx_dir, f"{base_name}_{self.dtype}.onnx")
            if os.path.exists(dtype_path):
                print(f"  Loading {base_name}_{self.dtype}.onnx")
                return ort.InferenceSession(dtype_path, providers=providers)
            print(f"  Warning: {base_name}_{self.dtype}.onnx not found, falling back to {base_name}.onnx")
        base_path = os.path.join(onnx_dir, f"{base_name}.onnx")
        print(f"  Loading {base_name}.onnx")
        return ort.InferenceSession(base_path, providers=providers)

    async def setup(self) -> None:
        def _load():
            # Load tokenizer / processor
            if self.vision:
                processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                tokenizer = processor.tokenizer
            else:
                tokenizer = PreTrainedTokenizerFast.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                )
                processor = None

            # Resolve local model path if cached
            model_path = self._resolve_model_path()
            onnx_dir = os.path.join(model_path, "onnx")

            if not os.path.exists(onnx_dir):
                raise RuntimeError(f"ONNX directory not found: {onnx_dir}")

            # Determine available provider
            available_providers = ort.get_available_providers()
            if self.provider in available_providers:
                providers = [self.provider]
            else:
                print(f"Warning: {self.provider} not available. Using CPU.")
                providers = ["CPUExecutionProvider"]

            print(f"Using ONNX Runtime providers: {providers}")
            if self.dtype:
                print(f"Using dtype variant: {self.dtype}")

            # Load embed tokens model
            embed_session = self._load_onnx_model(onnx_dir, "embed_tokens", providers)

            # Load vision encoder if vision mode
            vision_session = None
            if self.vision:
                vision_path = os.path.join(onnx_dir, "vision_encoder.onnx")
                if self.dtype:
                    dtype_vision = os.path.join(onnx_dir, f"vision_encoder_{self.dtype}.onnx")
                    if os.path.exists(dtype_vision):
                        vision_path = dtype_vision
                if os.path.exists(vision_path):
                    print(f"  Loading {os.path.basename(vision_path)}")
                    vision_session = ort.InferenceSession(vision_path, providers=providers)
                else:
                    print("  Warning: vision_encoder.onnx not found, vision disabled")
                    self.vision = False

            # Load decoder model
            decoder_session = self._load_onnx_model(onnx_dir, "decoder_model_merged", providers)

            return tokenizer, processor, embed_session, vision_session, decoder_session

        self.tokenizer, self.processor, self.embed_session, self.vision_session, self.decoder_session = (
            await asyncio.to_thread(_load)
        )
        self._loaded_at = int(time.time())

        self._reasoning_parser = get_parser(self.reasoning_parser)
        self._tools_parser = get_parser(self.toolcall_parser)

        print(f"ONNX Runtime backend loaded: {self.model_id}")

    def _strip_specials(self, text: str) -> str:
        """Remove the tokenizer's special tokens from free text.

        Wraps :func:`openai_http.parser.strip_special_tokens` with this
        backend's tokenizer; used after parsing when a parser required
        the specials to be kept through decoding.
        """
        return strip_special_tokens(text, self.tokenizer.all_special_tokens)

    def _resolve_model_path(self) -> str:
        """Resolve the local path to the cached model."""
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hub_dir = os.path.join(hf_home, "hub")

        cache_name = f"models--{self.model_id.replace('/', '--')}"
        cache_dir = os.path.join(hub_dir, cache_name)

        if os.path.exists(cache_dir):
            snapshots_dir = os.path.join(cache_dir, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    return os.path.join(snapshots_dir, snapshots[0])

        if os.path.exists(self.model_id):
            return self.model_id

        raise RuntimeError(f"Model not found in cache: {self.model_id}")

    def _init_past_states(self, batch_size: int = 1) -> dict[str, np.ndarray]:
        """Initialize empty past states for the first forward pass."""
        past_states = {}
        for layer_idx in range(self.num_layers):
            if layer_idx % 4 == 3:  # Full attention layers
                past_states[f"past_key_values.{layer_idx}.key"] = np.zeros(
                    (batch_size, 2, 0, 256), dtype=np.float32
                )
                past_states[f"past_key_values.{layer_idx}.value"] = np.zeros(
                    (batch_size, 2, 0, 256), dtype=np.float32
                )
            else:  # Linear attention layers
                past_states[f"past_conv.{layer_idx}"] = np.zeros(
                    (batch_size, 6144, 3), dtype=np.float32
                )
                past_states[f"past_recurrent.{layer_idx}"] = np.zeros(
                    (batch_size, 16, 128, 128), dtype=np.float32
                )
        return past_states

    def _update_past_states(
        self, past_states: dict[str, np.ndarray], decoder_outputs: list[np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Update past states from decoder outputs."""
        output_idx = 1
        for layer_idx in range(self.num_layers):
            if layer_idx % 4 == 3:  # Full attention layers
                past_states[f"past_key_values.{layer_idx}.key"] = decoder_outputs[output_idx]
                past_states[f"past_key_values.{layer_idx}.value"] = decoder_outputs[output_idx + 1]
                output_idx += 2
            else:  # Linear attention layers
                past_states[f"past_conv.{layer_idx}"] = decoder_outputs[output_idx]
                past_states[f"past_recurrent.{layer_idx}"] = decoder_outputs[output_idx + 1]
                output_idx += 2
        return past_states

    def _build_position_ids(self, positions: np.ndarray) -> np.ndarray:
        """Build M-RoPE position_ids [3, 1, seq_len]."""
        position_ids = np.stack([positions, positions, positions])
        return np.expand_dims(position_ids, 1)  # [3, 1, seq_len]

    def _run_decoder(
        self,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        past_states: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Run a single decoder forward pass."""
        num_logits_to_keep = np.array(1, dtype=np.int64)

        decoder_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "num_logits_to_keep": num_logits_to_keep,
            **past_states,
        }

        decoder_outputs = self.decoder_session.run(None, decoder_inputs)
        logits = decoder_outputs[0]
        past_states = self._update_past_states(past_states, decoder_outputs)

        return logits, past_states

    @staticmethod
    def _collect_images(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Image.Image]]:
        """Extract base64 data-URI images from OpenAI multimodal messages.

        Converts OpenAI ``image_url`` content parts into the transformers
        ``{"type": "image"}`` form and materializes each into a PIL image.
        Only inline base64 data URIs are supported; remote URLs are dropped.

        Args:
            messages: OpenAI-style chat messages, possibly with list content.

        Returns:
            A tuple of ``(normalized_messages, images)``.
        """
        images: list[Image.Image] = []
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
                    img = OnnxRuntimeBackend._load_data_uri_image(url)
                    if img is not None:
                        images.append(img)
                        new_parts.append({"type": "image"})
                    else:
                        new_parts.append({"type": "text", "text": "[unsupported image]"})
                else:
                    new_parts.append(part)
            normalized.append({**msg, "content": new_parts})
        return normalized, images

    @staticmethod
    def _load_data_uri_image(url: str) -> Optional[Image.Image]:
        """Decode a ``data:image/...;base64,...`` URI into a PIL Image.

        Returns None if the URL is not a decodable base64 data URI.
        """
        if not isinstance(url, str) or not url.startswith("data:"):
            return None
        _, _, data = url.partition(",")
        try:
            raw = base64.b64decode(data, validate=True)
        except Exception:
            return None
        try:
            return Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            return None

    def _merge_vision_embeddings(
        self,
        inputs_embeds: np.ndarray,
        input_ids: np.ndarray,
        image_features: np.ndarray,
    ) -> np.ndarray:
        """Replace image pad token embeddings with vision features.

        Args:
            inputs_embeds: Text embeddings [batch, seq_len, hidden_size].
            input_ids: Token IDs [batch, seq_len].
            image_features: Image features [num_features, hidden_size].

        Returns:
            Updated embeddings with image features inserted.
        """
        # Find image pad positions
        image_pad_positions = np.where(input_ids[0] == self.image_pad_id)[0]
        num_features = image_features.shape[0]
        num_pads = len(image_pad_positions)

        if num_pads == 0 or num_features == 0:
            return inputs_embeds

        if num_pads != num_features:
            # Pad or truncate to match
            if num_features > num_pads:
                image_features = image_features[:num_pads]
            else:
                # Pad with zeros if fewer features than pad positions
                padding = np.zeros((num_pads - num_features, image_features.shape[1]), dtype=image_features.dtype)
                image_features = np.concatenate([image_features, padding], axis=0)

        # Replace embeddings at image pad positions
        inputs_embeds = inputs_embeds.copy()
        for i, pos in enumerate(image_pad_positions):
            inputs_embeds[0, pos] = image_features[i]

        return inputs_embeds

    def _prepare_inputs(
        self,
        messages: list[dict[str, Any]],
        thinking: bool,
        **extra: Any,
    ) -> tuple[np.ndarray, int, str]:
        """Apply chat template, tokenize, embed, and optionally process vision.

        Returns:
            A tuple of ``(inputs_embeds, prompt_len, chat_text)``.
        """
        # Collect images if any
        normalized_messages, images = self._collect_images(messages)
        has_images = len(images) > 0 and self.vision and self.vision_session is not None

        chat_text = self._apply_template(normalized_messages, thinking, **extra)

        if has_images and self.processor is not None:
            # Use processor for multimodal inputs
            proc_inputs = self.processor(
                text=[chat_text],
                images=images,
                return_tensors="np",
            )
            input_ids = proc_inputs["input_ids"]
            prompt_len = input_ids.shape[1]

            # Get text embeddings
            embed_outputs = self.embed_session.run(None, {"input_ids": input_ids})
            inputs_embeds = embed_outputs[0]

            # Run vision encoder
            pixel_values = proc_inputs["pixel_values"].astype(np.float32)
            image_grid_thw = proc_inputs["image_grid_thw"].astype(np.int64)
            vision_outputs = self.vision_session.run(None, {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            })
            image_features = vision_outputs[0]

            # Merge image features into text embeddings
            inputs_embeds = self._merge_vision_embeddings(inputs_embeds, input_ids, image_features)
        else:
            # Text-only path
            input_ids = np.array([self.tokenizer.encode(chat_text)], dtype=np.int64)
            prompt_len = input_ids.shape[1]
            embed_outputs = self.embed_session.run(None, {"input_ids": input_ids})
            inputs_embeds = embed_outputs[0]

        return inputs_embeds, prompt_len, chat_text

    def _apply_template(
        self,
        messages: list[dict[str, Any]],
        thinking: bool,
        **extra: Any,
    ) -> str:
        """Apply a chat template, tolerating templates without enable_thinking.

        Args:
            messages: The chat messages.
            thinking: Whether to request ``enable_thinking`` in the template.
            **extra: Additional template kwargs (e.g. ``tools``, ``tool_choice``).

        Returns:
            The rendered prompt string.
        """
        common: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=thinking, **common, **extra
            )
        except (TypeError, ValueError):
            return self.tokenizer.apply_chat_template(messages, **common, **extra)

    def _generate(
        self,
        inputs_embeds: np.ndarray,
        prompt_len: int,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        skip_special_tokens: bool,
    ) -> tuple[str, int, int]:
        """Run the full generation loop."""
        sampler = Sampler(temperature=temperature, top_p=top_p)

        # Build position_ids for prompt
        positions = np.arange(prompt_len, dtype=np.int64)
        position_ids = self._build_position_ids(positions)

        # Attention mask for prompt
        attention_mask = np.ones((1, prompt_len), dtype=np.int64)

        # Initialize past states
        past_states = self._init_past_states(batch_size=1)

        # Prefill (first pass)
        logits, past_states = self._run_decoder(
            inputs_embeds, attention_mask, position_ids, past_states
        )
        next_token_id = sampler.sample(logits[0, -1, :])
        generated_tokens = [next_token_id]

        # Decode loop
        total_seq_len = prompt_len + 1
        for _ in range(max_new_tokens - 1):
            if next_token_id in self.eos_token_ids:
                break

            # Embed single token
            next_token_array = np.array([[next_token_id]], dtype=np.int64)
            embed_outputs = self.embed_session.run(None, {"input_ids": next_token_array})
            inputs_embeds = embed_outputs[0]

            # Position for new token
            position = np.array([total_seq_len - 1], dtype=np.int64)
            position_ids = self._build_position_ids(position)

            # Attention mask
            attention_mask = np.ones((1, total_seq_len), dtype=np.int64)

            # Decode step
            logits, past_states = self._run_decoder(
                inputs_embeds, attention_mask, position_ids, past_states
            )
            next_token_id = sampler.sample(logits[0, -1, :])
            generated_tokens.append(next_token_id)

            total_seq_len += 1

        # Decode result (keep special tokens for reasoning/tool parsing).
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=skip_special_tokens
        )
        completion_tokens = len(generated_tokens)

        print(f"[_generate] {generated_text}")

        return generated_text, prompt_len, completion_tokens

    async def generate(
        self,
        prompt: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict:
        thinking = bool(kwargs.get("enable_thinking", self.thinking))
        max_new_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        top_p = float(kwargs.get("top_p", self.top_p))

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

        inputs_embeds, prompt_len, chat_text = await asyncio.to_thread(
            self._prepare_inputs, prompt, thinking, **tc_extra
        )

        print(f"[generate] prompt:\n{chat_text}")

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
        generated_text, _, completion_tokens = await asyncio.to_thread(
            self._generate,
            inputs_embeds,
            prompt_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            skip_special_tokens=not keep_specials,
        )

        print(f"[generate] generated:\n{generated_text}")

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
        thinking = bool(kwargs.get("enable_thinking", self.thinking))
        max_new_tokens = int(kwargs.get("max_tokens", self.max_tokens))
        temperature = float(kwargs.get("temperature", self.temperature))
        top_p = float(kwargs.get("top_p", self.top_p))

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

        inputs_embeds, prompt_len, chat_text = await asyncio.to_thread(
            self._prepare_inputs, prompt, thinking, **tc_extra
        )

        print(f"[stream] prompt:\n{chat_text}")

        sampler = Sampler(temperature=temperature, top_p=top_p)

        # Prefill
        logits, past_states = self._run_decoder(
            inputs_embeds,
            np.ones((1, prompt_len), dtype=np.int64),
            self._build_position_ids(np.arange(prompt_len, dtype=np.int64)),
            self._init_past_states(batch_size=1),
        )
        next_token_id = sampler.sample(logits[0, -1, :])

        total_seq_len = prompt_len + 1
        completion_tokens = 1

        # Accumulate tokens for proper multi-byte character handling
        accumulated_tokens = [next_token_id]
        yielded_text = ""

        def _yield_new_text():
            nonlocal yielded_text
            full_text = self.tokenizer.decode(accumulated_tokens, skip_special_tokens=False)

            # Find common prefix between yielded_text and full_text
            common_len = 0
            min_len = min(len(yielded_text), len(full_text))
            for i in range(min_len):
                if yielded_text[i] == full_text[i]:
                    common_len += 1
                else:
                    break

            new_text = full_text[common_len:]

            # If new_text ends with replacement char, it might be incomplete
            if new_text and new_text.endswith('\ufffd'):
                clean_text = new_text.rstrip('\ufffd')
                yielded_text += clean_text
                return clean_text if clean_text else None

            yielded_text += new_text
            return new_text if new_text else None

        # Yield first token(s)
        new_text = _yield_new_text()
        if new_text:
            yield new_text

        # Decode loop
        for _ in range(max_new_tokens - 1):
            if next_token_id in self.eos_token_ids:
                print(f"[stream] finished:\n{yielded_text}")
                yield {"type": "finish", "reason": "stop"}
                return

            # Embed single token
            next_token_array = np.array([[next_token_id]], dtype=np.int64)
            embed_outputs = self.embed_session.run(None, {"input_ids": next_token_array})
            inputs_embeds = embed_outputs[0]

            # Position for new token
            position = np.array([total_seq_len - 1], dtype=np.int64)
            position_ids = self._build_position_ids(position)

            # Attention mask
            attention_mask = np.ones((1, total_seq_len), dtype=np.int64)

            # Decode step
            logits, past_states = self._run_decoder(
                inputs_embeds, attention_mask, position_ids, past_states
            )
            next_token_id = sampler.sample(logits[0, -1, :])
            accumulated_tokens.append(next_token_id)

            new_text = _yield_new_text()
            if new_text:
                yield new_text

            total_seq_len += 1
            completion_tokens += 1

        print(f"[stream] finished:\n{yielded_text}")
        yield {"type": "finish", "reason": "length"}

    async def list_models(self) -> list[dict]:
        return [
            {
                "id": self.model_name,
                "object": "model",
                "created": self._loaded_at or int(time.time()),
                "owned_by": "onnx-community",
            }
        ]

    async def get_model(self, model_id: str) -> Optional[dict]:
        if model_id != self.model_name:
            return None
        return {
            "id": self.model_name,
            "object": "model",
            "created": self._loaded_at or int(time.time()),
            "owned_by": "onnx-community",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Serve an ONNX model via ONNX Runtime with the OpenAI-compatible API."
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--name",
        "-n",
        default=None,
        help="Custom model name to serve (overrides --model for API responses)",
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
        "--provider",
        default="CUDAExecutionProvider",
        choices=["CUDAExecutionProvider", "CPUExecutionProvider"],
        help="ONNX Runtime execution provider (default: CUDAExecutionProvider)",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=["fp16", "q4", "q8", "q4f16", "base"],
        help="ONNX model dtype variant. When set, loads *_fp16.onnx, *_q4.onnx, "
        "*_q8.onnx, or *_q4f16.onnx variants. Falls back to the base model "
        "if the dtype-specific file does not exist. (default: base)",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=True,
        help="Enable thinking mode (default: True)",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_false",
        dest="thinking",
        help="Disable thinking mode",
    )
    parser.add_argument(
        "--reasoning-parser",
        default="qwen",
        choices=available_parsers(),
        help="Reasoning parser name (registered in openai_http.parser). "
        "Default: qwen.",
    )
    parser.add_argument(
        "--tool-call-parser",
        default="qwen",
        choices=available_parsers(),
        help="Tool-call parser name (registered in openai_http.parser). "
        "Default: qwen.",
    )
    parser.add_argument(
        "--vision",
        "-vl",
        action="store_true",
        help="Load as a vision-language model to accept image inputs (base64 data URIs)",
    )
    args = parser.parse_args()

    # Handle 'base' dtype as None
    _dtype = args.dtype if args.dtype and args.dtype != "base" else None

    backend = OnnxRuntimeBackend(
        model_id=args.model,
        name=args.name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        provider=args.provider,
        dtype=_dtype,
        thinking=args.thinking,
        reasoning_parser=args.reasoning_parser,
        toolcall_parser=args.tool_call_parser,
        vision=args.vision,
    )
    openai_http.run_server(backend=backend, host=args.host, port=args.port)
