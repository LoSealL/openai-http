# Transformers Backend Example

A working `openai_http` backend powered by HuggingFace Transformers.
Loads a causal language model and serves chat completions with proper
chat template handling, streaming, and real token usage counts.

Default model: **Qwen/Qwen3.5-0.8B** (~1.6 GB download).

## Install

These dependencies are *not* tracked by the main `openai_http` package
(would bloat it). Install them manually:

```bash
uv pip install torch transformers accelerate
```

For vision-language models (the `--vision` flag) you also need Pillow
(usually pulled in by `transformers`):

```bash
uv pip install Pillow
```

## Run

From the repo root:

```bash
# Default: Qwen3.5-0.8B, greedy sampling, thinking on
uv run python examples/transformers-backend/transformers_backend.py

# Custom model, temperature, and a reasoning budget
uv run python examples/transformers-backend/transformers_backend.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --temperature 0.7 \
  --reasoning-budget 40 \
  --max-tokens 2048 \
  --port 8000

# See all options
uv run python examples/transformers-backend/transformers_backend.py --help
```

The server listens on `http://0.0.0.0:8000`. The first request triggers
model download (cached in `~/.cache/huggingface/`).

## Try it

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

## Notes

- Uses real token counts from the tokenizer for the `usage` field.
- Auto-detects and uses CUDA, then Apple MPS, then CPU.
- **Thinking mode** is on by default. Toggle per-request with `"enable_thinking": false` in the request body (or the OpenAI SDK's `extra_body`).
- **Reasoning budget** (`--reasoning-budget PCT`, default 0 = off) caps how many of `max-tokens` may be spent on reasoning. Set e.g. `--reasoning-budget 50` to enable.
- **Tool calling** uses Qwen's built-in tool support. Pass `tools` and `tool_choice` in the chat completion request.
- **Parsers** (`--reasoning-parser` / `--tool-call-parser`, default `qwen`): see `openai_http.parser` for built-in parsers and how to register custom ones.
- **Vision** (`--vision` / `-vl`) loads the model as a vision-language model and accepts multimodal messages with `image_url` content parts (base64 data URIs only).
- **Embeddings** are not implemented — requests return HTTP 501.
