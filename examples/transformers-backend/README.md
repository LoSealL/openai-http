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


For CPU-only (smaller install):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers accelerate
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
  --reasoning-parser qwen \
  --tool-call-parser qwen \
  --port 8000

# Vision-language model (accepts image inputs via base64 data URIs)
uv run python examples/transformers-backend/transformers_backend.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --vision \
  --port 8000

# See all options
uv run python examples/transformers-backend/transformers_backend.py --help
```

The server listens on `http://0.0.0.0:8000`. The first request triggers
model download (cached in `~/.cache/huggingface/`).

## Try it

### Basic chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello!"}]}'
```

### Tool calling

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
```

Returns:
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_...",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"city\": \"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
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

Disable thinking for a single request (`enable_thinking` is forwarded
to the backend; thinking is on by default):

```python
resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=64,
    extra_body={"enable_thinking": False},
)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Vision / multimodal (requires `--vision`)

Start the server with a VLM and the `--vision` flag:

```bash
uv run python examples/transformers-backend/transformers_backend.py \
  --model Qwen/Qwen2.5-VL-3B-Instruct --vision
```

Send an image inline as a base64 `data:` URI (remote URLs are **not**
fetched). With the OpenAI Python SDK:

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

with open("photo.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ],
    }],
    max_tokens=256,
)
print(resp.choices[0].message.content)
```

Only base64 data URIs (`data:image/...;base64,...`) are supported for
`image_url.url`; `http(s)` URLs are ignored.


## Notes

- Uses real token counts from the tokenizer for the `usage` field
  (unlike the mock backend, which estimates by character count).
- Auto-detects and uses CUDA, then Apple MPS, then CPU.
- To use a different model, pass `--model` or edit `DEFAULT_MODEL` in
  `transformers_backend.py`. Note that some models (e.g. Llama 2)
  require `huggingface-cli login` for access.
- **Thinking mode** is on by default (reasoning via `<think>` tags,
  supported by Qwen3.5 and recent models). Toggle it per-request by
  sending ``"enable_thinking": false`` in the request body (or the
  OpenAI SDK's ``extra_body``).
- **Reasoning budget** (`--reasoning-budget PCT`, default 0 = off) caps
  how many of `max-tokens` may be spent on `<think>` reasoning. If the
  model is still thinking when the budget is hit, the `</think>` tag is
  force-injected and generation continues with the remaining tokens
  reserved for the answer. Set e.g. `--reasoning-budget 50` to enable.
  Works for both streaming and non-streaming responses.
- Every response prints a line to stdout in the form
  ``[generate:PATH] TEXT`` or ``[stream:PATH] TEXT``, where ``PATH`` is
  one of ``simple``, ``simple(no-thinking)``, ``reasoning-budget/natural``
  or ``reasoning-budget/forced``.
- **Tool calling** uses Qwen's built-in tool support
  (`apply_chat_template` with `tools=` parameter). Pass `tools` and
  `tool_choice` in the chat completion request.
- **Parsers** (`--reasoning-parser` / `--tool-call-parser`, default
  `qwen`): reasoning and tool-call output are parsed by a pluggable
  module discovered by name. The backend dynamically imports
  `<name>_parser.py` from this folder and calls its fixed-signature
  functions. The bundled `qwen_parser.py` handles Qwen 3 / 3.5
  (`</think>` reasoning split, `<tool_call>{json}</tool_call>` tools).

  To support another format, drop a `<name>_parser.py` file next to
  `qwen_parser.py` exposing these symbols:

  ```python
  REASONING_END_MARKER = "</think>"        # str; "" if reasoning isn't tagged

  def parse_reasoning(model_output: str) -> tuple[str | None, str]:
      """Return (reasoning, content). reasoning is None if no marker."""

  def parse_tool_calls(model_output: str) -> tuple[str, list[dict]]:
      """Return (content_without_tools, tool_calls)."""
  ```

  Then run with `--reasoning-parser <name>` / `--tool-call-parser <name>`.
  Run `--help` to list every parser discovered in the folder.
- **Vision** (`--vision` / `-vl`) loads the model as a
  vision-language model via `AutoProcessor` +
  `AutoModelForImageTextToText` and accepts multimodal messages with
  `image_url` content parts (base64 data URIs only). Text-only
  requests also work when `--vision` is on.
- **Embeddings** are not implemented — requests return HTTP 501.
