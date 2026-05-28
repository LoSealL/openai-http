# Transformers Backend Example

A working `openai_http` backend powered by HuggingFace Transformers.
Loads a causal language model and serves chat completions with proper
chat template handling, streaming, and real token usage counts.

Default model: **Qwen/Qwen2.5-0.5B-Instruct** (~1 GB download).

## Install

These dependencies are *not* tracked by the main `openai_http` package
(would bloat it). Install them manually:

```bash
uv pip install torch transformers accelerate
```

For CPU-only (smaller install):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers accelerate
```

## Run

From the repo root:

```bash
uv run python examples/transformers-backend/transformers_backend.py
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

## Notes

- Uses real token counts from the tokenizer for the `usage` field
  (unlike the mock backend, which estimates by character count).
- Auto-detects and uses CUDA, then Apple MPS, then CPU.
- To use a different model, edit `DEFAULT_MODEL` in
  `transformers_backend.py`. Note that some models (e.g. Llama 2)
  require `huggingface-cli login` for access.
- **Tool calling** is implemented via Qwen2.5's built-in tool support
  (`apply_chat_template` with `tools=` parameter). Pass `tools` and
  `tool_choice` in the chat completion request.
- **Embeddings** are not implemented — requests return HTTP 501.
