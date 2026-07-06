# ONNX Runtime Backend Example

A working `openai_http` backend powered by ONNX Runtime with manual token-by-token
generation. Loads ONNX exported models (e.g. from `onnx-community`) and serves
chat completions with temperature/top-p sampling, streaming, and proper chat
template handling.

Default model: **onnx-community/Qwen3.5-0.8B-ONNX-OPT** (~1.6 GB cached).

## Install

These dependencies are *not* tracked by the main `openai_http` package
(would bloat it). Install them manually with **system Python**:

```bash
# CPU-only
pip install onnxruntime transformers numpy Pillow

# GPU (CUDA)
pip install onnxruntime-gpu transformers numpy Pillow
```

> **Note:** This example uses **system Python**, not `uv`. Run with `python`
> directly, not `uv run python`.

## Run

From the repo root:

```bash
# Default: Qwen3.5-0.8B-ONNX-OPT, CPU provider, greedy sampling
python examples/onnxruntime-backend/onnxruntime_backend.py

# Custom model, temperature, GPU provider, and dtype
python examples/onnxruntime-backend/onnxruntime_backend.py \
  --model onnx-community/Qwen3.5-0.8B-ONNX-OPT \
  --provider CUDAExecutionProvider \
  --dtype q4 \
  --temperature 0.7 \
  --max-tokens 2048 \
  --port 8000

# Vision-language model (accepts image inputs via base64 data URIs)
python examples/onnxruntime-backend/onnxruntime_backend.py \
  --model onnx-community/Qwen3.5-0.8B-ONNX-OPT \
  --vision \
  --port 8000

# See all options
python examples/onnxruntime-backend/onnxruntime_backend.py --help
```

The server listens on `http://0.0.0.0:8000`.

## Try it

### Basic chat completion

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"onnx-community/Qwen3.5-0.8B-ONNX-OPT","messages":[{"role":"user","content":"Hello!"}]}'
```

With the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

resp = client.chat.completions.create(
    model="onnx-community/Qwen3.5-0.8B-ONNX-OPT",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    max_tokens=64,
)
print(resp.choices[0].message.content)
```

Streaming:

```python
stream = client.chat.completions.create(
    model="onnx-community/Qwen3.5-0.8B-ONNX-OPT",
    messages=[{"role": "user", "content": "Tell me a short story."}],
    max_tokens=200,
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Vision / multimodal (requires `--vision`)

Start the server with the `--vision` flag:

```bash
python examples/onnxruntime-backend/onnxruntime_backend.py \
  --model onnx-community/Qwen3.5-0.8B-ONNX-OPT --vision
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
    model="onnx-community/Qwen3.5-0.8B-ONNX-OPT",
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

- **ONNX dtype variants**: Use `--dtype` to select quantized variants:
  ```bash
  python examples/onnxruntime-backend/onnxruntime_backend.py --dtype q4
  ```
  Available: `fp16`, `q4`, `q8`, `q4f16`. Falls back to the base
  model if the dtype-specific file does not exist.
- **ONNX Runtime provider**: Use `--provider CUDAExecutionProvider` for GPU
  acceleration. Falls back to `CPUExecutionProvider` automatically if CUDA
  is not available.
- **Model caching**: The model is loaded from the HuggingFace cache
  (`~/.cache/huggingface/` or `HF_HOME`). Set `HF_HOME` env var if your cache
  is in a non-standard location.
- **Architecture**: This example implements a manual generation loop because
  `optimum` and `onnxruntime-genai` do not support the `qwen3_5` model type.
  The loop handles:
  - M-RoPE (multi-dimensional rotary position embeddings)
  - Hybrid KV-cache (linear attention + full attention)
  - Temperature/top-p sampling
  - Streaming with proper multi-byte character handling
- **Vision**: When `--vision` is enabled, the backend loads `AutoProcessor` and
  `vision_encoder.onnx` to process image inputs. Images are preprocessed by the
  processor, encoded by the vision encoder, and merged into text embeddings
  by replacing `<|image_pad|>` tokens.
- **Thinking**: On by default. Toggle with `--no-thinking` or per-request
  via `enable_thinking: false`.
- **Tool calling**: Uses Qwen's built-in tool support, parsed via the
  `openai_http.parser` registry (`--reasoning-parser` /
  `--tool-call-parser`, default `qwen`). Pass `tools` and `tool_choice`
  in the request.
- **Embeddings** are not implemented — requests return HTTP 501.

## How it works

The ONNX model consists of three parts (each with optional dtype suffixes):

1. `embed_tokens[_fp16|_q4|_q8|_q4f16].onnx` — converts token IDs to embeddings
2. `vision_encoder[_fp16|_q4|_q8|_q4f16].onnx` — encodes images into feature vectors
3. `decoder_model_merged[_fp16|_q4|_q8|_q4f16].onnx` — runs the decoder with KV-cache

Select dtype with `--dtype fp16|q4|q8|q4f16`. Falls back to base if the variant is missing.

The generation loop:

1. Apply chat template to messages (with `enable_thinking` / `tools` support)
2. If images are present, preprocess them with `AutoProcessor` and run
   `vision_encoder` to get image features
3. Merge image features into text embeddings by replacing `<|image_pad|>` tokens
4. Run `decoder` with initial past states (empty)
5. Sample from logits (greedy or temperature/top-p)
6. Feed the sampled token back through the loop
7. Repeat until EOS or max tokens

The model uses a hybrid architecture (Qwen3.5):
- 18 linear attention layers (with `conv` + `recurrent` states)
- 6 full attention layers (with traditional `key`/`value` caches)

All these states are managed automatically by the backend.
