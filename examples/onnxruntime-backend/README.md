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

# See all options
python examples/onnxruntime-backend/onnxruntime_backend.py --help
```

## Try it

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"onnx-community/Qwen3.5-0.8B-ONNX-OPT","messages":[{"role":"user","content":"Hello!"}]}'
```

## Notes

- **ONNX dtype variants**: Use `--dtype` to select quantized variants (`fp16`, `q4`, `q8`, `q4f16`). Falls back to the base model if the dtype-specific file does not exist.
- **ONNX Runtime provider**: Use `--provider CUDAExecutionProvider` for GPU acceleration. Falls back to `CPUExecutionProvider` automatically if CUDA is not available.
- **Model caching**: The model is loaded from the HuggingFace cache (`~/.cache/huggingface/` or `HF_HOME`).
- **Thinking**: On by default. Toggle with `--no-thinking` or per-request via `enable_thinking: false`.
- **Embeddings** are not implemented — requests return HTTP 501.
