"""
OpenAI Compatible HTTP Server
=============================
基于 FastAPI 实现的 OpenAI API 格式兼容服务器。

后端设计为 transformers，当前通过 MockTransformersBackend 绕过真实模型加载。

支持端点:
  - GET  /v1/models
  - GET  /v1/models/{model_id}
  - POST /v1/chat/completions    (stream / non-stream)
  - POST /v1/completions         (stream / non-stream)
  - POST /v1/embeddings          (mock)
  - POST /v1/moderations         (mock)

启动方式:
    python main.py
    # 或使用 uvicorn:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

import time
import uuid
import json
import random
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from mock_backend import MockTransformersBackend, AVAILABLE_MODELS


# ==============  lifespan 管理 ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时释放资源"""
    print("[Server] Starting up...")
    app.state.backend = MockTransformersBackend(
        model_name="mock-llm",
        device="cpu",
    )
    yield
    print("[Server] Shutting down...")
    app.state.backend = None


app = FastAPI(
    title="OpenAI-Compatible API Server",
    description="基于 transformers 后端的 OpenAI API 兼容服务（当前为 Mock 模式）",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== 请求模型 ==============
class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system/user/assistant/tool")
    content: str = Field(..., description="消息内容")
    name: Optional[str] = Field(None, description="发送者名称（可选）")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="模型 ID")
    messages: list[ChatMessage] = Field(..., description="对话消息列表")
    stream: bool = Field(False, description="是否使用流式传输")
    max_tokens: Optional[int] = Field(512, ge=1, le=8192, description="最大生成 token 数")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0, description="核采样参数")
    n: Optional[int] = Field(1, ge=1, le=10, description="生成候选数（当前仅支持 1）")
    stop: Optional[Any] = Field(None, description="停止序列")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0)
    user: Optional[str] = Field(None, description="用户标识")


class CompletionRequest(BaseModel):
    model: str = Field(..., description="模型 ID")
    prompt: str = Field(..., description="文本提示")
    stream: bool = Field(False, description="是否使用流式传输")
    max_tokens: Optional[int] = Field(512, ge=1, le=8192)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)
    suffix: Optional[str] = Field(None, description="后缀（用于 FIM）")
    n: Optional[int] = Field(1, ge=1, le=10)


class EmbeddingRequest(BaseModel):
    model: str = Field(..., description="模型 ID")
    input: str | list[str] = Field(..., description="输入文本")
    encoding_format: Optional[str] = Field("float", description="编码格式: float/base64")
    dimensions: Optional[int] = Field(None, description="嵌入维度")
    user: Optional[str] = Field(None)


# ============== vLLM 特有: /tokenize & /detokenize ==============
class TokenizeRequest(BaseModel):
    model: Optional[str] = Field(None, description="模型 ID")
    prompt: Optional[str] = Field(None, description="纯文本 prompt（completion 模式）")
    messages: Optional[list[ChatMessage]] = Field(None, description="对话消息列表（chat 模式）")
    add_special_tokens: bool = Field(False, description="是否添加特殊 token")
    add_generation_prompt: bool = Field(True, description="chat 模式下是否添加生成提示")
    continue_final_message: bool = Field(False, description="是否继续最后一条消息")
    chat_template: Optional[str] = Field(None, description="自定义 chat template")
    chat_template_kwargs: Optional[dict] = Field(None, description="chat template 额外参数")
    tools: Optional[list] = Field(None, description="工具定义（chat 模式）")
    return_token_strs: bool = Field(False, description="是否返回 token 字符串列表")


class DetokenizeRequest(BaseModel):
    model: Optional[str] = Field(None, description="模型 ID")
    tokens: list[int] = Field(..., description="token ID 列表")


# ============== 辅助函数 ==============
def _sse_chunk(
    request_id: str,
    model: str,
    created: int,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> str:
    """构建 SSE 格式的 chat.completion.chunk"""
    data = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": "mock-fp-001",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ============== API 端点 ==============
@app.get("/")
async def root():
    return {
        "message": "OpenAI Compatible API Server (Mock Mode)",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/v1/models")
async def list_models():
    """列出可用模型（对应 OpenAI /v1/models）"""
    return {"object": "list", "data": AVAILABLE_MODELS}


@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    """获取特定模型信息"""
    for model in AVAILABLE_MODELS:
        if model["id"] == model_id:
            return model
    raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    聊天补全接口（对应 OpenAI /v1/chat/completions）
    支持 stream=true 的 SSE 流式传输
    """
    backend = app.state.backend
    if not backend:
        raise HTTPException(status_code=503, detail="Model backend not available")

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    if request.stream:

        async def stream_generator():
            # 1. 发送 role
            yield _sse_chunk(
                request_id, request.model, created, delta={"role": "assistant"}
            )

            # 2. 流式发送内容
            async for chunk in backend.generate_stream(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ):
                yield _sse_chunk(
                    request_id, request.model, created, delta={"content": chunk}
                )

            # 3. 发送 finish_reason
            yield _sse_chunk(
                request_id,
                request.model,
                created,
                delta={},
                finish_reason="stop",
            )

            # 4. 结束标记
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
            },
        )
    else:
        # 非流式响应
        result = backend.generate(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        return {
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": request.model,
            "system_fingerprint": "mock-fp-001",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["generated_text"],
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": result["usage"],
        }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """
    文本补全接口（对应 OpenAI /v1/completions）
    """
    backend = app.state.backend
    if not backend:
        raise HTTPException(status_code=503, detail="Model backend not available")

    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # 将 prompt 包装为单条 user 消息
    messages = [{"role": "user", "content": request.prompt}]
    result = backend.generate(
        messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
    )

    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "text": result["generated_text"],
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": result["usage"],
    }


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """
    文本嵌入接口（Mock 实现）
    """
    if isinstance(request.input, str):
        inputs = [request.input]
    else:
        inputs = request.input

    dims = request.dimensions or 1536
    data = []
    total_tokens = 0

    for i, text in enumerate(inputs):
        tokens = len(text) // 4 + 1
        total_tokens += tokens

        # 生成确定性随机向量（基于文本哈希，保证相同输入相同输出）
        seed = hash(text) % (2**31)
        rng = random.Random(seed)
        embedding = [rng.uniform(-1.0, 1.0) for _ in range(dims)]

        data.append(
            {
                "object": "embedding",
                "index": i,
                "embedding": embedding,
            }
        )

    return {
        "object": "list",
        "data": data,
        "model": request.model,
        "usage": {
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        },
    }


@app.post("/v1/moderations")
async def create_moderation():
    """内容审核接口（Mock：全部通过）"""
    return {
        "id": f"modr-{uuid.uuid4().hex[:12]}",
        "model": "mock-moderation",
        "results": [
            {
                "flagged": False,
                "categories": {
                    "sexual": False,
                    "hate": False,
                    "violence": False,
                    "self-harm": False,
                    "sexual/minors": False,
                    "hate/threatening": False,
                    "violence/graphic": False,
                },
                "category_scores": {
                    "sexual": 0.0,
                    "hate": 0.0,
                    "violence": 0.0,
                    "self-harm": 0.0,
                    "sexual/minors": 0.0,
                    "hate/threatening": 0.0,
                    "violence/graphic": 0.0,
                },
            }
        ],
    }


@app.post("/tokenize")
async def tokenize(request: TokenizeRequest):
    """
    vLLM 特有的 /tokenize 端点。

    支持两种输入模式（互斥）：
      1. prompt: 纯文本 tokenization（completion 模式）
      2. messages: 先应用 chat template 再 tokenization（chat 模式）
    """
    backend = app.state.backend
    if not backend or not backend.tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not available")

    tokenizer = backend.tokenizer

    # 判断输入模式
    if request.prompt is not None and request.messages is not None:
        raise HTTPException(
            status_code=400, detail="Cannot specify both 'prompt' and 'messages'"
        )

    if request.prompt is not None:
        # Completion 模式
        text = request.prompt
    elif request.messages is not None:
        # Chat 模式：应用 chat template
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=request.add_generation_prompt,
        )
    else:
        raise HTTPException(
            status_code=400, detail="Must specify either 'prompt' or 'messages'"
        )

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=request.add_special_tokens)

    response = {
        "tokens": tokens,
        "count": len(tokens),
        "max_model_len": 8192,
    }

    if request.return_token_strs:
        response["token_strs"] = tokenizer.convert_ids_to_tokens(tokens)

    return response


@app.post("/detokenize")
async def detokenize(request: DetokenizeRequest):
    """
    vLLM 特有的 /detokenize 端点。
    将 token ID 列表还原为文本。
    """
    backend = app.state.backend
    if not backend or not backend.tokenizer:
        raise HTTPException(status_code=503, detail="Tokenizer not available")

    text = backend.tokenizer.decode(request.tokens, skip_special_tokens=True)
    return {"prompt": text}


# ============== 启动入口 ==============
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
