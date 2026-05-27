"""
Mock Transformers Backend
=========================
模拟基于 transformers 的模型推理后端。

设计用于在没有真实模型权重时提供 OpenAI API 兼容的测试响应。
未来可替换为真实的 transformers pipeline 或 AutoModel 加载逻辑。
"""

import time
import random
import asyncio
from typing import List, Dict, AsyncGenerator, Optional


class MockTokenizer:
    """
    模拟 transformers Tokenizer
    ==========================
    用于 /tokenize 和 /detokenize 端点。

    采用 Unicode code point 映射方案：
      - encode:  每个字符 -> ord(char) + OFFSET 作为 token ID
      - decode:  token ID -> chr(token_id - OFFSET) 还原字符
    完全可逆，无需维护词汇表，且不依赖真实模型文件。

    预留了真实 transformers AutoTokenizer 的接口位置。
    """

    OFFSET = 100  # 避开 special token ID 区域

    def __init__(self):
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.unk_token_id = 3
        self.vocab_size = 0x110000 + self.OFFSET + 10

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """将文本编码为 token ID 列表"""
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_token_id)
        for char in text:
            tokens.append(ord(char) + self.OFFSET)
        if add_special_tokens:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """将 token ID 列表解码为文本"""
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
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str:
        """模拟 tokenizer.apply_chat_template，将 messages 转为 prompt 字符串"""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|{role}|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """将 token IDs 转为可读的 token 字符串（对应 tokenizer.convert_ids_to_tokens）"""
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


class MockTransformersBackend:
    """
    模拟 transformers 后端的模型推理类。

    接口设计参考了 transformers 的 Pipeline 和 TextIteratorStreamer：
      - generate(): 对应 pipeline(..., return_full_text=True)
      - generate_stream(): 对应 TextIteratorStreamer 的异步迭代
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
        self.pipeline = None  # 占位：真实实现中会存储 transformers pipeline

        # 模拟模型加载过程
        self._mock_load()

    def _mock_load(self):
        """模拟 transformers 模型加载流程"""
        print(f"[MockBackend] Initializing transformers pipeline...")
        print(f"[MockBackend] Model config: {self.model_name}")
        print(f"[MockBackend] Device: {self.device}")

        # 模拟从 HuggingFace Hub 或本地路径加载的延迟
        time.sleep(0.05)

        # ==================== 真实代码示例（当前被 mock 绕过） ====================
        # from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path or self.model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_path or self.model_name,
        #     torch_dtype="auto",
        #     device_map=self.device,
        # )
        # self.pipeline = pipeline(
        #     "text-generation",
        #     model=self.model,
        #     tokenizer=self.tokenizer,
        # )
        # ========================================================================

        self.tokenizer = MockTokenizer()
        self.pipeline = True  # 标记为已加载
        print(f"[MockBackend] Mock model ready (no actual weights loaded)")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """粗略估算 token 数量（中文字符按 1.5 token，其他按 0.25 token）"""
        count = 0
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                count += 1.5
            else:
                count += 0.25
        return int(max(1, count))

    @staticmethod
    def _build_prompt(messages: List[Dict[str, str]]) -> str:
        """将 messages 列表转换为纯文本 prompt（模拟 tokenizer.apply_chat_template）"""
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

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> Dict:
        """
        模拟同步文本生成（对应 transformers pipeline 的 __call__ 方法）

        Returns:
            dict: 包含 generated_text 和 usage 的字典
        """
        if not self.pipeline:
            raise RuntimeError("Model not loaded")

        prompt = self._build_prompt(messages)
        prompt_tokens = self._estimate_tokens(prompt)

        # 模拟生成延迟（与 max_tokens 成正比）
        generation_time = 0.001 * min(max_tokens, 100)
        time.sleep(generation_time)

        # 提取最后一条用户消息用于构造回复
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = msg.get("content", "")
                break

        # 根据 temperature 调整随机性
        if temperature > 0.8:
            templates = [
                f"收到你的消息：「{last_user_msg[:30]}」。这是来自 mock 模型的随机回复。",
                f"[随机模式] 你说了「{last_user_msg[:20]}」... 有趣的问题！这是模拟生成的答案。",
                f"Mock模型回复：输入长度 {len(last_user_msg)} 字符，temperature={temperature:.2f}。",
            ]
            generated_text = random.choice(templates)
        else:
            templates = [
                f"这是一个模拟响应。你的输入是：{last_user_msg[:50]}",
                f"[Mock Mode] 已收到消息（{len(last_user_msg)} 字符）。当前未加载真实模型，此为测试输出。",
            ]
            generated_text = templates[0]

        # 截断到 max_tokens 估算长度（粗略：1 token ≈ 3 字符）
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
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        模拟流式文本生成（对应 transformers 的 TextIteratorStreamer）

        Yields:
            str: 生成的文本片段（逐 token/chunk）
        """
        result = self.generate(messages, max_tokens, temperature, **kwargs)
        full_text = result["generated_text"]

        idx = 0
        while idx < len(full_text):
            # 每块 1-4 个字符（模拟 token-by-token 生成）
            chunk_size = random.randint(1, 4)
            chunk = full_text[idx : idx + chunk_size]
            yield chunk
            idx += chunk_size

            # 模拟流式生成延迟
            await asyncio.sleep(random.uniform(0.01, 0.03))


# ============== 模型元数据（对应 /v1/models） ==============
AVAILABLE_MODELS = [
    {
        "id": "mock-gpt",
        "object": "model",
        "created": 1677610602,
        "owned_by": "mock-org",
    },
    {
        "id": "mock-llama",
        "object": "model",
        "created": 1677610602,
        "owned_by": "mock-org",
    },
]
