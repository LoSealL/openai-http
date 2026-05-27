"""
测试客户端：验证 OpenAI API 兼容性

运行前请先启动服务端:
    python main.py

本脚本使用 openai 官方 SDK 访问 mock 服务端。
"""

import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="mock-api-key",  # mock 模式下任意值均可
)


def test_models():
    print("=== Testing GET /v1/models ===")
    models = client.models.list()
    for model in models.data:
        print(f"  - {model.id}")


def test_chat_completion():
    print("\n=== Testing POST /v1/chat/completions (non-stream) ===")
    response = client.chat.completions.create(
        model="mock-gpt",
        messages=[{"role": "user", "content": "你好，请介绍一下自己"}],
        stream=False,
        max_tokens=100,
    )
    print(f"Response: {response.choices[0].message.content}")
    print(f"Usage: {response.usage}")


def test_chat_completion_stream():
    print("\n=== Testing POST /v1/chat/completions (stream) ===")
    stream = client.chat.completions.create(
        model="mock-gpt",
        messages=[{"role": "user", "content": "讲个笑话"}],
        stream=True,
        max_tokens=100,
    )
    print("Response: ", end="", flush=True)
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def test_completions():
    print("\n=== Testing POST /v1/completions ===")
    response = client.completions.create(
        model="mock-llama",
        prompt="Once upon a time",
        max_tokens=50,
    )
    print(f"Response: {response.choices[0].text}")
    print(f"Usage: {response.usage}")


def test_embeddings():
    print("\n=== Testing POST /v1/embeddings ===")
    response = client.embeddings.create(
        model="mock-embedding",
        input="Hello world",
    )
    print(f"Embedding dimensions: {len(response.data[0].embedding)}")
    print(f"Usage: {response.usage}")


def test_tokenize_prompt():
    print("\n=== Testing POST /tokenize (prompt mode) ===")
    import httpx
    r = httpx.post(
        "http://localhost:8000/tokenize",
        json={"model": "mock-gpt", "prompt": "Hello world", "return_token_strs": True},
    )
    data = r.json()
    print(f"Tokens count: {data['count']}")
    print(f"Token IDs (first 5): {data['tokens'][:5]}")
    print(f"Token strs (first 5): {data.get('token_strs', [])[:5]}")


def test_tokenize_chat():
    print("\n=== Testing POST /tokenize (chat mode) ===")
    import httpx
    r = httpx.post(
        "http://localhost:8000/tokenize",
        json={
            "model": "mock-gpt",
            "messages": [{"role": "user", "content": "Hi"}],
            "add_generation_prompt": True,
            "return_token_strs": True,
        },
    )
    data = r.json()
    print(f"Tokens count: {data['count']}")
    print(f"Token strs (first 5): {data.get('token_strs', [])[:5]}")


def test_detokenize():
    print("\n=== Testing POST /detokenize ===")
    import httpx
    # 先 encode 再 decode，验证可逆性
    encode_r = httpx.post(
        "http://localhost:8000/tokenize",
        json={"model": "mock-gpt", "prompt": "Hello vLLM"},
    )
    tokens = encode_r.json()["tokens"]
    decode_r = httpx.post(
        "http://localhost:8000/detokenize",
        json={"model": "mock-gpt", "tokens": tokens},
    )
    prompt = decode_r.json()["prompt"]
    print(f"Original: 'Hello vLLM'")
    print(f"Decoded:  '{prompt}'")
    assert prompt == "Hello vLLM", "Detokenize should be reversible"


if __name__ == "__main__":
    test_models()
    test_chat_completion()
    test_chat_completion_stream()
    test_completions()
    test_embeddings()
    test_tokenize_prompt()
    test_tokenize_chat()
    test_detokenize()
    print("\n[OK] All tests passed!")
