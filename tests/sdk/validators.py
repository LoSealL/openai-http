"""
Copyright (C) 2026 The OPENAI-HTTP Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Response schema validators for OpenAI v1 API responses.

Provides validation functions to ensure mock backend responses
match the OpenAI v1 specification.
"""

from typing import Any, Dict


def validate_model(model: Dict[str, Any]) -> bool:
    """Validate Model object schema."""
    required = {"id", "object", "created", "owned_by"}
    if not required.issubset(model.keys()):
        return False
    if model.get("object") != "model":
        return False
    return True


def validate_model_list(response: Dict[str, Any]) -> bool:
    """Validate ModelList object schema."""
    if response.get("object") != "list":
        return False
    if "data" not in response or not isinstance(response["data"], list):
        return False
    for model in response["data"]:
        if not validate_model(model):
            return False
    return True


def validate_usage(usage: Dict[str, Any]) -> bool:
    """Validate Usage object schema."""
    required = {"prompt_tokens", "completion_tokens", "total_tokens"}
    if not required.issubset(usage.keys()):
        return False
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        return False
    return True


def validate_chat_completion(response: Dict[str, Any]) -> bool:
    """Validate ChatCompletion object schema."""
    required = {"id", "object", "created", "model", "choices", "usage"}
    if not required.issubset(response.keys()):
        return False
    if response.get("object") != "chat.completion":
        return False
    if not response.get("id").startswith("chatcmpl-"):
        return False
    if not isinstance(response["choices"], list) or len(response["choices"]) == 0:
        return False

    for choice in response["choices"]:
        if not {"index", "message", "finish_reason"}.issubset(choice.keys()):
            return False
        message = choice["message"]
        if not {"role", "content"}.issubset(message.keys()):
            return False
        if message.get("role") != "assistant":
            return False

    if not validate_usage(response["usage"]):
        return False

    return True


def validate_chat_completion_chunk(chunk: Dict[str, Any]) -> bool:
    """Validate ChatCompletionChunk object schema."""
    required = {"id", "object", "created", "model", "choices"}
    if not required.issubset(chunk.keys()):
        return False
    if chunk.get("object") != "chat.completion.chunk":
        return False
    if not isinstance(chunk["choices"], list) or len(chunk["choices"]) == 0:
        return False

    for choice in chunk["choices"]:
        if not {"index", "delta", "finish_reason"}.issubset(choice.keys()):
            return False
        delta = choice["delta"]
        if not isinstance(delta, dict):
            return False

    return True


def validate_completion(response: Dict[str, Any]) -> bool:
    """Validate Completion object schema."""
    required = {"id", "object", "created", "model", "choices", "usage"}
    if not required.issubset(response.keys()):
        return False
    if response.get("object") != "text_completion":
        return False
    if not response.get("id").startswith("cmpl-"):
        return False
    if not isinstance(response["choices"], list) or len(response["choices"]) == 0:
        return False

    for choice in response["choices"]:
        if not {"index", "text", "finish_reason"}.issubset(choice.keys()):
            return False

    if not validate_usage(response["usage"]):
        return False

    return True


def validate_embedding(response: Dict[str, Any]) -> bool:
    """Validate Embedding object schema."""
    if response.get("object") != "embedding":
        return False
    if not {"index", "embedding"}.issubset(response.keys()):
        return False
    if not isinstance(response["embedding"], list):
        return False
    if len(response["embedding"]) == 0:
        return False
    if not all(isinstance(x, (int, float)) for x in response["embedding"]):
        return False
    return True


def validate_embedding_list(response: Dict[str, Any]) -> bool:
    """Validate EmbeddingList object schema."""
    if response.get("object") != "list":
        return False
    if "data" not in response or not isinstance(response["data"], list):
        return False
    if "model" not in response:
        return False
    if "usage" not in response:
        return False

    for embedding in response["data"]:
        if not validate_embedding(embedding):
            return False

    if not validate_usage(response["usage"]):
        return False

    return True


def validate_error_response(
    response: Dict[str, Any], expected_status: int = None
) -> bool:
    """Validate Error object schema."""
    if "error" not in response:
        return False
    error = response["error"]
    if not {"message", "type"}.issubset(error.keys()):
        return False
    return True
