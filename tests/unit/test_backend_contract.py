import pytest
from openai_http.backends.contract import (
    validate_generation,
    validate_model_info,
    validate_model_list,
    validate_stream_chunk,
)
from openai_http.backends.types import (
    ContentChunk,
    FinishChunk,
    GenerationResult,
    GenerationUsage,
    ModelInfo,
    ReasoningChunk,
)
from openai_http.errors import OpenAIError


VALID_USAGE = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


class TestGenerationResultValidation:
    def test_minimum_dict_validates(self):
        raw = {
            "generated_text": "hello",
            "usage": VALID_USAGE,
        }
        result = validate_generation(raw)
        assert isinstance(result, GenerationResult)
        assert result.generated_text == "hello"
        assert result.finish_reason == "stop"
        assert result.reasoning_content is None

    def test_invalid_finish_reason_raises_500(self):
        raw = {
            "generated_text": "x",
            "usage": VALID_USAGE,
            "finish_reason": "bogus",
        }
        with pytest.raises(OpenAIError) as exc:
            validate_generation(raw)
        assert exc.value.status_code == 500
        assert exc.value.error_type == "server_error"
        assert exc.value.code == "backend_contract_error"

    def test_missing_usage_raises_500(self):
        with pytest.raises(OpenAIError) as exc:
            validate_generation({"generated_text": "x"})
        assert exc.value.status_code == 500
        assert exc.value.code == "backend_contract_error"


    def test_pydantic_instance_passes_through(self):
        instance = GenerationResult(
            generated_text="x",
            usage=GenerationUsage(**VALID_USAGE),
        )
        assert validate_generation(instance) is instance

    def test_negative_token_count_rejected(self):
        raw = {
            "generated_text": "x",
            "usage": {
                "prompt_tokens": -1,
                "completion_tokens": 0,
                "total_tokens": -1,
            },
        }
        with pytest.raises(OpenAIError):
            validate_generation(raw)


class TestStreamChunkValidation:
    def test_string_passes_through(self):
        assert validate_stream_chunk("hi") == "hi"

    def test_reasoning_dict_validates(self):
        chunk = validate_stream_chunk({"type": "reasoning", "content": "..."})
        assert isinstance(chunk, ReasoningChunk)
        assert chunk.content == "..."

    def test_content_dict_validates(self):
        chunk = validate_stream_chunk({"type": "content", "content": "abc"})
        assert isinstance(chunk, ContentChunk)

    def test_finish_dict_validates(self):
        chunk = validate_stream_chunk({"type": "finish", "reason": "length"})
        assert isinstance(chunk, FinishChunk)
        assert chunk.reason == "length"

    def test_finish_invalid_reason_raises(self):
        with pytest.raises(OpenAIError):
            validate_stream_chunk({"type": "finish", "reason": "bogus"})

    def test_unknown_type_falls_back_to_content_validation(self):
        with pytest.raises(OpenAIError):
            validate_stream_chunk({"type": "garbage", "content": "abc"})

    def test_unsupported_type_raises_500(self):
        with pytest.raises(OpenAIError) as exc:
            validate_stream_chunk(42)
        assert exc.value.status_code == 500


class TestModelInfoValidation:
    def test_dict_validates(self):
        info = validate_model_info(
            {
                "id": "m",
                "object": "model",
                "created": 1,
                "owned_by": "me",
            }
        )
        assert isinstance(info, ModelInfo)

    def test_missing_field_raises(self):
        with pytest.raises(OpenAIError):
            validate_model_info({"id": "m", "owned_by": "me"})

    def test_list_validates(self):
        result = validate_model_list(
            [{"id": "m", "object": "model", "created": 1, "owned_by": "me"}]
        )
        assert len(result) == 1
        assert isinstance(result[0], ModelInfo)

    def test_non_list_raises(self):
        with pytest.raises(OpenAIError):
            validate_model_list({"not": "a list"})

