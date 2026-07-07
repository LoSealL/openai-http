"""Tests for the openai_http.parser registry and built-in parsers."""

import json

import pytest

from openai_http.parser import (
    ParserBase,
    available_parsers,
    get_parser,
    make_tool_call,
    register_parser,
    strip_special_tokens,
)
from openai_http.parser.base import ReasoningResult, ToolCall, ToolCallResult


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_builtin_parsers_registered():
    """qwen, cpm, lfm self-register on import."""
    names = available_parsers()
    assert "qwen" in names
    assert "cpm" in names
    assert "lfm" in names


def test_get_parser_returns_instance():
    p = get_parser("qwen")
    assert isinstance(p, ParserBase)
    assert p.REASONING_END_MARKER == "</think>"


def test_get_parser_unknown_raises_with_listings():
    with pytest.raises(KeyError) as exc_info:
        get_parser("nope")
    msg = str(exc_info.value)
    assert "qwen" in msg
    assert "nope" in msg


def test_register_custom_parser_overrides():
    class MyParser(ParserBase):
        def parse_reasoning(self, model_output: str) -> ReasoningResult:
            return ReasoningResult(reasoning=None, content=model_output)

        def parse_tool_calls(self, model_output: str) -> ToolCallResult:
            return ToolCallResult(content=model_output, tool_calls=[])

    custom = MyParser()
    register_parser("mytest", custom)
    try:
        assert get_parser("mytest") is custom
    finally:
        # Don't leak into other tests.
        register_parser("mytest", custom)  # idempotent; fine to leave


# ---------------------------------------------------------------------------
# Top-level export
# ---------------------------------------------------------------------------


def test_parserbase_exported_from_top_level():
    import openai_http

    assert openai_http.ParserBase is ParserBase
    assert "ParserBase" in openai_http.__all__


# ---------------------------------------------------------------------------
# make_tool_call helper
# ---------------------------------------------------------------------------


def test_make_tool_call_from_dict():
    tc = make_tool_call("f", {"x": 1})
    assert tc.name == "f"
    assert json.loads(tc.arguments) == {"x": 1}
    assert tc.id.startswith("call_")
    assert len(tc.id) == len("call_") + 24


def test_make_tool_call_from_json_string_passthrough():
    tc = make_tool_call("f", '{"x": 1}')
    assert tc.arguments == '{"x": 1}'


def test_tool_call_is_frozen():
    tc = make_tool_call("f", {})
    with pytest.raises(Exception):
        tc.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# QwenParser
# ---------------------------------------------------------------------------


def test_qwen_parse_reasoning_with_marker():
    p = get_parser("qwen")
    r = p.parse_reasoning("let me think</think>the answer is 4")
    assert r.reasoning == "let me think"
    assert r.content == "the answer is 4"


def test_qwen_parse_reasoning_no_marker():
    p = get_parser("qwen")
    r = p.parse_reasoning("no thinking here")
    assert r.reasoning is None
    assert r.content == "no thinking here"


def test_qwen_parse_reasoning_empty_marker_prefix():
    p = get_parser("qwen")
    # "</think>" at index 0 -> empty reasoning becomes None.
    r = p.parse_reasoning("</think>just answer")
    assert r.reasoning is None
    assert r.content == "just answer"


def test_qwen_parse_tool_calls_json():
    p = get_parser("qwen")
    res = p.parse_tool_calls(
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    )
    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert isinstance(call, ToolCall)
    assert call.name == "get_weather"
    assert json.loads(call.arguments) == {"city": "Paris"}


def test_qwen_parse_tool_calls_agent_format():
    p = get_parser("qwen")
    res = p.parse_tool_calls(
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
    )
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "get_weather"
    assert json.loads(res.tool_calls[0].arguments) == {"city": "Paris"}


def test_qwen_parse_tool_calls_strips_blocks():
    p = get_parser("qwen")
    res = p.parse_tool_calls(
        "I'll check.\n<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>"
    )
    assert "I'll check." in res.content
    assert "<tool_call>" not in res.content


def test_qwen_parse_tool_calls_none():
    p = get_parser("qwen")
    res = p.parse_tool_calls("no calls here")
    assert res.tool_calls == []
    assert res.content == "no calls here"


# ---------------------------------------------------------------------------
# CpmParser
# ---------------------------------------------------------------------------


def test_cpm_parse_reasoning():
    p = get_parser("cpm")
    r = p.parse_reasoning("step 1\nstep 2</think>\nresult")
    assert r.reasoning == "step 1\nstep 2"
    assert r.content == "result"


def test_cpm_parse_tool_calls():
    p = get_parser("cpm")
    res = p.parse_tool_calls(
        '<function name="get_weather"><param name="city">Paris</param></function>'
    )
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "get_weather"
    assert json.loads(res.tool_calls[0].arguments) == {"city": "Paris"}


def test_cpm_parse_tool_calls_cdata():
    p = get_parser("cpm")
    res = p.parse_tool_calls(
        '<function name="write_file">'
        '<param name="content"><![CDATA[line 1\nline 2]]></param>'
        "</function>"
    )
    assert len(res.tool_calls) == 1
    args = json.loads(res.tool_calls[0].arguments)
    assert args == {"content": "line 1\nline 2"}


def test_cpm_parse_tool_calls_multiple():
    p = get_parser("cpm")
    res = p.parse_tool_calls(
        '<function name="get_weather"><param name="city">Paris</param></function>'
        '<function name="get_time"><param name="zone">2</param></function>'
    )
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0].name == "get_weather"
    assert res.tool_calls[1].name == "get_time"


# ---------------------------------------------------------------------------
# LfmParser
# ---------------------------------------------------------------------------


def test_lfm_reasoning_markers_empty():
    p = get_parser("lfm")
    assert p.REASONING_END_MARKER == ""
    assert p.REASONING_START_MARKER == ""


def test_lfm_parse_reasoning_returns_full_output():
    p = get_parser("lfm")
    r = p.parse_reasoning("anything goes</think>not split")
    assert r.reasoning is None
    assert r.content == "anything goes</think>not split"


def test_lfm_parse_tool_calls_basic():
    p = get_parser("lfm")
    res = p.parse_tool_calls('[get_bureau_statistics(year=1954, category="veterans")]')
    assert len(res.tool_calls) == 1
    call = res.tool_calls[0]
    assert call.name == "get_bureau_statistics"
    args = json.loads(call.arguments)
    assert args == {"year": 1954, "category": "veterans"}


def test_lfm_parse_tool_calls_multiple():
    p = get_parser("lfm")
    res = p.parse_tool_calls(
        'First: [get_weather(city="Paris")] and second: [get_time(zone=2)]'
    )
    assert len(res.tool_calls) == 2
    assert res.tool_calls[0].name == "get_weather"
    assert res.tool_calls[1].name == "get_time"
    assert json.loads(res.tool_calls[1].arguments) == {"zone": 2}


def test_lfm_parse_tool_calls_dotted_name():
    p = get_parser("lfm")
    res = p.parse_tool_calls("[triangle_properties.get(side1=5, side2=4, side3=3)]")
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].name == "triangle_properties.get"
    assert json.loads(res.tool_calls[0].arguments) == {
        "side1": 5,
        "side2": 4,
        "side3": 3,
    }


def test_lfm_parse_tool_calls_strips_blocks():
    p = get_parser("lfm")
    res = p.parse_tool_calls('prefix [f(x=1)] suffix')
    assert "<" not in res.content  # no markup, but blocks stripped
    assert "prefix" in res.content
    assert "suffix" in res.content
    assert "[" not in res.content


def test_lfm_parse_tool_calls_none():
    p = get_parser("lfm")
    res = p.parse_tool_calls("plain text, no calls")
    assert res.tool_calls == []
    assert res.content == "plain text, no calls"


def test_lfm_parse_tool_calls_boolean_and_float():
    p = get_parser("lfm")
    res = p.parse_tool_calls("[f(flag=true, ratio=0.5)]")
    args = json.loads(res.tool_calls[0].arguments)
    assert args == {"flag": True, "ratio": 0.5}


# ---------------------------------------------------------------------------
# REQUIRES_SPECIAL_TOKENS (adaptive skip_special_tokens contract)
# ---------------------------------------------------------------------------


def test_requires_special_tokens_default_false():
    """ParserBase default is False; qwen and lfm inherit it."""
    assert ParserBase.REQUIRES_SPECIAL_TOKENS is False
    assert get_parser("qwen").REQUIRES_SPECIAL_TOKENS is False
    assert get_parser("lfm").REQUIRES_SPECIAL_TOKENS is False


def test_cpm_requires_special_tokens_true():
    """CpmParser opts in: its <function/<param markers are special tokens."""
    assert get_parser("cpm").REQUIRES_SPECIAL_TOKENS is True


def test_custom_parser_requires_special_tokens_inherited():
    """A subclass that does not set the flag inherits the False default."""

    class P(ParserBase):
        def parse_reasoning(self, model_output: str) -> ReasoningResult:
            return ReasoningResult(reasoning=None, content=model_output)

        def parse_tool_calls(self, model_output: str) -> ToolCallResult:
            return ToolCallResult(content=model_output, tool_calls=[])

    assert P().REQUIRES_SPECIAL_TOKENS is False


# ---------------------------------------------------------------------------
# strip_special_tokens (backend post-parse cleanup helper)
# ---------------------------------------------------------------------------


def test_strip_special_tokens_removes_listed():
    out = strip_special_tokens("hi<|im_end|>", ["<|im_end|>", "</s>"])
    assert out == "hi"


def test_strip_special_tokens_multiple_kinds():
    out = strip_special_tokens("a</s>b<|im_end|>c", ["<|im_end|>", "</s>"])
    assert out == "abc"


def test_strip_special_tokens_empty_list_passthrough():
    assert strip_special_tokens("hi<|im_end|>", []) == "hi<|im_end|>"


def test_strip_special_tokens_no_match_passthrough():
    assert strip_special_tokens("hello", ["<|im_end|>"]) == "hello"


def test_strip_special_tokens_handles_regex_meta_chars():
    """Special tokens may contain regex metacharacters (e.g. <|...|>)."""
    out = strip_special_tokens("x<|im_end|>y", ["<|im_end|>"])
    assert out == "xy"
