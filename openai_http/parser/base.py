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

Base class and typed result models for reasoning/tool-call parsers.

A parser turns a model's raw decoded text into structured parts:
``reasoning`` (the ``<think>``-style prefix, if any), ``content`` (the
answer the user sees), and ``tool_calls`` (function invocations the model
emitted inline). Parsers are pure text transforms; they never touch the
network, GPU, or tokenizers.

Built-in parsers live in :mod:`openai_http.parser.qwen`,
:mod:`openai_http.parser.cpm`, and :mod:`openai_http.parser.lfm`. External
parsers subclass :class:`ParserBase` and register via
:func:`openai_http.parser.register_parser`.
"""

import abc
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A single parsed tool/function call.

    Attributes:
        id: A generated unique call id (``call_<24 hex chars>``).
        name: The function name.
        arguments: The arguments as a JSON-encoded string, matching the
            OpenAI ``tool_calls[].function.arguments`` shape.
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Result of splitting reasoning from content.

    Attributes:
        reasoning: The reasoning/thinking text, or ``None`` when the
            model produced no reasoning marker.
        content: The remaining answer text after stripping reasoning.
    """

    reasoning: str | None
    content: str


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Result of extracting tool calls from model output.

    Attributes:
        content: The model output with tool-call blocks stripped (or the
            original text when no tool calls were found).
        tool_calls: The parsed tool calls; empty when none were found.
    """

    content: str
    tool_calls: list[ToolCall]


def make_tool_call(name: str, arguments: Mapping[str, object] | str) -> ToolCall:
    """Build a :class:`ToolCall` with a generated id and JSON arguments.

    Args:
        name: The function name.
        arguments: Either a mapping (JSON-encoded into a string) or a raw
            JSON string passed through unchanged.

    Returns:
        A frozen :class:`ToolCall` with a fresh ``call_`` id.
    """
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return ToolCall(
        id=f"call_{uuid.uuid4().hex[:24]}",
        name=name,
        arguments=args,
    )


def strip_special_tokens(text: str, special_tokens: list[str]) -> str:
    """Remove special token substrings from free text.

    Backends decode with ``skip_special_tokens=False`` whenever a parser
    in use sets :attr:`ParserBase.REQUIRES_SPECIAL_TOKENS`, so the parser
    sees the structural tags it needs. After parsing, this helper cleans
    the leftover framing specials (``<|im_end|>``, ``</s>``, ...) from
    the free text a user sees. It operates on token *strings*, so it
    stays independent of any tokenizer library; callers pass
    ``tokenizer.all_special_tokens``.

    Special-token stripping only ever applies to free text (reasoning /
    content), never to parsed tool-call arguments, which the parser
    already extracted from structured blocks.

    Args:
        text: The free text that may contain embedded special token
            strings.
        special_tokens: The exact special token strings to remove
            (typically ``tokenizer.all_special_tokens``).

    Returns:
        *text* with every special token substring removed.
    """
    if not special_tokens:
        return text
    pattern = re.compile("|".join(re.escape(tok) for tok in special_tokens))
    return pattern.sub("", text)


class ParserBase(abc.ABC):
    """Abstract base class for reasoning/tool-call text parsers.

    Subclasses set the ``REASONING_START_MARKER`` /
    ``REASONING_END_MARKER`` class attributes (empty string when the
    format uses no markers) and implement both abstract methods.
    """

    #: Reasoning open tag, or ``""`` when the format has none.
    REASONING_START_MARKER: str = ""
    #: Reasoning close tag, or ``""`` when the format has none.
    REASONING_END_MARKER: str = ""
    #: True when this parser's markers are special tokens in the model's
    #: tokenizer and ``skip_special_tokens=True`` would strip them, breaking
    #: parsing. Backends that respect this decode with
    #: ``skip_special_tokens=False`` while any parser in use sets it, then
    #: call :func:`strip_special_tokens` on the final free text after
    #: parsing completes. CpmParser sets ``True`` (its ``<function`` /
    #: ``<param`` tags are special tokens); qwen and lfm leave the default.
    REQUIRES_SPECIAL_TOKENS: bool = False

    @abc.abstractmethod
    def parse_reasoning(self, model_output: str) -> ReasoningResult:
        """Split reasoning from content in fully decoded model output.

        Args:
            model_output: The fully decoded model text (reasoning + answer).

        Returns:
            A :class:`ReasoningResult` whose ``reasoning`` is ``None``
            when no reasoning marker is present.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_tool_calls(self, model_output: str) -> ToolCallResult:
        """Extract tool calls from fully decoded model output.

        Args:
            model_output: The fully decoded model text.

        Returns:
            A :class:`ToolCallResult`; ``content`` is the text with
            tool-call blocks stripped, ``tool_calls`` holds the parsed
            calls (empty when none were found).
        """
        raise NotImplementedError
