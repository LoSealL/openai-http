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

MiniCPM5 reasoning + tool-call parser.

Reasoning format: same ``<think>...</think>`` convention as Qwen; the
chat template emits the opening ``<think>`` as part of the prompt.
``<think>``/``</think>`` are NOT special tokens, so they survive
``skip_special_tokens=True``.

Tool-call format::

    <function name="func-name">
        <param name="key">value</param>
        <param name="other"><![CDATA[raw value]]></param>
    </function>

.. _cpm-special-tokens:

.. note:: **Special-token handling (auto-managed by the backend).**

   MiniCPM5's tokenizer registers ``<function``, ``</function>``,
   ``<param``, ``</param>``, ``<tool_call>`` and ``</tool_call>`` as
   special tokens (ids 18-21, 2-3). The default
   ``skip_special_tokens=True`` strips them all, collapsing the
   structure to a bare `` name="f"> name="k">v`` fragment where function
   and parameter tags are indistinguishable and consecutive calls merge
   into one. The information is gone, so no stripped-text fallback can
   reconstruct it reliably.

   This parser therefore sets
   :attr:`ParserBase.REQUIRES_SPECIAL_TOKENS` = ``True``. Backends that
   respect the flag decode with ``skip_special_tokens=False`` so the
   parser sees the structural tags, then call
   :func:`strip_special_tokens` on the final free text (reasoning /
   content) after parsing completes. The reasoning split on
   ``</think>`` is unaffected either way, since ``<think>`` /
   ``</think>`` are not special tokens.
"""

import re

from openai_http.parser import register_parser
from openai_http.parser.base import (
    ParserBase,
    ReasoningResult,
    ToolCallResult,
    make_tool_call,
)

REASONING_START_MARKER = "<think>"
REASONING_END_MARKER = "</think>"

_FUNC_RE = re.compile(r'<function\s+name="([^"]+)">(.*?)</function>', re.DOTALL)
_PARAM_RE = re.compile(r'<param\s+name="([^"]+)">(.*?)</param>', re.DOTALL)
_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)


class CpmParser(ParserBase):
    """Parser for MiniCPM5 reasoning and tool-call syntax."""

    REASONING_START_MARKER = REASONING_START_MARKER
    REASONING_END_MARKER = REASONING_END_MARKER
    # MiniCPM5 registers <function / <param / <tool_call> / </tool_call> as
    # special tokens; skip_special_tokens=True would delete them and make
    # function and param tags indistinguishable. Backends read this flag,
    # decode with skip_special_tokens=False, and clean the leftover framing
    # specials from the final free text via strip_special_tokens().
    REQUIRES_SPECIAL_TOKENS = True

    def parse_reasoning(self, model_output: str) -> ReasoningResult:
        """Split reasoning from content on the first ``</think>`` marker.

        Args:
            model_output: Fully decoded model output (reasoning + answer).

        Returns:
            A :class:`ReasoningResult`; ``reasoning`` is ``None`` when
            no ``</think>`` marker is present.
        """
        idx = model_output.find(self.REASONING_END_MARKER)
        if idx == -1:
            return ReasoningResult(reasoning=None, content=model_output)
        reasoning = model_output[:idx]
        content = model_output[idx + len(self.REASONING_END_MARKER) :].lstrip("\n")
        return ReasoningResult(reasoning=(reasoning or None), content=content)

    def parse_tool_calls(self, model_output: str) -> ToolCallResult:
        """Extract MiniCPM5 ``<function name="...">`` blocks.

        Each ``<param>`` value is unwrapped from CDATA when present.

        Args:
            model_output: Fully decoded model output.

        Returns:
            A :class:`ToolCallResult`; ``content`` has tool-call blocks
            stripped, ``tool_calls`` holds the parsed calls.
        """
        tool_calls = []
        for func_match in _FUNC_RE.finditer(model_output):
            name = func_match.group(1).strip()
            body = func_match.group(2)
            arguments: dict[str, str] = {}
            for param_match in _PARAM_RE.finditer(body):
                param_name = param_match.group(1).strip()
                param_value = param_match.group(2).strip()
                cdata_match = _CDATA_RE.match(param_value)
                if cdata_match:
                    param_value = cdata_match.group(1)
                arguments[param_name] = param_value
            tool_calls.append(make_tool_call(name, arguments))
        if tool_calls:
            content = _FUNC_RE.sub("", model_output).strip()
        else:
            content = model_output
        return ToolCallResult(content=content, tool_calls=tool_calls)


register_parser("cpm", CpmParser())
