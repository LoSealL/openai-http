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

Qwen 3 / 3.5 reasoning + tool-call parser.

Reasoning format: the chat template emits ``<think>`` as part of the
prompt, so the generated text begins with reasoning and ends with
``</think>`` followed by the answer.

Tool-call formats (both accepted):

* Hermes/Qwen JSON inside ``<tool_call>...</tool_call>``::

      <tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>

* Qwen XML agent fallback::

      <tool_call><function=f><parameter=x>1</parameter></function></tool_call>
"""

import json
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

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_AGENT_FC_RE = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)
_AGENT_PARAM_RE = re.compile(r"<parameter=(.*?)>\s*(.*?)\s*</parameter>", re.DOTALL)


class QwenParser(ParserBase):
    """Parser for Qwen 3 / 3.5 reasoning and tool-call syntax."""

    REASONING_START_MARKER = REASONING_START_MARKER
    REASONING_END_MARKER = REASONING_END_MARKER

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
        """Extract Qwen/Hermes ``<tool_call>`` blocks.

        Each block is parsed first as JSON, then, on failure, as the
        Qwen XML agent format.

        Args:
            model_output: Fully decoded model output.

        Returns:
            A :class:`ToolCallResult`; ``content`` has tool-call blocks
            stripped, ``tool_calls`` holds the parsed calls.
        """
        tool_calls = []
        for match in _TOOL_CALL_RE.finditer(model_output):
            raw = match.group(1).strip()
            call = _parse_tool_json(raw)
            if call is None:
                call = _parse_tool_agent(raw)
            if call is not None:
                name, arguments = call
                tool_calls.append(make_tool_call(name, arguments))
        if tool_calls:
            content = _TOOL_CALL_RE.sub("", model_output).strip()
        else:
            content = model_output
        return ToolCallResult(content=content, tool_calls=tool_calls)


def _parse_tool_json(text: str) -> tuple[str, dict[str, object]] | None:
    """Parse *text* as a Qwen tool-call JSON object.

    Expects a JSON object with ``name`` (str) and ``arguments`` (object).
    Returns ``None`` when *text* is not valid JSON or not an object with
    a string ``name``.
    """
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded, dict):
        return None
    name = loaded.get("name")
    if not isinstance(name, str):
        return None
    raw_args = loaded.get("arguments", {})
    arguments = raw_args if isinstance(raw_args, dict) else {}
    return name, arguments


def _parse_tool_agent(text: str) -> tuple[str, dict[str, object]] | None:
    """Parse Qwen's XML agent format into ``(name, arguments)``.

    Format: ``<function=name><parameter=k>v</parameter>...</function>``.
    Returns ``None`` when no ``<function=...>`` element is found.
    """
    m = _AGENT_FC_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    body = m.group(2)
    args: dict[str, object] = {}
    for pm in _AGENT_PARAM_RE.finditer(body):
        args[pm.group(1).strip()] = pm.group(2).strip()
    return name, args


register_parser("qwen", QwenParser())
