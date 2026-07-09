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

LFM (Liquid Foundation Model) reasoning + tool-call parser.

LFM uses no reasoning markers, so all output is treated as content.

Tool-call format (bracketed calls)::

    [function_name(key1=value1, key2="value2", ...)]
    [module.function(key=42, flag=true)]

Argument values may be quoted strings, numbers, booleans, or bare words.
"""

import re

from openai_http.parser import register_parser
from openai_http.parser.base import (
    ParserBase,
    ReasoningResult,
    ToolCallResult,
    make_tool_call,
)

# Matches [function_name(key=value, ...)] or [module.function(...)]
_TOOL_CALL_RE = re.compile(r"\[([\w.]+)\((.*?)\)\]", re.DOTALL)
# Matches a single key=value pair; value may be quoted, numeric, boolean, or bare.
_PARAM_RE = re.compile(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,\)]+)')


class LfmParser(ParserBase):
    """Parser for LFM reasoning and tool-call syntax."""

    # LFM uses no reasoning markers.
    REASONING_START_MARKER = ""
    REASONING_END_MARKER = ""

    def parse_reasoning(self, model_output: str) -> ReasoningResult:
        """Return the full output as content; LFM has no reasoning markers.

        Args:
            model_output: Fully decoded model output.

        Returns:
            A :class:`ReasoningResult` with ``reasoning=None`` and the
            original text as ``content``.
        """
        return ReasoningResult(reasoning=None, content=model_output)

    def parse_tool_calls(self, model_output: str) -> ToolCallResult:
        """Extract LFM ``[function_name(args)]`` blocks.

        Argument values are coerced to int/float/bool when possible;
        quoted strings are unquoted; otherwise the bare word is kept.

        Args:
            model_output: Fully decoded model output.

        Returns:
            A :class:`ToolCallResult`; ``content`` has tool-call blocks
            stripped, ``tool_calls`` holds the parsed calls.
        """
        tool_calls = []
        for match in _TOOL_CALL_RE.finditer(model_output):
            name = match.group(1).strip()
            raw_args = match.group(2).strip()
            arguments = _parse_args(raw_args)
            tool_calls.append(make_tool_call(name, arguments))
        if tool_calls:
            content = _TOOL_CALL_RE.sub("", model_output).strip()
        else:
            content = model_output
        return ToolCallResult(content=content, tool_calls=tool_calls)


def _parse_args(raw: str) -> dict[str, object]:
    """Parse ``key=value`` pairs from LFM tool-call arguments.

    Handles quoted strings (``key="v"`` / ``key='v'``), ints, floats,
    booleans (``true``/``false``), and unquoted bare words.

    Args:
        raw: The raw argument text between the parentheses.

    Returns:
        A dict mapping each parameter name to its coerced value.
    """
    args: dict[str, object] = {}
    for match in _PARAM_RE.finditer(raw):
        key = match.group(1)
        value: object = match.group(2)
        if (
            isinstance(value, str)
            and len(value) >= 2
            and (
                (value[0] == '"' and value[-1] == '"')
                or (value[0] == "'" and value[-1] == "'")
            )
        ):
            value = value[1:-1]
        elif isinstance(value, str):
            value = _coerce_scalar(value)
        args[key] = value
    return args


def _coerce_scalar(value: str) -> object:
    """Coerce a bare-word string to int, float, or bool when possible.

    Falls back to the original string when no conversion applies.

    Args:
        value: The unquoted bare-word string.

    Returns:
        The coerced int/float/bool, or the original string.
    """
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return value


register_parser("lfm", LfmParser())
