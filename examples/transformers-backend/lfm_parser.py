"""
LFM reasoning + tool-call parser.

A parser is a plain module discovered by the transformers backend via
dynamic import: pass ``--reasoning-parser <name>`` /
``--tool-call-parser <name>`` and the backend loads ``<name>_parser.py``
from this folder, then looks up the callables/constants below by their
fixed names.

Required symbols (the backend looks these up by name):

    REASONING_END_MARKER   str    - reasoning close tag ("" if none)
    parse_reasoning(text)  -> (reasoning | None, content)
    parse_tool_calls(text) -> (content_without_tools, tool_calls)

Optional:

    REASONING_START_MARKER str    - reasoning open tag, for reference

To add a new format, drop a ``<name>_parser.py`` file next to this one
implementing the same callables, then pass ``--reasoning-parser <name>``
/ ``--tool-call-parser <name>``. No registry or subclassing required.

LFM tool call format:
    [function_name(key1=value1, key2="value2", ...)]
"""

import json
import re
import uuid

import loguru

# LFM does not use reasoning markers
REASONING_START_MARKER = ""
REASONING_END_MARKER = ""

# Matches [function_name(key=value, ...)] or [module.function(key=value, ...)]
_TOOL_CALL_RE = re.compile(r"\[([\w.]+)\((.*?)\)\]", re.DOTALL)
# Matches individual key=value pairs (handles quoted strings and numbers)
_PARAM_RE = re.compile(r'(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,\)]+)')

log = loguru.logger


def parse_reasoning(model_output: str) -> tuple[str | None, str]:
    """LFM does not use reasoning markers, returns full output as content.

    Args:
        model_output: Fully decoded model output.

    Returns:
        A ``(None, model_output)`` tuple.
    """
    log.debug(
        "parse_reasoning: no markers, returning full output ({} chars)",
        len(model_output),
    )
    return None, model_output


def parse_tool_calls(model_output: str) -> tuple[str, list[dict]]:
    """Extract LFM ``[function_name(args)]`` blocks.

    Args:
        model_output: Fully decoded model output.

    Returns:
        A ``(content, tool_calls)`` tuple. ``content`` is the output with
        tool-call blocks stripped; ``tool_calls`` is a list of
        OpenAI-shaped tool call dicts (id/type/function).
    """
    tool_calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(model_output):
        name = match.group(1).strip()
        raw_args = match.group(2).strip()
        arguments = _parse_args(raw_args)
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        )
    if tool_calls:
        content = _TOOL_CALL_RE.sub("", model_output).strip()
    else:
        content = model_output
    log.debug(
        "parse_tool_calls: found {} calls: {}",
        len(tool_calls),
        json.dumps(tool_calls, indent=2),
    )
    return content, tool_calls


def _parse_args(raw: str) -> dict:
    """Parse key=value pairs from LFM tool call arguments.

    Handles:
        - Quoted strings: key="value" or key='value'
        - Numbers: key=42, key=3.14
        - Booleans: key=true, key=false
        - Unquoted bare words: key=value
    """
    args: dict = {}
    for match in _PARAM_RE.finditer(raw):
        key = match.group(1)
        value = match.group(2)
        # Strip quotes if present
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        else:
            # Try to parse as number or boolean
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    if value.lower() == "true":
                        value = True
                    elif value.lower() == "false":
                        value = False
        args[key] = value
    log.info("_parse_args: raw={} -> parsed={}", raw, args)
    return args


if __name__ == "__main__":
    # Test basic tool call
    _, calls = parse_tool_calls(
        '[get_bureau_statistics(year=1954, category="veterans")]'
    )
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "get_bureau_statistics"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"year": 1954, "category": "veterans"}, args

    # Test multiple tool calls
    _, calls2 = parse_tool_calls(
        'First call: [get_weather(city="Paris")] and second: [get_time(zone=2)]'
    )
    assert len(calls2) == 2, calls2
    assert calls2[0]["function"]["name"] == "get_weather"
    assert calls2[1]["function"]["name"] == "get_time"

    # Test no tool calls
    content, calls3 = parse_tool_calls("No tool calls here")
    assert len(calls3) == 0
    assert content == "No tool calls here"

    # Test dotted function name (e.g., module.function)
    _, calls4 = parse_tool_calls(
        '[triangle_properties.get(side1=5, side2=4, side3=3)]'
    )
    assert len(calls4) == 1, calls4
    assert calls4[0]["function"]["name"] == "triangle_properties.get"
    args4 = json.loads(calls4[0]["function"]["arguments"])
    assert args4 == {"side1": 5, "side2": 4, "side3": 3}, args4

    print("lfm_parser self-test OK")
