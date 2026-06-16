"""
Qwen 3 / 3.5 reasoning + tool-call parser.

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

To add a new format (e.g. GLM, Llama, Mistral), drop a
``<name>_parser.py`` file next to this one implementing the same
callables, then pass ``--reasoning-parser <name>`` / ``--tool-call-parser
<name>``. No registry or subclassing required.
"""

import json
import re
import uuid

REASONING_START_MARKER = "<think>"
REASONING_END_MARKER = "</think>"

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_AGENT_FC_RE = re.compile(
    r"<function=(.*?)>(.*?)</function>", re.DOTALL
)
_AGENT_PARAM_RE = re.compile(
    r"<parameter=(.*?)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def parse_reasoning(model_output: str) -> tuple[str | None, str]:
    """Split reasoning from content on the first ``</think>`` marker.

    Qwen's chat template emits the opening ``<think>`` as part of the
    prompt, so the generated text begins with reasoning and ends with
    ``</think>`` followed by the answer.

    Args:
        model_output: Fully decoded model output (reasoning + answer).

    Returns:
        A ``(reasoning, content)`` tuple. ``reasoning`` is None when no
        ``</think>`` marker is present.
    """
    idx = model_output.find(REASONING_END_MARKER)
    if idx == -1:
        return None, model_output
    reasoning = model_output[:idx]
    content = model_output[idx + len(REASONING_END_MARKER):].lstrip("\n")
    return (reasoning or None), content


def parse_tool_calls(model_output: str) -> tuple[str, list[dict]]:
    """Extract Qwen/Hermes ``<tool_call>{json}</tool_call>`` blocks.

    Args:
        model_output: Fully decoded model output.

    Returns:
        A ``(content, tool_calls)`` tuple. ``content`` is the output with
        tool-call blocks stripped; ``tool_calls`` is a list of
        OpenAI-shaped tool call dicts (id/type/function).
    """
    tool_calls: list[dict] = []
    for match in _TOOL_CALL_RE.finditer(model_output):
        raw = match.group(1).strip()
        call = _parse_tool_json(raw)
        if call is None:
            call = _parse_tool_agent(raw)
        if call is not None:
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": call.get("name", ""),
                        "arguments": json.dumps(call.get("arguments", {})),
                    },
                }
            )
    if tool_calls:
        content = _TOOL_CALL_RE.sub("", model_output).strip()
    else:
        content = model_output
    return content, tool_calls


def _parse_tool_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _parse_tool_agent(text: str) -> dict | None:
    """Parse Qwen's XML agent format:
    <function=name><parameter=k1>v1</parameter>...</function>"""
    m = _AGENT_FC_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    body = m.group(2)
    args: dict[str, str] = {}
    for pm in _AGENT_PARAM_RE.finditer(body):
        args[pm.group(1).strip()] = pm.group(2).strip()
    return {"name": name, "arguments": args}


if __name__ == "__main__":
    r, c = parse_reasoning("let me think</think>the answer is 4")
    assert r == "let me think", r
    assert c == "the answer is 4", c

    r2, c2 = parse_reasoning("no thinking here")
    assert r2 is None and c2 == "no thinking here", (r2, c2)

    _, calls = parse_tool_calls(
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n'
        "</tool_call>"
    )
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Paris"}

    # Agent format: <function=get_weather><parameter=city>Paris</parameter></function>
    _, calls2 = parse_tool_calls(
        "<tool_call>\n"
        "<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert len(calls2) == 1, calls2
    assert calls2[0]["function"]["name"] == "get_weather"
    assert json.loads(calls2[0]["function"]["arguments"]) == {"city": "Paris"}

    print("qwen_parser self-test OK")
