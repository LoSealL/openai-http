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

Reasoning/tool-call parser registry.

Parsers split a model's decoded text into reasoning, content, and tool
calls. Built-in parsers (``qwen``, ``cpm``, ``lfm``) self-register when
their modules are imported below; external parsers register via
:func:`register_parser`. Resolve a parser by name with
:func:`get_parser` at backend load time.
"""

# pylint:disable=duplicate-code  # R0801: lfm/qwen tool-call assembly is structurally similar but semantically distinct

from .base import (
    ParserBase,
    ReasoningResult,
    ToolCall,
    ToolCallResult,
    make_tool_call,
    strip_special_tokens,
)

__all__ = [
    "ParserBase",
    "ReasoningResult",
    "ToolCall",
    "ToolCallResult",
    "make_tool_call",
    "strip_special_tokens",
    "register_parser",
    "get_parser",
    "available_parsers",
]

#: All registered parsers, keyed by name.
_parsers: dict[str, ParserBase] = {}


def register_parser(name: str, parser: ParserBase) -> None:
    """Register *parser* under *name*.

    Overwrites any existing entry registered under the same name, so
    later registrations win. Built-in parsers call this at import time;
    user code calls it to add custom formats or override defaults.

    Args:
        name: The lookup key (e.g. ``"qwen"``).
        parser: A :class:`ParserBase` instance.
    """
    _parsers[name] = parser


def get_parser(name: str) -> ParserBase:
    """Resolve a parser by name.

    Args:
        name: The registered parser name.

    Returns:
        The :class:`ParserBase` registered under *name*.

    Raises:
        KeyError: If no parser is registered under *name*. The error
            message lists the available names.
    """
    try:
        return _parsers[name]
    except KeyError:
        available = ", ".join(sorted(_parsers)) or "(none)"
        raise KeyError(
            f"No parser registered as {name!r}. Available: {available}"
        ) from None


def available_parsers() -> list[str]:
    """Return the sorted names of every registered parser."""
    return sorted(_parsers)


# Eagerly import built-ins so they self-register. These modules call
# register_parser("qwen"/"cpm"/"lfm", ...) as a side effect of import.
from . import cpm as _cpm  # noqa: E402,F401  pylint:disable=wrong-import-position,cyclic-import
from . import lfm as _lfm  # noqa: E402,F401  pylint:disable=wrong-import-position,cyclic-import
from . import qwen as _qwen  # noqa: E402,F401  pylint:disable=wrong-import-position,cyclic-import
