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

Public types for the radix KV cache.

The cache operates on token-id sequences and treats per-node values
(handles) as opaque. Backends supply a ``slice_fn`` to derive a
sub-handle when the tree splits an edge on a partial match.
"""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

H = TypeVar("H")


@dataclass
class PrefixMatch(Generic[H]):
    """Longest-prefix lookup result.

    Attributes:
        handle: Opaque backend handle covering the first ``num_tokens``
            tokens of the queried sequence. ``None`` on total miss.
        num_tokens: Number of leading tokens ``handle`` covers. ``0``
            on total miss or empty query.
    """

    handle: H | None
    num_tokens: int


class SliceFn(Protocol[H]):
    """Derive a handle covering the first ``n`` tokens of ``handle``.

    Backends implement this so the radix tree can split a node when a
    partial match terminates inside a stored segment. Must be
    composable: ``slice_fn(slice_fn(h, m), n) == slice_fn(h, n)`` for
    ``n <= m``.
    """

    def __call__(self, handle: H, n: int) -> H: ...


class Serializer(Protocol[H]):
    """Future disk-persistence seam. Not used by ``RadixKVCache`` in Tier 1.

    Defined so a future ``DiskKVStore`` can plug in without changing
    ``RadixKVCache``'s public API.
    """

    def dumps(self, handle: H) -> bytes: ...

    def loads(self, data: bytes) -> H: ...


@dataclass
class CacheStats:
    """Snapshot of cache state for metrics and logs.

    Attributes:
        total_tokens: Sum of edge lengths over all live (non-root) nodes.
        node_count: Number of non-root nodes.
        evictable_tokens: Tokens in leaf nodes with ``refcount == 0``.
        protected_tokens: Tokens in nodes with ``refcount > 0``.
        hits: Cumulative ``match`` calls returning ``num_tokens > 0``.
        misses: Cumulative ``match`` calls returning ``num_tokens == 0``.
        evictions: Cumulative token count evicted.
    """

    total_tokens: int = 0
    node_count: int = 0
    evictable_tokens: int = 0
    protected_tokens: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
