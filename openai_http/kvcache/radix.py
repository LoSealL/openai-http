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

Radix tree over token-id sequences with opaque backend-supplied handles.

Algorithm ported from SGLang's ``RadixCache`` (Apache 2.0), simplified
to the generic opaque-handle case: no torch, no memory pools, no
bigram/page/session/event features. The generic<->tensor bridge is a
single ``slice_fn`` callable the backend supplies.
"""

import threading
import time
from typing import Generic, Sequence, TypeVar

from openai_http.kvcache.types import CacheStats, PrefixMatch, SliceFn

H = TypeVar("H")


class _TreeNode(Generic[H]):
    """A node in the radix tree.

    ``key`` is the edge label (token ids on the edge from parent to
    this node). ``handle`` covers the cumulative prefix from root to
    the end of this node's edge (``prefix_len`` tokens total).
    ``refcount`` is the number of in-flight users protecting this node
    (and its ancestors) from eviction.
    """

    __slots__ = (
        "key",
        "handle",
        "parent",
        "children",
        "refcount",
        "last_used",
        "prefix_len",
    )

    def __init__(
        self,
        key: tuple[int, ...] = (),
        handle: H | None = None,
        parent: "_TreeNode | None" = None,
        prefix_len: int = 0,
    ) -> None:
        self.key: tuple[int, ...] = tuple(key)
        self.handle: H | None = handle
        self.parent: "_TreeNode | None" = parent
        self.children: dict[int, _TreeNode] = {}
        self.refcount: int = 0
        self.last_used: float = time.monotonic()
        self.prefix_len: int = prefix_len


def _common_prefix_len(a: tuple[int, ...], b: Sequence[int]) -> int:
    """Length of the shared leading prefix of ``a`` and ``b``."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class RadixKVCache(Generic[H]):
    """Generic in-memory radix tree over token-id sequences.

    Each node holds an edge label (token-id tuple), an opaque
    backend-supplied handle, a refcount (in-flight users, blocks
    eviction), and a ``last_used`` timestamp for LRU. The root
    represents the empty prefix and is never evicted.

    Thread-safe via a single ``threading.Lock`` around all public
    methods. Today the event loop is the only caller and the request
    queue serializes requests, so contention is zero — but the lock
    is cheap and removes a "works until concurrency changes" trap
    when Tier 2 batch scheduling arrives.
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        slice_fn: SliceFn[H],
    ) -> None:
        """Initialize the cache.

        Args:
            max_tokens: LRU token budget. ``None`` means unbounded.
                When set and an insert pushes total tokens over budget,
                leaf nodes with ``refcount == 0`` are evicted in
                least-``last_used`` order until under budget.
            slice_fn: Callable deriving a sub-handle. Used when an
                edge is split on a partial match.
        """
        self._max_tokens = max_tokens
        self._slice_fn = slice_fn
        self._root: _TreeNode[H] = _TreeNode(key=(), handle=None, prefix_len=0)
        self._total_tokens = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        # ponytail: global lock; per-node locks only if a future batch
        # scheduler contends. Cheap under serial execution today.
        self._lock = threading.Lock()

    def match(self, token_ids: Sequence[int]) -> PrefixMatch[H]:
        """Return the longest cached prefix of ``token_ids``."""
        with self._lock:
            node = self._root
            remaining = tuple(token_ids)
            now = time.monotonic()
            node.last_used = now
            while remaining:
                child = node.children.get(remaining[0])
                if child is None:
                    break
                k = _common_prefix_len(child.key, remaining)
                if k == len(child.key):
                    child.last_used = now
                    node = child
                    remaining = remaining[k:]
                elif k > 0:
                    # Partial edge match — splitting arrives in Task 2.
                    # For now, stop at the parent.
                    break
                else:
                    break
            if node is self._root:
                self._misses += 1
                return PrefixMatch(None, 0)
            self._hits += 1
            return PrefixMatch(node.handle, node.prefix_len)

    def insert(self, token_ids: Sequence[int], handle: H) -> None:
        """Insert (or refresh) a sequence -> handle mapping."""
        with self._lock:
            seq = tuple(token_ids)
            node = self._root
            now = time.monotonic()
            node.last_used = now
            pos = 0
            while pos < len(seq):
                first = seq[pos]
                child = node.children.get(first)
                if child is None:
                    remainder = seq[pos:]
                    leaf = _TreeNode(
                        key=remainder,
                        handle=handle,
                        parent=node,
                        prefix_len=node.prefix_len + len(remainder),
                    )
                    leaf.last_used = now
                    node.children[first] = leaf
                    self._total_tokens += len(remainder)
                    return
                k = _common_prefix_len(child.key, seq[pos:])
                if k == len(child.key):
                    child.last_used = now
                    node = child
                    pos += k
                elif k > 0:
                    # Splitting arrives in Task 2. For now, stop and
                    # refresh the deepest full-edge node.
                    node = child
                    pos += k
                    break
                else:
                    break
            node.handle = handle
            node.last_used = now

    def evict(self, num_tokens: int) -> int:
        """Force-evict leaf nodes. Implemented in Task 3."""
        return 0

    def pin(self, token_ids: Sequence[int]) -> None:
        """Increment refcount along the path. Implemented in Task 3."""
        return None

    def unpin(self, token_ids: Sequence[int]) -> None:
        """Decrement refcount along the path. Implemented in Task 3."""
        return None

    def stats(self) -> CacheStats:
        """Return a snapshot. Full implementation in Task 4."""
        with self._lock:
            return CacheStats(
                total_tokens=self._total_tokens,
                node_count=0,
                evictable_tokens=0,
                protected_tokens=0,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
