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

import heapq
import threading
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
        self.last_used: float = 0.0
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

    ``last_used`` is a logical clock (monotonic integer tick), not
    wall-clock time: ``time.monotonic()`` on Windows has ~15ms
    resolution, so back-to-back inserts in the same request would
    share a timestamp and break LRU ordering. A counter guarantees
    strict ordering under tight loops.
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
        # Logical clock: each tick is one observed operation. Cheaper
        # and finer-grained than time.monotonic() under tight loops.
        self._clock = 0

    def _tick(self) -> int:
        """Advance the logical clock and return the new timestamp."""
        self._clock += 1
        return self._clock

    def _split_node(self, child: _TreeNode[H], k: int, now: float) -> _TreeNode[H]:
        """Split ``child`` into a new internal node + ``child``.

        ``child.key`` is split at position ``k``: the new internal node
        gets ``child.key[:k]`` and a handle covering the cumulative
        prefix up to the split; ``child`` keeps ``child.key[k:]`` and
        its original handle.

        Args:
            child: The node to split.
            k: Number of leading tokens of ``child.key`` that match.
            now: Current monotonic timestamp for ``last_used``.

        Returns:
            The new internal node (which replaces ``child`` in its
            parent's children dict).
        """
        parent = child.parent
        assert parent is not None
        # A splittable child is never the root, so it always has a real handle.
        assert child.handle is not None
        split_len_cum = child.prefix_len - len(child.key) + k
        new_handle = self._slice_fn(child.handle, split_len_cum)
        new_node = _TreeNode[H](
            key=child.key[:k],
            handle=new_handle,
            parent=parent,
            prefix_len=split_len_cum,
        )
        new_node.refcount = child.refcount
        new_node.last_used = now
        child_remainder_first = child.key[k]
        new_node.children[child_remainder_first] = child
        child.key = child.key[k:]
        child.parent = new_node
        parent.children[new_node.key[0]] = new_node
        return new_node

    def _collect_evictable_leaves(self) -> list[_TreeNode[H]]:
        """Return all leaf nodes (non-root, no children) with refcount == 0."""
        result: list[_TreeNode[H]] = []
        stack: list[_TreeNode[H]] = [self._root]
        while stack:
            n = stack.pop()
            if n is not self._root and not n.children and n.refcount == 0:
                result.append(n)
            for c in n.children.values():
                stack.append(c)
        return result

    def _delete_leaf(self, leaf: _TreeNode[H]) -> None:
        """Detach ``leaf`` from its parent and prune childless ancestors."""
        parent = leaf.parent
        if parent is None:
            return
        del parent.children[leaf.key[0]]
        self._total_tokens -= len(leaf.key)
        self._evictions += len(leaf.key)
        # Prune now-childless internal nodes with refcount == 0.
        while (
            parent is not self._root
            and not parent.children
            and parent.refcount == 0
        ):
            grand = parent.parent
            assert grand is not None
            del grand.children[parent.key[0]]
            self._total_tokens -= len(parent.key)
            parent = grand

    def _evict_if_over_budget(self) -> None:
        """Lazy LRU eviction: evict oldest evictable leaves until under budget."""
        if self._max_tokens is None:
            return
        if self._total_tokens <= self._max_tokens:
            return
        leaves = self._collect_evictable_leaves()
        heap = [(lf.last_used, id(lf), lf) for lf in leaves]
        heapq.heapify(heap)
        while self._total_tokens > self._max_tokens and heap:
            _, _, leaf = heapq.heappop(heap)
            self._delete_leaf(leaf)

    def match(self, token_ids: Sequence[int]) -> PrefixMatch[H]:
        """Return the longest cached prefix of ``token_ids``."""
        with self._lock:
            node = self._root
            remaining = tuple(token_ids)
            now = self._tick()
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
                    node = self._split_node(child, k, now)
                    remaining = remaining[k:]
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
            now = self._tick()
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
                    self._evict_if_over_budget()
                    return
                k = _common_prefix_len(child.key, seq[pos:])
                if k == len(child.key):
                    child.last_used = now
                    node = child
                    pos += k
                elif k > 0:
                    new_internal = self._split_node(child, k, now)
                    node = new_internal
                    pos += k
                    # Continue the loop: next iteration either creates
                    # a new leaf for the divergent suffix or descends.
                else:
                    break
            node.handle = handle
            node.last_used = now
            self._evict_if_over_budget()

    def evict(self, num_tokens: int) -> int:
        """Force-evict at least ``num_tokens`` worth of leaf nodes.

        Evicts leaves with ``refcount == 0`` in least-``last_used``
        order. Returns the actual number of tokens evicted (may exceed
        the request).
        """
        with self._lock:
            leaves = self._collect_evictable_leaves()
            heap = [(lf.last_used, id(lf), lf) for lf in leaves]
            heapq.heapify(heap)
            evicted = 0
            while evicted < num_tokens and heap:
                _, _, leaf = heapq.heappop(heap)
                before = self._total_tokens
                self._delete_leaf(leaf)
                evicted += before - self._total_tokens
            return evicted

    def pin(self, token_ids: Sequence[int]) -> None:
        """Increment refcount along the path to the deepest matching node.

        Protects the matched prefix from eviction. Pair with ``unpin``
        when the caller is done using the handle.
        """
        with self._lock:
            node = self._deepest_node(tuple(token_ids))
            self._adjust_ref(node, +1)

    def unpin(self, token_ids: Sequence[int]) -> None:
        """Decrement refcount along the path to the deepest matching node."""
        with self._lock:
            node = self._deepest_node(tuple(token_ids))
            self._adjust_ref(node, -1)

    def _deepest_node(self, seq: tuple[int, ...]) -> _TreeNode[H]:
        """Walk the tree to the deepest node fully covered by ``seq``."""
        node = self._root
        remaining = seq
        while remaining:
            child = node.children.get(remaining[0])
            if child is None:
                break
            k = _common_prefix_len(child.key, remaining)
            if k == len(child.key):
                node = child
                remaining = remaining[k:]
            else:
                break
        return node

    def _adjust_ref(self, node: _TreeNode[H], delta: int) -> None:
        """Apply ``delta`` to refcount on ``node`` and all ancestors up to root."""
        n: _TreeNode[H] | None = node
        while n is not None and n is not self._root:
            n.refcount += delta
            if n.refcount < 0:
                n.refcount = 0
            n = n.parent

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
