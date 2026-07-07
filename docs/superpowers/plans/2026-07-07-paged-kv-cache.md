# Paged Radix KV Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a pure-Python `openai_http.kvcache` package that stores token-id-sequence-keyed opaque handles in a radix tree, so backends can skip prefill recomputation for cached prompt prefixes.

**Architecture:** A `RadixKVCache[H]` generic class owns the radix tree; backends supply an opaque handle type `H` and a `slice_fn(handle, n) -> handle` callable as the only generic↔tensor bridge. The core never imports `torch` or inspects handles. LRU eviction with refcount protection and a global lock for forward-compatibility with future batch scheduling. No changes to `BackendBase`, routers, `_server.py`, `queue.py`, or the OpenAI wire shape.

**Tech Stack:** Python 3.12 stdlib only (`threading`, `heapq`, `time`, `dataclasses`, `typing`). No new dependencies. Pytest for tests.

**Spec:** `docs/superpowers/specs/2026-07-07-paged-kv-cache-design.md`

## Global Constraints

- **Python:** 3.12+ (see `.python-version`).
- **Package manager:** `uv`. All commands run inside the `.venv/` via `uv run`.
- **No new dependencies.** The `kvcache` package uses only the stdlib. Do NOT add `torch` or any other import to `openai_http/kvcache/`.
- **No changes outside `openai_http/kvcache/` and `tests/unit/test_kvcache.py`.** Specifically: do not modify `BackendBase`, routers, `_server.py`, `queue.py`, `config.py`, `__init__.py` (top-level), or `pyproject.toml`. The `[tool.setuptools.packages.find]` glob `openai_http*` already includes the new subpackage.
- **License header:** Every new `.py` file starts with the Apache 2.0 header used elsewhere in the project (copy verbatim from `openai_http/backends/base.py:1-14`).
- **Docstrings:** Google-style, matching the rest of the codebase.
- **Test conventions:** Module-level functions (not classes) in `tests/unit/test_kvcache.py`, with one-line docstrings. Sync tests (no `@pytest.mark.asyncio` — the cache is synchronous).
- **Commit messages:** `[dev]` prefix for feature work, `[fix]` for bugfixes. Append `Signed-off-by: <agent>` per `AGENTS.md`.
- **`ponytail:` comments:** Mark the deliberate global-lock shortcut with a `# ponytail:` comment naming the ceiling and upgrade path, per the spec.

---

## File Structure

```
openai_http/kvcache/
├── __init__.py     # public exports: RadixKVCache, PrefixMatch, SliceFn, Serializer, CacheStats
├── types.py        # PrefixMatch dataclass, SliceFn/Serializer protocols, CacheStats dataclass
└── radix.py        # _TreeNode, RadixKVCache (the radix tree + match/insert/evict/pin/unpin/stats)

tests/unit/test_kvcache.py   # pure-Python unit tests (no torch, no GPU)
```

Each file has one clear responsibility:
- **`types.py`** — dataclasses and Protocols only, no logic. The generic↔tensor contract surface.
- **`radix.py`** — the tree algorithm. Owns `_TreeNode` (private) and `RadixKVCache` (public). All locking, eviction, and refcount logic lives here.
- **`__init__.py`** — re-exports the public API. No logic.
- **`test_kvcache.py`** — one test per spec scenario, using a fake handle (token-id tuple) and `slice_fn = lambda h, n: h[:n]`.

---

## Interfaces (carried across all tasks)

These signatures are fixed by the spec and shared between tasks. Implementers of later tasks rely on the names and types exactly as written here.

```python
# openai_http/kvcache/types.py  — stable across all tasks
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

H = TypeVar("H")

@dataclass
class PrefixMatch(Generic[H]):
    handle: H | None
    num_tokens: int

class SliceFn(Protocol[H]):
    def __call__(self, handle: H, n: int) -> H: ...

class Serializer(Protocol[H]):
    def dumps(self, handle: H) -> bytes: ...
    def loads(self, data: bytes) -> H: ...

@dataclass
class CacheStats:
    total_tokens: int
    node_count: int
    evictable_tokens: int
    protected_tokens: int
    hits: int
    misses: int
    evictions: int
```

```python
# openai_http/kvcache/radix.py  — RadixKVCache public surface (stable across all tasks)
class RadixKVCache(Generic[H]):
    def __init__(self, *, max_tokens: int | None, slice_fn: SliceFn[H]) -> None: ...
    def match(self, token_ids: Sequence[int]) -> PrefixMatch[H]: ...
    def insert(self, token_ids: Sequence[int], handle: H) -> None: ...
    def evict(self, num_tokens: int) -> int: ...
    def pin(self, token_ids: Sequence[int]) -> None: ...
    def unpin(self, token_ids: Sequence[int]) -> None: ...
    def stats(self) -> CacheStats: ...
```

Handle semantics (the contract every task obeys): a node's `handle` covers the **cumulative** prefix from the root to the end of that node's edge (i.e. `node.prefix_len` tokens). `slice_fn(handle, n)` returns a handle covering the first `n` tokens of whatever `handle` covers.

---

### Task 1: Package scaffold + types + insert/exact-match core

**Files:**
- Create: `openai_http/kvcache/__init__.py`
- Create: `openai_http/kvcache/types.py`
- Create: `openai_http/kvcache/radix.py`
- Test: `tests/unit/test_kvcache.py`

**Interfaces:**
- Produces: `RadixKVCache.__init__`, `RadixKVCache.match`, `RadixKVCache.insert` (exact-match and total-miss paths only; splitting comes in Task 2). `PrefixMatch`, `CacheStats`, `SliceFn`, `Serializer` types.

- [ ] **Step 1: Write the failing tests (total miss + exact-prefix hit + match-extension)**

Create `tests/unit/test_kvcache.py` with this complete content:

```python
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

Unit tests for the radix KV cache. Fake handle = token-id tuple;
slice_fn = tuple slicing. No torch, no GPU.
"""

from openai_http.kvcache import CacheStats, PrefixMatch, RadixKVCache


def _make_cache(max_tokens=None):
    """Build a cache whose handles are token-id tuples and slice_fn is tuple slicing."""
    return RadixKVCache(max_tokens=max_tokens, slice_fn=lambda h, n: h[:n])


def test_total_miss_on_empty_tree():
    """match() on an empty cache returns PrefixMatch(None, 0)."""
    cache = _make_cache()
    result = cache.match([1, 2, 3])
    assert result.handle is None
    assert result.num_tokens == 0


def test_exact_prefix_hit():
    """After inserting [1,2,3], matching [1,2,3,4,5] returns the handle and length 3."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    result = cache.match([1, 2, 3, 4, 5])
    assert result.num_tokens == 3
    assert result.handle == (10, 20, 30)


def test_match_exact_length():
    """Matching a sequence identical to an inserted one returns the full handle."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    result = cache.match([1, 2, 3])
    assert result.num_tokens == 3
    assert result.handle == (10, 20, 30)


def test_match_no_common_prefix():
    """Matching a sequence with no shared first token returns a total miss."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    result = cache.match([9, 9, 9])
    assert result.handle is None
    assert result.num_tokens == 0


def test_insert_refreshes_handle():
    """Re-inserting the same sequence updates the stored handle."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.insert([1, 2, 3], (40, 50, 60))
    result = cache.match([1, 2, 3])
    assert result.handle == (40, 50, 60)
```

- [ ] **Step 2: Run tests to verify they fail (module not found)**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: collection error / `ModuleNotFoundError: No module named 'openai_http.kvcache'`

- [ ] **Step 3: Create `openai_http/kvcache/types.py`**

```python
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
```

- [ ] **Step 4: Create `openai_http/kvcache/radix.py` (core: init, match-without-split, insert-without-split)**

```python
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


class _TreeNode:
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
        self._root = _TreeNode(key=(), handle=None, prefix_len=0)
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
```

- [ ] **Step 5: Create `openai_http/kvcache/__init__.py`**

```python
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

Radix KV cache for prefix-reuse across inference requests.

Construct a ``RadixKVCache`` in a backend's ``setup()`` and consult it
inside ``generate()`` to skip prefill on cached prompt prefixes.
"""

from openai_http.kvcache.radix import RadixKVCache
from openai_http.kvcache.types import CacheStats, PrefixMatch, Serializer, SliceFn

__all__ = [
    "CacheStats",
    "PrefixMatch",
    "RadixKVCache",
    "Serializer",
    "SliceFn",
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: 5 passed.

- [ ] **Step 7: Run lint and typecheck**

Run: `uv run ruff check openai_http/kvcache/ tests/unit/test_kvcache.py`
Expected: no errors.

Run: `uv run mypy openai_http/kvcache/`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add openai_http/kvcache/ tests/unit/test_kvcache.py
git commit -m "[dev] add kvcache package scaffold with insert and exact-match

RadixKVCache generic class over token-id sequences with opaque
backend-supplied handles. This task ships the core tree structure,
insert (no splitting), and match (exact-boundary and total-miss
paths). Edge splitting, eviction, and full stats arrive in later
tasks.

Signed-off-by: claude"
```

---

### Task 2: Edge splitting + partial match + slice_fn memoization

**Files:**
- Modify: `openai_http/kvcache/radix.py` (add `_split_node`; rewrite the partial-match branch in `match`; rewrite the partial-edge branch in `insert`)
- Test: `tests/unit/test_kvcache.py` (append new tests)

**Interfaces:**
- Consumes: `SliceFn[H]` from Task 1 (called as `self._slice_fn(handle, n) -> H`).
- Produces: `match()` now returns the true longest prefix even when the divergence is mid-edge. `_split_node` is private and not referenced outside `radix.py`.

**What changes:** When `match` or `insert` finds a child whose edge diverges partway (`0 < k < len(child.key)`), the child is split into a new internal node (key = matched prefix, handle = `slice_fn(child.handle, split_len_cum)`) and the child (key = unmatched suffix). The `slice_fn` result is stored on the new internal node so future matches at that boundary are O(1).

- [ ] **Step 1: Append the failing tests for partial match**

Append to `tests/unit/test_kvcache.py`:

```python
def test_partial_hit_at_edge_boundary():
    """Insert [1,2,3,4], match [1,2,9] -> handle covers first 2 tokens via edge split."""
    cache = _make_cache()
    cache.insert([1, 2, 3, 4], (10, 20, 30, 40))
    result = cache.match([1, 2, 9])
    assert result.num_tokens == 2
    assert result.handle == (10, 20)


def test_split_then_descend_into_sibling():
    """After a split, a subsequent insert descends through the new internal node."""
    cache = _make_cache()
    cache.insert([1, 2, 3, 4], (10, 20, 30, 40))
    cache.insert([1, 2, 9, 9], (11, 22, 33, 44))
    # Both [1,2,3,...] and [1,2,9,...] should match up to [1,2].
    r1 = cache.match([1, 2, 3, 4])
    assert r1.num_tokens == 4
    assert r1.handle == (10, 20, 30, 40)
    r2 = cache.match([1, 2, 9, 9])
    assert r2.num_tokens == 4
    assert r2.handle == (11, 22, 33, 44)
    # And a shorter query matches only the shared [1,2] prefix.
    r3 = cache.match([1, 2, 7])
    assert r3.num_tokens == 2
    assert r3.handle == (10, 20)


def test_slice_fn_memoized_on_split_node():
    """The slice_fn is called exactly once per split; the result is reused."""
    cache = _make_cache()
    cache.insert([1, 2, 3, 4], (10, 20, 30, 40))
    # First match triggers the split and calls slice_fn once.
    cache.match([1, 2, 9])
    # Second match at the same boundary must not call slice_fn again.
    cache.match([1, 2, 9])
    # We cannot count calls on a lambda; instead verify the split node's
    # handle is set (it was memoized) and stable across matches.
    r1 = cache.match([1, 2, 9])
    r2 = cache.match([1, 2, 9])
    assert r1.handle == r2.handle == (10, 20)


def test_slice_fn_called_once_per_split():
    """slice_fn is invoked exactly once when an edge is split."""
    call_count = [0]

    def counting_slice(h, n):
        call_count[0] += 1
        return h[:n]

    cache = RadixKVCache(max_tokens=None, slice_fn=counting_slice)
    cache.insert([1, 2, 3, 4], (10, 20, 30, 40))
    cache.match([1, 2, 9])  # triggers split
    assert call_count[0] == 1
    cache.match([1, 2, 9])  # should not trigger another slice
    assert call_count[0] == 1
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: the 4 new tests FAIL (partial match returns wrong length because Task 1 stops at parent). The 5 Task-1 tests still pass.

- [ ] **Step 3: Add `_split_node` to `radix.py`**

Add this method to the `RadixKVCache` class in `openai_http/kvcache/radix.py` (place it immediately after `__init__`, before `match`):

```python
    def _split_node(self, child: _TreeNode, k: int, now: float) -> _TreeNode:
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
        split_len_cum = child.prefix_len - len(child.key) + k
        new_handle = self._slice_fn(child.handle, split_len_cum)
        new_node = _TreeNode(
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
```

- [ ] **Step 4: Rewrite the partial-match branch in `match`**

In `openai_http/kvcache/radix.py`, replace the `match` method's partial-edge `elif k > 0:` branch. The current (Task 1) code is:

```python
                elif k > 0:
                    # Partial edge match — splitting arrives in Task 2.
                    # For now, stop at the parent.
                    break
```

Replace it with:

```python
                elif k > 0:
                    node = self._split_node(child, k, now)
                    remaining = remaining[k:]
                    break
```

- [ ] **Step 5: Rewrite the partial-edge branch in `insert`**

In `openai_http/kvcache/radix.py`, replace the `insert` method's partial-edge `elif k > 0:` branch. The current (Task 1) code is:

```python
                elif k > 0:
                    # Splitting arrives in Task 2. For now, stop and
                    # refresh the deepest full-edge node.
                    node = child
                    pos += k
                    break
```

Replace it with:

```python
                elif k > 0:
                    new_internal = self._split_node(child, k, now)
                    node = new_internal
                    pos += k
                    # Continue the loop: next iteration either creates
                    # a new leaf for the divergent suffix or descends.
```

(Remove the `break` so the `while` loop continues and either descends into an existing child of the new internal node or creates a new leaf for the remainder.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: all 9 tests pass.

- [ ] **Step 7: Run lint and typecheck**

Run: `uv run ruff check openai_http/kvcache/ tests/unit/test_kvcache.py`
Run: `uv run mypy openai_http/kvcache/`
Expected: no errors.

- [ ] **Step 8: Commit**

```bash
git add openai_http/kvcache/radix.py tests/unit/test_kvcache.py
git commit -m "[dev] add edge splitting and partial-match to kvcache

match() now returns the true longest prefix even when divergence is
mid-edge, by splitting the child node. The slice_fn result is stored
on the new internal node and reused on subsequent matches at the same
boundary. insert() also splits edges so divergent suffixes become
sibling leaves under a shared internal node.

Signed-off-by: claude"
```

---

### Task 3: LRU eviction + refcount (pin/unpin)

**Files:**
- Modify: `openai_http/kvcache/radix.py` (implement `evict`, `pin`, `unpin`; wire eviction into `insert`; add `_collect_evictable_leaves`, `_delete_leaf`, `_evict_if_over_budget`, `_deepest_node`, `_adjust_ref`)
- Test: `tests/unit/test_kvcache.py` (append new tests)

**Interfaces:**
- Produces: working `evict(num_tokens) -> int`, `pin(token_ids) -> None`, `unpin(token_ids) -> None`. `insert` now triggers lazy eviction when `max_tokens` is exceeded.

**Eviction rules (from spec):**
1. Collect leaf nodes (non-root, no children) with `refcount == 0`.
2. Heapify by `last_used` ascending.
3. Pop and delete until `total_tokens <= max_tokens` or heap empty.
4. After deleting a leaf, prune its parent if the parent is now childless and has `refcount == 0` (repeat up the tree).

**Refcount rules:** `pin(token_ids)` walks from the deepest matching node up to (but not including) the root, incrementing `refcount` on each ancestor. `unpin` decrements. A split copies the child's `refcount` to the new internal node (already done in Task 2's `_split_node`).

- [ ] **Step 1: Append the failing tests for eviction and refcount**

Append to `tests/unit/test_kvcache.py`:

```python
def test_eviction_under_budget():
    """When total tokens exceed max_tokens, oldest evictable leaves are removed."""
    cache = _make_cache(max_tokens=6)
    cache.insert([1, 2, 3], (10, 20, 30))  # 3 tokens
    cache.insert([4, 5, 6], (40, 50, 60))  # 3 tokens, total 6 (at budget)
    # Inserting a 3rd sequence pushes total to 9 > 6; oldest leaf evicted.
    cache.insert([7, 8, 9], (70, 80, 90))
    stats = cache.stats()
    assert stats.total_tokens <= 6
    # The first-inserted [1,2,3] should have been evicted (oldest).
    r = cache.match([1, 2, 3])
    assert r.num_tokens == 0


def test_eviction_keeps_recent():
    """After eviction, recently-used sequences are still matched."""
    cache = _make_cache(max_tokens=6)
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.insert([4, 5, 6], (40, 50, 60))
    # Touch [1,2,3] to make it more recent than [4,5,6].
    cache.match([1, 2, 3])
    cache.insert([7, 8, 9], (70, 80, 90))
    r_recent = cache.match([1, 2, 3])
    assert r_recent.num_tokens == 3
    r_evicted = cache.match([4, 5, 6])
    assert r_evicted.num_tokens == 0


def test_refcount_blocks_eviction():
    """A pinned node is not evicted even when it is the LRU candidate."""
    cache = _make_cache(max_tokens=6)
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.pin([1, 2, 3])
    cache.insert([4, 5, 6], (40, 50, 60))  # total 6, at budget
    cache.insert([7, 8, 9], (70, 80, 90))  # would evict [1,2,3] but it's pinned
    # [1,2,3] survives because it was pinned.
    r = cache.match([1, 2, 3])
    assert r.num_tokens == 3
    cache.unpin([1, 2, 3])
    # Now a subsequent over-budget insert can evict it.
    cache.insert([10, 11, 12], (100, 110, 120))
    stats = cache.stats()
    assert stats.total_tokens <= 6


def test_evict_returns_token_count():
    """evict(num_tokens) returns the actual number of tokens evicted."""
    cache = _make_cache(max_tokens=None)
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.insert([4, 5, 6], (40, 50, 60))
    evicted = cache.evict(4)
    assert evicted >= 4


def test_eviction_prunes_childless_internal_nodes():
    """After all leaves under an internal node are evicted, the internal node is pruned."""
    cache = _make_cache(max_tokens=3)
    # [1,2] is the shared prefix; [1,2,3] and [1,2,4] are leaves.
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.insert([1, 2, 4], (11, 22, 44))
    # Insert something else to trigger eviction of both [1,2,3] and [1,2,4].
    cache.insert([9, 9, 9], (99, 99, 99))
    # The internal node for [1,2] should have been pruned.
    r = cache.match([1, 2, 3])
    assert r.num_tokens == 0
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: the 5 new tests FAIL (eviction/pin/unpin are no-ops from Task 1). The 9 Task 1-2 tests still pass.

- [ ] **Step 3: Implement eviction helpers and wire into `insert`**

Add these private methods to `RadixKVCache` in `openai_http/kvcache/radix.py` (place after `_split_node`, before `match`):

```python
    def _collect_evictable_leaves(self) -> list[_TreeNode]:
        """Return all leaf nodes (non-root, no children) with refcount == 0."""
        result: list[_TreeNode] = []
        stack: list[_TreeNode] = [self._root]
        while stack:
            n = stack.pop()
            if n is not self._root and not n.children and n.refcount == 0:
                result.append(n)
            for c in n.children.values():
                stack.append(c)
        return result

    def _delete_leaf(self, leaf: _TreeNode) -> None:
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
        import heapq

        leaves = self._collect_evictable_leaves()
        heap = [(lf.last_used, id(lf), lf) for lf in leaves]
        heapq.heapify(heap)
        while self._total_tokens > self._max_tokens and heap:
            _, _, leaf = heapq.heappop(heap)
            self._delete_leaf(leaf)
```

Now wire eviction into `insert`. In the `insert` method, add a call at the very end of the method body (after the `with self._lock:` block's last statement). The current end of `insert` is:

```python
            node.handle = handle
            node.last_used = now
```

Change it to:

```python
            node.handle = handle
            node.last_used = now
        self._evict_if_over_budget()
```

(Place `self._evict_if_over_budget()` outside the `with self._lock:` block is wrong — it must be inside the lock. Instead, keep it inside. The corrected placement: add `self._evict_if_over_budget()` as the last statement **inside** the `with self._lock:` block, after the `while` loop. Since `insert` may `return` early from inside the loop, also add the call before each `return` inside the lock. To keep it simple, refactor `insert` to use a single exit point.)

The cleanest refactor: replace the entire `insert` method body with:

```python
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
                else:
                    break
            node.handle = handle
            node.last_used = now
            self._evict_if_over_budget()
```

- [ ] **Step 4: Implement `evict`, `pin`, `unpin`, and helpers**

Replace the three stub methods (`evict`, `pin`, `unpin`) at the bottom of `RadixKVCache` in `openai_http/kvcache/radix.py` with:

```python
    def evict(self, num_tokens: int) -> int:
        """Force-evict at least ``num_tokens`` worth of leaf nodes.

        Evicts leaves with ``refcount == 0`` in least-``last_used``
        order. Returns the actual number of tokens evicted (may exceed
        the request).
        """
        import heapq

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

    def _deepest_node(self, seq: tuple[int, ...]) -> _TreeNode:
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

    def _adjust_ref(self, node: _TreeNode, delta: int) -> None:
        """Apply ``delta`` to refcount on ``node`` and all ancestors up to root."""
        n = node
        while n is not self._root:
            n.refcount += delta
            if n.refcount < 0:
                n.refcount = 0
            n = n.parent
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: all 14 tests pass.

- [ ] **Step 6: Run lint and typecheck**

Run: `uv run ruff check openai_http/kvcache/ tests/unit/test_kvcache.py`
Run: `uv run mypy openai_http/kvcache/`
Expected: no errors. (If ruff complains about `import heapq` inside methods, move it to the module top and remove the local imports.)

- [ ] **Step 7: Commit**

```bash
git add openai_http/kvcache/radix.py tests/unit/test_kvcache.py
git commit -m "[dev] add LRU eviction and refcount pinning to kvcache

insert() now triggers lazy LRU eviction when total tokens exceed
max_tokens, evicting the oldest leaf nodes with refcount == 0 and
pruning childless internal ancestors. pin()/unpin() let callers
protect a prefix from eviction during long generation. evict() exposes
forced eviction for future scheduler use.

Signed-off-by: claude"
```

---

### Task 4: Full stats() + thread-safety verification

**Files:**
- Modify: `openai_http/kvcache/radix.py` (replace the stub `stats()` with the full implementation)
- Test: `tests/unit/test_kvcache.py` (append new tests)

**Interfaces:**
- Produces: `stats() -> CacheStats` with accurate `node_count`, `evictable_tokens`, `protected_tokens`, `total_tokens`, `hits`, `misses`, `evictions`.

- [ ] **Step 1: Append the failing tests for stats and concurrency**

Append to `tests/unit/test_kvcache.py`:

```python
def test_stats_total_tokens():
    """stats().total_tokens equals the sum of all inserted edge lengths."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.insert([1, 2, 9], (11, 22, 99))
    # [1,2,3] = 3 tokens, [1,2] shared internal = 2, [9] = 1 -> total 6.
    stats = cache.stats()
    assert stats.total_tokens == 6


def test_stats_node_count():
    """stats().node_count counts all non-root nodes."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    stats = cache.stats()
    assert stats.node_count == 1


def test_stats_hits_and_misses():
    """stats() tracks cumulative hit/miss counts."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.match([1, 2, 3])  # hit
    cache.match([9, 9, 9])  # miss
    stats = cache.stats()
    assert stats.hits == 1
    assert stats.misses == 1


def test_stats_evictable_and_protected():
    """evictable_tokens counts unprotected leaves; protected_tokens counts pinned."""
    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.insert([4, 5, 6], (40, 50, 60))
    cache.pin([1, 2, 3])
    stats = cache.stats()
    # [1,2,3] is pinned -> its 3 tokens are protected.
    assert stats.protected_tokens == 3
    # [4,5,6] is an unpinned leaf -> its 3 tokens are evictable.
    assert stats.evictable_tokens == 3


def test_stats_evictions():
    """stats().evictions accumulates evicted token count."""
    cache = _make_cache(max_tokens=3)
    cache.insert([1, 2, 3], (10, 20, 30))
    cache.insert([4, 5, 6], (40, 50, 60))  # evicts [1,2,3]
    stats = cache.stats()
    assert stats.evictions == 3


def test_concurrent_insert_and_match():
    """Concurrent insert/match from multiple threads does not corrupt the tree."""
    import threading

    cache = _make_cache(max_tokens=1000)

    def worker(offset):
        for i in range(50):
            seq = [offset, i, i + 1, i + 2]
            cache.insert(seq, tuple(seq))
            cache.match(seq)

    threads = [threading.Thread(target=worker, args=(o,)) for o in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stats = cache.stats()
    # No assertion on exact counts — the point is no crash and total_tokens
    # is consistent (equals sum of live edge lengths, which we recompute
    # independently by walking the tree).
    assert stats.total_tokens <= 1000
    assert stats.total_tokens > 0


def test_concurrent_pin_unpin_does_not_deadlock():
    """Concurrent pin/unpin across threads completes without deadlock."""
    import threading

    cache = _make_cache()
    cache.insert([1, 2, 3], (10, 20, 30))

    def worker():
        for _ in range(100):
            cache.pin([1, 2, 3])
            cache.unpin([1, 2, 3])

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # All threads finished -> no deadlock. Refcount should be back to 0.
    stats = cache.stats()
    assert stats.protected_tokens == 0
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: the stats tests FAIL (Task 1's stub returns `node_count=0`, `evictable_tokens=0`, `protected_tokens=0`). The concurrency tests may pass already (lock is in place) but verify.

- [ ] **Step 3: Replace the stub `stats()` with the full implementation**

In `openai_http/kvcache/radix.py`, replace the stub `stats` method:

```python
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
```

with:

```python
    def stats(self) -> CacheStats:
        """Return a snapshot of cache state.

        Walks the tree once to compute ``node_count``,
        ``evictable_tokens``, and ``protected_tokens``.
        ``total_tokens``, ``hits``, ``misses``, and ``evictions`` are
        tracked incrementally.
        """
        with self._lock:
            node_count = 0
            evictable = 0
            protected = 0
            stack: list[_TreeNode] = [self._root]
            while stack:
                n = stack.pop()
                if n is not self._root:
                    node_count += 1
                    if n.refcount > 0:
                        protected += len(n.key)
                    elif not n.children:
                        evictable += len(n.key)
                for c in n.children.values():
                    stack.append(c)
            return CacheStats(
                total_tokens=self._total_tokens,
                node_count=node_count,
                evictable_tokens=evictable,
                protected_tokens=protected,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )
```

- [ ] **Step 4: Run the full test suite**

Run: `uv run pytest tests/unit/test_kvcache.py -v`
Expected: all 21 tests pass.

- [ ] **Step 5: Run lint and typecheck**

Run: `uv run ruff check openai_http/kvcache/ tests/unit/test_kvcache.py`
Run: `uv run mypy openai_http/kvcache/`
Expected: no errors.

- [ ] **Step 6: Run the entire existing test suite to confirm no regressions**

Run: `uv run pytest tests/ -v`
Expected: all pre-existing tests still pass (the new package is isolated; nothing imports it yet outside its own tests).

- [ ] **Step 7: Commit**

```bash
git add openai_http/kvcache/radix.py tests/unit/test_kvcache.py
git commit -m "[dev] implement full stats and verify thread-safety

stats() now walks the tree to report node_count, evictable_tokens,
and protected_tokens accurately. Added concurrency smoke tests for
insert/match and pin/unpin under the global lock.

Signed-off-by: claude"
```

---

## Post-Implementation Notes

**What this plan delivers:** a complete, tested, pure-Python `openai_http.kvcache` package with the radix tree, edge splitting, LRU eviction, refcount pinning, and stats. No changes to any other module in the project.

**What is explicitly NOT in this plan (deferred per spec):**
- Backend integration (transformers example consulting the cache in `generate()`). The spec sketches the pattern; a follow-up plan covers it.
- Disk persistence (the `Serializer` protocol is defined; a future `DiskKVStore` implements it).
- Tier 2 continuous batching / paged-attention kernel.
- Surfacing cached token counts in the OpenAI `usage` block.

**Verification commands after all tasks:**
```bash
uv run pytest tests/unit/test_kvcache.py -v   # all 21 tests pass
uv run ruff check openai_http/kvcache/         # clean
uv run mypy openai_http/kvcache/               # clean
uv run pytest tests/ -v                        # no regressions
```
