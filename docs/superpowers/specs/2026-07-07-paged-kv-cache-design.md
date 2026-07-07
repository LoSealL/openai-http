# Feature Specification: Paged Radix KV Cache for `openai_http`

**Feature Branch**: `003-paged-kv-cache`

**Created**: 2026-07-07

**Status**: Draft

**Input**: User description: "Design a new feature: paged kv cache. Like PageAttention and RadixAttention, store users prompt's kvcache in radix and paged manner, skip recalculation if cache hit. The stored cache can be in memory (for now) and in disk (later)."

## Context

`openai_http` is an OpenAI v1-compatible server with a pluggable backend system
(`BackendBase`). The package does **not** own model weights or tensors — real
backends (transformers, onnx) live outside the package, in `examples/` and
user code. The package owns only the HTTP layer, request queue
(`queue.py`, `asyncio.Semaphore(1)` → serial execution today), and a mock
backend.

A radix KV cache stores prompt prefix tensors by their token-id sequence so
that subsequent requests sharing a prefix (system prompt, few-shot examples,
multi-turn history) skip recomputing the prefill for that prefix. SGLang's
RadixCache is the reference implementation; vLLM uses a paged block pool
without a radix tree.

**Research finding**: No standalone, dependency-free Python library exists for
this. SGLang's `RadixCache` is the proven algorithm but is unextractable as a
dependency — it hard-imports `torch`, stores `torch.Tensor` values into a GPU
`token_to_kv_pool_allocator`, and couples to ~10 features we do not need
(session radix, KV events, bigram/eagle, page alignment, host backup,
priority eviction, hash-based distributed sync). vLLM has no prefix-matching.
PyPI radix-trie libraries (`pygtrie` etc.) provide only generic prefix-tree
substrate, missing KV-cache semantics (LRU eviction, path-refcount,
edge-split on partial match, token-budget).

**Conclusion**: ship a minimal in-package pure-Python module that ports
SGLang's algorithm without its tensor coupling. The seam between generic
core and tensor backend is a single callable the backend supplies.

## Goals & Non-Goals

### Goals

1. **Generic radix KV cache** in `openai_http.kvcache`, pure Python, no
   `torch` dependency, fully testable without a GPU or model.
2. **Longest-prefix matching** over token-id sequences against cached
   entries, with opaque backend-supplied handles as values.
3. **Edge-splitting** on partial match so future requests can match at any
   internal boundary, not just original insert boundaries.
4. **LRU eviction** with a configurable token budget; in-flight requests
   (refcount > 0) are protected.
5. **Backend opt-in** with zero changes to `BackendBase`, routers, server,
   `queue.py`, or the OpenAI wire shape. Existing backends keep working.
6. **Prefix-reuse value** (Tier 1): when a request shares a token-id prefix
   with a cached entry, the backend can skip prefill for that prefix.
7. **Disk-persistence seam**: define the serialization protocol now, ship
   only the in-memory store.

### Non-Goals (deferred to Tier 2 / future work)

- **Live cross-request batch sharing / continuous batching / paged-attention
  kernels.** This is a scheduler rewrite and pulls GPU-kernel concerns into
  the package. The data structures chosen here leave the door open: a future
  batch scheduler can consult the same radix tree per-batch instead of
  per-request without rework.
- **Disk persistence implementation.** Protocol is defined; implementation
  is a future `DiskKVStore`.
- **Changes to the OpenAI `usage` block.** Cached token counts are logged,
  not surfaced on the wire.
- **A custom paged-attention CUDA kernel.**

## Architecture

### Module layout

A new top-level package `openai_http/kvcache/`:

```
openai_http/kvcache/
├── __init__.py     # public exports: RadixKVCache, PrefixMatch, SliceFn
├── radix.py        # the radix tree + match / insert / evict
└── types.py        # PrefixMatch dataclass, SliceFn / Serializer protocols, CacheStats
```

No changes to `BackendBase`, the routers, `_server.py`, `queue.py`, or
`config.py`. The cache is invisible to the server; backends opt in by
constructing one in `setup()`.

### Why core-owns-the-tree, backend-owns-the-tensors

The package cannot allocate or interpret torch tensors — that's the existing
`BackendBase` boundary (backends receive `messages` and own tokenization +
model state). The radix tree, however, operates purely on token-id sequences
and treats the per-node value as opaque. Splitting the design at this seam
keeps the core dependency-free and lets each backend store whatever handle
shape its framework uses (e.g. HuggingFace `past_key_values`, ONNX KV state).

This is the seam SGLang lacks: SGLang's `TreeNode.value` is a `torch.Tensor`
of GPU pool indices, which is why their `RadixCache` cannot be extracted
without dragging in the memory pool. Our generic↔tensor bridge is a single
callable the backend supplies (`slice_fn`), and the core never sees a
tensor.

### Request data flow (backend-side)

1. Backend tokenizes the rendered prompt → `token_ids: Sequence[int]`.
2. `match = cache.match(token_ids)` → `PrefixMatch(handle, num_tokens)`.
   The contract: `handle` covers exactly the first `num_tokens` tokens of
   `token_ids`.
3. If `num_tokens > 0`: backend reattaches `handle` as its framework's KV
   state, feeds only `token_ids[num_tokens:]` to the model, and the prefill
   for the cached prefix is skipped.
4. Backend generates the suffix; assembles the full KV for the entire
   `token_ids` (matched prefix + new suffix).
5. `cache.insert(token_ids, full_handle)` — splits tree edges as needed so
   future partial matches work at any boundary.

## Core API

```python
# openai_http/kvcache/types.py

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

H = TypeVar("H", covariant=True)  # opaque backend handle type


@dataclass
class PrefixMatch(Generic[H]):
    """Result of a longest-prefix lookup.

    Attributes:
        handle: Opaque backend handle covering the first ``num_tokens``
            tokens of the queried sequence. ``None`` on total miss.
        num_tokens: Number of leading tokens ``handle`` covers. ``0`` on
            total miss or empty query.
    """
    handle: H | None
    num_tokens: int


class SliceFn(Protocol[H]):
    """Callable that derives a handle covering the first ``n`` tokens.

    Backends supply this so the radix tree can split a node when a
    partial match terminates inside an inserted segment. Memoized on
    the node after first computation.
    """

    def __call__(self, handle: H, n: int) -> H: ...


class Serializer(Protocol[H]):
    """Future disk-persistence seam. Not implemented in Tier 1.

    Defined now so a ``DiskKVStore`` can plug in later without changing
    ``RadixKVCache``'s public API.
    """

    def dumps(self, handle: H) -> bytes: ...
    def loads(self, data: bytes) -> H: ...


@dataclass
class CacheStats:
    """Snapshot for /metrics and logs."""
    total_tokens: int          # sum of len(key) over all live nodes
    node_count: int
    evictable_tokens: int      # tokens in leaf nodes with refcount == 0
    protected_tokens: int      # tokens in nodes with refcount > 0
    hits: int                  # cumulative match calls with num_tokens > 0
    misses: int                # cumulative match calls with num_tokens == 0
    evictions: int             # cumulative evicted token count
```

```python
# openai_http/kvcache/radix.py

from typing import Generic, TypeVar

from openai_http.kvcache.types import CacheStats, PrefixMatch, SliceFn

H = TypeVar("H")


class RadixKVCache(Generic[H]):
    """Generic in-memory radix tree over token-id sequences.

    Each node holds an edge label (a tuple of token ids), an opaque
    backend-supplied handle, a refcount (in-flight users, blocks
    eviction), and a ``last_used`` monotonic timestamp for LRU. The
    root represents the empty prefix and is never evicted.

    Thread-safe via a single ``threading.Lock`` around ``match`` /
    ``insert`` / ``evict``. Today the event loop is the only caller
    and the request queue serializes requests, so the lock is
    belt-and-suspenders — but cheap, and removes a "works until
    concurrency changes" trap.
    """

    def __init__(
        self,
        *,
        max_tokens: int | None = None,
        slice_fn: SliceFn[H],
    ) -> None:
        """
        Args:
            max_tokens: LRU token budget. ``None`` means unbounded
                (useful for tests and single-user dev). When set and
                an insert pushes total tokens over budget, leaf nodes
                with ``refcount == 0`` are evicted in least-``last_used``
                order until under budget.
            slice_fn: Callable used to derive a handle for an internal
                node when an edge must be split (partial match
                terminates inside a stored segment). The tree calls
                ``slice_fn(existing_handle, split_len)`` and memoizes
                the result on the new internal node. Every backend
                must supply one — it is a one-liner for tuple-shaped
                handles (``lambda h, n: h[:n]``) and a few lines for
                HF ``past_key_values``.
        """

    def match(self, token_ids: Sequence[int]) -> PrefixMatch[H]:
        """Return the longest cached prefix of ``token_ids``.

        Walks the tree from the root, advancing through children whose
        edge labels match successive prefixes of ``token_ids``. When a
        child's edge label diverges partway, the child is split once
        to expose a precise boundary, and the new internal node's
        handle becomes ``slice_fn(child.handle, split_len)``
        (memoized on the node).

        Updates ``last_used`` on every node touched along the matched
        path. Increments the cumulative hit/miss counters on
        ``CacheStats``.
        """

    def insert(self, token_ids: Sequence[int], handle: H) -> None:
        """Insert (or refresh) a sequence → handle mapping.

        Walks the tree, splitting edges as needed so that the inserted
        sequence ends at its own leaf. Existing overlapping nodes are
        left in place; only the new suffix becomes a new node. After
        insertion, runs lazy LRU eviction if over budget.
        """

    def evict(self, num_tokens: int) -> int:
        """Force-evict at least ``num_tokens`` worth of leaf nodes.

        Evicts leaves with ``refcount == 0`` in least-``last_used``
        order, pruning now-childless internal nodes. Returns the
        actual number of tokens evicted (may exceed the request).
        """

    def stats(self) -> CacheStats: ...
```

### The slicing contract

For partial matches to work across different prefixes, the cache must
sometimes produce a handle for an internal node covering `[0:k]` of an
existing segment when only a handle for `[0:N]` was stored. The cache
calls `slice_fn(full_handle, k)` lazily on the first split that needs
it and memoizes the result on the new internal node. Subsequent
matches at that boundary are O(1).

The contract for `slice_fn(handle, n)`:

- Input `handle` covers some prefix `P` of length `N >= n`.
- Returns a handle covering exactly the first `n` tokens of `P`.
- Must be composable: `slice_fn(slice_fn(h, m), n)` for `n <= m <= N`
  yields the same result as `slice_fn(h, n)`.
- For HuggingFace `past_key_values`, this is a tuple of
  `t[..., :n, :]` slices per layer — the backend owns this detail, the
  core never sees it.

`slice_fn` is required, not optional. Every backend that wants
prefix-reuse can supply one trivially; a backend that doesn't want
caching simply doesn't construct a `RadixKVCache`.

## Eviction

**Lazy LRU, triggered on `insert`.** When `max_tokens` is set and total
cached tokens exceed it after an insert:

1. Collect all leaf nodes with `refcount == 0`.
2. Heapify by `last_used` ascending.
3. Pop and delete leaves (plus prune now-childless internal parents) until
   total tokens ≤ `max_tokens` or no evictable leaves remain.

Budget unit is **tokens**, not bytes — the core cannot see tensor sizes.
This is the natural unit the core can enforce; backends tune the
tokens→bytes ratio via their model config (hidden size, num layers).

A `evict(num_tokens)` method exposes forced eviction for future use (e.g.
pre-flight cleanup before a known-large request, or a future Tier 2
scheduler managing pool capacity). Tier 1 does not call it directly.

## Thread-safety

A single `threading.Lock` guards `match` / `insert` / `evict`. Today
the event loop is the only caller and the request queue's semaphore
serializes requests, so contention is zero — but the lock is cheap and
removes a "works until concurrency changes" trap when Tier 2 batch
scheduling arrives.

```python
# ponytail: global lock; per-node locks only if a future batch scheduler contends.
```

## Disk-persistence seam (deferred)

The core cannot serialize opaque backend handles. The seam is the
`Serializer[H]` protocol (`dumps` / `loads`) the backend supplies. A
future `DiskKVStore` will:

1. Persist the radix tree topology (edges + node ids) as JSON or msgpack.
2. Use `Serializer.dumps(handle)` to write handle bytes alongside.
3. On load, reconstruct the tree and call `Serializer.loads` per node.

Tier 1 ships only the in-memory `RadixKVCache` and the protocol
definition. `RadixKVCache`'s public API (`match` / `insert` / `evict` /
`stats`) will not change when disk support lands.

## Backend integration pattern

Backends opt in by constructing a cache in `setup()` and consulting it
inside `generate()` / `generate_stream()`. The cache stores opaque
handles as-is — no serialization for in-memory use. The pattern (sketch
only — real integration lands in the transformers example, not the core
package):

```python
class CachingTransformersBackend(BackendBase):
    async def setup(self) -> None:
        # ... existing model load ...
        self.kv_cache = RadixKVCache(
            max_tokens=8192,
            slice_fn=lambda pkv, n: tuple(
                tuple(layer[..., :n, :] for layer in layer_pair)
                for layer_pair in pkv
            ),
        )

    async def generate(self, prompt, **kwargs) -> dict:
        token_ids = self._prepare_inputs(prompt, ...)   # -> list[int]
        match = self.kv_cache.match(token_ids)

        with torch.no_grad():
            if match.num_tokens > 0 and match.handle is not None:
                # Skip prefill on the cached prefix
                out = self.model.generate(
                    input_ids=torch.tensor([token_ids[match.num_tokens:]]),
                    past_key_values=match.handle,
                    max_new_tokens=kwargs["max_tokens"],
                )
            else:
                out = self.model.generate(
                    input_ids=torch.tensor([token_ids]),
                    max_new_tokens=kwargs["max_tokens"],
                )

        # Store the full KV directly — the cache treats it as opaque.
        self.kv_cache.insert(token_ids, out.past_key_values)
        # ... decode, parse, return ...
```

The handle stored is whatever the backend passes — typically the
framework's native KV state object (e.g. HF `past_key_values`). The
core never inspects or copies it.

## Testing

**Pure-Python unit tests** (`tests/unit/test_kvcache.py`), no GPU, no
torch, no model. Fake handle = the token-id tuple itself;
`slice_fn = lambda h, n: h[:n]`. Cases:

1. Total miss (empty tree) → `PrefixMatch(None, 0)`.
2. Exact-prefix hit — insert `[1,2,3]`, query `[1,2,3,4,5]` →
   `PrefixMatch(handle_for_123, 3)`.
3. Partial hit at edge boundary — insert `[1,2,3,4]`, query `[1,2,9]`
   → `PrefixMatch(handle_for_12, 2)`, with edge split creating a new
   internal node at `[1,2]`.
4. Edge splitting creates a memoized `slice_fn` result on the new
   internal node.
5. LRU eviction under budget: insert until over `max_tokens`, verify
   oldest leaves with `refcount == 0` are removed and total ≤ budget.
6. Refcount blocks eviction: a node with `refcount > 0` is not
   evicted even when it is the LRU candidate.
7. `stats()` returns consistent counts after a sequence of
   `match`/`insert`/`evict` operations.
8. Concurrent `match`/`insert` from multiple threads does not corrupt
   tree topology (smoke test for the lock).

**Integration smoke test** in the transformers example: serve two
requests with a shared system prompt, observe (via logs) that the second
request's prefill token count drops by the system prompt length.

## User Scenarios & Acceptance

### User Story 1 — Multi-turn chat skips prefill on prior turns (P1)

**Given** a backend that consults `RadixKVCache`, **when** a client
sends request N+1 of a multi-turn conversation whose first N turns
identical to request N (a common chat pattern), **then** the backend's
prefill compute covers only the new turn — the prior turns' KV is
reused from the cache. Logs report the cached token count.

**Independent test**: unit test #2 (exact-prefix hit) plus the
transformers-example integration smoke test.

### User Story 2 — Shared system prompt across unrelated requests (P1)

**Given** many requests share the same long system prompt (e.g. a
fixed persona or instruction set), **when** the second and subsequent
requests arrive within the cache's LRU window, **then** the system
prompt prefix is matched and its prefill is skipped.

**Independent test**: unit test #3 (partial hit at edge boundary)
covers this structurally.

### User Story 3 — Cache stays within memory budget (P2)

**Given** `max_tokens` is set, **when** the working set exceeds it,
**then** the least-recently-used leaf entries are evicted so total
tokens stays at or below `max_tokens`. In-flight requests whose KV is
still referenced are never evicted.

**Independent test**: unit tests #6 and #7.

### User Story 4 — Opt-in does not break existing backends (P1)

**Given** a backend that does not construct a `RadixKVCache`, **when**
the server runs, **then** behavior is identical to before — no
imports, no API changes, no perf regression. The cache module is
imported only by backends that use it.

**Independent test**: existing test suite passes unchanged; no new
imports in `BackendBase`, routers, `_server.py`, or `queue.py`.

### User Story 5 — Disk-persistence seam is forward-compatible (P3)

**Given** a `Serializer[H]` protocol is defined, **when** a future
`DiskKVStore` is added, **then** `RadixKVCache`'s public API
(`match` / `insert` / `evict` / `stats`) does not change.

**Independent test**: the protocol is defined in `types.py` and
referenced only by future code. No Tier 1 code imports `Serializer`.

## References

- SGLang `RadixCache`: `python/sglang/srt/mem_cache/radix_cache.py` at
  https://github.com/sgl-project/sglang — reference algorithm
  (`RadixKey.match` with exponential-search gallop, `TreeNode` with
  `lock_ref` + `last_access_time`, `_match_prefix_helper` with on-split,
  `_insert_helper` with edge split, `evict` with leaf-heap). Our
  implementation ports the algorithm without the torch coupling,
  memory-pool coupling, bigram/eagle/page-alignment, session radix,
  KV events, host backup, priority eviction, or hash-based distributed
  sync.
- vLLM block manager: paged block pool, no radix tree — different
  shape, not reused.
- RadixAttention paper: Zhong et al., "RadixAttention: Towards
  Structure-Aware Sharing of KV Cache for LLMs" (2023).

## Out of Scope (recap)

- Continuous batching / live cross-request batch sharing (Tier 2).
- Disk persistence implementation (seam only).
- Paged-attention CUDA kernel.
- Changes to `BackendBase`, routers, `_server.py`, `queue.py`,
  `config.py`, or the OpenAI wire shape.
- Surfacing cached token counts in the `usage` block (logged only).
