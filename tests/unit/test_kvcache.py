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

from openai_http.kvcache import RadixKVCache


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
