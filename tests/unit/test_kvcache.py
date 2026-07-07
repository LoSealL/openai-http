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
