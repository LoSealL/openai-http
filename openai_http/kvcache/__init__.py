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
