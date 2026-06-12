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

Request queue for concurrent request handling.

Uses asyncio.Semaphore for GPU serialization and asyncio.Queue for FIFO bound.
Returns 429 when queue is full.
"""

import asyncio
from contextlib import asynccontextmanager

from openai_http.errors import RateLimitError


class RequestQueue:
    """
    Bounded FIFO request queue with GPU serialization.

    Only one request executes at a time (semaphore).
    Additional requests wait in queue up to max depth.
    Returns 429 Too Many Requests when queue is full.
    """

    def __init__(self, max_depth: int = 32, retry_after: int = 5):
        """Initialize the request queue.

        Args:
            max_depth: Maximum number of requests allowed in the queue
                before rejecting with 429.
            retry_after: Suggested retry-after seconds sent with the
                429 response.
        """
        self.semaphore = asyncio.Semaphore(1)
        self.queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_depth)
        self.active_count = 0
        self.retry_after = retry_after

    @property
    def pending(self) -> int:
        """Return the number of requests currently waiting in the queue."""
        return self.queue.qsize()

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire queue slot and GPU access.

        Raises RateLimitError if queue is full.
        """
        if self.queue.full():
            raise RateLimitError(
                message=f"Request queue full ({self.queue.maxsize} pending). Please retry later.",
                code="queue_full",
            )

        await self.queue.put(1)
        try:
            async with self.semaphore:
                self.active_count += 1
                try:
                    yield
                finally:
                    self.active_count -= 1
        finally:
            await self.queue.get()
