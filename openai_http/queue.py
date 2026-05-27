"""
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
        self.semaphore = asyncio.Semaphore(1)
        self.queue = asyncio.Queue(maxsize=max_depth)
        self.active_count = 0
        self.retry_after = retry_after

    @property
    def pending(self) -> int:
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
