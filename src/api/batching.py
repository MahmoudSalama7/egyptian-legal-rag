"""Adaptive micro-batching for embedding and reranking inference.

Collects individual requests into micro-batches for higher GPU/CPU
throughput.  When the batch window elapses or the max-batch-size is reached
the accumulated items are processed together and each caller gets its own
result via an ``asyncio.Future``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class _PendingItem:
    """A single item waiting in the micro-batch queue."""

    payload: Any
    future: asyncio.Future = field(
        default_factory=lambda: asyncio.get_event_loop().create_future()
    )
    enqueue_time: float = field(default_factory=time.monotonic)


class MicroBatcher:
    """Adaptive micro-batcher that groups incoming requests.

    Parameters
    ----------
    process_fn:
        Synchronous callable that accepts ``list[Any]`` (list of payloads) and
        returns ``list[Any]`` (corresponding results, same length & order).
    max_batch_size:
        Hard cap on items per batch.
    max_wait_ms:
        Maximum milliseconds to wait before flushing a partial batch.
    """

    def __init__(
        self,
        process_fn: Callable[[list[Any]], list[Any]],
        max_batch_size: int = 16,
        max_wait_ms: float = 50.0,
    ) -> None:
        self._process_fn = process_fn
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._queue: asyncio.Queue[_PendingItem] = asyncio.Queue()
        self._flush_task: asyncio.Task | None = None
        self._running = False
        self._stats_batches = 0
        self._stats_items = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Drain remaining items and stop."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Flush any remaining items
        await self._flush_once()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def submit(self, payload: Any) -> Any:
        """Submit a single item and await its result."""
        item = _PendingItem(payload=payload)
        await self._queue.put(item)
        return await item.future

    @property
    def stats(self) -> dict[str, int]:
        return {
            "batches_processed": self._stats_batches,
            "items_processed": self._stats_items,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    async def _flush_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._max_wait_ms / 1000.0)
            await self._flush_once()

    async def _flush_once(self) -> None:
        items: list[_PendingItem] = []
        while not self._queue.empty() and len(items) < self._max_batch_size:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not items:
            return

        payloads = [it.payload for it in items]
        loop = asyncio.get_event_loop()

        try:
            # Run the (sync) processing function in a thread so we don't
            # block the event loop.
            results = await loop.run_in_executor(None, self._process_fn, payloads)
            for it, res in zip(items, results):
                if not it.future.done():
                    it.future.set_result(res)
            self._stats_batches += 1
            self._stats_items += len(items)
            logger.debug(
                "Flushed micro-batch: size=%d  latency=%.1fms",
                len(items),
                (time.monotonic() - items[0].enqueue_time) * 1000,
            )
        except Exception as exc:
            for it in items:
                if not it.future.done():
                    it.future.set_exception(exc)
            logger.exception("Micro-batch processing failed")
