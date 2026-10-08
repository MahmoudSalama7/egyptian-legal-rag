"""Tests for the adaptive micro-batching engine."""

from __future__ import annotations

import asyncio

import pytest

from src.api.batching import MicroBatcher


def _double_batch(payloads: list[int]) -> list[int]:
    """Simple batch function that doubles each item."""
    return [x * 2 for x in payloads]


def _failing_batch(payloads: list) -> list:
    """Batch function that always raises."""
    raise ValueError("Intentional batch failure")


@pytest.mark.asyncio
async def test_microbatcher_single_item():
    """Single-item submit should return the correct result."""
    batcher = MicroBatcher(process_fn=_double_batch, max_batch_size=4, max_wait_ms=10)
    await batcher.start()
    try:
        result = await batcher.submit(5)
        assert result == 10
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_microbatcher_multiple_items():
    """Multiple concurrent submits should all resolve correctly."""
    batcher = MicroBatcher(process_fn=_double_batch, max_batch_size=16, max_wait_ms=20)
    await batcher.start()
    try:
        tasks = [batcher.submit(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        assert sorted(results) == [i * 2 for i in range(10)]
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_microbatcher_stats():
    """Stats should reflect processed batches and items."""
    batcher = MicroBatcher(process_fn=_double_batch, max_batch_size=4, max_wait_ms=10)
    await batcher.start()
    try:
        await batcher.submit(1)
        await asyncio.sleep(0.05)
        stats = batcher.stats
        assert stats["items_processed"] >= 1
        assert stats["batches_processed"] >= 1
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_microbatcher_error_propagation():
    """Errors in the batch function should propagate to callers."""
    batcher = MicroBatcher(process_fn=_failing_batch, max_batch_size=4, max_wait_ms=10)
    await batcher.start()
    try:
        with pytest.raises(ValueError, match="Intentional batch failure"):
            await batcher.submit("anything")
    finally:
        await batcher.stop()
