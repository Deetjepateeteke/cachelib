#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_eviction.py - Tests for LRU/LFU eviction logic.

These tests cover:
    - LRU eviction logic in MemoryCache and DiskCache
    - LFU eviction logic in MemoryCache and DiskCache

To run:
    python -m pytest .\tests\test_eviction.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from cachelib import DiskCache, MemoryCache, eviction
from pathlib import Path
import pytest

path = Path("tests", "test_file.db")


def create_lru_memory_cache():
    return MemoryCache(eviction_policy=eviction.LRU, max_size=2)

def create_lfu_memory_cache():
    return MemoryCache(eviction_policy=eviction.LFU, max_size=2)

def create_lru_disk_cache():
    return DiskCache(path, eviction_policy=eviction.LRU, max_size=2)

def create_lfu_disk_cache():
    return DiskCache(path, eviction_policy=eviction.LFU, max_size=2)


@pytest.mark.parametrize("lru_cache", [create_lru_memory_cache, create_lru_disk_cache])
def test_lru_eviction(lru_cache):
    lru_cache = lru_cache()

    try:
        lru_cache.set("key", "value")  # Least recently accessed
        lru_cache.set("k", "v")

        lru_cache.max_size = 1
        assert "key" not in lru_cache
        assert "k" in lru_cache

        lru_cache.clear()

        lru_cache.max_size = 2

        lru_cache.set("key", "value")
        lru_cache.set("k", "v")  # Least recently accessed
        lru_cache.get("key")

        lru_cache.max_size = 1

        assert "key" in lru_cache
        assert "k" not in lru_cache

    finally:
        if isinstance(lru_cache, DiskCache):
            with lru_cache._lock:
                lru_cache.close()

                if path.exists():
                    path.unlink()


@pytest.mark.parametrize("lfu_cache", [create_lfu_memory_cache, create_lfu_disk_cache])
def test_lfu_eviction(lfu_cache):
    lfu_cache = lfu_cache()

    try:
        lfu_cache.set("key", "value")  # Most frequently accessed
        lfu_cache.set("k", "v")

        for _ in range(5):
            lfu_cache.get("key")

        lfu_cache.get("k")

        lfu_cache.max_size = 1

        assert "key" in lfu_cache
        assert "k" not in lfu_cache

    finally:
        if isinstance(lfu_cache, DiskCache):
            with lfu_cache._lock:
                lfu_cache.close()

                if path.exists():
                    path.unlink()
