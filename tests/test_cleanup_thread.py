#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_cleanup_thread.py - Unit tests for the CleanupThread in a MemoryCache and a DiskCache.

These tests cover:
    - Cleanup functionality
    - Modifying the cleanup thread's interval

To run:
    python -m pytest .\tests\test_cleanup_thread.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import pytest
import time

from cachelib import DiskCache
from cachelib.errors import CleanupThreadConfigurationError
from tests.utils import create_disk_cache, create_memory_cache, teardown_cache

raises = pytest.raises


def test_cleanup_thread(cache):
    cache.set("key", "value", ttl=0.1)
    time.sleep(2)
    assert "key" not in cache

    cache.set("key", "value", ttl=100)
    assert "key" in cache


@pytest.mark.parametrize("create_cache", [create_disk_cache, create_memory_cache])
def test_invalid_calls(create_cache):
    cache = create_cache()

    try:
        with raises(CleanupThreadConfigurationError):
            cache.cleanup_thread.interval = 1

        with raises(CleanupThreadConfigurationError):
            cache.cleanup_thread.set_interval(interval="1")

        with raises(CleanupThreadConfigurationError):
            cache.cleanup_thread.set_interval(interval=-1)

    finally:
        if isinstance(cache, DiskCache):
            teardown_cache(cache)
