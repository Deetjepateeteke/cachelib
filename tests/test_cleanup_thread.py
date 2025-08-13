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

from cachelib.errors import CleanupThreadConfigurationError

raises = pytest.raises


def test_cleanup_thread(cache):
    cache.set("key", "value", ttl=0.1)
    time.sleep(2)
    assert "key" not in cache

    cache.cleanup_thread.interval = 10
    cache.set("key", "value", ttl=0.1)
    assert "key" in cache


def test_invalid_calls(cache):
    with raises(CleanupThreadConfigurationError):
        cache.cleanup_thread.interval = "1"

    with raises(CleanupThreadConfigurationError):
        cache.cleanup_thread.interval = -1
