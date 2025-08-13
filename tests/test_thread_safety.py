#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_thread_safety.py - Tests for the thread safety of cachelib caches.

To run:
    python -m pytest .\tests\test_thread_safety.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from pathlib import Path
import threading
import uuid

from cachelib import DiskCache, MemoryCache


def test_thread_safety_memory_cache():
    cache = MemoryCache()

    def set_item():
        for _ in range(500):
            key = str(uuid.uuid4())
            cache.set(key, "")

    threads = []

    for _ in range(20):
        thread = threading.Thread(target=set_item)
        threads.append(thread)
        thread.start()

    # Wait for all threads to end
    for thread in threads:
        thread.join()

    assert len(cache) == 10000


def test_thread_safery_disk_cache():
    path = Path("tests", "test_file.db")

    def set_item():
        cache = DiskCache(path)
        for _ in range(100):
            key = str(uuid.uuid4())
            cache.set(key, "")
        cache.close()

    threads = []
    for _ in range(10):
        thread = threading.Thread(target=set_item)
        threads.append(thread)
        thread.start()

    # Wait for all threads to end
    for t in threads:
        t.join()

    cache = DiskCache(path)
    assert len(cache) == 1000

    cache.close()
    if path.exists():
        path.unlink()
