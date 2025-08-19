#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_thread_safety.py - Tests for the thread safety of cachelib caches.

To run:
    python -m pytest .\tests\test_thread_safety.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import threading
import uuid

from tests.utils import create_disk_cache, create_memory_cache, teardown_cache


def test_thread_safety_memory_cache():
    cache = create_memory_cache()

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


def test_thread_safety_disk_cache():

    def set_item():
        cache = create_disk_cache()
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

    cache = create_disk_cache()
    assert len(cache) == 1000

    teardown_cache(cache)
