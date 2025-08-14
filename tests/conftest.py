#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
conftest.py - Defines pytest fixtures.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from pathlib import Path
import pytest

from cachelib import DiskCache, MemoryCache

path = Path("tests", "test_file.db")

# Delete the path's contents, if the path exists.
if path.exists():
    path.unlink()

@pytest.fixture
def memory_cache():
    memory_cache: MemoryCache = MemoryCache()

    try:
        yield memory_cache
    finally:
        memory_cache.clear()
        memory_cache.cleanup_thread.stop()


@pytest.fixture
def disk_cache():
    disk_cache: DiskCache = DiskCache(path=path)

    try:
        yield disk_cache
    finally:
        with disk_cache._lock:
            disk_cache.clear()
            disk_cache.close()
            if path.exists():
                path.unlink()


@pytest.fixture(params=["memory_cache", "disk_cache"])
def cache(request):
    return request.getfixturevalue(request.param)
