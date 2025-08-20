#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_api.py - Tests for the MemoryCache and DiskCache API and lru/lfu eviction.

These tests cover:
    - Basic cache API:
        - Basic set and get implementation
        - Delete and clear implementation
        - Cache.ttl and cache.inspect implementation
        - Cache.keys() and cache.values() implementation
    - Per-cache eviction logic
    - Memoize decorator
    - Change max_size
    - Cache overflow
    - Read-only mode
    - Persistance
    - In-background cleanup thread

To run:
    python -m pytest .\tests\test_api.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from pathlib import Path
import pytest

from cachelib import DiskCache, MemoryCache, eviction
from cachelib.errors import (
    CacheConfigurationError,
    CacheOverflowError,
    PathError,
    KeyExpiredError,
    KeyNotFoundError,
    ReadOnlyError
)
from tests.utils import PATH, teardown_cache

raises = pytest.raises
path = Path("tests", "test_file.db")


def test_set_and_get(cache):
    cache.set("key", "value")
    assert cache.get("key") == "value"

    # __setitem__ and __getitem__
    cache["k"] = "v"
    assert cache["k"] == "v"

    # Get nonexistent key
    assert cache.get("non-existent") is None

    # Set key - invalid calls
    with raises(CacheConfigurationError):
        cache.set("some key")

    with raises(CacheConfigurationError):
        cache.set(1, 2, 3, 4)


def test_update(cache):
    cache.set("key", "value")
    assert cache.get("key") == "value"

    cache.set("k", "other value")
    assert cache.get("k") == "other value"

    # update with __setitem__ and __getitem__
    cache["k"] = "v"
    assert cache["k"] == "v"

    cache["k"] = "k"
    assert cache["k"] == "k"

    # Update key - invalid call
    with raises(CacheConfigurationError):
        cache.set("k")


def test_delete(cache):
    cache.set("key", "value")
    cache.delete("key")
    assert "key" not in cache

    # __setitem__ and __getitem__
    cache["k"] = "v"
    assert cache["k"] == "v"

    del cache["k"]
    assert "k" not in cache

    # Delete nonexistent key
    with raises(KeyNotFoundError):
        cache.delete("nonexistent")


def test_clear(cache):
    cache.set("key", "value")
    assert len(cache) == 1

    cache.clear()
    assert "key" not in cache
    assert len(cache) == 0


def test_get_many(cache):
    cache.set("key", "value")
    cache.set("k", "v")

    assert cache.get_many(("key", "k")) == ("value", "v")


def test_ttl(cache):

    cache.set("k", "v", ttl=10)
    assert cache.ttl("k") == 10

    cache.clear()

    cache.set("k", "v", ttl=None)
    assert cache.ttl("k") is None

    cache.clear()

    # Get ttl of an expired key
    cache.set("k", "v", ttl=0)
    with raises(KeyExpiredError):
        cache.ttl("k")

    # Get ttl of a nonexistent key
    with raises(KeyNotFoundError):
        cache.ttl("non-existent")


def test_inspect(cache, mocker):
    mock_obj = mocker.patch("cachelib.caches.base.time.time", return_value=0)

    # Not expired key
    cache.set("k", "v", ttl=10)
    info = cache.inspect("k")

    assert info["key"] == "k"
    assert info["value"] == "v"
    assert info["ttl"] == 10

    cache.clear()

    # Expired key
    cache.set("k", "v", ttl=10)

    mock_obj.return_value = 11

    with raises(KeyExpiredError):
        info = cache.inspect("k")

    with raises(KeyNotFoundError):
        info = cache.inspect("non-existent")

    # Inspect nonexistent key
    with raises(KeyNotFoundError):
        cache.inspect("non-existent")


def test_keys_and_values(cache):
    cache.set("foo", "bar")

    assert "foo" in cache.keys()
    assert "bar" in cache.values()


def test_memoize(cache, mocker):
    mock_obj = mocker.patch("cachelib.caches.base.time.time", return_value=0)

    calls = 0

    @cache.memoize(1)
    def add(a, b):
        nonlocal calls
        calls += 1
        return a + b

    assert add(1, 2) == 3 and calls == 1
    assert add(1, 2) == 3 and calls == 1  # Key isn't expired

    print(cache.keys())

    mock_obj.return_value = 1.1

    assert add(1, 2) == 3 and calls == 2  # Key is expired
    assert add(1, 2) == 3 and calls == 2  # Key isn't expired


def test_change_max_size():

    def create_lfu_caches():
        memory_lfu = MemoryCache(eviction_policy=eviction.LFU, max_size=2)
        disk_lru = DiskCache(path, eviction_policy=eviction.LRU, max_size=2)

        return memory_lfu, disk_lru

    def create_lru_caches():
        memory_lru = MemoryCache(eviction_policy=eviction.LRU, max_size=2)
        disk_lru = DiskCache(path, eviction_policy=eviction.LRU, max_size=2)

        return memory_lru, disk_lru

    for cache_init in (create_lfu_caches, create_lru_caches):
        disk_cache, memory_cache = cache_init()

        for cache in (disk_cache, memory_cache):
            cache.set("key", "value")
            cache.set("k", "v")

            cache.max_size = 1

            assert "key" not in cache

            # Invalid calls
            with raises(CacheConfigurationError):
                cache.max_size = -1

            with raises(CacheConfigurationError):
                cache.max_size = "invalid type"

            if isinstance(cache, DiskCache):
                cache.close()


def test_max_memory_memory_cache():
    cache = MemoryCache(max_memory="150b", eviction_policy=eviction.LRU)

    cache.set("key", "value")
    assert "key" in cache

    cache.set("k", "v")
    assert "key" not in cache
    assert "k" in cache

    # Test 0 max-memory
    cache.max_memory = 0
    cache.set("key", "value")

    assert "k" not in cache
    assert "key" not in cache

    cache.clear()

    # Test unlimited max-memory
    cache.max_memory = None

    for i in range(100):
        cache.set(i, i)

    assert len(cache) == 100


def test_max_memory_disk_cache():
    cache = DiskCache(path=PATH, max_memory="15kb", eviction_policy=eviction.LRU)

    try:
        for i in range(200):
            cache.set(i, i)

        # Assert that some items were deleted.
        assert len(cache) < 200

        cache.max_memory = 0
        assert len(cache) == 0
    finally:
        teardown_cache(cache)


def test_max_memory_invalid_calls():
    # Invalid calls
    with raises(CacheConfigurationError):
        MemoryCache(max_memory="10bt", eviction_policy=eviction.LRU)

    with raises(CacheConfigurationError):
        MemoryCache(max_memory="mb10", eviction_policy=eviction.LRU)

    with raises(CacheConfigurationError):
        MemoryCache(max_memory=-1, eviction_policy=eviction.LRU)


def test_read_only(cache):
    cache.set("k", "v")
    with cache.read_only():

        with raises(ReadOnlyError):
            cache.set("key", "value")
        assert "key" not in cache

        with raises(ReadOnlyError):
            cache.delete("k")
        assert "k" in cache

        with raises(ReadOnlyError):
            cache.clear()
        assert len(cache) == 1


def test_persistance(memory_cache):
    path = Path("tests", "test_file.pkl")

    memory_cache.set("key", "value", ttl=10)
    memory_cache.save(path)

    memory_cache.clear()

    newCache = MemoryCache.load(path)
    assert newCache.get("key") == "value"

    path.unlink()

    # Invalid calls
    with raises(PathError, match="expected a '.pkl' file, got '.*': .*"):
        invalid_path = Path("tests", "test_file.txt")
        memory_cache.save(invalid_path)
        MemoryCache.load(invalid_path)

    with raises(PathError, match="'path' must be a str or Path, got .*"):
        invalid_path = True
        memory_cache.save(invalid_path)
        MemoryCache.load(invalid_path)


def test_global_ttl():
    cache = MemoryCache(ttl=10)
    cache.set("key", "value")
    assert cache.ttl("key") == 10


def test_cache_overflow(disk_cache, memory_cache):
    for cache in (disk_cache, memory_cache):
        cache.max_size = 0

        with raises(CacheOverflowError):
            cache.set("key", "value")
