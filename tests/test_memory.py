#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_memory.py - Tests for the MemoryCache API and lru/lfu eviction.

These tests cover:
    - Basic cache API:
        - Basic set and get implementation
        - Delete and clear implementation
        - Cache.ttl and cache.inspect implementation
    - Per-cache eviction logic
    - Memoize decorator
    - Change max_size
    - Read-only mode
    - Persistance
    - In-background cleanup thread

To run:
    python -m pytest .\tests\test_memory.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from pathlib import Path
import pytest
import time

from cachelib import MemoryCache
from cachelib.errors import (
    CacheConfigurationError,
    CacheOverflowError,
    CachePathError,
    KeyExpiredError,
    KeyNotFoundError,
    ReadOnlyError
)

raises = pytest.raises


@pytest.fixture
def cache():
    cache = MemoryCache()
    yield cache
    cache.clear()


@pytest.fixture
def lru():
    lru = MemoryCache(max_size=5, eviction_policy="lru")
    yield lru
    lru.clear()


@pytest.fixture
def lfu():
    lfu = MemoryCache(max_size=5, eviction_policy="lfu")
    yield lfu
    lfu.clear()


def test_set_get(cache):
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

    assert cache.get_many(("key", "k")) == {"key": "value", "k": "v"}


def test_ttl(cache):
    cache.set("k", "v", ttl=10)
    assert cache.ttl("k") == 10

    cache.set("k", ttl=None)
    assert cache.ttl("k") is None

    cache.set("k", "v", ttl=0)

    with raises(KeyExpiredError):
        cache.ttl("k")

    # Get ttl of nonexistent key
    with raises(KeyNotFoundError):
        cache.ttl("non-existent")


def test_inspect(cache, mocker):
    mock_obj = mocker.patch("cachelib.base.time.time", return_value=0)

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


def test_memoize(cache, mocker):
    mock_obj = mocker.patch("cachelib.base.time.time", return_value=0)

    calls = 0

    @cache.memoize(1)
    def add(a, b):
        nonlocal calls
        calls += 1
        return a+b

    assert add(1, 2) == 3 and calls == 1
    assert add(1, 2) == 3 and calls == 1  # Key isn't expired

    mock_obj.return_value = 1.1

    assert add(1, 2) == 3 and calls == 2  # Key is expired
    assert add(1, 2) == 3 and calls == 2  # Key isn't expired


def test_change_max_size(lru, lfu):
    for cache in (lru, lfu):
        cache.set("key", "value")
        cache.set("k", "v")

        cache.max_size = 1

        assert "key" not in cache

        # Invalid calls
        with raises(CacheConfigurationError):
            cache.max_size = -1

        with raises(CacheConfigurationError):
            cache.max_size = "invalid type"


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


def test_persistance(cache):
    path = Path("tests", "test_file.pkl")

    cache.set("key", "value", ttl=10)
    cache.save(path)
    cache.clear()
    newCache = cache.load(path)
    assert newCache.get("key") == "value"

    path.unlink()

    # Invalid calls
    with raises(CachePathError, match="expected a '.pkl' file, got '.*': .*"):
        invalid_path = Path("tests", "test_file.txt")
        cache.save(invalid_path)
        cache.load(invalid_path)

    with raises(CachePathError, match="'path' must be a str or Path, got .*"):
        invalid_path = True
        cache.save(invalid_path)
        cache.load(invalid_path)


def test_cleanup_thread(cache):
    cache.set("key", "value", ttl=0.01)
    time.sleep(2)
    assert "key" not in cache


def test_global_ttl(cache):
    cache._ttl = 10
    cache.set("key", "value")
    assert cache.ttl("key") == 10


"""def test_load_lru(lru):
    with raises(TypeError, match="expected instance of .*, got .*"):
        path = Path("tests", "test_file.pkl")

        lru.save(path)
        MemoryCache.load(path)

    path.unlink()


def test_load_lfu(lfu):
    with raises(TypeError, match="expected instance of .*, got .*"):
        path = Path("tests", "test_file.pkl")

        lfu.save(path)
        MemoryCache.load(path)

    path.unlink()"""


def test_lru_eviction_logic(lru):
    lru.max_size = 2

    lru.set("key", "value")  # Least recently accessed
    lru.set("k", "v")

    lru.max_size = 1
    assert "key" not in lru
    assert "k" in lru

    lru.clear()

    lru.max_size = 2

    lru.set("key", "value")
    lru.set("k", "v")  # Least recently accessed
    lru.get("key")

    lru.max_size = 1
    assert "key" in lru
    assert "k" not in lru


def test_lfu_eviction_logic(lfu):
    lfu.max_size = 2

    lfu.set("key", "value")  # Most frequently accessed
    lfu.set("k", "v")

    for _ in range(5):
        lfu.get("key")

    lfu.get("k")

    lfu.max_size = 1
    assert "key" in lfu
    assert "k" not in lfu


def test_cache_overflow_error(cache):
    cache.max_size = 0
    with raises(CacheOverflowError):
        cache.set("key", "value")
