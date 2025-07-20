#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_cache.py - Tests for the BaseCache API and the per-cache eviction logic.

These tests cover:
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
    python -m pytest .\tests\test_cache.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from pathlib import Path
import pytest
import time

from cachelib import LFUCache, LRUCache

raises = pytest.raises


@pytest.fixture
def caches():
    lfu = LFUCache()
    lru = LRUCache()
    caches = [lru, lfu]

    yield caches

    for cache in caches:
        cache.clear()


@pytest.fixture
def lru():
    lru = LRUCache()
    yield lru
    lru.clear()


@pytest.fixture
def lfu():
    lfu = LFUCache()
    yield lfu
    lfu.clear()


def test_set_get(caches):
    for cache in caches:
        cache.set("key", "value")
        assert cache.get("key") == "value"

        # __setitem__ and __getitem__
        cache["k"] = "v"
        assert cache["k"] == "v"

        # Get nonexistent key
        assert cache.get("non-existent") is None

        # Set key - invalid calls
        with raises(TypeError, match="expected 'value' argument"):
            cache.set("some key")

        with raises(TypeError, match="expected at most 3 arguments, got .*"):
            cache.set(1, 2, 3, 4)


def test_update(caches):
    for cache in caches:
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
        with raises(TypeError, match="expected either 'value' or 'ttl' as arguments"):
            cache.set("k")


def test_delete(caches):
    for cache in caches:
        cache.set("key", "value")
        cache.delete("key")
        assert "key" not in cache

        # __setitem__ and __getitem__
        cache["k"] = "v"
        assert cache["k"] == "v"

        del cache["k"]
        assert "k" not in cache

        # Delete nonexistent key
        with raises(KeyError, match="key '.*' not found in cache"):
            cache.delete("nonexistent")


def test_clear(caches):
    for cache in caches:
        cache.set("key", "value")
        assert len(cache) == 1

        cache.clear()
        assert "key" not in cache
        assert len(cache) == 0


def test_ttl(caches):
    for cache in caches:
        cache.set("k", "v", ttl=10)
        assert cache.ttl("k") == 10

        cache.set("k", ttl=None)
        assert cache.ttl("k") is None

        # Get ttl of nonexistent key
        with raises(KeyError, match="key '.*' not found in cache"):
            cache.ttl("non-existent")


def test_inspect(caches, mocker):
    mock_obj = mocker.patch("cachelib.base.time.time")

    for cache in caches:
        mock_obj.return_value = 0

        # Not expired key
        cache.set("k", "v", ttl=10)
        info = cache.inspect("k")

        assert info["key"] == "k"
        assert info["value"] == "v"
        assert not info["expired"]
        assert info["ttl"] == 10

        cache.clear()

        # Expired key
        cache.set("k", "v", ttl=10)

        mock_obj.return_value = 11
        info = cache.inspect("k")

        assert info["key"] == "k"
        assert info["value"] == "v"
        assert info["expired"]
        assert info["ttl"] == 10

        # Inspect nonexistent key
        with raises(KeyError, match="key '.*' not found in cache"):
            cache.inspect("non-existent")


def test_memoize(caches, mocker):
    mock_obj = mocker.patch("cachelib.base.time.time")

    for cache in caches:
        mock_obj.return_value = 0

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


def test_change_max_size(caches):
    for cache in caches:
        cache.set("key", "value")
        cache.set("k", "v")

        cache.max_size = 1

        assert "key" not in cache

        # Invalid calls
        with raises(ValueError, match=".*.max_size should be non-negative or None"):
            cache.max_size = -1

        with raises(TypeError, match=".*.max_size should be of type: int or NoneType"):
            cache.max_size = "invalid type"


def test_read_only(caches):
    for cache in caches:
        cache.set("k", "v")
        with cache.read_only():

            with raises(RuntimeError):
                cache.set("key", "value")
            assert "key" not in cache

            with raises(RuntimeError):
                cache.delete("k")
            assert "k" in cache

            with raises(RuntimeError):
                cache.clear()
            assert len(cache) == 1


def test_persistance(caches):
    for cache in caches:
        path = Path("tests", "test_file.pkl")

        cache.set("key", "value", ttl=10)
        cache.save(path)
        cache.clear()
        newCache = cache.load(path)
        assert newCache.get("key") == "value"

        path.unlink()

        # Invalid calls
        with raises(ValueError, match="expected a '.pkl' file, got '.*': .*"):
            invalid_path = Path("tests", "test_file.txt")
            cache.save(invalid_path)
            cache.load(invalid_path)

        with raises(TypeError, match="'path' must be a str or Path, got .*"):
            invalid_path = True
            cache.save(invalid_path)
            cache.load(invalid_path)


def test_cleanup_thread(caches):
    for cache in caches:
        cache.set("key", "value", ttl=0.01)
        time.sleep(2)
        assert "key" not in cache


def test_global_ttl_lru():
    cache = LRUCache(ttl=10)
    cache.set("key", "value")
    assert cache.ttl("key") == 10


def test_global_ttl_lfu():
    cache = LFUCache(ttl=10)
    cache.set("key", "value")
    assert cache.ttl("key") == 10


def test_load_lru(lru):
    with raises(TypeError, match="expected instance of .*, got .*"):
        path = Path("tests", "test_file.pkl")

        lru.save(path)
        LFUCache.load(path)

    path.unlink()


def test_load_lfu(lfu):
    with raises(TypeError, match="expected instance of .*, got .*"):
        path = Path("tests", "test_file.pkl")

        lfu.save(path)
        LRUCache.load(path)

    path.unlink()


def test_lru_iter(lru):
    lru.set("key", "value")  # Most recently accessed
    lru.set("k", "v")
    lru.get("key")

    for i, key in enumerate(lru):
        if i == 0:
            assert key == ("key", "value")
        elif i == 1:
            assert key == ("k", "v")


def test_lru_reversed(lru):
    lru.set("key", "value")
    lru.set("k", "v")  # Least recently accessed
    lru.get("key")

    for i, key in enumerate(reversed(lru)):
        if i == 0:
            assert key == ("k", "v")
        elif i == 1:
            assert key == ("key", "value")


def test_lfu_iter(lfu):
    lfu.set("key", "value")  # Most frequently accessed
    lfu.set("k", "v")
    lfu.get("key")

    for i, key in enumerate(lfu):
        if i == 0:
            assert key == ("key", "value")
        elif i == 1:
            assert key == ("k", "v")


def test_lfu_reversed(lfu):
    lfu.set("key", "value")
    lfu.set("k", "v")  # Least frequently accessed
    lfu.get("key")

    for i, key in enumerate(reversed(lfu)):
        if i == 0:
            assert key == ("k", "v")
        elif i == 1:
            assert key == ("key", "value")


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
