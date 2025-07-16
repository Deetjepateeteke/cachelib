#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Unit tests for the LRUCache class

These tests cover:
    - Basic set and get implementation (with edge cases)
    - Basic delete, clear and inspect implementation
    - TTL expiration
    - Eviction logic
    - Change max-size logic
    - Memoize decorator
    - Cache persistance
    - LRUCache.stats
    - Read-only mode

Uses tests.utils.MockTimeContextManager() to simulate a passage
in time. This is used to speed up testing TTL-related tests.

To run:
    python -m unittest .\tests\test_lru.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import time
from pathlib import Path
import unittest
from unittest.mock import patch

from cachelib import LRUCache
from tests.utils import MockTimeContextManager


class TestLRUCacheSet(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(max_size=2)

    def test_set_key(self):
        self.cache.set("foo", "bar")  # without ttl
        self.assertEqual(len(self.cache), 1)

        self.cache.set("foo", "bar", ttl=5)  # with ttl
        self.assertEqual(len(self.cache), 1)

    def test_update_existing_key(self):
        self.cache.set("foo", "bar")
        self.cache.set("foo", "notbar")

        self.assertEqual(self.cache.get("foo"), "notbar")

    def test_eviction(self):
        self.cache.set("a", 1)
        self.cache.set("b", 2)

        self.assertEqual(len(self.cache), 2)

        self.cache.set("c", 3)  # evict ("a", 1)

        self.assertEqual(len(self.cache), 2)
        self.assertIsNone(self.cache.get("a"))
        self.assertEqual(self.cache.stats.evictions, 1)

    def test_set_read_only_raises_runtimeerror(self):
        with self.cache.read_only():
            self.assertRaises(RuntimeError, self.cache.set, "foo", "bar")


class TestLRUCacheGet(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(max_size=2)

    def test_get_existing_key(self):
        self.cache.set("foo", "bar")
        self.assertEqual(self.cache.get("foo"), "bar")
        self.assertEqual(self.cache.stats.hits, 1)

    def test_get_nonexistent_key(self):
        self.assertIsNone(self.cache.get("foo"))
        self.assertEqual(self.cache.stats.misses, 1)

    def test_get_key_with_ttl(self):
        self.cache.set("foo", "bar", ttl=5)

        # Used to replace time.time() with future_time
        future_time = time.time() + 5

        with patch("time.time", return_value=future_time):
            self.assertIsNone(self.cache.get("foo"))
            self.assertEqual(self.cache.stats.evictions, 1)

    def test_cant_delete_key_with_ttl_during_read_only(self):
        self.cache.set("foo", "bar", ttl=5)

        with MockTimeContextManager(future_time=5):
            with self.cache.read_only():
                # The key doesn't get deleted, because read_only mode is enabled.
                self.cache.get("foo")

                info = self.cache.inspect("foo")
                self.assertIsNotNone(info)

                self.assertTrue(info["expired"])
                self.assertAlmostEqual(round(info["ttl"], 2), round(time.time(), 2))

            # When read-only mode is disabled, the key gets deleted.
            self.assertIsNone(self.cache.get("foo"))


class TestLRUCacheDelete(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(max_size=2)

    def test_delete_existing_key(self):
        self.cache.set("foo", "bar")
        self.cache.delete("foo")

        self.assertIsNone(self.cache.get("foo"))

    def test_delete_nonexistent_key(self):
        self.assertRaises(KeyError, self.cache.delete, "foo")

    def test_delete_read_only_raises_runtimeerror(self):
        self.cache.set("foo", "bar")
        with self.cache.read_only():
            self.assertRaises(RuntimeError, self.cache.delete, "foo")
        self.assertEqual(self.cache.get("foo"), "bar")


class TestLRUCacheClear(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(max_size=2)

    def test_clear(self):
        self.cache.set("foo", "bar")
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)

    def test_clear_read_only_raises_runtimeerror(self):
        self.cache.set("foo", "bar")
        with self.cache.read_only():
            self.assertRaises(RuntimeError, self.cache.clear)
        self.assertEqual(self.cache.get("foo"), "bar")


class TestLRUCacheInspect(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(max_size=2)

    def test_inspect_existing_key_no_ttl(self):
        self.cache.set("foo", "bar")
        info = self.cache.inspect("foo")

        self.assertEqual(info["key"], "foo")
        self.assertEqual(info["value"], "bar")
        self.assertFalse(info["expired"])
        self.assertIsNone(info["ttl"])

    def test_inspect_existing_key_with_ttl(self):
        self.cache.set("foo", "bar", ttl=5)

        # Used to replace time.time() with future_time
        future_time = time.time() + 5

        # Mock time.time() to simulate 10 seconds later
        with patch("time.time", return_value=future_time):
            info = self.cache.inspect("foo")

            self.assertEqual(info["key"], "foo")
            self.assertEqual(info["value"], "bar")
            self.assertTrue(info["expired"])
            self.assertAlmostEqual(round(info["ttl"], 2), round(future_time, 2))

    def test_inspect_nonexistent_key(self):
        self.assertIsNone(self.cache.inspect("foo"))


class TestLRUCacheMaxSize(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(max_size=2)

    def test_change_max_size(self):
        self.cache.max_size = 5
        self.assertEqual(self.cache.max_size, 5)
        self.assertEqual(self.cache.stats.max_size, 5)

    def test_change_max_size_with_eviction(self):
        self.cache.set("a", 1)
        self.cache.set("b", 2)
        self.cache.max_size = 1

        self.assertIsNone(self.cache.get("a"))

    def test_change_max_size_with_eviction_raises_runtimeerror(self):
        self.cache.set("a", 1)
        self.cache.set("b", 2)

        with self.cache.read_only():
            self.assertRaises(RuntimeError, self.cache.__setattr__, "max_size", 1)


class TestMemoizeDecorator(unittest.TestCase):
    def setUp(self):
        self.cache = LRUCache(max_size=2)

    def test_decorator_caches_function_result(self):
        call_counter = {"count": 0}

        @self.cache.memoize()
        def add(a, b):
            call_counter["count"] += 1
            return a + b

        add(1, 2)  # first call, gets computed

        result = add(1, 2)  # second call, gets retrieved from cache
        self.assertEqual(result, 3)
        self.assertEqual(call_counter["count"], 1)

        add(2, 3)  # first call, gets computed

        result = add(2, 3)
        self.assertEqual(result, 5)
        self.assertEqual(call_counter["count"], 2)

        add(3, 4)

        result = add(3, 4)
        self.assertEqual(result, 7)
        self.assertEqual(call_counter["count"], 3)

        # Check if the first cached result got evicted, due to max_size
        self.assertIsNone(self.cache.get((1, 2)))

    def test_decorator_caches_function_result_with_ttl(self):

        @self.cache.memoize(ttl=5)
        def identity(x):
            return x

        identity(1)

        with MockTimeContextManager(future_time=5):
            info = self.cache.inspect((1,))
            self.assertIsNotNone(info)

            self.assertTrue(info["expired"])
            self.assertAlmostEqual(round(info["ttl"], 2), round(time.time(), 2))

    def test_decorator_read_only_raises_runtimeerror(self):

        @self.cache.memoize()
        def identity(x):
            return x

        with self.cache.read_only():
            self.assertRaises(RuntimeError, identity, 1)


class TestCachePersistance(unittest.TestCase):
    def setUp(self):
        self.test_file = Path(r"tests\test_file.pkl")
        self.cache = LRUCache(2)
        self.cache.set("foo", "bar")

    def tearDown(self):
        if self.test_file.exists():
            self.test_file.unlink()

    def test_save_and_load_cache(self):
        self.cache.save(self.test_file)

        new_cache = LRUCache.load(self.test_file)
        self.assertEqual(new_cache.get("foo"), "bar")

    def test_path_is_not_pkl_file(self):
        file = r"tests\test_file.txt"

        self.assertRaises(ValueError, self.cache.save, file)
        self.assertRaises(ValueError, self.cache.load, file)

    def test_path_is_str_type(self):
        file = Path(r"tests\test_file.pkl")

        self.cache.save(file)
        new_cache = LRUCache.load(file)

        self.assertEqual(new_cache.get("foo"), "bar")

    def test_path_is_invalid_type(self):
        file = None

        self.assertRaises(TypeError, self.cache.save, file)
        self.assertRaises(TypeError, self.cache.load, file)


if __name__ == "__main__":
    unittest.main()
