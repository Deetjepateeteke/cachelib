#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_multi_level_cache.py - Tests for the multi-level cache.

These tests cover:
    - Cache to cache transfers
    - Move node to top after accessing it
    - Cache overflow

To run:
    python -m pytest .\tests\test_multi_level_cache.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import pytest

from cachelib.errors import CacheOverflowError
from tests.utils import (
    create_exclusive_multi_level_cache,
    create_inclusive_multi_level_cache,
    teardown_cache
)


raises = pytest.raises


def test_transfer_inclusive(inclusive_mlc):
    # L1 max_size = 1
    # L2 max_size = 2
    # L3 max_size = inf
    inclusive_mlc.set("a", 1)
    inclusive_mlc.set("b", 2)

    assert "a" not in inclusive_mlc.l1
    assert "a" in inclusive_mlc.l2
    assert "a" in inclusive_mlc.l3

    assert "b" in inclusive_mlc.l1
    assert "b" in inclusive_mlc.l2
    assert "b" in inclusive_mlc.l3

    inclusive_mlc.set("c", 3)

    assert "a" not in inclusive_mlc.l1
    assert "a" not in inclusive_mlc.l2
    assert "a" in inclusive_mlc.l3

    assert "b" not in inclusive_mlc.l1
    assert "b" in inclusive_mlc.l2
    assert "b" in inclusive_mlc.l3

    assert "c" in inclusive_mlc.l1
    assert "c" in inclusive_mlc.l2
    assert "c" in inclusive_mlc.l3


def test_transfer_exclusive(exclusive_mlc):
    exclusive_mlc.set("a", 1)
    exclusive_mlc.set("b", 2)

    assert "a" not in exclusive_mlc.l1
    assert "a" in exclusive_mlc.l2
    assert "a" not in exclusive_mlc.l3

    assert "b" in exclusive_mlc.l1
    assert "b" not in exclusive_mlc.l2
    assert "b" not in exclusive_mlc.l3

    exclusive_mlc.set("c", 3)

    assert "a" not in exclusive_mlc.l1
    assert "a" in exclusive_mlc.l2
    assert "a" not in exclusive_mlc.l3

    assert "b" not in exclusive_mlc.l1
    assert "b" in exclusive_mlc.l2
    assert "b" not in exclusive_mlc.l3

    assert "c" in exclusive_mlc.l1
    assert "c" not in exclusive_mlc.l2
    assert "c" not in exclusive_mlc.l3

    exclusive_mlc.set("d", 4)

    assert "a" not in exclusive_mlc.l1
    assert "a" not in exclusive_mlc.l2
    assert "a" in exclusive_mlc.l3

    assert "b" not in exclusive_mlc.l1
    assert "b" in exclusive_mlc.l2
    assert "b" not in exclusive_mlc.l3

    assert "c" not in exclusive_mlc.l1
    assert "c" in exclusive_mlc.l2
    assert "c" not in exclusive_mlc.l3

    assert "d" in exclusive_mlc.l1
    assert "d" not in exclusive_mlc.l2
    assert "d" not in exclusive_mlc.l3


def test_move_to_top_inclusive(inclusive_mlc):
    inclusive_mlc.set("a", 1)
    inclusive_mlc.set("b", 2)

    inclusive_mlc.get("a")

    assert "a" in inclusive_mlc.l1
    assert "b" not in inclusive_mlc.l1

    assert "a" in inclusive_mlc.l2
    assert "b" in inclusive_mlc.l2

    assert "a" in inclusive_mlc.l3
    assert "b" in inclusive_mlc.l3

    inclusive_mlc.set("c", 3)

    assert "a" not in inclusive_mlc.l1
    assert "b" not in inclusive_mlc.l1
    assert "c" in inclusive_mlc.l1

    assert "a" in inclusive_mlc.l2
    assert "c" in inclusive_mlc.l2

    assert "a" in inclusive_mlc.l3
    assert "b" in inclusive_mlc.l3
    assert "c" in inclusive_mlc.l3

    inclusive_mlc.set("d", 4)

    assert "a" not in inclusive_mlc.l1
    assert "b" not in inclusive_mlc.l1
    assert "c" not in inclusive_mlc.l1
    assert "d" in inclusive_mlc.l1

    assert "a" not in inclusive_mlc.l2
    assert "b" not in inclusive_mlc.l2
    assert "c" in inclusive_mlc.l2
    assert "d" in inclusive_mlc.l2

    assert "a" in inclusive_mlc.l3
    assert "b" in inclusive_mlc.l3
    assert "c" in inclusive_mlc.l3
    assert "d" in inclusive_mlc.l3


def test_move_to_top_exclusive(exclusive_mlc):
    exclusive_mlc.set("a", 1)
    exclusive_mlc.set("b", 2)

    exclusive_mlc.get("a")

    assert "a" in exclusive_mlc.l1
    assert "b" not in exclusive_mlc.l1

    assert "a" not in exclusive_mlc.l2
    assert "b" in exclusive_mlc.l2

    exclusive_mlc.set("c", 3)

    assert "a" not in exclusive_mlc.l1
    assert "b" not in exclusive_mlc.l1
    assert "c" in exclusive_mlc.l1

    assert "a" in exclusive_mlc.l2
    assert "b" in exclusive_mlc.l2
    assert "c" not in exclusive_mlc.l2

    assert "a" not in exclusive_mlc.l3
    assert "b" not in exclusive_mlc.l3
    assert "c" not in exclusive_mlc.l3

    exclusive_mlc.set("d", 4)

    assert "a" not in exclusive_mlc.l1
    assert "b" not in exclusive_mlc.l1
    assert "c" not in exclusive_mlc.l1
    assert "d" in exclusive_mlc.l1

    assert "a" in exclusive_mlc.l2
    assert "b" not in exclusive_mlc.l2
    assert "c" in exclusive_mlc.l2
    assert "d" not in exclusive_mlc.l2

    assert "a" not in exclusive_mlc.l3
    assert "b" in exclusive_mlc.l3
    assert "c" not in exclusive_mlc.l3
    assert "d" not in exclusive_mlc.l3


@pytest.mark.parametrize("mlc", [create_inclusive_multi_level_cache, create_exclusive_multi_level_cache])
def test_cache_overflow(mlc):
    mlc = mlc()

    try:
        mlc.l1.max_size = 0
        mlc.l2.max_size = 0
        mlc.l3.max_size = 0

        with raises(CacheOverflowError):
            mlc.set("key", "value")

        mlc.l3.max_size = 1
        mlc.set("key", "value")

        with raises(CacheOverflowError):
            mlc.set("k", "v")

    finally:
        teardown_cache(mlc)
