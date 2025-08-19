#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
conftest.py - Defines pytest fixtures.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import pytest

from tests.utils import (
    create_disk_cache,
    create_inclusive_multi_level_cache,
    create_memory_cache,
    create_exclusive_multi_level_cache,
    teardown_cache
)


@pytest.fixture
def memory_cache():
    memory_cache = create_memory_cache()

    try:
        yield memory_cache
    finally:
        memory_cache.clear()
        memory_cache.cleanup_thread.stop()


@pytest.fixture
def disk_cache():
    disk_cache = create_disk_cache()

    try:
        yield disk_cache
    finally:
        teardown_cache(disk_cache)


@pytest.fixture
def inclusive_mlc():
    multi_level_cache = create_inclusive_multi_level_cache()

    try:
        yield multi_level_cache
    finally:
        teardown_cache(multi_level_cache)


@pytest.fixture
def exclusive_mlc():
    multi_level_cache = create_exclusive_multi_level_cache()

    try:
        yield multi_level_cache
    finally:
        teardown_cache(multi_level_cache)


@pytest.fixture(params=["disk_cache", "memory_cache", "inclusive_mlc", "exclusive_mlc"])
def cache(request):
    return request.getfixturevalue(request.param)
