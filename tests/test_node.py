#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
test_node.py - tests for the Node class.

These tests cover:
    - Test node creation
    - TTL expiration
    - Test Node.expires_at after reassigning Node.value or Node.ttl
    - Edge cases:
        - Node.ttl is zero
        - Node.ttl is None

To run:
    python -m pytest .\tests\test_node.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import pytest

from cachelib.node import Node

raises = pytest.raises


@pytest.fixture
def node():
    node = Node("key", "value")
    yield node


def test_set_and_get(mocker):
    mock_obj = mocker.patch("cachelib.node.time.time", return_value=0)

    node = Node("key", "value", ttl=10)
    assert node.key == "key"
    assert node.value == "value"
    assert node.ttl == 10
    assert not node.is_expired()

    mock_obj.return_value = 11

    assert node.is_expired()

    # Invalid calls
    with raises(ValueError, match="Node.ttl must be non-negative or None"):
        node.ttl = -1

    with raises(TypeError, match="Node.ttl must be of type: int, float or NoneType"):
        node.ttl = True


def test_reset_expires_at(mocker):
    mock_obj = mocker.patch("cachelib.node.time.time", return_value=0)

    node = Node("key", "value", ttl=10)
    assert node.ttl == 10
    assert node.expires_at == 10

    # Reassign node.value
    node.value = "other value"
    assert node.value == "other value"
    assert node.expires_at == 10

    mock_obj.return_value = 10

    # Reassign node.ttl
    node.ttl = 20
    assert node.ttl == 20
    assert node.expires_at == 30


def test_ttl_zero():
    node = Node("key", "value", ttl=0)
    assert node.ttl == 0
    assert node.is_expired()


def test_ttl_none():
    node = Node("key", "value", ttl=None)
    assert node.ttl is None
    assert not node.is_expired()
