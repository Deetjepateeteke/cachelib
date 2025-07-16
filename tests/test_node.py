#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Unit tests for the Node class.

These tests cover:
    - Check for valid ttl-value.
    - TTL expiration.
    - Reset expiration timestamp when Node.value or
      Node.ttl gets reassigned.
    - Edge cases (ttl=0 and ttl=None)

Uses tests.utils.MockTimeContextManager() to simulate
passage of time to speed up testing  TTL-related tests.

To run:
    python -m unittest .\tests\test_node.py

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import time
import unittest

from cachelib.node import Node
from tests.utils import MockTimeContextManager


class TestNodeWithoutTTL(unittest.TestCase):
    def setUp(self):
        self.node = Node("foo", "bar")

    def test_ttl_setter_raises_exceptions(self):
        self.assertRaises(ValueError, self.node.__setattr__, "ttl", -1)
        self.assertRaises(TypeError, self.node.__setattr__, "ttl", "some node")


class TestNodeWithTTL(unittest.TestCase):
    def setUp(self):
        self.node = Node("foo", "bar", ttl=5)

    def test_node_ttl_expiration(self):
        self.assertFalse(self.node.is_expired())

        with MockTimeContextManager(future_time=5):
            self.assertAlmostEqual(round(self.node.expires_at, 2),
                                   round(time.time(), 2))
            self.assertTrue(self.node.is_expired())

    def test_reset_expires_at_when_value_reassigned(self):
        with MockTimeContextManager(future_time=5):
            self.assertTrue(self.node.is_expired())
            self.node.value = "some val"
            self.assertFalse(self.node.is_expired())
            self.assertAlmostEqual(round(self.node.expires_at, 2),
                                   round(time.time() + 5, 2))

    def test_reset_expires_at_when_ttl_reassigned(self):
        with MockTimeContextManager(future_time=5):
            self.assertTrue(self.node.is_expired())
            self.node.ttl = 10
            self.assertFalse(self.node.is_expired())
            self.assertAlmostEqual(round(self.node.expires_at, 2),
                                   round(time.time() + 10, 2))

    def test_ttl_zero(self):
        self.node.ttl = 0
        self.assertTrue(self.node.is_expired())

    def test_ttl_none(self):
        self.node.ttl = None

        with MockTimeContextManager(future_time=5):
            self.assertFalse(self.node.is_expired())
