# -*- coding: utf-8 -*-

"""
node.py - Node class implementation

This module provides a Node class with ttl-support that
is used in LRUCache and LFUCache.

Classes:
    Node: The main Node class

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from __future__ import annotations
import time
from typing import Any, Hashable, Optional, Union, Self

from .errors import CacheConfigurationError


class Node:
    """
    A node representing a key/value pair in a linked list of a cache.
    This node supports ttl.

    Attributes:
        - key: the node's key
        - value: the node's value (can't be None)
        - ttl: the node's ttl in seconds
        - prev
        - next
    """
    __slots__ = ["key", "_value", "_ttl", "_expires_at", "prev", "next"]

    def __init__(self,
                 key: Hashable,
                 value: Any,
                 ttl: Optional[Union[int, float]] = None,
                 prev: Optional[Self] = None,  # LRUCache.head.prev=None
                 next: Optional[Self] = None  # LRUCache.tail.next=None
                 ):
        self.key: Hashable = key
        self._value: Any = value

        self._ttl: Optional[Union[int, float]] = None
        self._expires_at: Optional[float] = None

        # Set self._ttl and self._expires_at
        # self.reset_expires_at() gets called when ttl is reassigned.
        self.ttl = ttl

        self.prev: Optional[Self] = prev
        self.next: Optional[Self] = next

    def is_expired(self) -> bool:
        """
        Check if the node's ttl is expired.

        Returns:
            bool: Return True if the node is expired; otherwise False.
        """
        if self._expires_at is not None:
            return time.time() > self._expires_at
        return False  # no ttl, so never expired

    def reset_expires_at(self) -> None:
        """
        Reset the expiration timestamp of the node.
        This gets called after either Node.value or Node.ttl gets changed.
        """
        if self._ttl is not None:
            self._expires_at = time.time() + self._ttl
        else:
            self._expires_at = None

    @property
    def value(self) -> Any:
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        """
        When the value gets updated, node._expires_at also gets updated.
        """
        # Whenever the node's value gets updated, reset the starttime.
        self._value = value
        self.reset_expires_at()

    @property
    def ttl(self) -> Optional[Union[int, float]]:
        return self._ttl

    @ttl.setter
    def ttl(self, ttl: Optional[Union[int, float]]) -> None:
        """
        When node.ttl gets updated, node._expires_at also gets updated.

        Raises:
            ValueError: When the given ttl is negative.
            TypeError: When the given ttl is not of type: int, float or NoneType.
        """
        # Check if ttl is valid
        if ttl is not None:
            if type(ttl) in (int, float):
                if ttl < 0:
                    raise CacheConfigurationError("ttl must be a positive int or float, or None")
            else:
                raise CacheConfigurationError("ttl must be a positive int or float, or None")

        self._ttl = ttl

        # Whenever the node's ttl gets updated, reset expires_at.
        self.reset_expires_at()

    @property
    def expires_at(self) -> Optional[float]:
        return self._expires_at

    def __repr__(self) -> str:
        """
        Examples:
            >>> repr(Node)
            NodeLRU(key='foo', value='bar', ttl=5s, prev='some_node', next='some_other_node')
        """
        return (
            f"{self.__class__.__name__}(key={self.key!r}, value={self.value!r}, "
            f"ttl={str(self.ttl) + ("s" if not isinstance(self.ttl, type(None)) else "")}, "
            f"expires_at={self._expires_at!r}, "
            f"prev={(self.prev.key if self.prev else None)}, "
            f"next={(self.next.key if self.next else None)})"
        )
