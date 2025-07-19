# -*- coding: utf-8 -*-

"""
lru.py - LRU (Least Recently Used) Cache implementation.

This module provides an in-memory LRU cache with optional ttl-support.
Entries are evicted based on usage order and (optionally) expiration
time.

Classes:
    LRUCache: The main LRU cache class.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from typing import Any, Iterator, Hashable, Optional, Union

from .base import BaseCache
from .node import Node


__all__ = ["LRUCache"]


class LRUCache(BaseCache):
    """
    An in-memory LRUCache (Least Recently Used) with optional ttl-support.
    """

    def __init__(self,
                 name: Optional[str] = "",
                 max_size: Optional[int] = None,
                 verbose: bool = False,
                 thread_safe: bool = True
                 ):
        # set the name to cachelib.LRUCache.name
        name = f"cachelib.{self.__class__.__name__}" + (f".{name}" if name else "")
        super().__init__(name=name,
                         max_size=max_size,
                         verbose=verbose,
                         thread_safe=thread_safe
                         )
        self._head: Node = Node(None, None)
        self._tail: Node = Node(None, None)

        self._head.next = self._tail
        self._tail.prev = self._head

        self._logger.info(f"Initialized LRUCache with max-size={self._max_size}")

    def _add_node(self,
                  key: Optional[Hashable],
                  value: Optional[Any],
                  ttl: Optional[Union[int, float]]
                  ) -> Node:
        # Create a node
        node: Node = Node(key, value, ttl)

        # Update linked list
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

        # Store in cache
        self._cache[key] = node

        return node

    def _remove_node(self, node: Node) -> None:
        # Remove node from liked list
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

        # Remove node from cache
        del self._cache[node.key]

    def _get_evict_node(self) -> Node:
        current_node = self._tail.prev
        while current_node.is_expired():
            current_node = current_node.prev

        return current_node

    def _update_cache_state(self, node: Node) -> None:
        # Place the node at the start of the linked list
        # Remove node
        node.prev.next = node.next
        node.next.prev = node.prev

        # Add node
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def __iter__(self) -> Iterator[tuple[Hashable, Any]]:
        """
        Yield the (key, value) pairs in order of most recently used.

        Returns:
            Iterator (Iterator[tuple[Hashable, Any]]): Iterates over the (key, value) pairs.
        """
        with self._lock:
            node = self._head.next
            while node != self._tail:
                yield (node.key, node.value)
                node = node.next

    def __reversed__(self) -> Iterator[tuple[Hashable, Any]]:
        """
        Yield the (key, value) pairs in order of least recently used.

        Returns:
            Iterator (Iterator[tuple[Hashable, Any]]): Iterates over the (key, value) pairs.
        """
        with self._lock:
            node = self._tail.prev
            while node != self._head:
                yield (node.key, node.value)
                node = node.prev
