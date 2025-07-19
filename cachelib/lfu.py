# -*- coding: utf-8 -*-

"""
lfu.py - LFU (Least Frequently Used) Cache implementation.

This module provides an in-memory LFU cache with optional ttl-support.
Entries are evicted based on usage frequency and (optionally) expiration
time.

Classes:
    LFUCache: The main LFU cache class.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from typing import Any, Iterator, Hashable, Optional, Union

from .base import BaseCache
from .node import Node


__all__ = ["LFUCache"]


class LFUCache(BaseCache):
    """
    An in-memory LFUCache (Least Frequently Used) with optional ttl-support.
    """

    def __init__(self,
                 name: Optional[str] = "",
                 max_size: Optional[int] = None,
                 verbose: bool = False,
                 thread_safe: bool = True
                 ):
        # set the name to cachelib.LFUCache.name
        name = f"cachelib.{self.__class__.__name__}" + (f".{name}" if name else "")
        super().__init__(name=name,
                         max_size=max_size,
                         verbose=verbose,
                         thread_safe=thread_safe
                         )
        self._lookup_freq: dict[Hashable, int] = {}

        self._head: Node = Node(None, None)
        self._tail: Node = Node(None, None)

        self._head.next = self._tail
        self._tail.prev = self._head

        self._logger.info(f"Initialized LFUCache with max-size={self._max_size}")

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
        self._lookup_freq[key] = 0

        # update cache.stats.size
        self._stats._size = self.__len__()

        return node

    def _remove_node(self, node: Node) -> None:

        # Remove node from linked list
        node.prev.next = node.next
        node.next.prev = node.prev
        node.prev = node.next = None

        # Remove node from cache
        del self._cache[node.key]
        del self._lookup_freq[node.key]

        self._stats._size = self.__len__()  # update cache.stats.size

    def _get_evict_node(self) -> Node:
        sorted_freq = sorted(self._lookup_freq, key=lambda k: self._lookup_freq[k])

        least_freq_keys = []

        # Get the amount of keys that are equally least accessed
        for key in sorted_freq:
            if not self._cache[key].is_expired():
                if len(least_freq_keys):
                    if self._lookup_freq[key] == self._lookup_freq[least_freq_keys[0]]:
                        least_freq_keys.append(key)
                else:
                    least_freq_keys.append(key)

        # If there are multiple keys that are the least accessed one,
        # evict based on least recently accessed.
        if len(least_freq_keys) > 1:
            current_node = self._tail
            while current_node.key not in least_freq_keys:
                current_node = current_node.prev

            return current_node
        return self._cache[least_freq_keys[0]]

    def _update_cache_state(self, node: Node) -> None:
        # Increase the node's lookup counter
        self._lookup_freq[node.key] += 1

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
        Yield the (key, value) pairs in order of most recently accessed.

        Returns:
            Iterator (Iterator[tuple[Hashable, Any]]): Iterates over the (key, value) pairs.
        """
        keys = self._cache.copy()
        sorted_freq = sorted(self._lookup_freq, key=lambda k: self._lookup_freq[k], reverse=True)

        for _ in range(self.__len__()):
            most_freq_keys = []

            # Get the amount of keys that are equally least accessed
            for key in sorted_freq:
                if not keys[key].is_expired():
                    if len(most_freq_keys):
                        if self._lookup_freq[key] == self._lookup_freq[most_freq_keys[0]]:
                            most_freq_keys.append(key)
                    else:
                        most_freq_keys.append(key)

            # If there are multiple keys that are the least accessed one,
            # evict based on least recently accessed.
            if len(most_freq_keys) > 1:
                current_node = self._tail
                while current_node.key not in most_freq_keys:
                    current_node = current_node.prev

                yield (current_node.key, current_node.value)

                del keys[current_node]
                del sorted_freq[current_node]
            else:
                node = keys[most_freq_keys[0]]

                yield (node.key, node.value)

                del keys[most_freq_keys[0]]
                del sorted_freq[sorted_freq.index(most_freq_keys[0])]

    def __reversed__(self) -> Iterator[tuple[Hashable, Any]]:
        """
        Yield the (key, value) pairs in order of least recently accessed.

        Returns:
            Iterator (Iterator[tuple[Hashable, Any]]): Iterates over the (key, value) pairs.
        """
        reversed = []

        # Transform the __iter__ iterator to a list
        for key in self.__iter__():
            reversed.append(key)
        # And reverse it
        reversed.reverse()

        for key in reversed:
            yield key
