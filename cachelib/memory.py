# -*- coding: utf-8 -*-

"""
memory.py - In-memory cache implementation.

This module provides an in-memory cache with lru/lfu eviction
and optional ttl-support.

Classes:
    MemoryCache: The in-memory cache class.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from typing import Any, Hashable, Optional, Union

from .base import BaseCache
from .errors import *
from .node import Node


__all__ = ["MemoryCache"]


class MemoryCache(BaseCache):
    """
    An in-memory cache with lru/lfu eviction and optional ttl-support.
    """
    def __init__(self,
                 name: str = "",
                 max_size: Optional[int] = None,
                 eviction_policy: str = "",
                 ttl: Optional[Union[int, float]] = None,
                 verbose: bool = False,
                 thread_safe: bool = True
                 ):
        """
        Initialize a new MemoryCache.

        Args:
            name (str): The cache's name.
            max_size (Optional[int]): The maximum amount of entries that
                fit in the cache.
            eviction_policy (str): The eviction policy to use:
                - '': No eviction
                - 'lru': Least Recently Used
                - 'lfu': Least Frequently Used
            ttl (Optional[Union[int, float]]): The default ttl of every entry
                in the cache.
            verbose (bool): Debug mode
            thread_safe (bool): Make the cache thread safe.
        """
        super().__init__(
            name=name,
            max_size=max_size,
            ttl=ttl,
            verbose=verbose,
            thread_safe=thread_safe
        )

        # Check for a valid eviction policy
        if eviction_policy in ("lfu", "lru", ""):
            if max_size is None and eviction_policy in ("lfu", "lru"):
                raise ValueError("can only set eviction_policy when max_size is not infinite")
            else:
                self._eviction_policy: str = eviction_policy
        else:
            raise ValueError(f"eviction_policy should be 'lru' or 'lfu', got '{eviction_policy}'")

        self._cache: dict[Hashable, Any] = {}

        if self._eviction_policy == "lfu":
            # Used to hold track of access frequency in LFU
            self._access_freq: dict[Hashable, int] = {}

        if self._eviction_policy in ("lfu", "lru"):
            # Use a linked list to track recency
            self._head: Node = Node(None, None)
            self._tail: Node = Node(None, None)

            self._head.next = self._tail
            self._tail.prev = self._head

        # Start in-background cleanup thread
        self._cleanup_thread.start()
        self._logger.info(f"Initialized {self.__class__.__name__} with eviction-policy={self._eviction_policy}")

    def _add_node(self,
                  key: Hashable,
                  value: Any,
                  ttl: Optional[Union[int, float]]
                  ) -> Node:
        with self._lock:
            # Create a node
            node: Node = Node(key, value, ttl)

            # Store in cache
            self._cache[key] = node

            if self._eviction_policy in ("lfu", "lru"):
                # Update linked list
                node.prev = self._head
                node.next = self._head.next
                self._head.next.prev = node
                self._head.next = node

            if self._eviction_policy == "lfu":
                self._access_freq[key] = 0

            return node

    def _update_node(self, node: Node, params: dict[str, Any]) -> None:
        with self._lock:
            # Update the node's value, if provided
            if "value" in params.keys():
                node.value = params["value"]
            # Update the node's ttl, if provided
            if "ttl" in params.keys():
                node.ttl = params["ttl"]

    def _get_node(self, key: Hashable) -> Node:
        # When the requested node doesn't exist,
        # it will raise a KeyError that will be handled
        # by BaseCache.get() by returning None.
        with self._lock:
            return self._cache[key]

    def _remove_node(self, node: Node) -> None:
        with self._lock:

            # Remove node from cache
            del self._cache[node.key]

            if self._eviction_policy in ("lfu", "lru"):
                # Remove node from liked list
                node.prev.next = node.next
                node.next.prev = node.prev
                node.prev = node.next = None

            if self._eviction_policy == "lfu":
                del self._access_freq[node.key]

    def _get_evict_node(self) -> Node:
        with self._lock:
            if self._eviction_policy == "lfu":
                return self._lfu_eviction()
            elif self._eviction_policy == "lru":
                return self._lru_eviction()
            else:
                raise CacheOverflowError(max_size=self._max_size)

    def _update_cache_state(self, node: Node) -> None:
        with self._lock:
            # Update the node's access frequency
            if self._eviction_policy == "lfu":
                self._access_freq[node.key] += 1

            # Place the node at the start of the linked list
            if self._eviction_policy in ("lfu", "lru"):
                # Remove node
                node.prev.next = node.next
                node.next.prev = node.prev

                # Add node
                node.prev = self._head
                node.next = self._head.next
                self._head.next.prev = node
                self._head.next = node

    def _lru_eviction(self) -> Node:
        """
        LRU (Least Recently Used) eviction.
        This is the eviction method called when eviction_policy='lru'.
        """
        with self._lock:
            current_node = self._tail.prev
            while current_node.is_expired():
                current_node = current_node.prev

            return current_node

    def _lfu_eviction(self) -> Node:
        """
        LFU (Least Frequently Used) eviction.
        This is the eviction method called when eviction_policy='lfu'.
        """
        with self._lock:
            # Sort the keys by frequency in ascending order
            sorted_freq = sorted(self._access_freq, key=lambda k: self._access_freq[k])

            least_freq_keys = []

            # Get the amount of keys that are equally least accessed
            for key in sorted_freq:
                if not self._get_node(key).is_expired():
                    if len(least_freq_keys):
                        if self._access_freq[key] == self._access_freq[least_freq_keys[0]]:
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
            return self._get_node(least_freq_keys[0])
