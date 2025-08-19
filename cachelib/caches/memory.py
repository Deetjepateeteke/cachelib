# -*- coding: utf-8 -*-

"""
memory.py - In-memory cache implementation.

This module provides an in-memory cache with lru/lfu eviction
and optional ttl-support.

Classes:
    MemoryCache: The in-memory cache class.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from pathlib import Path
import pickle
from typing import Any, Hashable, Optional, Self, Union

from .base import BaseCache
from ..errors import (
    CacheLoadError,
    CacheSaveError,
    KeyNotFoundError,
    ReadOnlyError
)
from ..eviction import EvictionPolicy, _LFUEviction
from ..node import Node


__all__ = ["MemoryCache"]


class MemoryCache(BaseCache):
    """
    An in-memory cache with lru/lfu eviction and optional ttl-support.
    """
    def __init__(self,
                 name: str = "",
                 max_size: Optional[int] = None,
                 eviction_policy: Optional[EvictionPolicy] = None,
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
                - None: No eviction
                - cachelib.eviction.LRU: Least Recently Used
                - cachelib.eviction.LFU: Least Frequently Used
            ttl (Optional[Union[int, float]]): The default ttl of every entry
                in the cache.
            verbose (bool): Debug mode
            thread_safe (bool): Make the cache thread safe.
        """
        super().__init__(
            name=name,
            max_size=max_size,
            eviction_policy=eviction_policy,
            ttl=ttl,
            verbose=verbose,
            thread_safe=thread_safe
        )

        self._cache: dict[Hashable, Any] = {}

        if isinstance(self._eviction_policy, _LFUEviction):
            # Used to hold track of access frequency in LFU
            self._access_freq: dict[Hashable, int] = {}

        # Use a linked list to track recency
        if isinstance(self._eviction_policy, EvictionPolicy):
            self._head: Node = Node(None, None)
            self._tail: Node = Node(None, None)

            self._head.next = self._tail
            self._tail.prev = self._head

        # Initialize self.cleanup_thread
        self._create_cleanup_thread()

        self.logger.info(f"Initialized {self.__class__.__name__} with eviction-policy={self._eviction_policy}")

    def _add_node(self,
                  key: Hashable,
                  value: Any,
                  ttl: Optional[Union[int, float]]) -> Node:
        with self._lock:

            # Create a new node
            node: Node = Node(key, value, ttl)

            # Store in cache
            self._cache[key] = node

            if isinstance(self._eviction_policy, EvictionPolicy):
                # Update linked list
                node.prev = self._head
                node.next = self._head.next
                self._head.next.prev = node
                self._head.next = node

            if isinstance(self._eviction_policy, _LFUEviction):
                # Initialize frequency table
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
        with self._lock:
            try:
                return self._cache[key]
            except KeyError as exc:
                raise KeyNotFoundError(key) from exc

    def _remove_node(self, node: Node) -> None:
        with self._lock:

            # Remove node from cache
            del self._cache[node.key]

            if isinstance(self._eviction_policy, EvictionPolicy):
                # Remove node from liked list
                node.prev.next = node.next
                node.next.prev = node.prev
                node.prev = node.next = None

            if isinstance(self._eviction_policy, _LFUEviction):
                del self._access_freq[node.key]

    def _move_to_top(self, node: Node) -> None:
        with self._lock:
            # Update the node's access frequency
            if isinstance(self._eviction_policy, _LFUEviction):
                self._access_freq[node.key] += 1

            # Place the node at the start of the linked list
            if isinstance(self._eviction_policy, EvictionPolicy):
                # Remove node
                node.prev.next = node.next
                node.next.prev = node.prev

                # Add node
                node.prev = self._head
                node.next = self._head.next
                self._head.next.prev = node
                self._head.next = node

    def clear(self) -> None:
        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                self._cache = {}

                if isinstance(self._eviction_policy, _LFUEviction):
                    self._access_freq = {}

                self._head = self._tail = Node(None, None)
                self._head.next = self._tail
                self._tail.prev = self._head

                self.logger.debug("CLEAR CACHE")
            else:
                raise ReadOnlyError()

    def keys(self) -> tuple[Hashable]:
        with self._lock:
            return tuple(self._cache.keys())

    def values(self) -> tuple[Any]:
        with self._lock:
            nodes = tuple(self._cache.values())
            return tuple(map(lambda node: node.value, nodes))

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the cache as a .pkl file.

        Args:
            path (Union[str, Path]): The path to the .pkl file
                        in which the cache will be stored.

        Returns:
            None

        Raises:
            ValueError: When the given path isn't a .pkl file.
            TypeError: When path isn't a str or a pathlib.Path().
        """
        path = self._check_path_valid(path, suffix=".pkl")

        with self._lock:
            try:
                with open(path, "wb") as f:
                    pickle.dump(self, f)

                self.logger.debug(f"SAVE path='{path}'")
            except Exception as exc:
                raise CacheSaveError(exc) from exc

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """
        Load a cache from a .pkl file.

        Args:
            path (Union[str, Path]): The path to the saved cache.

        Returns:
            Self: The cache object (a subclass of BaseCache).

        Raises:
            CacheLoadError: When something went wrong while reading the .pkl file.
            CachePathError: When the given path isn't a .pkl file.
            CachePathError: When path isn't a str or a pathlib.Path().
        """
        path = cls._check_path_valid(path, suffix=".pkl")

        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception as exc:
            raise CacheLoadError(exc) from exc

        # Check if the imported cache is of the same type
        # as the class load() got called from.
        # Eg. LRUCache.load() -> LRUCache
        if not isinstance(obj, cls):
            raise CacheLoadError(
                f"expected instance of {cls.__name__}, "
                f"got {obj.__class__.__name__}"
            )
        return obj

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def _lru_eviction(self) -> Node:
        """
        LRU (Least Recently Used) eviction.
        This is the eviction method called when eviction_policy=cachelib.eviction.LRU.
        """
        with self._lock:
            current_node = self._tail.prev
            while current_node.is_expired():
                current_node = current_node.prev

            return current_node

    def _lfu_eviction(self) -> Node:
        """
        LFU (Least Frequently Used) eviction.
        This is the eviction method called when eviction_policy=cachelib.eviction.LFU.
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

    @staticmethod
    def _create_cache_key(key):
        """
        The key doesn't need to be modified to be stored in a hashmap.
        """
        return key
