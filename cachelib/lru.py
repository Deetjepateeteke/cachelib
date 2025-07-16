# -*- coding: utf-8 -*-

"""
lru.py - LRU (Least Recently Used) Cache implementation.

This module provides an in-memory LRU cache with optional ttl-support.
Entries are evicted based on usage order and (optionally) expiration
time.

Classes:
    LRUCache: The main LRU cache class.

Usage:
    >>> cache = LRUCache()
    >>> cache.set('foo', 'bar')
    >>> cache.get('foo')
    'bar'

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Hashable, Optional, Self, Union

from .base import BaseCache
from .node import Node


__all__ = ["LRUCache"]


class LRUCache(BaseCache):
    """
    An in-memory LRUCache (Least Recently Used) with optional ttl-support.

    Methods:
        get(key): Retrieve a value by key.
        set(key, value, ttl=None): Set the value for the given key.
        delete(key): Remove an entry from the cache.
        clear(): Reset the cache.
        inspect(key): Get the information of a key.
        memoize(ttl=None): A decorator function that caches the result from a function.
        save(path): Save the cache to a .pkl file.
        read_only(): A context manager that enables read-only mode.
        set_read_only(read_only): The manual version of read_only().
        verbose(): A context manager that enables debug mode.
        set_verbose(verbose): The manual version of verbose().
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

    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve the value for the given key (if the key exists).
        Then move the key to the most recently accessed position.

        Args:
            Key (Hashable): The requested key

        Returns:
            Optional[Any]: If the key is found in the cache,
                        return its value, None otherwise.
        """
        try:
            with self._lock:
                node = self._cache[key]

                if not node.is_expired():
                    self._move_to_head(node)

                    self._stats._hits += 1
                    self._logger.debug(f"GET key='{key}' (hit)")

                    return node.value
                else:
                    if not self._read_only:
                        # remove node due to ttl
                        self.delete(node.key)

                        self._stats._evictions += 1
                        self._logger.debug(f"EVICT key='{node.key}' due to ttl")
        except KeyError:
            self._logger.debug(f"GET key='{key}' (miss)")
            self._stats._misses += 1
            return None

    def set(self,
            key: Hashable,
            value: Any,
            ttl: Optional[Union[int, float]] = None
            ) -> None:
        """
        Set the value for the given key (if it doesn't exist yet),
        update the value otherwise. Then move the key to
        the most recently accessed position.

        Parameters:
            key (Any): The key
            value (Any): The value
            ttl (Optional[Union[int, float]]): The time to live in seconds

        Returns:
            None

        Raises:
            RuntimeError: When LRUCache.set() is called when read-only mode is enabled.
        """
        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                if key not in self._cache.keys():
                    # Create a new entry in the cache
                    node = Node(key, value, ttl)
                    self._cache[key] = node
                    self._add_node(node)
                    self._logger.debug("SET key='%s' %s (adding new key)"
                                       % (key, f"with ttl={ttl}" if ttl else ""))
                else:
                    # Update an existing entry
                    node = self._cache[key]
                    node.value = value
                    if ttl:
                        node.ttl = ttl  # update ttl
                    self._move_to_head(node)
                    self._logger.debug("SET key='%s' %s (updating value)"
                                       % (key, f"with ttl={ttl}" if ttl else ""))

                # Evict the node tail.prev if the capacity gets exceeded.
                # Don't evict nodes if capacity is None (infinite capacity).
                if (self._max_size is not None) and (self.__len__() > self._max_size):
                    # Remove the tail node
                    tail = self._tail.prev
                    self.delete(tail.key)

                    self._stats._evictions += 1
                    self._logger.debug(f"EVICT key='{tail.key}' due to capacity")
            else:
                raise self._ReadOnlyContext.error

    def delete(self, key: Hashable) -> None:
        """
        Delete a key from the cache.

        Args:
            key (Any): The key that has to be deleted

        Returns:
            None

        Raises:
            RuntimeError: When LRUCache.delete() is called when read-only mode is enabled.
        """
        with self._lock:
            if not self._read_only:
                try:
                    node = self._cache[key]
                    del self._cache[key]
                    self._remove_node(node)

                    self._logger.debug(f"REMOVE key='{key}'")
                except KeyError:
                    raise KeyError("Invalid key")
            else:
                raise self._ReadOnlyContext.error

    def clear(self) -> None:
        """
        Removes all elements from the cache.

        Returns:
            None

        Raises:
            RuntimeError: When LRUCache.clear() is called when read-only mode is enabled.
        """
        with self._lock:
            if not self._read_only:
                self._cache = {}
                self._head = Node(None, None)
                self._tail = Node(None, None)

                self._stats._size = 0
                self._logger.debug("CLEAR CACHE")
            else:
                raise self._ReadOnlyContext.error

    def inspect(self, key: Hashable) -> Optional[dict[str, Any]]:
        """
        Inspect the key in the cache. Returns a dict with
        the key's information, if the key exists in the cache;
        None otherwise. Afterwards, place the key as the
        most recently used.

        Args:
            key (Hashable): The requested key

        Returns:
            dict[str, Any] or None: Return a dict with the key's information,
                                    if the key is found, None otherwise.
                                    The dict contains key name, value,
                                    expired and ttl.

        """
        if key not in self._cache.keys():
            return None

        node = self._cache[key]
        self._move_to_head(node)

        return {
            "key": key,
            "value": node.value,
            "expired": node.is_expired(),
            "ttl": node._expires_at
        }

    def memoize(self, ttl: Optional[Union[int, float]] = None) -> Any:

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                def _make_cache_key(args, kwargs) -> tuple:
                    """
                    Transform the args and kwargs into a hashable tuple
                    so it can be used as a dict key. The tuple looks like:
                    (a, b, (c, 1), (d, 2))
                    """
                    kwargs_key = tuple(sorted(kwargs.items()))
                    return args + kwargs_key

                cache_key = _make_cache_key(args, kwargs)

                with self._lock:
                    if cache_key in self._cache.keys():
                        return self.get(cache_key)
                    else:
                        result = func(*args, **kwargs)
                        if not self._read_only:
                            self.set(cache_key, result, ttl=ttl)
                        else:
                            raise self._ReadOnlyContext.error

                        return result

            return wrapper
        return decorator

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """
        Load a LRUCache from a .pkl file.
        """
        obj = super().load(path)

        # The imported cache must be a LRUCache.
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected instance of {cls.__name__}, "
                f"got {obj.__name__}."
            )
        return obj

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    def max_size(self, max_size: Optional[int]) -> None:
        """
        Modify the cache's max-size.

        Args:
            max_size (Optional[int]): The new max_size of the cache. If max_size=0,
                                        the cache's max_size will be infinite.

        Raises:
            RuntimeError: When LRUCache.max_size gets modified when read-only mode is enabled.
        """
        # Check valid type
        super()._check_max_size_valid(max_size)

        with self._lock:
            if not self._read_only:
                self._max_size = max_size
                self._logger.debug(f"CHANGE max-size={max_size}")

                if not isinstance(self._max_size, type(None)):
                    while self._max_size and self.__len__() > self._max_size:
                        self.delete(self._tail.prev.key)
                        self._logger.debug(f"EVICT key='{self._tail.prev.key}' \
                                        due to max-size")

                self._stats._max_size = self._max_size  # update cache.stats.max_size
            else:
                raise self._ReadOnlyContext.error

    def _add_node(self, node: Node) -> None:
        """
        [HEAD] <-> [new node] <-> [old most recently used node] <-> ... <-> [TAIL]
        """
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

        self._stats._size = self.__len__()  # update cache.stats.size

    def _remove_node(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

        node.prev = node.next = None

        self._stats._size = self.__len__()  # update cache.stats.size

    def _move_to_head(self, node: Node) -> None:
        self._remove_node(node)
        self._add_node(node)

    # Dunder methods
    def __contains__(self, key: Hashable) -> bool:
        """
        Check if the given key is in the cache.

        Args:
            key (Any): The requested key

        Returns:
            bool: Returns True if the key is found in the cache,
                False otherwise.
        """
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """
        Return the amount of entries in the cache.

        Returns:
            int: The amount of entries in the cache.
        """
        with self._lock:
            return len(self._cache)

    def __iter__(self) -> Iterator[Any]:
        """
        Yield the (key, value) pairs in order of most recently used.

        Returns:
            Iterator[Any]: The iterator with (key, value) pairs.
        """
        with self._lock:
            node = self._head.next
            while node != self._tail:
                yield (node.key, node.value)
                node = node.next

    def __reversed__(self) -> Iterator[Any]:
        """
        Yield the (key, value) pairs in order of least recently used.

        Returns:
            Iterator[Any]: The iterator with (key, value) pairs.
        """
        with self._lock:
            node = self._tail.prev
            while node != self._head:
                yield (node.key, node.value)
                node = node.prev

    def __repr__(self) -> str:
        """
        Examples:
            >>> repr(LRUCache)
            LRUCache(max-size=5, size=4, keys=['foo', 'bar', 'foo2', ...])
        """
        keys = list(self._cache.keys())
        preview = ", ".join([f"{key!r}" for key in keys[:3]])
        if len(keys) > 3:
            preview += ", ..."

        return (
            f"{self.__class__.__name__}(max-size={self._max_size!r}, "
            f"size={self.__len__()!r}, keys=[{preview}])"
        )
