# -*- coding: utf-8 -*-

"""
multi_level_cache.py - Multi-level cache implementation.

Classes:
    MultiLevelCache: The multi-level cache class.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from functools import wraps
from threading import RLock
from typing import Any, Callable, Hashable, Iterable, Literal, NoReturn, Optional, Union

from .base import BaseCache
from ..errors import (
    CacheConfigurationError,
    KeyNotFoundError,
    ReadOnlyError
)
from ..node import Node
from ..utils import create_json_cache_key, extract_items_from_args


__all__ = ["MultiLevelCache"]


class MultiLevelCache(BaseCache):
    """
    A multi-level cache that spreads it key/value pairs over multiple caches
    to improve performance. The cache supports 'inclusive' or 'exclusive' mode.
    """
    def __init__(self,
                 levels: list[BaseCache],
                 inclusivity: Literal["inclusive", "exclusive"] = "inclusive"
                 ):
        if inclusivity not in ["inclusive", "exclusive"]:
            raise ValueError("MultiLevelCache.inclusivity should be either 'inclusive' or 'exclusive'")
        self._inclusivity: Literal["inclusive", "exclusive"] = inclusivity

        self._init_levels(levels)

        # Thread safety
        self._lock: RLock = RLock()
        self._read_only: bool = False

    def _init_levels(self, levels: list[BaseCache]) -> None:
        """ Initialize self.levels with the given subcaches. """
        # Check if levels is valid
        if len(levels) < 2 or len(levels) > 3:
            raise CacheConfigurationError(f"There must be between 2 and 3 subcaches in a MultiLevelCache, got {levels}")

        self.l1: BaseCache = levels[0]
        self.l2: BaseCache = levels[1]
        self.l3: Optional[BaseCache] = levels[2] if len(levels) == 3 else None

        if self.l3 is not None:
            self.levels: tuple[BaseCache] = (self.l1, self.l2, self.l3)
        else:
            self.levels: tuple[BaseCache] = (self.l1, self.l2)

    def _add_node(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError()

    def _update_node(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError()

    def _get_node(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError()

    def _remove_node(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError()

    def _get_evict_node(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError()

    def _create_cache_key(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError()

    def _move_to_top(self, node: Node) -> None:
        with self._lock:
            self.delete(node.key)
            self.set(node.key, node.value, node.ttl)

    def get(self, key: Hashable) -> Any:
        with self._lock:
            for cache in self.levels:
                cache_key = cache._create_cache_key(key)

                if cache_key in cache:
                    node: Node = cache._get_node(cache_key)
                    cache._move_to_top(node)

                    if not node.is_expired():
                        # Update the node's position in the cache.
                        self._move_to_top(node)

                    else:
                        # Evict the key from the cache due to ttl.
                        if not self._read_only:
                            self.delete(cache_key)

                    return node.value
            # The key doesn't existin the cache.
            return None

    def get_many(self, keys: Iterable) -> tuple[Any]:
        values = tuple()

        with self._lock:
            for key in keys:
                values += (self.get(key),)
        return values

    def set(self, key: Hashable, *args, **kwargs) -> None:

        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:

                params = extract_items_from_args(*args, **kwargs)

                # Can't create a node without a value
                if "value" not in params.keys():
                    raise CacheConfigurationError("expected 'value' argument")

                if not self.__contains__(key) and not self.__contains__(create_json_cache_key(key)):
                    # Add a new entry

                    # When 'ttl' is not found, default to the cache's global ttl
                    value, ttl = params["value"], self.l1._ttl
                    if "ttl" in params.keys():
                        ttl = params["ttl"]

                    # If inclusivity is 'inclusive', add the key to all the caches.
                    if self._inclusivity == "inclusive":
                        for cache in self.levels:
                            cache_key = cache._create_cache_key(key)
                            cache._add_node(cache_key, value, ttl)

                    # If inclusivity is 'exclusive', add the key only to the first cache.
                    elif self._inclusivity == "exclusive":
                        cache_key = self.l1._create_cache_key(key)
                        self.l1._add_node(cache_key, value, ttl)

                    for i, cache in enumerate(self.levels):
                        if cache._exceeds_max_size():
                            # When max_size gets exceeded, evict the node
                            evict_node = cache._get_evict_node()
                            cache._remove_node(evict_node)

                            if i != len(self.levels) - 1:
                                next_cache = self.levels[i+1]

                                if evict_node.key not in next_cache:
                                    next_cache._add_node(evict_node.key, evict_node.value, evict_node.ttl)
                else:
                    for cache in self.levels:
                        cache_key = cache._create_cache_key(key)

                        if cache_key in cache:
                            node = cache._get_node(cache_key)
                            params = extract_items_from_args(*args, **kwargs)

                            # Check for either 'value' or 'ttl' in args
                            if "value" not in params.keys() and "ttl" not in params.keys():
                                raise CacheConfigurationError("expected either 'value' or 'ttl' as arguments")

                            cache._update_node(node, params)
                            cache._move_to_top(node)
                            self._move_to_top(node)
            else:
                raise ReadOnlyError()

    def delete(self, key: Hashable) -> None:
        with self._lock:
            # Do not allow changes while read-only is enabled
            if not self._read_only:
                if self.__contains__(key) or self.__contains__(create_json_cache_key(key)):
                    for cache in self.levels:
                        cache_key = cache._create_cache_key(key)

                        if cache_key in cache:
                            cache.delete(cache_key)

                            # When keys get stored in only a single cache, break from loop
                            if self._inclusivity == "exclusive":
                                break

                else:
                    # When the key isn't found in any cache, raise an exception
                    raise KeyNotFoundError(key)
            else:
                raise ReadOnlyError()

    def clear(self) -> None:
        with self._lock:
            # Do not allow changes while read-only is enabled
            if not self._read_only:
                for cache in self.levels:
                    cache.clear()
            else:
                raise ReadOnlyError()

    def ttl(self, key: Hashable) -> Optional[Union[int, float]]:
        with self._lock:
            for cache in self.levels:
                cache_key = cache._create_cache_key(key)

                if cache_key in cache:
                    return cache.ttl(cache_key)
            raise KeyNotFoundError(key)

    def inspect(self, key: Hashable) -> Optional[dict[str, Any]]:
        with self._lock:
            for cache in self.levels:
                cache_key = cache._create_cache_key(key)

                if cache_key in cache:
                    return cache.inspect(cache_key)
            raise KeyNotFoundError(key)

    def keys(self) -> tuple[Hashable]:
        with self._lock:
            keys = set()
            for cache in self.levels:
                keys.update(cache.keys())

            return tuple(keys)

    def values(self) -> tuple[Any]:
        with self._lock:
            values = set()
            for cache in self.levels:
                values.update(cache.values())

            return tuple(values)

    def memoize(self, ttl: Optional[Union[int, float]] = None) -> Any:
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:

                # Transform the given args into a sorted tuple
                key = tuple(args) + tuple(sorted(kwargs.items()))

                with self._lock:

                    # Try to retrieve cached function result.
                    for cache in self.levels:
                        cache_key = cache._create_cache_key(key)

                        if cache_key in cache:
                            node = cache._get_node(cache_key)
                            if not node.is_expired():
                                return node.value

                    # When the key is not found in the cache, compute it.
                    result = func(*args, **kwargs)
                    if not self._read_only:
                        self.set(key, result, ttl=ttl)
                    else:
                        raise ReadOnlyError()

                    return result
            return wrapper
        return decorator

    def __len__(self) -> int:
        with self._lock:
            return len(self.keys())

    def close(self) -> None:
        with self._lock:
            for cache in self.levels:
                try:
                    cache.close()
                except Exception:
                    continue
