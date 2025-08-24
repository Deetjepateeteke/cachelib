# -*- coding: utf-8 -*-

"""
base.py - BaseCache implementation

This module provides an abstract BaseCache that provides the foundational
interface for every cache in cachelib.

Classes:
    BaseCache(): The foundational BaseCache class.
    Stats(): Used to store statistics, it is used in BaseCache.stats.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import wraps
import logging
from pathlib import Path
import re
from threading import RLock
import time
from typing import Any, Callable, Hashable, Optional, Union

from ..cleanup_thread import CleanupThread, DiskCleanupThread, MemoryCleanupThread
from ..errors import (
    CacheConfigurationError,
    CacheOverflowError,
    PathError,
    DeserializationError,
    KeyExpiredError,
    KeyNotFoundError,
    ReadOnlyError,
    SerializationError
)
from ..eviction import EvictionPolicy, FIFO, LFU, LRU
from ..node import Node
from ..utils import extract_items_from_args, NullContext, NullValue

__all__ = ["BaseCache", "Stats"]


class BaseCache(ABC):
    """
    An abstract base cache that provides the foundational methods
    and attributes for every cache in cachelib.

    Methods that should be implemented in subclasses of this class:
        - _add_node
        - _update_node
        - _get_node
        - _remove_node
        - _get_evict_node
        - _move_to_top
        - _make_cache_key
    """

    @abstractmethod
    def __init__(self,
                 name: str = "",
                 max_size: Optional[int] = None,
                 max_memory: Optional[Union[int, str]] = NullValue(),
                 eviction_policy: Optional[EvictionPolicy] = None,
                 ttl: Optional[Union[int, float]] = None,
                 verbose: bool = False,
                 thread_safe: bool = True):

        self.name: str = name

        if isinstance(max_memory, NullValue) and isinstance(max_size, NullValue):
            raise CacheConfigurationError("max_size and max_memory cannot be of type NullValue at the same time")

        # Initialize self._max_size and self._max_memory.
        if not isinstance(max_memory, NullValue):
            self._max_memory: Optional[Union[int, str]] = self._parse_max_memory(max_memory)
            self._max_size: NullValue = NullValue()

        elif self._check_max_size_valid(max_size):
            self._max_size: Optional[int] = max_size
            self._max_memory: NullValue = NullValue()

        self._ttl: Optional[Union[int, float]] = ttl
        self._thread_safe: bool = thread_safe
        self._lock: Union[NullContext, RLock] = \
            RLock() if self._thread_safe else NullContext()  # thread safety
        self.logger: logging.Logger = logging.getLogger(self.name)
        self._stats: Stats = Stats(self)  # statistics
        self._read_only: bool = False  # read-only mode

        # Check for a valid eviction policy
        if isinstance(eviction_policy, (EvictionPolicy, type(None))):
            if isinstance(eviction_policy, EvictionPolicy):
                if self._max_size is None or self._max_memory is None:
                    raise CacheConfigurationError("can only set eviction_policy when max_size is not infinite")
        else:
            raise CacheConfigurationError(
                f"Eviction policy must be of type "
                f"{EvictionPolicy.__module__}.{EvictionPolicy.__qualname__}"
                f", got {type(eviction_policy).__name__}"
            )
        self._eviction_policy: EvictionPolicy = eviction_policy

        self.set_verbose(verbose)

    def _create_cleanup_thread(self):
        # Initialize self.cleanup_thread
        self._cleanup_thread_interval = 1.0
        if self.__class__.__name__ == "MemoryCache":
            self.cleanup_thread: CleanupThread = MemoryCleanupThread(self, self._cleanup_thread_interval)
        elif self.__class__.__name__ == "DiskCache":
            self.cleanup_thread: CleanupThread = DiskCleanupThread(self, self._cleanup_thread_interval)
        self.cleanup_thread.start()

    @abstractmethod
    def _add_node(self,
                  key: Hashable,
                  value: Any,
                  ttl: Optional[Union[int, float]] = None
                  ) -> Node:
        """
        Add a node to the cache.

        Args:
            key (Hashable): The node's key.
            value (Any): The node's value.
            ttl (Optional[Union[int, float]]): The node's ttl in seconds.

        Returns:
            Node: The newly created node.
        """
        ...

    @abstractmethod
    def _update_node(self, node: Node, params: dict[str, Any]) -> None:
        """
        Update an existing node.

        Args:
            node (Node): The node that will be updated.
            params (dict[str, Any]): A dict that has 'value' and/or
                'ttl' as keys and as value the node's new value or ttl.

        Returns:
            None
        """
        ...

    @abstractmethod
    def _get_node(self, key: Hashable) -> Optional[Node]:
        """
        Retrieve the node of the given key.

        Args:
            key (Hashable): The given key.

        Returns:
            Node: The key's node.

        Raises:
            KeyNotFoundError: When the node is not found.
        """
        ...

    @abstractmethod
    def _remove_node(self, node: Node) -> None:
        """
        Remove a node.

        Args:
            node (Node): The node that has to be removed.

        Returns:
            None
        """
        ...

    def _get_evict_node(self) -> Node:
        """
        Get the next node to be evicted.

        Returns:
            Node: The node to evict.
        """
        with self._lock:
            if self._eviction_policy is LFU:
                return self._lfu_eviction()
            elif self._eviction_policy is LRU:
                return self._lru_eviction()
            elif self._eviction_policy is FIFO:
                return self._fifo_eviction()
            else:
                raise CacheOverflowError(max_size=self._max_size)

    @abstractmethod
    def _move_to_top(self, node: Node) -> None:
        """
        This function gets called whenever a node
        is accessed in the cache. Eg. cache.get(node).

        Args:
            node (Node): The node that got accessed.
        """
        ...

    @abstractmethod
    def _get_cache_size(self) -> int:
        """
        Get the cache's size in bytes.

        Returns:
            int: The cache's size in bytes.
        """

    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve the value for the given key (if the key exists).
        If the key isn't found in the cache, None gets returned.

        Args:
            Key (Hashable): The requested key.

        Returns:
            Optional[Any]: If the key is found in the cache,
                        return its value; None otherwise.
        """
        cache_key = self._create_cache_key(key)

        with self._lock:
            if self.__contains__(cache_key):
                node: Node = self._get_node(cache_key)

                # Check if the node is expired
                if not node.is_expired():
                    # Update the node's position in the cache.
                    self._move_to_top(node)

                    self._stats._hits += 1
                    self.logger.debug(f"GET key='{key}' (hit)")
                else:
                    # Evict the key from the cache due to ttl.
                    if not self._read_only:
                        self._remove_node(node)

                        self._stats._evictions += 1
                        self.logger.debug(f"EVICT key='{node.key}' due to ttl")

                return node.value
            else:
                # The key doesn't exist in the cache.
                self._stats._misses += 1
                self.logger.debug(f"GET key='{key}' (miss)")

                return None

    def get_many(self, keys: tuple[Hashable]) -> tuple[Any]:
        """
        Retrieve the corresponding values for the given keys, if they exist.

        Args:
            keys (tuple[Hashable]): The requested keys.

        Returns:
            tuple[Any]: The corresponding values to the
              requested keys.
        """
        values = tuple()

        with self._lock:
            for key in keys:
                values += (self.get(key),)
        return values

    def set(self, key: Hashable, *args, **kwargs) -> None:
        """
        Set the value (and ttl) for the given key (if it doesn't exist yet),
        update the value (and ttl) otherwise.

        Args:
            key (Hashable): The key.
            value (Any): The node's value.
            ttl (Optional[Union[int, float]]): The node's ttl.

        Returns:
            None

        Raises:
            CacheConfigurationError: When the given arguments aren't valid.
            ReadOnlyError: When cache.set() is called while
                        read-only mode is enabled.
        """
        key = self._create_cache_key(key)

        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                if key not in self.keys():
                    # Add a new key

                    params = extract_items_from_args(*args, **kwargs)

                    # Can't create a node without a value
                    if "value" not in params.keys():
                        raise CacheConfigurationError("expected 'value' argument")

                    # When 'ttl' is not found, default to the global ttl
                    if "ttl" not in params.keys():
                        params["ttl"] = self._ttl

                    # Create a new entry in the cache
                    node = self._add_node(key, params["value"], params["ttl"])

                    self.logger.debug("SET key='%s' %s (adding new key)"
                                      % (key, f"with ttl={node.ttl}" if node.ttl else ""))

                    # Evict a node if the capacity gets exceeded.
                    # Don't evict nodes if capacity is None (infinite capacity).
                    if self._exceeds_max_size():
                        node = self._get_evict_node()

                        self._remove_node(node)

                        self._stats._evictions += 1
                        self.logger.debug(f"EVICT key='{node.key}' due to capacity")

                else:
                    # Update an existing entry

                    node = self._get_node(key)
                    params = extract_items_from_args(*args, **kwargs)

                    # Check for either 'value' or 'ttl' in args
                    if "value" not in params.keys() and "ttl" not in params.keys():
                        raise CacheConfigurationError("expected either 'value' or 'ttl' as arguments")

                    self._update_node(node, params)
                    self._move_to_top(node)

                    self.logger.debug("SET key='%s' %s (updating value)"
                                      % (key, f"with ttl={node.ttl}" if node.ttl else ""))

            else:
                raise ReadOnlyError()

    def delete(self, key: Hashable) -> None:
        """
        Delete a key from the cache.

        Args:
            key (Hashable): The key that has to be deleted.

        Returns:
            None

        Raises:
            KeyNotFoundError: When the given key is not found in the cache.
            ReadOnlyError: When cache.delete() is called while
                        read-only mode is enabled.
        """
        cache_key = self._create_cache_key(key)

        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                if self.__contains__(cache_key):
                    self._remove_node(self._get_node(cache_key))

                    self.logger.debug(f"REMOVE key='{cache_key}'")
                else:
                    # Invalid key
                    raise KeyNotFoundError(cache_key)
            else:
                raise ReadOnlyError()

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all entries from the cache.

        Returns:
            None

        Raises:
            ReadOnlyError: When cache.clear() is called while
                        read-only mode is enabled.
        """
        ...

    def ttl(self, key: Hashable) -> Optional[Union[int, float]]:
        """
        Get the ttl (Time To Live) of the given key.

        Args:
            key (Hashable): The key.

        Returns:
            Optional[Union[int, float]]: The key's ttl.

        Raises:
            KeyNotFoundError: When the given key is nonexistent.
            KeyExpiredError: When the given key is expired.
        """
        cache_key = self._create_cache_key(key)

        with self._lock:
            if self.__contains__(cache_key):
                node = self._get_node(cache_key)

                if node.is_expired():
                    self._remove_node(node)
                    raise KeyExpiredError(key)

                return node.ttl
            else:
                raise KeyNotFoundError(cache_key)

    def inspect(self, key: Hashable) -> Optional[dict[str, Any]]:
        """
        Inspect the given key in the cache. Returns a dict with
        the key's information.

        Args:
            key (Hashable): The requested key.

        Returns:
            dict[str, Any]: Return a dict with the key's information.

        Raises:
            KeyNotFoundError: When the given key is nonexistent.
            KeyExpiredError: When the given key is expired.
        """
        cache_key = self._create_cache_key(key)

        with self._lock:
            if self.__contains__(cache_key):
                node: Node = self._get_node(cache_key)

                if node.is_expired():
                    self._remove_node(node)
                    raise KeyExpiredError(cache_key)

                return {
                    "key": cache_key,
                    "value": node.value,
                    "ttl": node._expires_at
                }
            else:
                raise KeyNotFoundError(cache_key)

    @abstractmethod
    def keys(self) -> tuple[Hashable]:
        """
        Get the cached keys.

        Returns:
            tuple[Hashable]: The tuple with the cached keys.
        """
        ...

    @abstractmethod
    def values(self) -> tuple[Any]:
        """
        Get the cached values.

        Returns:
            tuple[Any]: The tuple with the cached values.
        """
        ...

    def memoize(self, ttl: Optional[Union[int, float]] = None) -> Any:
        """
        A decorator.

        Cache the return value of a function with its arguments.
        When the function gets called again with the same arguments,
        the cached result will get returned.

        Args:
            ttl (Optional[Union[int, float]]): The ttl of every
                                            cached function result.

        Returns:
            Any: The cached or computed outcome of the function.

        Raises:
            ReadOnlyError: When a function call gets stored While
                        read-only mode is enabled
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:

                # Transform the given args into a sorted tuple
                cache_key = self._create_cache_key(tuple(args) + tuple(sorted(kwargs.items())))

                with self._lock:
                    if self.__contains__(cache_key) and not self._get_node(cache_key).is_expired():
                        return self.get(cache_key)
                    else:
                        result = func(*args, **kwargs)
                        if not self._read_only:
                            self.set(cache_key, result, ttl=ttl)
                        else:
                            raise ReadOnlyError()

                    return result
            return wrapper
        return decorator

    @property
    def max_size(self) -> Optional[int]:
        return self._max_size

    @max_size.setter
    def max_size(self, max_size: Optional[int]) -> None:
        """
        Modify the cache's max-size.

        Args:
            max_size (Optional[int]): The new max_size of the cache.
              If max_size equals None, the cache's max_size will be unlimited.

        Returns:
            None

        Raises:
            CacheConfigurationError: If max_size is not an int or None.
            CacheConfigurationError: If max_size is negative.
            CacheConfigurationError: When cache.max_size gets modified
                        while read-only mode is enabled.
        """
        # Check valid type
        self._check_max_size_valid(max_size)

        with self._lock:
            # Do not allow change while read-only is enabled.
            if not self._read_only:
                self._max_size = max_size

                self.logger.debug(f"CHANGE max-size={max_size}")

                # Evict nodes if max-size gets exceeded.
                while self._exceeds_max_size():
                    evict_node = self._get_evict_node()
                    self._remove_node(evict_node)

                    self.logger.debug(f"EVICT key='{evict_node.key}' due to max-size")
            else:
                raise ReadOnlyError()

    @property
    def max_memory(self) -> Optional[int]:
        return self._max_memory

    @max_memory.setter
    def max_memory(self, max_memory: Optional[int]) -> None:
        """
        Modify the cache's max_memory.

        Args:
            max_memory (Optional[int]): The new max_memory of the cache.
              If max_memory is None, the cache's memory will be unlimited.

        Returns:
            None

        Raises:
            CacheConfigurationError: When max_memory is an invalid type.
            CacheConfigurationError: When max_memory is a negative int.
            CacheConfigurationError: When max_memory is an invalid str format.
            CacheConfigurationError: When an invalid unit was detected in max_memory.
        """
        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                self._max_memory = self._parse_max_memory(max_memory)

                self.logger.debug(f"CHANGE max-memory={max_memory}")

                # Evict nodes if max-memory gets exceeded.
                while self._exceeds_max_size():
                    evict_node = self._get_evict_node()
                    print(evict_node)
                    self._remove_node(evict_node)

                    self.logger.debug(f"EVICT key='{evict_node.key}' due to max-memory")
            else:
                raise ReadOnlyError()

    def _parse_max_memory(self, max_memory: Optional[Union[int, str]]) -> Optional[int]:
        UNITS = {
            "b": 1,
            "kb": 1024,
            "mb": 1024**2,
            "gb": 1024**3,
            "tb": 1024**4
        }

        if max_memory is None:
            return max_memory

        elif isinstance(max_memory, int):
            if max_memory < 0:
                raise CacheConfigurationError(f"max_memory shouldn't be a negative int, got {max_memory}")

            return max_memory

        elif isinstance(max_memory, str):
            match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", max_memory.strip())

            if not match:
                raise CacheConfigurationError(f"max_memory is an invalid format: {max_memory!r}")

            value, unit = match.groups()
            value = float(value)
            unit = unit.lower()

            if unit not in UNITS:
                raise CacheConfigurationError(f"when parsing max_memory, an invalid unit was detected: {unit!r}")

            return int(value * UNITS[unit])
        else:
            raise CacheConfigurationError(f"max_memory should be of type str, int or None, got {max_memory.__class__.__name__}")

    def set_verbose(self, verbose: bool) -> None:
        """
        Enable or disable debug mode manually. While
        debug mode is enabled, debug logs will be shown.
        It is preferred to use cache.verbose() as
        a context manager.

        Args:
            verbose (bool): True to enable debug mode,
                            False to disable debug mode.

        Returns:
            None

        Usage:
            >>> cache.set_verbose(True)  # Enable debug mode
            >>> ... some code with debug logs
            >>> cache.set_verbose(False)  # Disable debug mode

        Raises:
            CacheConfigurationError: When verbose isn't a bool.
        """
        if not isinstance(verbose, bool):
            raise CacheConfigurationError("Verbose should be of type: bool.")

        if verbose:
            self.logger.setLevel(logging.DEBUG)

            # Check if there isn't a handler yet.
            # If not, then add handler, so it doesn't overwrite logging.basicConfig
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger.setLevel(logging.WARNING)

    def set_read_only(self, read_only: bool) -> None:
        """
        Enable or disable read-only mode manually.
        The cache cannot be modified while read-only mode
        is enabled. It is preferred to use cache.read_only() as
        a context manager.

        Args:
            read_only (bool): True to enable read-only mode,
                            False to disable read-only mode.

        Returns:
            None

        Usage:
            >>> cache.set_read_only(True)  # Enable read-only mode
            >>> ... some code while cache is read-only
            >>> cache.set_read_only(False)  # Disable read-only mode

        Raises:
            CacheConfigurationError: When read_only isn't a bool.
        """
        if not isinstance(read_only, bool):
            raise CacheConfigurationError(f"{self.__class__.__name__}.read_only should be of type: bool.")

        self._read_only = read_only

    @property
    def cache(self) -> dict[Hashable, Optional[Any]]:
        """ read-only """
        return self._cache

    @property
    def stats(self) -> Stats:
        """ read-only """
        return self._stats

    def get_stats(self) -> dict[str, Any]:
        """
        Get the cache's stats.

        Returns:
            dict[str, Any]: The cache's stats.
        """
        return self._stats._as_dict()

    def __contains__(self, key: Hashable) -> bool:
        """
        Returns True if the given key exists in the cache; False otherwise.
        """
        return key in self.keys()

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the amount of entries in the cache.
        """
        ...

    def __getitem__(self, key: Hashable) -> Any:
        """
        Call self.get(key)
        """
        return self.get(key)

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        Call self.set(key, value)
        """
        self.set(key, value)

    def __delitem__(self, key: Hashable) -> None:
        """
        Call self.delete(key)
        """
        self.delete(key)

    def __getstate__(self) -> dict[str, object]:
        """
        Copy the cache's state while excluding the lock and the cleanup
        thread, so it will not interfere with saving it as a .pkl file.

        Raises:
            SerializationError: When something went wrong during serialization.
        """
        try:
            state = self.__dict__.copy()
            del state["_lock"]
            del state["cleanup_thread"]

            return state

        except Exception as exc:
            raise SerializationError(exc) from exc

    def __setstate__(self, state) -> None:
        """
        Set the object's state after loading it from a .pkl file
        and add a threading.RLock() and the cleanup thread.

        Raises:
            DeserializationError: When something went wrong during deserialization.
        """
        try:
            self.__dict__.update(state)

            if self._thread_safe:
                self._lock = RLock()
            else:
                self._lock = NullContext()

            # Add the cleanup thread
            if self.__class__.__name__ == "MemoryCache":
                self.cleanup_thread: CleanupThread = MemoryCleanupThread(self, self._cleanup_thread_interval)
            elif self.__class__.__name__ == "DiskCache":
                self.cleanup_thread: CleanupThread = DiskCleanupThread(self, self._cleanup_thread_interval)
            self.cleanup_thread.start()

        except Exception as exc:
            raise DeserializationError(exc) from exc

    def read_only(self):
        """
        A context manager to enable read-only mode.
        The cache cannot be modified while read-only mode is enabled.

        Usage:
            >>> with cache.read_only():
            >>>     some code while cache is read-only

        Raises:
            ReadOnlyError: When there is an attempt to modify the cache
                            while read-only mode is enabled.
        """
        class _ReadOnlyContext:
            __slots__ = ("_cache", "_default_mode")

            def __init__(self, cache: BaseCache):
                self._cache = cache
                self._default_mode = self._cache._read_only

            def __enter__(self):
                self._cache.set_read_only(True)

            def __exit__(self, *exc: Any):
                self._cache.set_read_only(self._default_mode)

        return _ReadOnlyContext(self)

    def verbose(self):
        """
        A context manager to enable debug mode. When debug
        mode is enabled, debug logs will be shown.

        Usage:
            >>> with cache.verbose():
            >>>     some code with debug logs
        """
        class _VerboseContext:
            __slots__ = ("_cache", "_default_mode")

            def __init__(self, cache: BaseCache):
                self._cache = cache
                self._default_mode = self._cache._verbose

            def __enter__(self):
                self._cache.set_verbose(True)

            def __exit__(self):
                self._cache.set_verbose(self._default_mode)

        return _VerboseContext(self)

    def _exceeds_max_size(self) -> bool:
        # In DiskCache, if max_memory = 0, even if there are no items cached,
        # the file size will be exceeded. Therefore check if there are no items cached.
        if self.__len__() == 0:
            return False

        if not isinstance(self._max_size, NullValue):
            if (self._max_size is not None) and (self.__len__() > self._max_size):
                return True

        if not isinstance(self._max_memory, NullValue):
            if (self._max_memory is not None) and (self._get_cache_size() > self._max_memory):
                return True

        return False

    @abstractmethod
    def _create_cache_key(key):
        """Transform the given args and kwargs to a format that
        is suitable to store in the cache. """
        ...

    @staticmethod
    def _check_max_size_valid(max_size) -> True:
        """ Returns True if no exceptions were raised. """
        if isinstance(max_size, (int, type(None))):
            if max_size is not None and max_size < 0:
                raise CacheConfigurationError(f"max_size must be a non-negative int or None, got {max_size!r}")
        else:
            raise CacheConfigurationError(f"max_size should be a non-negative int or None, got {max_size!r}")

        return True

    @staticmethod
    def _check_path_valid(path: Union[str, Path], suffix: str) -> Path:
        # Convert str to pathlib.Path
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise PathError(f"'path' must be a str or Path, got {type(path).__name__}")

        if path.suffix != suffix:
            raise PathError(f"expected a '{suffix}' file, got '{path.suffix}': {path.name}")

        return path

    def __repr__(self) -> str:
        """
        Examples:
            >>> repr(BaseCache)
            <BaseCache.Stats(hits=2, misses=1, evictions=1, size=2, max-size=2, uptime=23s)>
        """
        return f"<{self.__class__.__name__}.{repr(self._stats)[1:-1]}>"


class Stats:
    """
    Stores the statistics of the cache.

    Statistics that get tracked:
        - hits
        - misses
        - evictions
        - size
        - max_size
        - uptime

    Methods:
        _as_dict(): Returns the statistics as a dict.
    """
    __slots__ = (
        "_parent",
        "_hits",
        "_misses",
        "_evictions",
        "_starttime"
    )

    def __init__(self, parent: BaseCache):
        self._parent: BaseCache = parent

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._starttime: float = time.time()

    @property
    def type(self) -> str:
        return self._parent.__class__.__name__

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def size(self) -> int:
        return len(self._parent)

    @property
    def max_size(self) -> Optional[int]:
        return self._parent._max_size

    @property
    def uptime(self) -> float:
        return time.time() - self._starttime

    def _as_dict(self) -> dict[str, Union[str, int]]:
        """
        Get the cache's stats as a dict.

        Returns:
            dict[str, Union[str, int]]
        """
        return {
            "type": self.type,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "capacity": self.max_size,
            "uptime": self.uptime
        }

    def __repr__(self) -> str:
        """
        Examples:
            >>> repr(BaseCache.stats)
            <Stats(hits=2, misses=1, evictions=1, size=2, max-size=2, uptime=23s)>
        """
        return (
            f"<{self.__class__.__name__}(hits={self._hits!r}, misses={self._misses!r}, "
            f"evictions={self._evictions!r}, size={self.size!r}, "
            f"max-size={self.max_size!r}, uptime={self.uptime!r}s)>"
        )
