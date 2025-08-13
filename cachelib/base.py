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
from threading import RLock
import time
from typing import Any, Callable, Hashable, Optional, Union

from .cleanup_thread import CleanupThread, DiskCleanupThread, MemoryCleanupThread
from .errors import (
    CacheConfigurationError,
    CacheOverflowError,
    CachePathError,
    DeserializationError,
    KeyExpiredError,
    KeyNotFoundError,
    ReadOnlyError,
    SerializationError
)
from .eviction import EvictionPolicy, _LFUEviction, _LRUEviction
from .utils import NullContext
from .node import Node

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
        - _update_cache_state
    """

    @abstractmethod
    def __init__(self,
                 name: str = "",
                 max_size: Optional[int] = None,
                 eviction_policy: Optional[EvictionPolicy] = None,
                 ttl: Optional[Union[int, float]] = None,
                 verbose: bool = False,
                 thread_safe: bool = True):

        self.name: str = name

        self._check_max_size_valid(max_size)
        self._max_size: Optional[int] = max_size
        self._ttl: Optional[Union[int, float]] = ttl
        self._thread_safe: bool = thread_safe
        self._lock: Union[NullContext, RLock] = \
            RLock() if self._thread_safe else NullContext()  # thread safety
        self.logger: logging.Logger = logging.getLogger(self.name)
        self._stats: Stats = Stats(self)  # statistics
        self._read_only: bool = False  # read-only mode

        # Check for a valid eviction policy
        if isinstance(eviction_policy, (EvictionPolicy, type(None))):
            if max_size is None and isinstance(eviction_policy, EvictionPolicy):
                raise CacheConfigurationError("can only set eviction_policy when max_size is not infinite")
            else:
                self._eviction_policy: EvictionPolicy = eviction_policy
        else:
            raise CacheConfigurationError(
                f"Eviction policy must be of type "
                f"{EvictionPolicy.__module__}.{EvictionPolicy.__qualname__}"
                f", got {type(eviction_policy).__name__}"
            )

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
            if isinstance(self._eviction_policy, _LFUEviction):
                return self._lfu_eviction()
            elif isinstance(self._eviction_policy, _LRUEviction):
                return self._lru_eviction()
            else:
                raise CacheOverflowError(max_size=self._max_size)

    @abstractmethod
    def _update_cache_state(self, node: Node) -> None:
        """
        This function gets called whenever a node
        is accessed in the cache. Eg. cache.get(node).

        Args:
            node (Node): The node that got accessed.
        """
        ...

    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve the value for the given key (if the key exists).

        Args:
            Key (Hashable): The requested key.

        Returns:
            Optional[Any]: If the key is found in the cache,
                        return its value; None otherwise.
        """
        try:
            with self._lock:
                node: Node = self._get_node(key)

                # Check if the node is expired
                if not node.is_expired():
                    # Get the key's value

                    self._update_cache_state(node)

                    self._stats._hits += 1
                    self.logger.debug(f"GET key='{key}' (hit)")

                    return node.value

                else:
                    # Evict the key from the cache

                    if not self._read_only:
                        # Remove node due to ttl
                        self._remove_node(node)

                        self._stats._evictions += 1
                        self.logger.debug(f"EVICT key='{node.key}' due to ttl")
        # The key isn't found
        except KeyNotFoundError:
            self._stats._misses += 1
            self.logger.debug(f"GET key='{key}' (miss)")

            return None

    def get_many(self,
                 keys: tuple[Hashable]
                 ) -> dict[Hashable, Optional[Any]]:
        """
        Retrieve the (key, value) pairs for the given keys, if they exist.

        Args:
            keys (tuple[Hashable]): The requested keys.

        Returns:
            dict[Hashable, Optional[Any]]: The (key, value) pairs for the
                                        requested keys.
        """
        pairs: dict[Hashable, Any] = {}
        with self._lock:
            for key in keys:
                node = self._get_node(key)
                if node.is_expired():
                    self._remove_node(node)
                else:
                    pairs[key] = node.value
        return pairs

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
        # Check for 'value' and/or 'ttl' in args
        if len(args) + len(kwargs) > 2:
            raise CacheConfigurationError(f"expected at most 3 arguments, got {len(args) + len(kwargs) + 1}")

        def extract_items_from_args(*args, **kwargs) -> dict[str, Optional[Any]]:
            params = {}  # The params that have been found

            # Look for 'value' in args
            if args:
                params["value"] = args[0]
            # If not found in args, look for 'value' in kwargs
            elif "value" in kwargs.keys():
                params["value"] = kwargs["value"]

            # Look for 'ttl' in args
            if len(args) == 2:
                params["ttl"] = args[1]
            # If not found in args, look for 'ttl' in kwargs
            elif "ttl" in kwargs.keys():
                params["ttl"] = kwargs["ttl"]

            return params

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

                else:
                    # Update an existing entry

                    node = self._get_node(key)
                    params = extract_items_from_args(*args, **kwargs)

                    if "value" not in params.keys() and "ttl" not in params.keys():
                        raise CacheConfigurationError("expected either 'value' or 'ttl' as arguments")

                    self._update_node(node, params)
                    self._update_cache_state(node)

                    self.logger.debug("SET key='%s' %s (updating value)"
                                      % (key, f"with ttl={node.ttl}" if node.ttl else ""))

                # Evict a node if the capacity gets exceeded.
                # Don't evict nodes if capacity is None (infinite capacity).
                if (self._max_size is not None) and (self.__len__() > self._max_size):
                    node = self._get_evict_node()

                    self._remove_node(node)

                    self._stats._evictions += 1
                    self.logger.debug(f"EVICT key='{node.key}' due to capacity")
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
        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                try:
                    self._remove_node(self._get_node(key))

                    self.logger.debug(f"REMOVE key='{key}'")
                except KeyError as exc:
                    # Invalid key
                    raise KeyNotFoundError(key) from exc
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
        with self._lock:
            try:
                node = self._get_node(key)

                if node.is_expired():
                    self._remove_node(node)
                    raise KeyExpiredError(key)

                return node.ttl
            except KeyError as exc:
                raise KeyNotFoundError(key) from exc

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
        try:
            node: Node = self._get_node(key)

            if node.is_expired():
                self._remove_node(node)
                raise KeyExpiredError(key)

            return {
                "key": key,
                "value": node.value,
                "ttl": node._expires_at
            }

        except KeyError as exc:
            raise KeyNotFoundError(key) from exc

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
        the cached result will getreturned.

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

                cache_key = self._make_cache_key(*args, **kwargs)

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
                                    If max_size equals None, the cache's
                                    max_size will be infinite.

        Raises:
            TypeError: If max_size is not an int or None.
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

                # Evict nodes if max-size gets exceeded
                while self._max_size is not None and self.__len__() > self._max_size:
                    evict_node: Node = self._get_evict_node()
                    self._remove_node(evict_node)

                    self.logger.debug(f"EVICT key='{evict_node.key}' \
                                    due to max-size")
            else:
                raise ReadOnlyError()

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
            __slots__ = ["_cache", "_default_mode"]

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
            __slots__ = ["_cache", "_default_mode"]

            def __init__(self, cache: BaseCache):
                self._cache = cache
                self._default_mode = self._cache._verbose

            def __enter__(self):
                self._cache.set_verbose(True)

            def __exit__(self):
                self._cache.set_verbose(self._default_mode)

        return _VerboseContext(self)

    @staticmethod
    def _check_max_size_valid(max_size):
        if isinstance(max_size, (int, type(None))):
            if max_size is not None and max_size < 0:
                raise CacheConfigurationError(f"max_size must be a non-negative int or None, got {max_size!r}")
        else:
            raise CacheConfigurationError(f"max_size should be a non-negative int or None, got {max_size!r}")

    @staticmethod
    def _check_path_valid(path: Union[str, Path], suffix: str) -> Path:
        # Convert str to pathlib.Path
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise CachePathError(f"'path' must be a str or Path, got {type(path).__name__}")

        if path.suffix != suffix:
            raise CachePathError(f"expected a '{suffix}' file, got '{path.suffix}': {path.name}")

        return path

    def __repr__(self) -> str:
        """
        Examples:
            >>> repr(BaseCache)
            BaseCache.Stats(hits=2, misses=1, evictions=1, size=2, max-size=2, uptime=23s)
        """
        return f"{self.__class__.__name__}.{self._stats!r}"


class Stats:
    """
    Stores the statistics of the cache.

    Instances:
        Stats.hits
        Stats.misses
        Stats.evictions
        Stats.size
        Stats.max_size
        Stats.uptime

    Methods:
        _as_dict(): Returns the statistics as a dict.
    """
    __slots__ = [
        "_parent",
        "_hits",
        "_misses",
        "_evictions",
        "_starttime"
    ]

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
            Stats(hits=2, misses=1, evictions=1, size=2, max-size=2, uptime=23s)
        """
        return (
            f"{self.__class__.__name__}(hits={self._hits!r}, misses={self._misses!r}, "
            f"evictions={self._evictions!r}, size={self.size!r}, "
            f"max-size={self.max_size!r}, uptime={self.uptime!r}s)"
        )
