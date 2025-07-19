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
import pickle
from threading import RLock
import time
from typing import Any, Callable,  Hashable, Iterator, Optional, Self, Union

from .utils import NullContext
from .node import Node

__all__ = ["BaseCache", "Stats"]


class BaseCache(ABC):
    """
    An abstract base cache that provides the foundational methods
    and attributes for every cache in cachelib.

    Methods that should be implemented in subclasses of this class:
        - _add_node
        - _remove_node
        - _get_evict_node
        - _update_cache_state
        - __iter__
        - __reversed__

    Base methods:
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
        get_stats(): Get the cache's stats.
    """

    read_only_error = RuntimeError("Cannot modify cache: read-only mode is enabled.")

    @abstractmethod
    def __init__(self,
                 name: str = "",
                 max_size: Optional[int] = None,
                 verbose: bool = False,
                 thread_safe: bool = True):

        self.name: str = name
        self._cache: dict[Hashable, Node] = {}
        self._max_size: Optional[int] = max_size
        self._thread_safe: bool = thread_safe
        self._lock: Union[NullContext, RLock] = \
            RLock() if self._thread_safe else NullContext()  # thread safety
        self._logger: logging.Logger = logging.getLogger(self.name)
        self._stats: Stats = Stats(self)  # statistics
        self._read_only: bool = False  # read-only mode

        self.set_verbose(verbose)

        self._stats._max_size = self._max_size

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
            ttl (Optional[Union[int, float]]): The node's ttl.

        Returns:
            Node: The new node.
        """
        pass

    @abstractmethod
    def _remove_node(self, node: Node) -> None:
        """
        Remove a node.

        Args:
            node (Node): The node that has to be removed.

        Returns:
            None
        """
        pass

    @abstractmethod
    def _get_evict_node(self) -> Node:
        """
        Get the next node to be evicted.

        Returns:
            Node: The node to evict.
        """
        pass

    @abstractmethod
    def _update_cache_state(self, node: Node) -> None:
        """
        This is the function that gets called whenever a node
        is accessed in the cache. Eg. cache.get(node).

        Args:
            node (Node): The node that got accessed.
        """
        pass

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
                node: Node = self._cache[key]

                # Check if the node is expired
                if not node.is_expired():
                    # Get the key's value

                    self._update_cache_state(node)

                    self._stats._hits += 1
                    self._logger.debug(f"GET key='{key}' (hit)")

                    return node.value

                else:
                    # Evict the key from the cache

                    if not self._read_only:
                        # Remove node due to ttl
                        self._remove_node(node)

                        self._stats._evictions += 1
                        self._logger.debug(f"EVICT key='{node.key}' due to ttl")
        # The key isn't found
        except KeyError:
            self._stats._misses += 1
            self._logger.debug(f"GET key='{key}' (miss)")

            return None

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
            TypeError: When the given arguments aren't valid.
            RuntimeError: When cache.set() is called while
                        read-only mode is enabled.
        """
        # Check for 'value' and/or 'ttl' in args
        if len(args) + len(kwargs) > 2:
            raise TypeError(f"expected at most 3 arguments, got {len(args) + len(kwargs) + 1}")

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
                if key not in self._cache.keys():
                    # Add a new key

                    params = extract_items_from_args(*args, **kwargs)

                    # Can't create a node without a value
                    if "value" not in params.keys():
                        raise TypeError("expected 'value' argument")

                    # When 'ttl' is not found, default to None
                    if "ttl" not in params.keys():
                        params["ttl"] = None

                    # Create a new entry in the cache
                    node = self._add_node(key, params["value"], params["ttl"])

                    self._logger.debug("SET key='%s' %s (adding new key)"
                                       % (key, f"with ttl={node.ttl}" if node.ttl else ""))

                else:
                    # Update an existing entry

                    node = self._cache[key]
                    params = extract_items_from_args(*args, **kwargs)

                    if "value" not in params.keys() and "ttl" not in params.keys():
                        raise TypeError("expected either 'value' or 'ttl' as arguments")

                    # Update the node's value, if provided
                    if "value" in params.keys():
                        node.value = params["value"]
                    # Update the node's ttl, if provided
                    if "ttl" in params.keys():
                        node.ttl = params["ttl"]

                    self._update_cache_state(node)

                    self._logger.debug("SET key='%s' %s (updating value)"
                                       % (key, f"with ttl={node.ttl}" if node.ttl else ""))

                # Evict a node if the capacity gets exceeded.
                # Don't evict nodes if capacity is None (infinite capacity).
                if (self._max_size is not None) and (self.__len__() > self._max_size):
                    node = self._get_evict_node()

                    self._remove_node(node)

                    self._stats._evictions += 1
                    self._logger.debug(f"EVICT key='{node.key}' due to capacity")
            else:
                raise self.read_only_error

    def delete(self, key: Hashable) -> None:
        """
        Delete a key from the cache.

        Args:
            key (Hashable): The key that has to be deleted.

        Returns:
            None

        Raises:
            KeyError: When the given key is not found in the cache.
            RuntimeError: When cache.delete() is called while
                        read-only mode is enabled.
        """
        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                try:
                    self._remove_node(self._cache[key])

                    self._logger.debug(f"REMOVE key='{key}'")
                except KeyError:
                    # Invalid key
                    raise KeyError(f"key '{key}' not found in cache")
            else:
                raise self.read_only_error

    def clear(self) -> None:
        """
        Clear all entries from the cache.

        Returns:
            None

        Raises:
            RuntimeError: When cache.clear() is called while
                        read-only mode is enabled.
        """
        with self._lock:
            # Do not allow changes while read-only is enabled.
            if not self._read_only:
                self._cache = {}

                # Only LFUCache has a _lookup_freq table
                if self.__class__.__name__ == "LFUCache":
                    self._lookup_freq = {}

                self._head = self._tail = Node(None, None)
                self._head.next = self._tail
                self._tail.prev = self._head

                self._stats._size = 0
                self._logger.debug("CLEAR CACHE")
            else:
                raise self.read_only_error

    def ttl(self, key: Hashable) -> Optional[Union[float, int]]:
        """
        Get the ttl (Time To Live) of the given key.

        Args:
            key (Hashable): The key.

        Returns:
            Union[float, int] or None: The key's ttl.

        Raises:
            KeyError: When the given key is nonexistent or has expired.
        """
        with self._lock:
            try:
                node = self._cache[key]

                if node.is_expired():
                    self._remove_node(node)
                    raise KeyError(f"key '{key}' not found in cache")

                return self._cache[key].ttl
            except KeyError:
                raise KeyError(f"key '{key}' not found in cache")

    def inspect(self, key: Hashable) -> Optional[dict[str, Any]]:
        """
        Inspect the given key in the cache. Returns a dict with
        the key's information.

        Args:
            key (Hashable): The requested key.

        Returns:
            dict[str, Any]: Return a dict with the key's information.

        Raises:
            KeyError: When the given key is nonexistent or has expired.
        """
        try:
            node: Node = self._cache[key]
            self._update_cache_state(node)

            return {
                "key": key,
                "value": node.value,
                "expired": node.is_expired(),
                "ttl": node._expires_at
            }

        except KeyError:
            raise KeyError(f"key '{key}' not found in cache")

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
            RuntimeError: When a function call gets stored While
                        read-only mode is enabled
        """
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
                    if cache_key in self._cache.keys() and not self._cache[cache_key].is_expired():
                        return self.get(cache_key)
                    else:
                        result = func(*args, **kwargs)
                        if not self._read_only:
                            self.set(cache_key, result, ttl=ttl)
                        else:
                            raise self.read_only_error

                        return result
            return wrapper
        return decorator

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
        path = self._check_path_valid(path)

        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(self, f)

            self._logger.debug(f"SAVE path='{path}'")

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """
        Load a cache from a .pkl file.

        Args:
            path (Union[str, Path]): The path to the saved cache.

        Returns:
            Self: The cache object (a subclass of BaseCache).

        Raises:
            ValueError: When the given path isn't a .pkl file.
            TypeError: When path isn't a str or a pathlib.Path().
        """
        path = cls._check_path_valid(path)

        with open(path, "rb") as f:
            obj = pickle.load(f)

        # Check if the imported cache is of the same type
        # as the class load() got called from.
        # Eg. LRUCache.load() -> LRUCache
        if not isinstance(obj, cls):
            raise TypeError(
                f"expected instance of {cls.__name__}, "
                f"got {obj.__class__.__name__}"
            )
        return obj

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
            ValueError: If max_size is negative.
            RuntimeError: When cache.max_size gets modified
                        while read-only mode is enabled.
        """
        # Check valid type
        self._check_max_size_valid(max_size)

        with self._lock:
            # Do not allow change while read-only is enabled.
            if not self._read_only:
                self._max_size = max_size

                self._logger.debug(f"CHANGE max-size={max_size}")

                # Evict nodes if max-size gets exceeded
                while self._max_size is not None and self.__len__() > self._max_size:
                    least_freq_node: Node = self._get_evict_node()
                    self._remove_node(least_freq_node)

                    self._logger.debug(f"EVICT key='{least_freq_node.key}' \
                                    due to max-size")

                self._stats._max_size = self._max_size  # update cache.stats.max_size
            else:
                raise self.read_only_error

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
            TypeError: When verbose isn't a bool.
        """
        if not isinstance(verbose, bool):
            raise TypeError("Verbose should be of type: bool.")

        if verbose:
            self._logger.setLevel(logging.DEBUG)

            # Check if there isn't a handler yet.
            # If not, then add handler, so it doesn't overwrite logging.basicConfig
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
        else:
            self._logger.setLevel(logging.WARNING)

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
            TypeError: When read_only isn't a bool.
            RuntimeError: When read-only mode is enabled and there
                            was an attempt to modify the cache.
        """
        if not isinstance(read_only, bool):
            raise TypeError(f"{self.__class__.__name__}.read_only should be of type: bool.")

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

    def __contains__(self, key: Hashable) -> bool:
        """
        Returns True if the given key exists in the cache; False otherwise.
        """
        return key in self._cache.keys()

    def __len__(self) -> int:
        """
        Returns the amount of entries in the cache.
        """
        return len(self._cache)

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

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[Hashable, Any]]:
        """
        Yield the (key, value) pairs in order.

        Returns:
            Iterator (Iterator[tuple[Hashable, Any]]): Iterates over the (key, value) pairs.
        """
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[tuple[Hashable, Any]]:
        """
        Yield the (key, value) pairs in reversed order.

        Returns:
            Iterator (Iterator[tuple[Hashable, Any]]): Iterates over the (key, value) pairs.
        """
        pass

    def __getstate__(self) -> dict[str, object]:
        """
        Copy the cache's state while excluding the lock,
        so it will not interfere with saving it as a .pkl file.
        """
        state = self.__dict__.copy()
        del state["_lock"]

        return state

    def __setstate__(self, state) -> None:
        """
        Set the object's state after loading it from a .pkl file
        and add a threading.RLock().
        """
        self.__dict__.update(state)

        if self._thread_safe:
            self._lock = RLock()
        else:
            self._lock = NullContext()

    def read_only(self):
        """
        A context manager to enable read-only mode.
        The cache cannot be modified while read-only mode is enabled.

        Usage:
            >>> with cache.read_only():
            >>>     some code while cache is read-only

        Raises:
            RuntimeError: When there is an attempt to modify the cache
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
            if max_size < 0:
                raise ValueError("LRUCache.max_size should be non-negative or None")
        else:
            raise TypeError("LRUCache.max_size should be of type: int or NoneType")

    @staticmethod
    def _check_path_valid(path: Union[str, Path]) -> bool:
        # Convert str to pathlib.Path
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError(f"'path' must be a str or Path, got {type(path).__name__}")

        if path.suffix != ".pkl":
            raise ValueError(f"expected a '.pkl' file, got '{path.suffix}': {path.name}")

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
        "_size",
        "_max_size",
        "_starttime"
    ]

    def __init__(self, parent):
        self._parent = parent

        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._size = 0
        self._max_size = 0
        self._starttime = time.time()

    @property
    def type(self):
        return self._parent.__class__.__name__

    @property
    def hits(self):
        return self._hits

    @property
    def misses(self):
        return self._misses

    @property
    def evictions(self):
        return self._evictions

    @property
    def size(self):
        return self._size

    @property
    def max_size(self):
        return self._max_size

    @property
    def uptime(self):
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
            f"evictions={self._evictions!r}, size={self._size!r}, "
            f"max-size={self._max_size!r}, uptime={self.uptime!r}s)"
        )
