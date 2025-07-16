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
import logging
from pathlib import Path
import pickle
from threading import RLock
import time
from typing import Any, Hashable, Iterator, Optional, Self, Union

from .utils import NullContext
from .node import Node

__all__ = ["BaseCache", "Stats"]


class BaseCache(ABC):
    """
    An abstract base cache that provides the foundational methods
    and attributes for every cache in cachelib.
    """

    @abstractmethod
    def __init__(self,
                 name: str = "",
                 max_size: Optional[int] = None,
                 verbose: bool = False,
                 thread_safe: bool = True):

        self.name: str = name  # used in debug messages
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
    def get(self, key: Hashable) -> Optional[Any]:
        """
        Retrieve the value for the given key.
        Returns None if key is not found or expired.
        """
        pass

    @abstractmethod
    def set(self,
            key: Hashable,
            value: Any,
            ttl: Optional[Union[int, float]] = None
            ) -> None:
        """
        Set the value for the given key.
        """
        pass

    @abstractmethod
    def delete(self, key: Hashable) -> None:
        """
        Remove the given key with its value from the cache.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all entries from the cache.
        """
        pass

    @abstractmethod
    def inspect(self, key: Hashable) -> Optional[dict[str, Any]]:
        """
        Inspect the given key, returns a dict with the key's information.
        Returns None if the key doesn't exist in the cache.
        """
        pass

    @abstractmethod
    def memoize(self, ttl: Optional[Union[int, float]] = None) -> Any:
        """
        Cache the return value of a function with its parameters.
        It serves as a decorator function.
        """
        pass

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the cache as a .pkl file.

        Args:
            path (Path): The path to the .pkl file
                        in which the cache will be stored.

        Returns:
            None

        Raises:
            ValueError: When the given path isn't a .pkl file.
            TypeError: When path isn't a str or ap athlib.Path().
        """
        path = self._check_path_valid(path)

        with self._lock:
            with open(path, "wb") as f:
                pickle.dump(self, f)

            self._logger.debug(f"SAVE path='{path}'")

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> Self:
        """
        Load a cache from a .pkl file.

        Args:
            path (Union[str, Path]): The path where the cache object is saved.

        Returns:
            Self: The cache object (a subclass of BaseCache)

        Raises:
            ValueError: When the given path isn't a .pkl file.
            TypeError: When path isn't a str or ap athlib.Path().
        """
        path = cls._check_path_valid(path)

        with open(path, "rb") as f:
            return pickle.load(f)

    @property
    def max_size(self):
        return self._max_size

    @max_size.setter
    @abstractmethod
    def max_size(self, max_size: Optional[int]) -> None:
        """
        Modify the max-size of the cache.
        """
        pass

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
            raise TypeError(f"{self.__class__.__name__}.verbose should be of type: bool.")

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
    def cache(self):
        """ read-only """
        return self._cache

    @property
    def stats(self):
        """ read-only """
        return self._stats

    @abstractmethod
    def __contains__(self, key: Hashable) -> bool:
        """
        Returns True if the given key exists in the cache, False otherwise.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the amount of entries in the cache.
        """
        pass

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
    def __iter__(self) -> Iterator[Any]:
        """
        Yield the (key, value) pairs in order.
        """
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[Any]:
        """
        Yield the (key, value) pairs in reversed order.
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

    def __repr__(self) -> str:
        """
        Display the stats (BaseCache.stats).

        Examples:
            >>> repr(BaseCache)
            BaseCache.Stats(hits=2, misses=1, evictions=1, size=2, max-size=2, uptime=23s)
        """
        return f"{self.__class__.__name__}.{self._stats!r}"

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
        return self._ReadOnlyContext(self)

    class _ReadOnlyContext:
        __slots__ = ["cache", "default_mode"]

        error = RuntimeError("Cannot modify cache: read-only mode is enabled.")

        def __init__(self, cache: BaseCache):
            self.cache = cache
            self.default_mode = self.cache._read_only

        def __enter__(self):
            self.cache.set_read_only(True)

        def __exit__(self, *args):
            self.cache.set_read_only(self.default_mode)

    def verbose(self):
        """
        A context manager to enable debug mode. When debug
        mode is enabled, debug logs will be shown.

        Usage:
            >>> with cache.verbose():
            >>>     some code with debug logs
        """
        return self._VerboseContext(self)

    class _VerboseContext:
        __slots__ = ["cache", "default_mode"]

        def __init__(self, cache: BaseCache):
            self.cache = cache
            self.default_mode = self.cache._verbose

        def __enter__(self):
            self.cache.set_verbose(True)

        def __exit__(self):
            self.cache.set_verbose(self.default_mode)

    @staticmethod
    def _check_max_size_valid(max_size):
        if isinstance(max_size, (int, type(None))):
            if max_size < 0:
                raise ValueError("LRUCache.max_size should be non-negative or None.")
        else:
            raise TypeError("LRUCache.max_size should be of type: int or NoneType.")

    @staticmethod
    def _check_path_valid(path: Union[str, Path]) -> bool:
        # Convert str to pathlib.Path
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise TypeError(f"'path' must be a str or Path, got {type(path).__name__}")

        if path.suffix != ".pkl":
            raise ValueError(f"Expected a '.pkl' file, got '{path.suffix}' instead: {path.name}")

        return path


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
        as_dict(): Returns the statistics as a dict.
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

    def as_dict(self) -> dict[str, Union[str, int]]:
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
