# -*- coding: utf-8 -*-

"""
cleanup_thread.py - MemoryCleanupThread and DiskCleanupThread implementation.

The implementation for the in-background cleanup threads that evict
expired items from the cache.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from abc import ABC, abstractmethod
import sqlite3 as sqlite
from threading import Event, Thread
from typing import NoReturn, Union

from .errors import CleanupThreadConfigurationError
from .node import Node


__all__ = ["CleanupThread", "MemoryCleanupThread", "DiskCleanupThread"]


class CleanupThread(ABC, Thread):
    """
    An in-background thread that ensures that expired keys get evicted.
    """
    __slots__ = ("_cache", "_interval", "_stop_event")

    def __init__(self, cache, interval: Union[int, float]):
        super().__init__(daemon=True)

        self._cache = cache
        self.set_interval(interval)  # Initialize self._interval
        self._stop_event = Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            if not self._cache._read_only:
                self.cleanup()
                self._stop_event.wait(self.interval)

    def stop(self) -> None:
        """
        Stops the thread and waits untill it's done.
        """
        self._stop_event.set()
        self.join()

    @abstractmethod
    def cleanup(self) -> None:
        """
        Check for expired nodes and remove them from the cache.

        Returns:
            None
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__()}(parent={self._cache._name} interval={self._interval})>"

    @property
    def interval(self) -> Union[int, float]:
        return self._interval

    @interval.setter
    def interval(self, _) -> NoReturn:
        raise CleanupThreadConfigurationError("To change a cleanup thread's interval, use set_interval() instead")

    def set_interval(self, interval: Union[int, float]) -> None:
        """
        Change the cleanup thread's interval.

        Args:
            interval (Union[int, float]): The new interval (in seconds)

        Returns:
            None
        """
        # Check for a valid interval value
        if not isinstance(interval, (int, float)):
            self.stop()
            raise CleanupThreadConfigurationError(f"Expected interval to be of type int or float, got {type(interval).__name__}")

        if interval < 0:
            self.stop()
            raise CleanupThreadConfigurationError(f"Interval must be a non-negative number, got {interval}")

        self._interval = interval


class MemoryCleanupThread(CleanupThread):

    __slots__ = ("_cache", "_interval", "_stop_event")

    def cleanup(self) -> None:
        with self._cache._lock:
            expired_nodes = []

            # Check for expired nodes
            for key in self._cache.keys():
                node = self._cache._get_node(key)
                if node.is_expired():
                    expired_nodes.append(node)

            # Evict the found expired nodes
            for node in expired_nodes:
                self._cache._remove_node(node)

                self._cache._stats._evictions += 1
                self._cache.logger.debug(f"EVICT key='{node.key}' due to ttl")


class DiskCleanupThread(CleanupThread):

    __slots__ = ("_cache", "_interval", "_stop_event")

    def cleanup(self) -> None:
        with self._cache._lock:
            if self._cache._path.exists():
                conn = sqlite.connect(self._cache._path)
                try:
                    cursor = conn.cursor()

                    expired_nodes = []  # List with the expired nodes that will get evicted

                    # Check for expired nodes
                    cached_keys = cursor.execute(self._cache.KEYS_QUERY).fetchall()
                    for key in cached_keys:
                        key = key[0]
                        value, ttl, expires_at = cursor.execute(self._cache.GET_QUERY, (key,)).fetchone()

                        node = Node(key, value, ttl)
                        node._expires_at = expires_at

                        if node.is_expired():
                            expired_nodes.append(node)

                    # Evict the found expired nodes
                    for node in expired_nodes:
                        cursor.execute(self._cache.DELETE_QUERY, (node.key,))

                        self._cache._stats._evictions += 1
                        self._cache.logger.debug(f"EVICT key='{node.key}' due to ttl")

                    conn.commit()
                finally:
                    cursor.close()
                    conn.close()
                    conn = None
            else:
                raise CleanupThreadConfigurationError(
                    "When running an in-background cleanup thread,"
                    f"the cache's path was not found: {self._cache._path}"
                )
