# -*- coding: utf-8 -*-

"""
disk.py - On-disk cache implementation.

This module provides an on-disk cache with lru/lfu eviction
and optional ttl-support.

Classes:
    DiskCache: The on-disk cache class.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3 as sqlite
import time
from typing import Any, Final, Hashable, Optional, Union

from .base import BaseCache
from .errors import (
    KeyNotFoundError,
    ReadOnlyError
)
from .eviction import EvictionPolicy
from .node import Node


__all__ = ["DiskCache"]


class DiskCache(BaseCache):
    """
    An on-disk cache with lru/lfu eviction and optional ttl-support.
    """

    # SQL queries
    CACHE_CREATION_QUERY: Final = """
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value BLOB,
            ttl DECIMAL,
            expires_at DECIMAL,
            last_accessed DECIMAL NOT NULL,
            access_frequency INTEGER DEFAULT 0
        );
    """
    SET_QUERY: Final = """
        INSERT INTO cache
        (key, value, ttl, expires_at, last_accessed)
        VALUES (?, ?, ?, ?, ?);
    """
    UPDATE_QUERY: Final = """
        UPDATE cache
        SET value=?, ttl=?, expires_at=?
        WHERE key=?;
    """
    GET_QUERY: Final = """
        SELECT value, ttl, expires_at
        FROM cache
        WHERE key=?;
    """
    DELETE_QUERY: Final = """
        DELETE FROM cache
        WHERE key=?;
    """
    CLEAR_QUERY: Final = """
        DELETE FROM cache;
    """
    UPDATE_STATE_QUERY: Final = """
        UPDATE cache
        SET expires_at=?, last_accessed=?, access_frequency=access_frequency + 1
        WHERE key=?;
    """
    KEYS_QUERY: Final = """
        SELECT key
        FROM cache;
    """
    VALUES_QUERY: Final = """
        SELECT value
        FROM cache;
    """
    LEN_QUERY: Final = """
        SELECT COUNT(key)
        FROM cache;"""

    def __init__(self,
                 path: Union[str, Path],
                 name: str = "",
                 max_size: Optional[int] = None,
                 eviction_policy: Optional[EvictionPolicy] = None,
                 ttl: Optional[Union[int, float]] = None,
                 verbose: bool = False,
                 thread_safe: bool = True
                 ):
        """
        Initialize a new DiskCache.

        Args:
            path (Union[str, Path]): The path where the cache will be stored.
            name (str): The cache's name.
            max_size (Optional[int]): The maximum amount of entries that
                fit in the cache.
            eviction_policy (Optional[cachelib.EvictionPolicy]): The eviction policy to use:
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

        self._path = self._check_path_valid(path, suffix=".db")
        self._conn: sqlite.Connection = sqlite.connect(self._path)

        # Initialize the db tables
        self._create_sql_backend()

        # Initialize self.cleanup_thread
        self._create_cleanup_thread()

        self.logger.info(f"Initialized {self.__class__.__name__} with eviction-policy={self._eviction_policy}")

    def _create_sql_backend(self) -> None:
        with self._get_cursor() as cursor:
            cursor.execute(self.CACHE_CREATION_QUERY)

    @staticmethod
    def _create_node(key: Hashable,
                     value: Any,
                     ttl: Optional[Union[int, float]],
                     expires_at: Optional[float]) -> Node:
        """ Function to create a node. """
        node = Node(key, value, ttl)
        node._expires_at = expires_at

        return node

    def _add_node(self,
                  key: Hashable,
                  value: Any,
                  ttl: Optional[Union[int, float]]) -> Node:

        with self._get_cursor() as cursor:
            expires_at = None if ttl is None else time.time() + ttl
            cursor.execute(self.SET_QUERY, (key, value, ttl, expires_at, time.time()))

        return self._create_node(key, value, ttl, expires_at)

    def _update_node(self, node: Node, params: dict) -> None:

        value = node.value  # Default value
        ttl = node.ttl  # Default ttl

        if "value" in params.keys():
            value = params["value"]
        if "ttl" in params.keys():
            ttl = params["ttl"]

        expires_at = None if ttl is None else time.time() + ttl

        with self._get_cursor() as cursor:
            cursor.execute(self.UPDATE_QUERY, (value, ttl, expires_at, node.key))

        return self._create_node(node.key, value, ttl, expires_at)

    def _get_node(self, key: Hashable) -> Node:
        with self._get_cursor() as cursor:
            cursor.execute(self.GET_QUERY, (key,))

            try:
                value, ttl, expires_at = cursor.fetchone()

                return self._create_node(key, value, ttl, expires_at)
            except TypeError as exc:
                raise KeyNotFoundError(key) from exc

    def _remove_node(self, node: Node) -> None:
        with self._get_cursor() as cursor:
            cursor.execute(self.DELETE_QUERY, (node.key,))

    def _update_cache_state(self, node: Node) -> None:
        with self._get_cursor() as cursor:
            expires_at = None if node.ttl is None else time.time() + node.ttl
            cursor.execute(self.UPDATE_STATE_QUERY, (expires_at, time.time(), node.key))

    def clear(self) -> None:
        with self._lock:
            # Do not allow changes while read-only is enabled
            if not self._read_only:
                with self._get_cursor() as cursor:
                    cursor.execute(self.CLEAR_QUERY)

                self.logger.debug("CLEAR CACHE")
            else:
                raise ReadOnlyError()

    def keys(self) -> tuple[Hashable]:
        with self._get_cursor() as cursor:
            cursor.execute(self.KEYS_QUERY)
            keys = cursor.fetchall()
            return tuple([key[0] for key in keys])

    def values(self) -> tuple[Any]:
        with self._get_cursor() as cursor:
            cursor.execute(self.VALUES_QUERY)
            values = cursor.fetchall()
            return tuple([value[0] for value in values])

    def close(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.commit()
                self._conn.close()
                self._conn = None

            self.cleanup_thread.stop()

    def __len__(self) -> int:
        with self._get_cursor() as cursor:
            cursor.execute(self.LEN_QUERY)
            return cursor.fetchone()[0]

    def _lru_eviction(self) -> Node:

        def get_least_recently_used_node() -> Node:
            with self._lock:
                query = """
                    SELECT key, value, ttl, expires_at
                    FROM cache
                    WHERE last_accessed = (
                        SELECT MIN(last_accessed)
                        FROM cache
                        WHERE expires_at > ? OR expires_at IS NULL
                    )
                    LIMIT 1;
                """

                with self._get_cursor() as cursor:
                    cursor.execute(query, (time.time(),))
                    key, value, ttl, expires_at = cursor.fetchone()

                    return self._create_node(key, value, ttl, expires_at)
        return get_least_recently_used_node()

    def _lfu_eviction(self) -> Node:

        def get_least_frequently_used_node() -> Node:
            with self._lock:
                query = """
                    SELECT key, value, ttl, expires_at
                    FROM cache
                    WHERE access_frequency = (
                        SELECT MIN(access_frequency)
                        FROM cache
                        WHERE expires_at > ? OR expires_at IS NULL
                    )
                    LIMIT 1;
                """

                with self._get_cursor() as cursor:
                    cursor.execute(query, (time.time(),))
                    key, value, ttl, expires_at = cursor.fetchone()

                    return self._create_node(key, value, ttl, expires_at)
        return get_least_frequently_used_node()

    @contextmanager
    def _get_cursor(self):
        """
        A context manager used to interact with a SQLITE-db.

        Usage:
            >>> with self._get_cursor() as cursor:
            >>>     ...
        """
        with self._lock:
            cursor = self._conn.cursor()
            try:
                yield cursor
            finally:
                self._conn.commit()
                cursor.close()

    @staticmethod
    def _make_cache_key(*args: Optional[tuple], **kwargs: Optional[dict]) -> str:
        """
        Transform the args and kwargs into a json-format
        so it can be saved in a SQL table.

        Args:
            args (Optional[tuple]):
            kwargs (Optional[dict])
        """
        kwargs_key = tuple(sorted(kwargs.items()))
        return json.dumps(tuple(args) + kwargs_key)
