# -*- coding: utf-8 -*-

"""
errors.py - Custom cachelib errors.

Usage:
    >>> try:
    >>>     ...
    >>> except CacheError:  # Catches all errors raised by cachelib.
    >>>     ...

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from typing import Hashable

__all__ = [
    "CacheError",
    "KeyErrorBase",
    "CachePersistenceErrorBase",
    "SerializationError",
    "CacheConfigurationError",
    "CleanupThreadConfigurationError",
    "CacheOverflowError",
    "ReadOnlyError",
    "KeyNotFoundError",
    "KeyExpiredError",
    "CacheLoadError",
    "CacheSaveError",
    "CachePathError",
    "SerializationError",
    "DeserializationError"
]


class CacheError(Exception): ...  # The master `cachelib` exception

class KeyErrorBase(CacheError): ...
class CachePersistenceErrorBase(CacheError): ...
class SerializationErrorBase(CacheError): ...


class CacheConfigurationError(CacheError): ...
class CleanupThreadConfigurationError(CacheError): ...

class CacheOverflowError(CacheError):
    """Raised when the cache is full and no eviction policy is available."""
    def __init__(self, max_size: int, msg: str = None):
        if msg is None:
            msg = f"cache has reached its max-size of {max_size} and no eviction policy is set"
        super().__init__(msg)

class ReadOnlyError(CacheError):
    """Raised when the cache gets modified when read-only mode is enabled."""
    def __init__(self, msg: str = None):
        if msg is None:
            msg = "cannot modify cache when read-only mode is enabled"
        super().__init__(msg)


class KeyNotFoundError(KeyErrorBase):
    """Raised when the given key was not found in the cache."""
    def __init__(self, key: Hashable, msg: str = None):
        if msg is None:
            msg = f"key='{key}' not found in cache"
        super().__init__(msg)

class KeyExpiredError(KeyErrorBase):
    """Raised when the given key is found to be expired."""
    def __init__(self, key: Hashable, msg: str = None):
        if msg is None:
            msg = f"key='{key}' was found to be expired"
        super().__init__(msg)


class CacheLoadError(CachePersistenceErrorBase): ...
class CacheSaveError(CachePersistenceErrorBase): ...
class CachePathError(CachePersistenceErrorBase): ...


class SerializationError(SerializationErrorBase): ...
class DeserializationError(SerializationErrorBase): ...
