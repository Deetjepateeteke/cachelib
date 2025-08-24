# -*- coding: utf-8 -*-

"""
eviction.py - Cachelib eviction classes.

The classes defined in this module represent the supported eviction
methods used in cachelib.DiskCache and cachelib.MemoryCache.

Usage:
    >>> cache = MemoryCache(eviction=cachelib.eviction.LRU, max_size=5)

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

__all__ = ["EvictionPolicy", "FIFO", "LRU", "LFU"]


class EvictionPolicy():  # The master eviction policy class
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class _LRUEviction(EvictionPolicy): __slots__ = ()
class _LFUEviction(EvictionPolicy): __slots__ = ()
class _FIFOEviction(EvictionPolicy): __slots__ = ()


LRU: EvictionPolicy = _LRUEviction()
LFU: EvictionPolicy = _LFUEviction()
FIFO: EvictionPolicy = _FIFOEviction()
