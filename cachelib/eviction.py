# -*- coding: utf-8 -*-

"""
eviction.py - Cachelib eviction classes.

The classes defined in this module represent the supported eviction
methods used in cachelib.DiskCache and cachelib.MemoryCache.

Usage:
    >>> cache = MemoryCache(eviction=cachelib.eviction.LRU, max_size=5)

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

__all__ = ["EvictionPolicy", "LRU", "LFU"]


class EvictionPolicy(): ...  # The master eviction policy class

class _LRUEviction(EvictionPolicy): ...
class _LFUEviction(EvictionPolicy): ...


LRU: EvictionPolicy = _LRUEviction()
LFU: EvictionPolicy = _LFUEviction()
