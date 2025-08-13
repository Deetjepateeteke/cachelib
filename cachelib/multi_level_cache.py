"""

Usage:
    >>> l1 = MemoryCache(max_size=128)
    >>> l2 = MemoryCache(max_size=1024)
    >>> l3 = DiskCache(max_memory='10mb')

    >>> cache = MultiLevelCache(
            levels=[l1, l2, l3],
            inclusivity='inclusive'
        )
"""

from threading import RLock
from typing import Any, Hashable, Optional, Union

from .base import BaseCache


class MultiLevelCache:
    def __init__(self,
                 levels: list,
                 inclusivity: str["inclusive" | "exclusive"] = "inclusive",
                 ):
        if inclusivity not in ["inclusive", "exclusive"]:
            raise ValueError("MultiLevelCache.inclusivity should be either 'inclusive' or 'exclusive'")
        self._inclusivity = inclusivity

        self.l1: BaseCache = levels[0]
        self.l2: BaseCache = levels[1]
        self.l3: BaseCache = levels[2]

        # Thread safety
        self._lock = RLock()

    def set(self,
            key: Hashable,
            value: Any,
            ttl: Optional[Union[int, float]] = None
            ) -> None:
        with self._lock:
            if self._inclusivity == "inclusive":
                    self.l1.set(key, value, ttl)
                    self.l2.set(key, value, ttl)
                    self.l3.set(key, value, ttl)



            elif self._inclusivity == "exclusive":
                pass

    def get(self, key: Hashable):
        with self._lock:
            # Look for value in L1 cache
            value = self.l1.get(key)
            if value is not None:
                return value
            
            # Look for value in L2 cache
            value = self.l2.get(key)
            if value is not None:
                return value
            
            # Look for value in L3 cache
            value = self.l3.get(key)
            if value is not None:
                return value
            
            # Value is not found
            # TODO: update stats
            return value
