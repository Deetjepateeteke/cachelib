#!/usr/bin/python
# -*- coding: utf-8 -*-

from .caches.base import BaseCache
from .caches.disk import DiskCache
from .caches.memory import MemoryCache
from .caches.multi_level_cache import MultiLevelCache
from . import errors, eviction
from .node import Node


__all__ = [
    "BaseCache",
    "DiskCache",
    "Node",
    "errors",
    "eviction",
    "MemoryCache",
    "MultiLevelCache"
]
__version__ = "0.5.1"
__author__ = "Deetjepateeteke <https://github.com/Deetjepateeteke>"
__license__ = "MIT"
