#!/usr/bin/python
# -*- coding: utf-8 -*-

from .base import BaseCache
from .disk import DiskCache
from .node import Node
from . import errors, eviction
from .memory import MemoryCache


__all__ = [
    "BaseCache",
    "DiskCache",
    "Node",
    "errors",
    "eviction",
    "MemoryCache"
]
__version__ = "0.3.1"
__author__ = "Deetjepateeteke <https://github.com/Deetjepateeteke>"
__license__ = "MIT"
