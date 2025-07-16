# -*- coding: utf-8 -*-

"""
utils.py - Tools functions and classes for cachelib.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from __future__ import annotations
from contextlib import AbstractContextManager
from typing import Any

__all__ = ["NullContext"]


class NullContext(AbstractContextManager):
    """
    Empty context manager

    Used as a replacement for a normal context manager,
    when a block of code only is used sometimes with a normal context manager.
    """
    def __enter__(self) -> NullContext:
        """ empty method ... """
        return self

    def __exit__(self, *exc: Any) -> False:
        """ empty method ..."""
        return False  # Don't suppress any exceptions.
