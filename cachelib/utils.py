# -*- coding: utf-8 -*-

"""
utils.py - Miscellaneous functions and classes for cachelib.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

from __future__ import annotations
from contextlib import AbstractContextManager
import json
from typing import Any, Optional

from .errors import CacheConfigurationError

__all__ = ["extract_items_from_args", "NullContext", "NullValue"]


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


class NullValue:
    """
    Empty value container

    Used as a replacement for None when something is undefined.
    """
    pass


def extract_items_from_args(*args, **kwargs) -> dict[str, Optional[Any]]:
    """ Check for 'value' and/or 'ttl' in args. """

    # Check for a valid amount of args
    if len(args) + len(kwargs) > 2:
        raise CacheConfigurationError(f"expected at most 3 arguments, got {len(args) + len(kwargs) + 1}")
    params = {}  # The params that have been found

    # Look for 'value' in args
    if args:
        params["value"] = args[0]
    # If not found in args, look for 'value' in kwargs
    elif "value" in kwargs.keys():
        params["value"] = kwargs["value"]

    # Look for 'ttl' in args
    if len(args) == 2:
        params["ttl"] = args[1]
    # If not found in args, look for 'ttl' in kwargs
    elif "ttl" in kwargs.keys():
        params["ttl"] = kwargs["ttl"]

    return params


def create_json_cache_key(key: Any) -> str:
    """Transform the given args and kwargs to a json format that
    is suitable to store on a disk cache. """
    return json.dumps(key)
