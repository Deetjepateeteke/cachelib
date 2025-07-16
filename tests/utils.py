# -*- coding: utf-8 -*-

"""
utils.py - Tools used while testing.

Author: Deetjepateeteke <https://github.com/Deetjepateeteke>
"""

import time
from unittest.mock import patch

__all__ = ["MockTimeContextManager"]


class MockTimeContextManager:
    """
    Used to replace time.time() with a custom time, so future events can
    be simulated without having to wait.

    Usage:
        with MockTimeContextManager(future_time=10):
            time.time() will behave as if 10 seconds have passed
    """
    def __init__(self, future_time):
        self._patcher = patch(
            "time.time",
            return_value=time.time() + future_time
        )

    def __enter__(self):
        self._patcher.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._patcher.stop()
        return False
