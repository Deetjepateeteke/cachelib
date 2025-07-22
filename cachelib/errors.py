
__all__ = ["CacheOverflowError", "ReadOnlyError"]

class CacheOverflowError(Exception):
    """Raised when the cache is full and no eviction policy is available."""
    def __init__(self, max_size: int, msg: str =None):
        if msg is None:
            msg = f"cache has reached its max-size of {max_size} and no eviction policy is set"
        super().__init__(msg)

class ReadOnlyError(Exception):
    """Raised when the cache gets modified when read-only mode is enabled."""
    def __init__(self, msg: str = None):
        if msg is None:
            msg = "cannot modify cache when read-only mode is enabled"
        super().__init__(msg)