# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

- FIFO eviction
- `LazyCache`
- Namespace table in `SQL`
- Docs

## [0.5.1] - 2025-08-20

### Added

- Added `max_memory` in DiskCache and MemoryCache

## [0.5.0] - 2025-08-19

### Added

- Added `MultiLevelCache` and its tests
- Added abstractmethod `_make_cache_key()` in BaseCache
- Added `NullValue` type in `utils.py`

### Changed

- Moved `DiskCache` and `MemoryCache` into `cachelib.caches` folder
- Renamed `cachelib.errors.CachePathError` to `cachelib.errors.PathError`
- Replaced `cleanup_thread@interval.setter` by `cleanup_thread.set_interval()` method

## [0.4.0] - 2025-08-13

### Added

- `cachelib.DiskCache` with lru/lfu eviction and optional ttl-support
- Added `cachelib.Node`, `cachelib.eviction` and `cachelib.errors` to `__init__.py`
- Added `cleanup_thread.py` which implements a `DiskCleanupThread` and a `MemoryCleanupThread`
- Added `eviction.py` which implements LRU and LFU eviction
- `cache.keys()` and `cache.values()` feature
- Added test cases: `test_api.py`, `test_cleanup_thread.py`, `test_thread_safety.py`

## [0.3.1] - 2025-07-24

### Added

- `cachelib.errors`: custom errors

### Fixed

- Fixed bug where `cache._get_evict_node()` returned `None` when no eviction policy was set

## [0.3.0] - 2025-07-22

### Added

- Added `MemoryCache` with lru and lfu support
- Added global cache-ttl in `BaseCache`
- Added `BaseCache().get_many()` implementation
- Added `README.md`

### Changed

- Replaced `LRUCache` and `LFUCache` with `MemoryCache`
- Renamed `tests\test_cache.py` to `tests\test_memory.py`

### Removed

- Removed `__iter__` and `__reversed__` methods for every `BaseCache` subclass

## [0.2.1] - 2025-07-20

### Added

- In-background cleanup thread (`_CleanupThread`)
- Type hints in `BaseCache.Stats`

## [0.2.0] - 2025-07-19

### Added

- In memory LFU cache with ttl-support
- Initialized `pyproject.toml` with poetry

### Changed

- Improved `BaseCache` code coverage: less code needed in sub-classes
- Migrated from `unittest` to `pytest`

### Removed

- `tests\test_lru.py`: gets replaced by `tests\test_cache.py`
- `tests\utils.py`

## [0.1.0] - 2025-07-16

### Added

- In-memory LRU cache with ttl-support
- Cache persistance with `Pickle`
- Memoize decorator
- Track stats
- Basic logger (debug mode)
- Read-only mode
