# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- `MultiLevelCache` (l1, l2, l3)
- `LazyCache`
- `DiskCache`
- Add `max_memory`
- CLI
- Docs
- Add warnings

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