# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Lazy cache
- Disk cache
- CLI
- Docs
- cachelib errors
- Global cache ttl
- Cleanup thread that runs in background
- Add warnings

## [0.2.0] - 2025-07-19
### Added
- In memory LFU cache with ttl-support
- Initialized pyproject.toml with poetry

### Changed
- Improved BaseCache code coverage: less code needed in sub-classes
- Migrated from unittest to pytest

### Removed
- tests\test_lru.py: gets replaced by tests\test_cache.py
- tests\utils.py

## [0.1.0] - 2025-07-16
### Added
- In-memory LRU cache with ttl-support
- Cache persistance with Pickle
- Memoize decorator
- Track stats
- Basic logger (debug mode)
- Read-only mode