# Cachelib: Python cache library

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Version](https://img.shields.io/badge/version-0.5.1-blue.svg)

Cachelib is a safe and lightweight caching package, written in pure Python,
so it doesn't rely on any third-party dependencies, which makes it easily embeddable in any Python program or script.

## Install

To clone the repository:

```bash
git clone https://github.com/Deetjepateeteke/cachelib.git
cd cachelib
```

## Features

- In-memory cache and on-disk cache
- Multi-level cache (L1, L2, L3)
- TTL support
- LRU/LFU eviction
- Thread-safe

## Usage

Here's a basic example of how to use `cachelib`:

```python
import cachelib

# Create an in-memory cache
cache = cachelib.MemoryCache()

# Set a value with an optional TTL (in seconds)
cache.set("foo", "bar", ttl=100)

# Retrieve a cached value
cache.get("foo")  # -> returns 'bar'

# Check if a key exists
if "foo" in cache:
    print("'foo' is cached")

# Delete a cached value
cache.delete("foo")

# Clear the entire cache
cache.clear()
```

## Links

- Source Code: https://github.com/Deetjepateeteke/cachelib/

## License

This project is licensed under the [MIT License](LICENSE).
