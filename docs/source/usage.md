# Usage Guide

This guide shows how to use **cachelib**.

Caching is a technique where the result of expensive computations or functions that take a while (eg. loading webpages) get stored so they can easily be retrieved later without having to compute them again.

---

## Quickstart

Here is a basic usage example of how to use `cachelib`:

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

## Cache Expiration (TTL)

In cachelib, every cache has its own cleanup thread, that evicts expired keys automatically.

```python
cache = MemoryCache(ttl=60)

cache.set("key", "value")
```

In the above example, the cache is assigned a **global TTL (Time To Live)**. This means that every key/value pair in the cache will expire in 60 seconds after creation.

```python
cache = MemoryCache()

cache.set("foo", "bar", ttl=60)  # Will expire
cache.set("foo2", "bar2")  # Won't expire

```

Here, instead of assigning a global TTL, the first item has been given a TTL, while the second uses the global TTL. Since the global TTL defaults to `None`, meaning an unlimited time to live, the second item won't expire.

## Eviction Policies

When the **max-size** of a cache gets exceeded, space needs to be freed up for other items to get added to the cache. This is where eviction policies come in.

### LRU (Least Recently Used)

```python
import cachelib
from cachelib.eviction import LRU

cache = cachelib.MemoryCache(max_size=2, eviction_policy=LRU)

```

In the example above, **LRU (Least Recently Used)** is set as the eviction policy. When the cache's max-size gets exceeded, the least recently used item will get evicted from the cache.

```python
cache.set("a", 1)
cache.set("b", 2)  # Least Recently Used

cache.get("a")

cache.set("c", 3)  # Max-size gets exceeded, so evict LRU

assert "b" not in cache
```

In this example, you can see that when `cache.get("a")` is called, "b" becomes the least recently accessed. And when "c" is added to the cache, the cache's max-size gets exceeded and "b" gets evicted from the cache.

### LFU (Least Frequently Used)

```python
import cachelib
from cachelib.eviction import LFU

cache = cachelib.MemoryCache(max_size=2, eviction_policy=LFU)

```

Here, **LFU (Least Frequently Used)** is set as the eviction policy. When the cache's max-size gets exceeded, the least frequently used item will get evicted from the cache.

```python
cache.set("a", 1)
cache.set("b", 2)  # Least Frequently Used

for _ in range(5):
    cache.get("a")

cache.set("c", 3)  # Max-size gets exceeded, so evict LFU

assert "b" not in cache
```

In this example, `cache.get("a")` is called 5 times, which makes it the most frequently accessed. When "c" is added to the cache, the max-size gets exceeded and "b" gets evicted.

> [!INFO] Items equally accessed
> When there are two items accessed an equal amount of times, the item to evict will be decided by LRU based eviction.

## Cache backends

A cache backend is the way the items in the cache are stored. This can be **on disk** or in **memory**. The implementations for these two options are `cachelib.DiskCache` and `cachelib.MemoryCache`.

### DiskCache

DiskCache needs a `path` argument. This is the path to the file where the cache will be stored. It takes as input a `str` or a `pathlib.Path` object.

```python
import cachelib
from cachelib.eviction import LRU
from pathlib import Path

PATH = Path("example.db")

cache = cachelib.DiskCache(path=PATH, max_size=5, eviction_policy=LRU)
```

### MemoryCache

```python
import cachelib
from cachelib.eviction import LRU

cache = cachelib.MemoryCache(max_size=5, eviction_policy=LFU)
```

Since MemoryCache is an in memory cache, the cache won't be saved when the program gets exited. Therefore you can save the cache to a `.pkl` file.

```python
import cachelib
from cachelib.eviction import LRU
from pathlib import Path

PATH = Path("example.pkl")

cache = cachelib.MemoryCache(max_size=5, eviction_policy=LFU)
cache.set("foo", "bar")

# Save the cache as a .pkl file
cache.save(PATH)

# Load a cache from a .pkl file
loaded_cache = cachelib.MemoryCache.load(PATH)

# Check if "foo" (set in the cache before saving) is in the loaded cache
assert "foo" in loaded_cache
```

## Multi-level cache

A multi-level cache is a cache that exists out of multiple caches. Instead of having one huge cache, the cache is divided into multiple smaller caches. This improves the performance when retrieving cached items.

```python
from cachelib import DiskCache, MemoryCache, MultiLevelCache
from cachelib.eviction import LRU, LFU
from pathlib import Path

PATH = Path("example.db")

cache = MultiLevelCache(
    levels=[
        MemoryCache(max_size=32, eviction_policy=LFU),
        MemoryCache(max_size=64, eviction_policy=LFU),
        DiskCache(path=PATH, max_size=128, eviction_policy=LRU)
    ],
    inclusivity="inclusive"
)
```

This creates a multi-level cache existing of two MemoryCaches and one DiskCache. When the first cache is full, the evicted items will be transferred to the second cache and if that one is also full, the evicted items will be transferred to the DiskCache before getting removed for good.

`Inclusivity` can either be **"inclusive"** or **"exclusive"**:
    **"inclusive"**: when an item gets added to the cache, it gets added to every cache in the multi-level cache.
    **"exclusive"**: when an item gets added to the cache, it only gets added to the first cache.

The different caches in a multi-level cache are defined as `cache.l1`, `cache.l2` and `cache.l3`. Whenever `cache.get(key)` is called, the cache tries to find the key in its L1 cache first. If it isn't found there, it looks for it in its L2 cache and then its L3 cache.

When a key is accessed, the key gets moved to the L1 cache. This is the reason why a multi-level cache helps to improve performance, since the frequently accessed keys and recently added keys are stored in the L1 cache, where will be looked first when trying to retrieve a key.
