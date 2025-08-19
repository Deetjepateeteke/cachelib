from pathlib import Path
from cachelib import eviction, DiskCache, MemoryCache, MultiLevelCache

PATH = Path("tests", "test_file.db")


def teardown_cache(cache):
    cache.clear()
    cache.close()

    with cache._lock:
        if PATH.exists():
            PATH.unlink()


def create_memory_cache() -> MemoryCache:
    return MemoryCache()

def create_disk_cache() -> DiskCache:
    return DiskCache(path=PATH)

def create_lru_memory_cache():
    return MemoryCache(eviction_policy=eviction.LRU, max_size=2)

def create_lfu_memory_cache():
    return MemoryCache(eviction_policy=eviction.LFU, max_size=2)

def create_lru_disk_cache():
    return DiskCache(PATH, eviction_policy=eviction.LRU, max_size=2)

def create_lfu_disk_cache():
    return DiskCache(PATH, eviction_policy=eviction.LFU, max_size=2)


def create_multi_level_cache(inclusivity: str) -> MultiLevelCache:

    multi_level_cache = MultiLevelCache(
        levels=[
            MemoryCache(max_size=1, eviction_policy=eviction.LRU),
            MemoryCache(max_size=2, eviction_policy=eviction.LRU),
            DiskCache(path=PATH)
        ],
        inclusivity=inclusivity
    )
    multi_level_cache.clear()

    return multi_level_cache

def create_inclusive_multi_level_cache():
    return create_multi_level_cache(inclusivity="inclusive")

def create_exclusive_multi_level_cache():
    return create_multi_level_cache(inclusivity="exclusive")
