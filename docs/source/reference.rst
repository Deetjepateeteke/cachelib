API Reference
================

cachelib.MemoryCache
-------------------------------

.. autoclass:: cachelib.MemoryCache
   :members: __getitem__, __setitem__, __delitem__, get, get_many, set, delete, clear, ttl, inspect, keys, values, memoize, set_verbose, set_read_only, get_stats, __contains__, __len__, read_only, verbose, load, save
   :show-inheritance:

cachelib.DiskCache
-------------------------------
.. autoclass:: cachelib.DiskCache
   :members: __getitem__, __setitem__, __delitem__, get, get_many, set, delete, clear, ttl, inspect, keys, values, memoize, set_verbose, set_read_only, get_stats, __contains__, __len__, read_only, verbose, close
   :show-inheritance:

cachelib.MultiLevelCache
-------------------------------
.. autoclass:: cachelib.MultiLevelCache
   :members: __getitem__, __setitem__, __delitem__, get, get_many, set, delete, clear, ttl, inspect, keys, values, memoize, __len__, __contains__, close
   :show-inheritance:

cachelib.eviction
-------------------------------
.. autoclass:: cachelib.eviction.LRU
.. autoclass:: cachelib.eviction.LFU
.. autoclass:: cachelib.eviction.FIFO