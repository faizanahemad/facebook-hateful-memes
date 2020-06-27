from diskcache import Cache, Index
import numpy as np
import joblib
import os
import gc
import sys

cache_dir = os.path.join(os.getcwd(), 'cache')
if os.path.exists(cache_dir):
    assert os.path.isdir(cache_dir)
else:
    os.mkdir(cache_dir)

args = dict(eviction_policy='none', sqlite_cache_size=2**16, sqlite_mmap_size=2**34, disk_min_file_size=2**18)

from gevent.server import StreamServer
from mprpc import RPCServer
class CacheProxy(RPCServer):
    def __init__(self, cache):
        super().__init__()
        self.cache = cache

    def get(self, key):
        return self.cache[key]

    def set(self, key, value):
        self.cache[key] = value
        return True


try:
    server = StreamServer(('0.0.0.0', 6000), CacheProxy(Cache(cache_dir, **args)))
    server.serve_forever()
except Exception as e:
    print(e)
    del server
finally:
    del server
    print("Closed Server")
