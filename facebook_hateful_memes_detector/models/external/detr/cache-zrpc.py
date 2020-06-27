from gevent import monkey
monkey.patch_all()
from diskcache import Cache, Index
import numpy as np
import joblib
import os
import gc
import sys
import zerorpc

cache_dir = os.path.join(os.getcwd(), 'cache')
if os.path.exists(cache_dir):
    assert os.path.isdir(cache_dir)
else:
    os.mkdir(cache_dir)

args = dict(eviction_policy='none', sqlite_cache_size=2**16, sqlite_mmap_size=2**34, disk_min_file_size=2**18)


class CacheProxy:
    def __init__(self, cache):
        super().__init__()
        self.cache = cache

    def get(self, keys):
        if type(keys) == str:
            keys = [keys]

        values = {key: self.cache[key] if key in self.cache else None for key in keys}
        return values

    def set(self, kv_pairs):
        assert type(kv_pairs) == dict
        for k, v in kv_pairs.items():
            self.cache[k] = v
        return True


def start_server(port):
    assert type(port) == int
    try:
        server = zerorpc.Server(CacheProxy(Cache(cache_dir, **args)), heartbeat=None, pool_size=256)
        server.bind("tcp://0.0.0.0:%s" % port)
        server.run()
    except Exception as e:
        print(e)
        server.close()
    finally:
        server.close()
        print("Closed Server")


if __name__ == "__main__":
    argv = sys.argv[1:]
    print(argv)
    port_start = int(argv[0]) if len(argv) >= 1 else 4242
    start_server(port_start)
