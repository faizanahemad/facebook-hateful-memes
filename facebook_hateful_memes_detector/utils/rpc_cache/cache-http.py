from gevent import monkey
monkey.patch_all()
from flask import Flask
from flask import request
from flask import json
app = Flask(__name__)
from gevent import monkey
monkey.patch_all()
from diskcache import Cache, Index
import numpy as np
import joblib
import os
import gc
import sys
import zerorpc
from gevent.pywsgi import WSGIServer
from gevent.pool import Pool

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


proxy = CacheProxy(Cache(cache_dir, **args))


@app.route('/', methods=['POST'])
def serve():
    data = request.get_json()
    method = data["method"]
    data = data["data"]
    if method.lower() == "get":
        return proxy.get(data)
    elif method.lower() == "set":
        return {"result": proxy.set(data)}


@app.route('/get', methods=['POST'])
def get():
    data = request.get_json()
    return proxy.get(data)


@app.route('/set', methods=['POST'])
def setfn():
    data = request.get_json()
    return {"result": proxy.set(data)}


def start_server(port):
    assert type(port) == int
    try:
        pool = Pool(256)
        server = WSGIServer(('', port), app, spawn=pool)
        server.serve_forever()
    except Exception as e:
        print(e)
        server.close()
        del pool
        del server
    finally:
        server.close()
        del pool
        del server
        print("Closed Server")


if __name__ == "__main__":
    argv = sys.argv[1:]
    print(argv)
    port_start = int(argv[0]) if len(argv) >= 2 else 5000
    start_server(port_start)



