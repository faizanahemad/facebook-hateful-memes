
import collections
import logging
import contextlib
import gevent
from gevent import Greenlet
import threading
from threading import current_thread
# from gevent import monkey
# monkey.patch_all()
import zerorpc
import random
from greendb import Client
from more_itertools import chunked
from typing import List, Dict
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock
import requests
# https://github.com/coleifer/greendb


def htime():
    return float(str(time.time()).split(".")[0][-3:] + "." + str(time.time()).split(".")[1][:3])


class RpcCacheClient:
    def __init__(self, server, zrpc_port, mprpc_port, http_port, pool_size=1):
        self.server = server
        self.zrpc_port = zrpc_port
        self.mprpc_port = mprpc_port
        self.http_port = http_port
        self.pool = ThreadPoolExecutor(pool_size, thread_name_prefix='cache_')
        self.pool_size = pool_size

    def __getitem__(self, item):
        assert type(item) == str
        return self.__exec__("get", item)

    def __setitem__(self, key, value):
        assert type(key) == str
        return self.__exec__("set", key, value)

    def __exec_mprpc__(self, args):
        from mprpc import RPCClient, RPCPoolClient

        method = args[0]
        args = args[1]
        assert method in ["get", "set"]
        try:
            client = RPCClient(self.server, self.mprpc_port)
            result = client.call(method, args)
            client.close()
            return result
        except Exception as e:
            print(e)
            try:
                client.close()
            except:
                pass

    def __exec_zrpc__(self, args):
        method = args[0]
        args = args[1]
        assert method in ["get", "set"]
        try:
            client = zerorpc.Client(heartbeat=None)
            client.connect("tcp://%s:%s" % (self.server, self.zrpc_port))
            result = client.get(args) if method == "get" else client.set(args)
            client.close()
            return result
        except Exception as e:
            print(e)
            try:
                client.close()
            except:
                pass

    def __exec_http__(self, args):
        method = args[0]
        args = args[1]
        assert method in ["get", "set"]
        return requests.post('http://%s:%s/%s' % (self.server, self.http_port, method), json=args).json()

    def __clever_exec__(self, args):
        choice = random.randint(0, 1)
        if choice == 0:
            return self.__exec_http__(args)
        else:
            return self.__exec_zrpc__(args)

    def get_batch(self, items: List):
        chunk_size = int(len(items) / self.pool_size)  # 2
        results = self.pool.map(self.__exec_zrpc__, [("get", c) for c in chunked(items, chunk_size)])
        results = {k: v for d in results for k, v in d.items()}
        return results

    def set_batch(self, item_dict: Dict):
        results = self.pool.map(self.__exec_zrpc__, [("set", dict(c)) for c in chunked(item_dict.items(), 2)])
        results = all(results)
        assert results


if __name__ == "__main__":
    client = RpcCacheClient('dev-dsk-ahemf-cache-r5-12x-e48a86de.us-west-2.amazon.com', 4242, 6000, 5000, 32)
    import time
    kl = list(map(str, range(32)))
    kl2 = kl + kl
    kl4 = kl2 + kl2
    kl8 = kl4 + kl4
    kl16 = kl8 + kl8

    iters = 5
    s = time.time()
    res = all([all([v is not None for v in client.get_batch(kl).values()]) for _ in range(iters)])
    e = time.time() - s
    print(len(kl), res, "%.3f" % (e / iters))

    s = time.time()
    res = all([all([v is not None for v in client.get_batch(kl2).values()]) for _ in range(iters)])
    e = time.time() - s
    print(len(kl2), res, "%.3f" % (e / iters))

    s = time.time()
    res = all([all([v is not None for v in client.get_batch(kl4).values()]) for _ in range(iters)])
    e = time.time() - s
    print(len(kl4), res, "%.3f" % (e / iters))

    s = time.time()
    res = all([all([v is not None for v in client.get_batch(kl8).values()]) for _ in range(iters)])
    e = time.time() - s
    print(len(kl8), res, "%.3f" % (e / iters))

    s = time.time()
    res = all([all([v is not None for v in client.get_batch(kl16).values()]) for _ in range(iters)])
    e = time.time() - s
    print(len(kl16), res, "%.3f" % (e / iters))




