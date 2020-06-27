
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


def htime():
    return float(str(time.time()).split(".")[0][-3:] + "." + str(time.time()).split(".")[1][:3])


class RpcCacheClient:
    def __init__(self, server, zrpc_port, mprpc_port, pool_size=1):
        self.server = server
        self.zrpc_port = zrpc_port
        self.mprpc_port = mprpc_port
        self.pool = ThreadPoolExecutor(pool_size, thread_name_prefix='cache_')

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

    def __clever_exec__(self, args):
        choice = random.randint(0, 1)
        if choice == 0:
            return self.__exec_mprpc__(args)
        else:
            return self.__exec_zrpc__(args)


    def get_batch(self, items: List):
        results = self.pool.map(self.__exec_zrpc__, [("get", c) for c in chunked(items, 2)])
        results = {k: v for d in results for k, v in d.items()}
        return results

    def set_batch(self, item_dict: Dict):
        results = self.pool.map(self.__exec_zrpc__, [("set", dict(c)) for c in chunked(item_dict.items(), 2)])
        results = all(results)
        assert results


if __name__ == "__main__":
    client = RpcCacheClient('dev-dsk-ahemf-cache-r5-12x-e48a86de.us-west-2.amazon.com', 4242, 6000, 32)
    import time

    s = time.time()
    print(type(client.get_batch(list(map(str, range(32))) + list(map(str, range(32))))))
    e = time.time() - s
    print(e)

    s = time.time()
    print(type(client.get_batch(list(map(str, range(32))) + list(map(str, range(32))))))
    e = time.time() - s
    print(e)




