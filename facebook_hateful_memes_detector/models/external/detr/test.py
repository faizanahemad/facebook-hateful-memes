
import collections
import logging
import contextlib
import gevent
from gevent import Greenlet
import threading
from threading import current_thread
# from gevent import monkey
# monkey.patch_all()


from gsocketpool.exceptions import ConnectionNotFoundError, PoolExhaustedError
from typing import List, Dict
import asyncio
import random
from mprpc import RPCClient, RPCPoolClient
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from threading import Lock


def htime():
    return float(str(time.time()).split(".")[0][-3:] + "." + str(time.time()).split(".")[1][:3])


class RpcCacheClient:
    def __init__(self, server, port, pool_size=1):
        self.server = server
        self.port = port
        # self.client_pool = Pool(RPCPoolClient, dict(host=server, port=port), pool_size) # RPCPoolClient
        # self.pool = ThreadPoolExecutor(pool_size, thread_name_prefix='cache_')
        self._free = collections.deque(list(range(pool_size)))
        # self.lock = Lock()
        self.thread2Conn = dict()
        self.pool = ProcessPoolExecutor(pool_size)

    def __getitem__(self, item):
        assert type(item) == str
        return self.__exec__("get", item)

    def __setitem__(self, key, value):
        assert type(key) == str
        return self.__exec__("set", key, value)

    def __get_connection__(self):
        name = current_thread().name
        client = None
        if name in self.thread2Conn:
            conn_data = self.thread2Conn[name]
            assert len(conn_data) == 2
            client = conn_data[0]
            port = conn_data[1]
            assert port not in self._free
            print(name, port, client, client.is_connected())
            if client.is_connected():
                return client
            else:
                del conn_data
                del self.thread2Conn[name]
                self._free.append(port)

        if client is None:
            port = self._free.popleft()
            # print((name, port, self.thread2Conn.keys()))
            client = RPCClient(self.server, self.port + port)
            self.thread2Conn[name] = (client, port)
            print(name, port, client, client.is_connected())
            return client

    def __exec__(self, *args):
        assert len(args) > 1
        assert type(args[0]) == str

        t = htime()
        try:
            # self.lock.acquire()
            client = self.__get_connection__()
            # self.lock.release()
            # port = self._free.popleft()
            # client = RPCClient(self.server, self.port + port)

            result = client.call(*args)
            # with self.client_pool.connection() as client:
            #     result = client.call(*args)
            e = htime()
            tot = e - t

            # print((args[1], t, e, "%.3f" % tot))
            client.close()
            # self._free.append(port)
            return result
        except Exception as e:
            print(e)
            try:
                client.close()
            except:
                pass

    def get_batch(self, items: List):
        results = self.pool.map(self.__getitem__, items)
        return dict(zip(items, results))
        # loop = asyncio.get_event_loop()
        # futures = [(i, loop.run_in_executor(self.pool, self.__getitem__, i)) for i in items]
        # waiter = asyncio.wait([f for i, f in futures])
        # res, exceptions = loop.run_until_complete(waiter)
        # results = dict([(i, f.result()) for i, f in futures])
        # assert results.keys() == set(items)
        # return results

    def set_batch(self, item_dict: Dict):
        self.client_pool.drop_expired()
        loop = asyncio.get_event_loop()
        futures = [(k, loop.run_in_executor(self.pool, self.__setitem__, k, v)) for k, v in item_dict.items()]
        waiter = asyncio.wait([f for i, f in futures])
        loop.run_until_complete(waiter)
        results = all([f.result() for i, f in futures])
        assert results
        return results


client = RpcCacheClient('dev-dsk-ahemf-cache-r5-12x-e48a86de.us-west-2.amazon.com', 6000, 2)
import time
s = time.time()
print(type(client.get_batch(["a"])))
e = time.time() - s
print(e)


s = time.time()
print(type(client.get_batch(["a", "b", "c", "d", "e", "f", "g", "h"])))
e = time.time() - s
print(e)

s = time.time()
print(type(client.get_batch(list(map(str, range(32))))))
e = time.time() - s
print(e)

s = time.time()
print(type(client.get_batch(list(map(str, range(32))))))
e = time.time() - s
print(e)




