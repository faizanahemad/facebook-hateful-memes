
import collections
import logging
import contextlib
import gevent
from gevent import Greenlet
import threading
from threading import current_thread
from gevent import monkey
monkey.patch_all()


from gsocketpool.exceptions import ConnectionNotFoundError, PoolExhaustedError


class Pool(object):
    """Connection pool.

    Usage:
        Communicating echo server running on localhost 2000:

        >>>
        >>> from gsocketpool import Pool
        >>> from gsocketpool import TcpConnection
        >>> options = dict(host='localhost', port=2000)
        >>> pool = Pool(TcpConnection, options)
        >>>
        >>> with pool.connection() as conn:
        ...     conn.send('hello')
        ...     print conn.recv()
        hello

    :param factory: :class:`Connection <gsocketpool.connection.Connection>`
        class or a callable that creates
        :class:`Connection <gsocketpool.connection.Connection>` instance.
    :param dict options: (optional) Options to pass to the factory.
    :param int initial_connections: (optional) The number of connections that
        are initially established.
    :param int max_connections: (optional) The maximum number of connections.
    :param bool reap_expired_connections: (optional) If set to True, a
        background thread (greenlet) that periodically kills expired connections
        will be launched.
    :param int reap_interval: (optional) The interval to run to kill expired
        connections.
    """

    def __init__(self, factory, options={}, initial_connections=1,
                 max_connections=8):

        self._factory = factory
        self._options = options
        self._max_connections = max_connections

        self._pool = collections.deque()
        self._using = collections.deque()
        self._global_pool = list()

        assert initial_connections <= max_connections, "initial_connections must be less than max_connections"

        for i in range(initial_connections):
            self._pool.append(self._create_connection())

    def __del__(self):
        try:
            for conn in self._pool:
                conn.close()
            self._pool = None

            for conn in self._using:
                conn.close()
            self._using = None
        except:
            pass

    @contextlib.contextmanager
    def connection(self):
        conn = self.acquire()
        try:
            yield conn

        finally:
            self.release(conn)

    @property
    def size(self):
        """Returns the pool size."""

        return len(self._pool) + len(self._using)

    def acquire(self, retry=10, retried=0):
        """Acquires a connection from the pool.

        :param int retry: (optional) The maximum number of times to retry.
        :returns: :class:`Connection <gsocketpool.connection.Connection>` instance.
        :raises:  :class:`PoolExhaustedError <gsocketpool.exceptions.PoolExhaustedError>`
        """

        self.drop_expired()
        local = threading.local()
        if len(self._pool):
            conn = self._pool.popleft()
            self._using.append(conn)

            return conn

        else:
            if len(self._pool) + len(self._using) < self._max_connections:
                conn = self._create_connection()
                self._using.append(conn)
                return conn

            else:
                if retried >= retry:
                    raise PoolExhaustedError()
                retried += 1

                gevent.sleep(0.1)

                return self.acquire(retry=retry, retried=retried)

    def release(self, conn):
        """Releases the connection.

        :param Connection conn: :class:`Connection <gsocketpool.connection.Connection>` instance.
        :raises: :class:`ConnectionNotFoundError <gsocketpool.exceptions.ConnectionNotFoundError>`
        """

        if conn in self._using:
            self._using.remove(conn)
            self._pool.append(conn)

        else:
            raise ConnectionNotFoundError()

    def drop(self, conn):
        """Removes the connection from the pool.

        :param Connection conn: :class:`Connection <gsocketpool.connection.Connection>` instance.
        :raises: :class:`ConnectionNotFoundError <gsocketpool.exceptions.ConnectionNotFoundError>`
        """
        print("dropping", conn)
        if conn in self._pool:
            self._pool.remove(conn)
            if conn.is_connected():
                conn.close()

        else:
            raise ConnectionNotFoundError()

    def drop_expired(self):
        """Removes all expired connections from the pool.

        :param Connection conn: :class:`Connection <gsocketpool.connection.Connection>` instance.
        """
        try:
            expired_conns = [conn for conn in self._pool if conn.is_expired()]
        except AttributeError:
            expired_conns = [conn for conn in self._pool if not conn.is_connected()]

        for conn in expired_conns:
            self.drop(conn)

    def _create_connection(self):
        conn = self._factory(**self._options)
        try:
            conn.open()
        except AssertionError:
            pass
        return conn

from typing import List, Dict
import asyncio

class RpcCacheClient:
    def __init__(self, server, port, pool_size=1):
        from mprpc import RPCClient, RPCPoolClient
        self.server = server
        self.port = port
        self.client_pool = Pool(RPCPoolClient, dict(host=server, port=port), pool_size) # RPCPoolClient
        from concurrent.futures import ThreadPoolExecutor
        self.pool = ThreadPoolExecutor(pool_size)


    def __getitem__(self, item):
        assert type(item) == str
        return self.__exec__("get", item)

    def __setitem__(self, key, value):
        assert type(key) == str
        return self.__exec__("set", key, value)

    def __exec__(self, *args):
        from mprpc import RPCClient, RPCPoolClient
        assert len(args) > 1
        assert type(args[0]) == str
        t = time.time()
        try:
            # client = RPCClient(self.server, self.port)
            # result = client.call(*args)
            # client.close()
            with self.client_pool.connection() as client:
                result = client.call(*args)
            e = time.time() - t
            # print(args[1], "%.2f" % e)
            return result
        except Exception as e:
            print(e)

    def get_batch(self, items: List):
        # results = self.pool.map(self.__getitem__, items)
        # return dict(zip(items, results))
        self.client_pool.drop_expired()
        loop = asyncio.get_event_loop()
        futures = [(i, loop.run_in_executor(self.pool, self.__getitem__, i)) for i in items]
        waiter = asyncio.wait([f for i, f in futures])
        res, exceptions = loop.run_until_complete(waiter)
        results = dict([(i, f.result()) for i, f in futures])
        assert results.keys() == set(items)
        return results

    def set_batch(self, item_dict: Dict):
        self.client_pool.drop_expired()
        loop = asyncio.get_event_loop()
        futures = [(k, loop.run_in_executor(self.pool, self.__setitem__, k, v)) for k, v in item_dict.items()]
        waiter = asyncio.wait([f for i, f in futures])
        loop.run_until_complete(waiter)
        results = all([f.result() for i, f in futures])
        assert results
        return results

    def __call__(self, items: List):
        if type(items) == list or type(items) == tuple:
            return self.get_batch(items)
        else:
            assert type(items) == str
            return self[items]


client = RpcCacheClient('dev-dsk-ahemf-cache-r5-12x-e48a86de.us-west-2.amazon.com', 6000, 32)
print("Created Conns")
import time
s = time.time()
print(type(client.get_batch(["a", "b", "c", "d", "e", "f", "g", "h"])))
e = time.time() - s
print(e)
s = time.time()
print(type(client.get_batch(list(map(str, range(32))))))
e = time.time() - s
print(e)



