from __future__ import annotations

import asyncio

import pytest

from lmcache_compat.connector_patch import recv_exact, transaction


class FragmentedSocket:
    def __init__(self, responses):
        self.responses = responses
        self.pending = bytearray()
        self.sent = []

    def sendall(self, request):
        assert not self.pending
        self.sent.append(request)
        self.pending.extend(self.responses[request])

    def recv_into(self, view):
        count = min(len(view), len(self.pending), 2)
        view[:count] = self.pending[:count]
        del self.pending[:count]
        return count


@pytest.mark.parametrize("calls", [((b"c", False), (b"x", False)), ((b"g", True), (b"h", True)), ((b"c", False), (b"g", True), (b"x", False), (b"h", True))])
def test_fragmented_metadata_and_concurrent_operations_are_serialized(calls):
    async def run():
        sock = FragmentedSocket({b"c": b"YES", b"g": b"004data", b"x": b"NO!", b"h": b"003end"})
        lock = asyncio.Lock()
        recoveries = []

        async def call(request, body):
            return await transaction(
                lock, lambda: sock, lambda: recoveries.append(request), request, 3, bytes,
                lambda meta: (meta, recv_exact(sock, int(meta))) if body else meta,
            )

        results = await asyncio.gather(*(call(*item) for item in calls))
        expected = {b"c": b"YES", b"g": (b"004", b"data"), b"x": b"NO!", b"h": (b"003", b"end")}
        assert results == [expected[request] for request, _body in calls]
        assert sock.sent == [request for request, _body in calls]
        assert not recoveries

    asyncio.run(run())


def test_protocol_eof_recovers_connection_for_the_next_operation():
    async def run():
        sockets = [FragmentedSocket({b"bad": b"x"}), FragmentedSocket({b"ok": b"YES"})]
        current = 0

        def recover():
            nonlocal current
            current += 1

        with pytest.raises(ConnectionError, match="missing"):
            await transaction(asyncio.Lock(), lambda: sockets[current], recover, b"bad", 3, bytes)
        assert await transaction(asyncio.Lock(), lambda: sockets[current], recover, b"ok", 3, bytes) == b"YES"
        assert current == 1

    asyncio.run(run())
