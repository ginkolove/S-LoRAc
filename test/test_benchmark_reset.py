"""Fault injection for the benchmark's destructive, post-replay reset."""
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import uuid

from aiohttp import web
from aiohttp.test_utils import TestServer
from benchmarks.lowra_saturation_exp import finalize_point, reset_runtime
from slora.server import api_server
from slora.server.httpserver.manager import HttpServerManager
from slora.server.io_struct import LowRAResetAck


def point():
    return dict(result=dict(valid=True, completed_requests=878),
                runtime=dict(samples=40), preemption=dict(preempted_requests=1))


class ResetTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = HttpServerManager.__new__(HttpServerManager)
        self.manager.reset_operations = {}
        self.manager.full_reprefill = True
        self.manager.shared_prefix_ids = (1,)
        self.manager.send_to_router = Mock()
        self.proc = Mock()
        self.proc.poll.return_value = None
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    async def server(self, handler):
        app = web.Application()
        app.router.add_post("/lowra/reset_runtime", handler)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        return str(server.make_url("/")).rstrip("/")

    async def test_response_lost_after_reset_does_not_zero_snapshot(self):
        router_calls, http_calls = [], []
        async def send(req):
            router_calls.append(req.req_id)
            hits = 17 if len(router_calls) == 1 else 0
            rank = dict(free_slots_after=64, total_slots=64,
                        prefix_access_before=dict(batch_unique_hits=hits, batch_unique_misses=3))
            self.manager._finish_reset(LowRAResetAck(req.req_id, True, "ok", dict(per_rank=[rank])))
        self.manager.send_to_router.send_pyobj = send

        async def handle(request):
            op = uuid.UUID(request.query["operation_id"])
            http_calls.append(op)
            with patch.object(api_server, "isFirst", False), patch.object(
                    api_server, "httpserver_manager", self.manager, create=True):
                response = await api_server.lowra_reset_runtime(op)
            if len(http_calls) == 1:
                self.assertTrue((self.root / "checkpoints/point.json").is_file())
                request.transport.abort()  # Backend reset succeeded; HTTP response never arrives.
            return web.Response(body=response.body, status=response.status_code, content_type="application/json")

        await finalize_point(point(), self.root, "point", await self.server(handle), self.proc)
        final = json.loads((self.root / "point.json").read_text())
        self.assertEqual(len(router_calls), 1)
        self.assertEqual(len(http_calls), 2)
        self.assertEqual(http_calls[0], http_calls[1])
        self.assertEqual(final["reset"], dict(operation_id=str(http_calls[0]), status="confirmed"))
        self.assertEqual(final["runtime"]["prefix_cache_stats"]["hit_ratio"], 0.85)
        self.assertFalse((self.root / "checkpoints/point.json").exists())

    async def test_busy_retries_but_leak_is_not_a_valid_point(self):
        ops = []
        async def handle(request):
            ops.append(request.query["operation_id"])
            if len(ops) == 1:
                return web.json_response(dict(ok=False, operation_id=ops[-1], message="busy"), status=409)
            return web.json_response(dict(ok=True, operation_id=ops[-1],
                                         stats=dict(per_rank=[dict(total_slots=64, free_slots_after=63)])))
        with self.assertRaisesRegex(RuntimeError, "all slots free"):
            await finalize_point(point(), self.root, "point", await self.server(handle), self.proc)
        self.assertEqual(ops, [ops[0]] * 2)
        saved = json.loads((self.root / "checkpoints/point.json").read_text())
        self.assertTrue(saved["replay_valid"])
        self.assertFalse(saved["result"]["valid"])
        self.assertEqual(saved["result"]["completed_requests"], 878)
        self.assertEqual(saved["reset"]["status"], "unconfirmed")
        self.assertFalse((self.root / "point.json").exists())

    async def test_cancelled_waiter_duplicate_and_busy_ack(self):
        sent = asyncio.Event()
        async def send(req):
            sent.set()
        self.manager.send_to_router.send_pyobj.side_effect = send
        first = asyncio.create_task(self.manager.reset_lowra_runtime("op"))
        await sent.wait()
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        duplicate = asyncio.create_task(self.manager.reset_lowra_runtime("op"))
        await asyncio.sleep(0)
        self.manager._finish_reset(LowRAResetAck("op", False, "busy", {"running_batch_size": 1}))
        self.assertEqual((await duplicate)[0], 409)
        self.assertEqual(self.manager.send_to_router.send_pyobj.call_count, 1)
        sent.clear()
        retry = asyncio.create_task(self.manager.reset_lowra_runtime("op"))
        await sent.wait()
        self.manager._finish_reset(LowRAResetAck("op", True, "ok"))
        self.assertEqual((await retry)[0], 200)
        self.manager.shared_prefix_ids = (2,)
        self.assertEqual((await self.manager.reset_lowra_runtime("op"))[0], 200)
        self.assertEqual(self.manager.shared_prefix_ids, (2,))
        self.assertEqual(self.manager.send_to_router.send_pyobj.call_count, 2)

    async def test_dead_server_fails_immediately(self):
        self.proc.poll.return_value = self.proc.returncode = 7
        with self.assertRaisesRegex(RuntimeError, "server exited.*7"):
            await reset_runtime("http://unused", str(uuid.uuid4()), self.proc)
