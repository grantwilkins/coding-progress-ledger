"""
Claim:
Stage 1b starts from the validated old Apptainer sandbox path, keeps source and
sink vLLM instances separate, and replaces privileged kernel tc with one
user-space bandwidth bucket shared by the source-egress KV/API proxy routes.

Plausible wrong implementations:
- Reintroduce Docker/latest-cu129 or LMCacheMPConnector.
- Start source/sink with colliding ports, long TMPDIRs, or shared cache dirs.
- Shape each route independently instead of enforcing one shared link.
- Bill both directions instead of the simulated source-egress directions.
"""

from __future__ import annotations

import asyncio
import csv
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

import stage1b_drain_sink as s


def cmd_text(cmd):
    return s.shell(cmd)


def test_vllm_commands_pin_validated_sandbox_flags_and_roles():
    cfg = s.Config()

    source = cmd_text(s.vllm_cmd(cfg, "source"))
    sink = cmd_text(s.vllm_cmd(cfg, "sink"))
    smoke = cmd_text(s.vllm_cmd(cfg, "smoke1"))

    assert "vllm-openai-v0.10.1.1.sandbox" in source
    assert "APPTAINERENV_CUDA_VISIBLE_DEVICES=0" in source
    assert "APPTAINERENV_CUDA_VISIBLE_DEVICES=1" in sink
    assert "--port 8100" in source
    assert "--port 8200" in sink
    assert "--port 8120" in smoke
    assert "LMCacheConnectorV1" in source
    assert "LMCacheMPConnector" not in source
    assert "kv_producer" in source and "engine_id\":\"s0" in source
    assert "kv_consumer" in sink and "engine_id\":\"d0" in sink
    assert "kv_both" in smoke and "engine_id\":\"e0" in smoke
    assert "LMCACHE_REMOTE_URL=lm://127.0.0.1:5655" in source
    assert "LMCACHE_REMOTE_URL=lm://127.0.0.1:8300" in sink
    assert "LMCACHE_LMCACHE_INSTANCE_ID=stage1b_src" in source
    assert "LMCACHE_LMCACHE_INSTANCE_ID=stage1b_sink" in sink
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in source
    assert "LMCACHE_MAX_LOCAL_CPU_SIZE=4" in source
    assert "TMPDIR=/tmp/qh-src-" in source
    assert "TMPDIR=/tmp/qh-sink-" in sink
    assert "VLLM_RPC_BASE_PATH=/tmp/qh-src-" in source
    assert "VLLM_RPC_BASE_PATH=/tmp/qh-sink-" in sink
    assert "--enforce-eager" in source
    assert "--disable-frontend-multiprocessing" in source
    assert "--async-scheduling" not in source
    assert "stage1b-src" in source and "stage1b-sink" in sink


def test_lmcache_and_proxy_use_host_commands_not_docker_or_tc():
    cfg = s.Config()

    lmcache = cmd_text(s.lmcache_cmd(cfg))
    proxy = cmd_text(s.proxy_cmd(cfg, 1000.0))

    assert "stage1b_drain_sink.py lmcache-server --host 127.0.0.1 --port 5655" in lmcache
    assert "apptainer" not in lmcache
    assert "--nv" not in lmcache
    assert "APPTAINERENV_CUDA_VISIBLE_DEVICES" not in lmcache
    assert "lmcache.v1.server" not in lmcache
    assert "stage1b_drain_sink.py proxy" in proxy
    assert "--kv-listen 127.0.0.1:8300 --kv-target 127.0.0.1:5655" in proxy
    assert "--api-listen 127.0.0.1:8400 --api-target 127.0.0.1:8200" in proxy
    assert "--mbps 1000.0" in proxy
    assert "docker" not in proxy.lower()
    assert " tc " not in proxy.lower()


def test_lmcache_wait_reports_process_exit_log(tmp_path):
    log = tmp_path / "lmcache.log"
    log.write_text("first\nlast\n")
    proc = subprocess.Popen(["false"])
    with pytest.raises(RuntimeError, match="last"):
        s.wait_tcp_process("127.0.0.1", 1, 5, proc, log)


def test_custom_apptainer_image_path_is_wired():
    source = cmd_text(s.vllm_cmd(s.Config(sandbox="/tmp/qh.sif"), "source"))

    assert "/tmp/qh.sif" in source


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _lmc_request(port, command, key="k", data=b""):
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        header = s.LMCACHE_CLIENT_META.pack(command, len(data), 4, 6, 0, len(data), 0, 0, 0, key.encode().ljust(s.LMCACHE_MAX_KEY_LENGTH))
        sock.sendall(header + data)
        if command == s.LMCACHE_CLIENT_PUT:
            return b""
        meta = sock.recv(s.LMCACHE_SERVER_META.size)
        code, length, *_rest = s.LMCACHE_SERVER_META.unpack(meta)
        body = sock.recv(length) if length else b""
        return code, body


def test_lite_lmcache_server_put_get_and_flush(tmp_path):
    port = _free_port()
    log = tmp_path / "lmcache.log"
    proc = subprocess.Popen([sys.executable, "queue-haul/stage1b_drain_sink.py", "lmcache-server", "--host", "127.0.0.1", "--port", str(port)], stdout=log.open("w"), stderr=subprocess.STDOUT, start_new_session=True)
    try:
        s.wait_tcp_process("127.0.0.1", port, 5, proc, log)
        _lmc_request(port, s.LMCACHE_CLIENT_PUT, data=b"abc")
        for _ in range(20):
            if _lmc_request(port, s.LMCACHE_CLIENT_GET) == (s.LMCACHE_SERVER_SUCCESS, b"abc"):
                break
            time.sleep(0.05)
        else:
            pytest.fail("PUT was not visible to GET")
        os.kill(proc.pid, signal.SIGUSR1)
        time.sleep(0.2)
        assert _lmc_request(port, s.LMCACHE_CLIENT_GET) == (s.LMCACHE_SERVER_FAIL, b"")
    finally:
        s.stop_proc(proc)


def test_duplicate_ports_and_passthrough_overrides_hard_fail():
    with pytest.raises(ValueError, match="duplicate ports"):
        s.validate_ports(s.Config(src_port=8100, sink_port=8100))

    with pytest.raises(ValueError, match="duplicates typed flag"):
        s.vllm_cmd(s.Config(), "source", ["--max-model-len=4096"])

    with pytest.raises(ValueError, match="unknown role"):
        s.vllm_cmd(s.Config(), "bad")


def test_token_bucket_reserves_one_shared_timeline():
    bucket = s.TokenBucket(rate_bps=100.0, burst_s=1.0)
    bucket.updated = 0.0

    assert bucket.reserve(50, 0.0) == pytest.approx(0.5)
    assert bucket.updated == pytest.approx(0.5)
    assert bucket.reserve(50, 0.0) == pytest.approx(0.5)
    assert bucket.updated == pytest.approx(1.0)
    assert bucket.reserve(75, 2.0) == 0


def test_source_egress_billing_directions_only():
    assert s.billable("api", "client_to_target")
    assert s.billable("kv", "target_to_client")
    assert not s.billable("api", "target_to_client")
    assert not s.billable("kv", "client_to_target")


def test_proxy_relay_shapes_billable_bytes_and_logs(tmp_path):
    async def run():
        payload = b"x" * 512
        got = asyncio.get_running_loop().create_future()

        async def target(reader, writer):
            got.set_result(await reader.readexactly(len(payload)))
            writer.close()
            await writer.wait_closed()

        target_server = await asyncio.start_server(target, "127.0.0.1", 0)
        target_port = target_server.sockets[0].getsockname()[1]
        log = tmp_path / "proxy.csv"
        servers, byte_log = await s.start_proxy(
            [s.Route("api", "127.0.0.1", 0, "127.0.0.1", target_port)],
            rate_bps=2048.0,
            log=log,
        )
        proxy_port = servers[0].sockets[0].getsockname()[1]
        _reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        t0 = time.monotonic()
        writer.write(payload)
        await writer.drain()
        writer.write_eof()
        assert await got == payload
        elapsed = time.monotonic() - t0
        writer.close()
        await writer.wait_closed()
        for server in servers:
            server.close()
            await server.wait_closed()
        target_server.close()
        await target_server.wait_closed()
        byte_log.close()
        return elapsed, log

    elapsed, log = asyncio.run(run())

    assert elapsed >= 0.20
    rows = list(csv.DictReader(log.open()))
    assert sum(int(r["bytes"]) for r in rows if r["direction"] == "client_to_target") == 512
    assert {r["route"] for r in rows} == {"api"}
    assert {r["billed"] for r in rows} == {"1"}


def test_smoke2_live_cli_is_wired_with_1gbps_default():
    args = s.parse_args(["smoke2-live", "--run-root", "/tmp/live-proof"])

    assert args.cmd == "smoke2-live"
    assert args.mbps == 1000.0
    assert str(args.run_root) == "/tmp/live-proof"
