"""
Claim:
Stage 1b starts from the validated old Apptainer sandbox path, keeps source and
sink vLLM instances separate, and replaces privileged kernel tc with one
user-space bandwidth bucket shared by the KV and API proxy routes.

Plausible wrong implementations:
- Reintroduce Docker/latest-cu129 or LMCacheMPConnector.
- Start source/sink with colliding ports, long TMPDIRs, or shared cache dirs.
- Shape each route independently instead of enforcing one shared link.
- Generate a two-GPU plan that is not runnable as host orchestration around
  Apptainer children.
"""

from __future__ import annotations

import asyncio
import csv
import time
from pathlib import Path

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
    assert "VLLM_USE_FLASHINFER_SAMPLER=0" in source
    assert "LMCACHE_MAX_LOCAL_CPU_SIZE=0.25" in source
    assert "TMPDIR=/tmp/t" in source
    assert "--enforce-eager" in source
    assert "--async-scheduling" not in source
    assert "stage1b-src" in source and "stage1b-sink" in sink


def test_lmcache_and_plan_use_host_proxy_not_docker_or_tc():
    cfg = s.Config()

    lmcache = cmd_text(s.lmcache_cmd(cfg))
    text = s.plan_text(cfg, "smoke2", 100.0, [])

    assert "python3 -m lmcache.v1.server 127.0.0.1 5655 cpu" in lmcache
    assert "lmcache server --host" not in lmcache
    assert "stage1b smoke2: host orchestrator" in text
    assert "stage1b_drain_sink.py proxy" in text
    assert "--kv-listen 127.0.0.1:8300 --kv-target 127.0.0.1:5655" in text
    assert "--api-listen 127.0.0.1:8400 --api-target 127.0.0.1:8200" in text
    assert "docker" not in text.lower()
    assert " tc " not in text.lower()


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


def test_proxy_relay_shapes_bytes_and_logs(tmp_path):
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
            [s.Route("kv", "127.0.0.1", 0, "127.0.0.1", target_port)],
            rate_bps=2048.0,
            log=log,
        )
        proxy_port = servers[0].sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
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
    assert {r["route"] for r in rows} == {"kv"}
