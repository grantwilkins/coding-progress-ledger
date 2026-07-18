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
- Trust reset HTTP 200 after vLLM logged that the reset failed.
- Hide transfers in a private CPU cache or log every socket read as a sample.
"""

from __future__ import annotations

import asyncio
import csv
import json
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
    assert "CUDA_VISIBLE_DEVICES=0" in source
    assert "APPTAINERENV_CUDA_VISIBLE_DEVICES=0" in source
    assert "NVIDIA_VISIBLE_DEVICES=0" in source
    assert "APPTAINERENV_CUDA_VISIBLE_DEVICES=1" in sink
    assert "NVIDIA_VISIBLE_DEVICES=1" in sink
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
    assert "VLLM_SERVER_DEV_MODE=1" in source
    assert "PYTHONPATH=" in source and "lmcache_compat" in source
    assert "nvidia/cu13/lib" in source
    assert "${LD_LIBRARY_PATH:-}" in source
    assert "LMCACHE_LOCAL_CPU=False" in source
    assert "LMCACHE_MAX_LOCAL_CPU_SIZE=4" in source
    assert "TMPDIR=/tmp/qh-src-" in source
    assert "TMPDIR=/tmp/qh-sink-" in sink
    assert "VLLM_RPC_BASE_PATH=/tmp/qh-src-" in source
    assert "VLLM_RPC_BASE_PATH=/tmp/qh-sink-" in sink
    assert "--enforce-eager" in source
    assert "--enable-sleep-mode" in source
    assert "--enable-sleep-mode" not in sink
    assert "--enable-sleep-mode" not in smoke
    assert "--disable-frontend-multiprocessing" not in source
    assert "--async-scheduling" not in source
    assert "stage1b-src" in source and "stage1b-sink" in sink


def test_vllm_commands_honor_slurm_gpu_ids(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    source = cmd_text(s.vllm_cmd(s.Config(), "source"))
    sink = cmd_text(s.vllm_cmd(s.Config(), "sink"))

    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "CUDA_VISIBLE_DEVICES=3" in sink
    assert s.gpu_count() == 2


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


def test_lmcache_health_wait_reports_process_exit_log(tmp_path):
    log = tmp_path / "vllm.log"
    log.write_text("health died\n")
    proc = subprocess.Popen(["false"])
    proc.wait()
    with pytest.raises(RuntimeError, match="health died"):
        s.wait_health_process("127.0.0.1", 1, 5, proc, log)


def test_custom_apptainer_image_path_is_wired():
    source = cmd_text(s.vllm_cmd(s.Config(sandbox="/tmp/qh.sif"), "source"))

    assert "/tmp/qh.sif" in source


def test_cli_sandbox_default_honors_env(monkeypatch):
    monkeypatch.setenv("QH_APPTAINER_IMAGE", "/tmp/qh-env.sif")
    args = s.parse_args(["preflight"])

    assert str(args.sandbox) == "/tmp/qh-env.sif"


def test_port_offset_honors_env_for_config_and_cli(monkeypatch):
    monkeypatch.setenv("QH_PORT_OFFSET", "100")
    cfg = s.Config()
    args = s.parse_args(["preflight"])

    assert cfg.src_port == 8200
    assert cfg.sink_port == 8300
    assert cfg.lmc_port == 5755
    assert cfg.kv_proxy_port == 8400
    assert cfg.api_proxy_port == 8500
    assert cfg.smoke_port == 8220
    assert args.src_port == 8200
    assert args.lmc_port == 5755
    assert "\"kv_port\":14679" in cmd_text(s.vllm_cmd(cfg, "source"))
    assert "\"kv_port\":14680" in cmd_text(s.vllm_cmd(cfg, "sink"))


def test_port_offset_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("QH_PORT_OFFSET", "60000")
    with pytest.raises(ValueError, match="invalid QH_PORT_OFFSET"):
        s.Config()


def test_apptainer_gpu_mode_can_use_nvccli(monkeypatch):
    monkeypatch.setenv("QH_APPTAINER_GPU_MODE", "nvccli")

    cmd = s.vllm_cmd(s.Config(), "source")
    assert "--nvccli" in cmd
    assert "--nv" not in cmd
    monkeypatch.setenv("QH_APPTAINER_GPU_MODE", "bad")
    with pytest.raises(ValueError, match="unknown QH_APPTAINER_GPU_MODE"):
        s.vllm_cmd(s.Config(), "source")


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
        s.flush_lmcache(s.Stack(proc, None, None, None, tmp_path))
        assert s.LMCACHE_CLEAR_MARKER in log.read_text()
        assert _lmc_request(port, s.LMCACHE_CLIENT_GET) == (s.LMCACHE_SERVER_FAIL, b"")
    finally:
        s.stop_proc(proc)


def test_reset_vllm_caches_requires_success_from_both_logs(monkeypatch, tmp_path):
    calls = []
    logs = tmp_path / "source.log", tmp_path / "sink.log"
    for log in logs:
        log.write_text("")

    def reset(host, port, method, path):
        calls.append((host, port, method, path))
        logs[port == 8200].write_text("Successfully reset prefix cache\n")

    monkeypatch.setattr(s, "http_text", reset)

    s.reset_vllm_caches(s.Config(), logs)

    assert calls == [
        ("127.0.0.1", 8100, "POST", "/reset_prefix_cache"),
        ("127.0.0.1", 8200, "POST", "/reset_prefix_cache"),
    ]


def test_reset_result_rejects_failed_http_200_log():
    assert s.reset_result("Failed to reset prefix cache because blocks remain") is False
    assert s.reset_result("Successfully reset prefix cache") is True
    assert s.reset_result("POST /reset_prefix_cache 200 OK") is None


@pytest.mark.parametrize(("initial", "target", "path"), [(False, True, "/sleep?level=1"), (True, False, "/wake_up")])
def test_source_sleep_transitions_are_verified(monkeypatch, initial, target, path):
    states, calls = iter([initial, target]), []

    def fake_http(host, port, method, path):
        calls.append((host, port, method, path))
        return json.dumps({"is_sleeping": next(states)}) if method == "GET" else ""

    monkeypatch.setattr(s, "http_text", fake_http)
    s.set_source_sleep(s.Config(), target)

    assert calls == [
        ("127.0.0.1", 8100, "GET", "/is_sleeping"),
        ("127.0.0.1", 8100, "POST", path),
        ("127.0.0.1", 8100, "GET", "/is_sleeping"),
    ]


def test_source_sleep_transition_hard_fails(monkeypatch):
    monkeypatch.setattr(s, "http_text", lambda *_args: '{"is_sleeping": false}')

    with pytest.raises(RuntimeError, match="sleeping"):
        s.set_source_sleep(s.Config(), True)


def test_runtime_versions_are_pinned(monkeypatch):
    commands = []
    monkeypatch.setattr(s.subprocess, "check_output", lambda command, **_kwargs: commands.append(command) or "QH_RUNTIME_VERSIONS 0.10.1.1 0.3.3\n")

    assert s.runtime_versions(s.Config()) == s.RUNTIME_VERSIONS
    assert "LMCServerConnector._qh_patched" in s.shell(commands[0])
    assert "LMCacheConnectorV1Impl._qh_bypass_patched" in s.shell(commands[0])


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


def test_token_bucket_credits_scheduler_overshoot():
    bucket = s.TokenBucket(rate_bps=1_000_000.0)
    bucket.updated = 0.0

    assert bucket.reserve(100, 0.0) == pytest.approx(0.0001)
    assert bucket.reserve(100, 0.001) == 0


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
        await byte_log.close()
        return elapsed, log

    elapsed, log = asyncio.run(run())

    assert elapsed >= 0.20
    rows = list(csv.DictReader(log.open()))
    assert sum(int(r["bytes"]) for r in rows if r["direction"] == "client_to_target") == 512
    assert {r["route"] for r in rows} == {"api"}
    assert {r["billed"] for r in rows} == {"1"}
    assert len(rows) <= 2
    connections = list(csv.DictReader((tmp_path / "proxy_connections.csv").open()))
    assert len(connections) == 1
    assert int(connections[0]["client_to_target_bytes"]) == 512


def test_smoke2_live_cli_is_wired_with_1gbps_default():
    args = s.parse_args(["smoke2-live", "--run-root", "/tmp/live-proof"])

    assert args.cmd == "smoke2-live"
    assert args.mbps == 1000.0
    assert str(args.run_root) == "/tmp/live-proof"
