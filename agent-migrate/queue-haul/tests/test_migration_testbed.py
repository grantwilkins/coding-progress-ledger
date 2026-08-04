"""
Claim:
The migration testbed pins each validated legacy/MP runtime, keeps source and
sink vLLM instances separate, and replaces privileged kernel tc with one
user-space bandwidth bucket shared by the source-egress KV/API proxy routes.

Plausible wrong implementations:
- Run the bounded campaign on the legacy connector or an unpinned MP image.
- Start source/sink with colliding ports, long TMPDIRs, or shared cache dirs.
- Shape each route independently instead of enforcing one shared link.
- Bill both directions instead of the simulated source-egress directions.
- Aggregate concurrent connections so apparent overlap cannot be attributed.
- Read MP source keys before asynchronous chunk storage completes.
- Trust reset HTTP 200 after vLLM logged that the reset failed.
- Rely on a release-specific prefix-cache default for warm-session measurements.
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
from pathlib import Path

import pytest

import migration_testbed as s


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
    assert "--enable-prefix-caching" in source
    assert "--enable-sleep-mode" in source
    assert "--enable-sleep-mode" not in sink
    assert "--enable-sleep-mode" not in smoke
    assert "--disable-frontend-multiprocessing" not in source
    assert "--async-scheduling" not in source
    assert "stage1b-src" in source and "stage1b-sink" in sink


def test_mp_runtime_uses_release_image_and_shipped_connector(monkeypatch):
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    source = cmd_text(s.vllm_cmd(s.Config(), "source"))

    assert "lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif" in source
    assert "LMCacheMPConnector" in source
    assert "lmcache.integration.vllm.lmcache_mp_connector" in source
    assert "lmcache.mp.host" in source and "lmcache.mp.port" in source
    assert "engine_driven" in source
    assert "lmcache_compat" not in source
    assert "cuda-12.9/compat" in source
    assert "--gpu-memory-utilization 0.75" in source
    assert "--block-size 16" in source
    assert "--disable-hybrid-kv-cache-manager" in source
    assert "--enable-prompt-tokens-details" in source
    assert "--enable-sleep-mode" not in source
    assert s.expected_runtime_versions() == ("0.22.0+cu129", "0.5.1")


def test_native_mp_runtime_uses_host_stack_and_gpu_assignment(monkeypatch):
    monkeypatch.setenv("QH_RUNTIME", "native")
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")

    source = cmd_text(s.vllm_cmd(s.Config(), "source"))
    sink = cmd_text(s.vllm_cmd(s.Config(), "sink"))
    cache = cmd_text(s.mp_server_cmd(s.Config(), "source"))
    redis = cmd_text(s.redis_cmd(s.Config()))

    assert "apptainer" not in source + sink + cache + redis
    assert "CUDA_VISIBLE_DEVICES=2" in source
    assert "CUDA_VISIBLE_DEVICES=3" in sink
    assert str(Path(s.sys.executable).parent) in source
    assert "LD_LIBRARY_PATH" not in source
    assert redis.startswith("valkey-server --bind 127.0.0.1 --port 5655")
    assert s.expected_runtime_versions() == ("0.22.0", "0.5.1")


def test_native_runtime_rejects_legacy_and_unknown_modes(monkeypatch):
    monkeypatch.setenv("QH_RUNTIME", "native")
    with pytest.raises(ValueError, match="requires QH_LMCACHE_MODE=mp"):
        s.runtime_mode()

    monkeypatch.setenv("QH_RUNTIME", "other")
    with pytest.raises(ValueError, match="unknown QH_RUNTIME"):
        s.runtime_mode()


def test_native_preflight_requires_host_commands_and_pinned_versions(monkeypatch, tmp_path):
    monkeypatch.setenv("QH_RUNTIME", "native")
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").touch()
    monkeypatch.setattr(s, "model_snapshot_dir", lambda *_args: snapshot)
    monkeypatch.setattr(s, "port_free", lambda *_args: True)
    monkeypatch.setattr(s, "gpu_count", lambda: 2)
    monkeypatch.setattr(s.shutil, "which", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError) as error:
        s.preflight(s.Config(), 2)
    for executable in ("vllm", "lmcache", "valkey-server"):
        assert f"{executable} not found" in str(error.value)

    monkeypatch.setattr(s.shutil, "which", lambda executable, **_kwargs: f"/bin/{executable}")
    monkeypatch.setattr(s, "runtime_versions", lambda _cfg: ("0.22.1", "0.5.1"))
    with pytest.raises(RuntimeError, match="need vLLM/LMCache.*0.22.1"):
        s.preflight(s.Config(), 2)


def test_health_timeout_is_configurable(monkeypatch):
    monkeypatch.setenv("QH_HEALTH_TIMEOUT_S", "7200")
    assert s.health_timeout() == 7200


def test_mp_tokenization_uses_the_exact_chat_completion_renderer(monkeypatch):
    seen = {}

    def request(_host, _port, _method, path, payload):
        seen["path"] = path
        seen.update(payload)
        return {"token_ids": [1, 2]}

    monkeypatch.setattr(s, "http_json", request)
    messages = [{"role": "user", "content": "state"}]

    assert s.mp_chat_tokens(s.Config(), messages) == [1, 2]
    assert seen == {
        "path": "/v1/chat/completions/render", "model": s.Config().model,
        "messages": messages, "max_tokens": 512, "temperature": 0,
        "reasoning_effort": "low", "stream": True,
        "stream_options": {"include_usage": True},
    }


def test_resp_transfer_is_immediately_visible(tmp_path):
    async def check():
        log = s.ByteLog(tmp_path / "proxy_bytes.csv")
        await log.resp_transfer(["c", "SET", "k", 1, 2, 3, 4, 5, 6])
        assert list(csv.DictReader((tmp_path / "resp_transfers.csv").open()))[0]["key_hashes"] == "k"
        await log.close()

    asyncio.run(check())


def test_mp_storage_wait_aggregates_chunked_writes(tmp_path):
    log = tmp_path / "lmcache.log"
    prefix = "LMCache ✓\n".encode()
    log.write_bytes(prefix + b"Stored 8192 tokens\nStored 4096 tokens\n")

    s.mp_wait_stored(log, len(prefix), 12288)


def test_mp_storage_wait_requires_exact_resp_set_keys(tmp_path):
    log = tmp_path / "lmcache.log"
    log.write_text("Stored 512 tokens\n")
    transfers = tmp_path / "resp_transfers.csv"
    transfers.write_text(
        "connection_id,command,key_hashes,start_ns,end_ns,request_wire_bytes,"
        "response_wire_bytes,request_body_bytes,payload_bytes\n"
        "a,SET,old,0,1,1,1,1,1\n"
    )
    offset = transfers.stat().st_size
    with transfers.open("a") as handle:
        handle.write("a,SET,k1,0,3,1,1,1,1\na,SET,k2,4,5,1,1,1,1\n")

    assert s.mp_wait_source_keys(log, 0, transfers, offset, 512) == {"k1", "k2"}


def test_mp_wait_idle_uses_bounded_stability(tmp_path):
    transfers = tmp_path / "resp_transfers.csv"
    transfers.write_text("header\n")
    started = time.monotonic()
    s.mp_wait_idle(transfers, .01)
    assert time.monotonic() - started >= .01


def test_mp_source_keys_excludes_known_keys(tmp_path):
    log = tmp_path / "lmcache.log"
    log.write_text("")
    transfers = tmp_path / "resp_transfers.csv"
    transfers.write_text(
        "connection_id,command,key_hashes,start_ns,end_ns,request_wire_bytes,"
        "response_wire_bytes,request_body_bytes,payload_bytes\n"
        "a,SET,old,0,1,1,1,1,1\na,SET,k1,2,3,1,1,1,1\n"
    )
    assert s.mp_wait_source_keys(log, 0, transfers, 0, 256, {"old"}) == {"k1"}


def test_mp_source_keys_include_generated_tail_key(tmp_path):
    log = tmp_path / "lmcache.log"
    log.write_text("Stored 512 tokens\n")
    transfers = tmp_path / "resp_transfers.csv"
    transfers.write_text(
        "connection_id,command,key_hashes,start_ns,end_ns,request_wire_bytes,"
        "response_wire_bytes,request_body_bytes,payload_bytes\n"
        "a,SET,k1,0,1,1,1,1,1\na,SET,k2,2,3,1,1,1,1\n"
        "a,SET,generated,4,5,1,1,1,1\n"
    )

    assert s.mp_wait_source_keys(log, 0, transfers, 0, 512) == {
        "k1", "k2", "generated",
    }


def test_mp_request_hit_uses_byte_offset(tmp_path):
    log = tmp_path / "lmcache.log"
    prefix = "LMCache ✓\n".encode()
    log.write_bytes(prefix + b"2/2 retained keys (2 L1, 0 L2), external_request_id=req,\n")

    assert s.mp_request_hit(log, len(prefix), "req") == 512
    log.write_text("1/2 retained keys (1 L1, 0 L2), external_request_id=req,\n")
    assert s.mp_request_hit(log, 0, "req", False) == 256


def test_bounded_campaign_pins_validated_mp_transport():
    text = Path(s.__file__).with_name("bounded_hardware_campaign.sbatch").read_text()

    assert "QH_LMCACHE_MODE=mp" in text
    assert "lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif" in text


def test_mp_chat_disables_reasoning_without_changing_legacy(monkeypatch):
    cfg = s.Config()
    assert "reasoning_effort" not in json.loads(s.chat_payload(cfg, "x"))
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    assert json.loads(s.chat_payload(cfg, "x"))["reasoning_effort"] == "low"


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

    assert "migration_testbed.py lmcache-server --host 127.0.0.1 --port 5655" in lmcache
    assert "apptainer" not in lmcache
    assert "--nv" not in lmcache
    assert "APPTAINERENV_CUDA_VISIBLE_DEVICES" not in lmcache
    assert "lmcache.v1.server" not in lmcache
    assert "migration_testbed.py proxy" in proxy
    assert "--kv-listen 127.0.0.1:8300 --kv-target 127.0.0.1:5655" in proxy
    assert "--api-listen 127.0.0.1:8400 --api-target 127.0.0.1:8200" in proxy
    assert "--mbps 1000.0" in proxy
    assert "docker" not in proxy.lower()
    assert " tc " not in proxy.lower()


def test_mp_cache_services_use_redis_l2_through_proxy(monkeypatch):
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    cfg = s.Config()
    redis = cmd_text(s.redis_cmd(cfg))
    source = cmd_text(s.mp_server_cmd(cfg, "source"))
    sink = cmd_text(s.mp_server_cmd(cfg, "sink"))

    assert "redis-7.4.2-bookworm.sif" in redis
    assert "--port 5655" in redis
    assert "lmcache server" in source and "--port 5555" in source
    assert "--supported-transfer-mode engine_driven" in source
    assert '"port":8300' in source
    assert "--port 5556" in sink
    assert '"port":8300' in sink
    assert "--http-port 8080" in source and "--http-port 8081" in sink
    assert "--nv" not in source and "CUDA_VISIBLE_DEVICES=" in source


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
    proc = subprocess.Popen(
        [sys.executable, s.__file__, "lmcache-server", "--host", "127.0.0.1",
         "--port", str(port)], stdout=log.open("w"),
        stderr=subprocess.STDOUT, start_new_session=True,
    )
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


def test_mp_flush_uses_explicit_config(monkeypatch, tmp_path):
    cfg = s.Config(lmc_port=6380, src_lmc_http_port=8088, sink_lmc_http_port=8089)
    calls = []

    class Sock:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def sendall(self, data):
            calls.append(data)

        def recv(self, _size):
            return b"+OK\r\n"

    monkeypatch.setattr(s, "lmcache_mode", lambda: "mp")
    monkeypatch.setattr(s.socket, "create_connection", lambda address: calls.append(address) or Sock())
    monkeypatch.setattr(s, "http_text", lambda host, port, method, path: calls.append((host, port, method, path)))
    s.flush_lmcache(s.Stack(type("Proc", (), {"poll": lambda self: None})(), None, None, None, tmp_path), cfg)

    assert (cfg.host, 6380) in calls
    assert (cfg.host, 8088, "POST", "/cache/clear") in calls
    assert (cfg.host, 8089, "POST", "/cache/clear") in calls


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
    assert bucket.reserve(50, 0.0) == pytest.approx(1.0)
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


def test_kv_connection_key_hash_ignores_protocol_padding():
    key = b"session-block"
    header = s.LMCACHE_CLIENT_META.pack(
        s.LMCACHE_CLIENT_GET, 0, 1, 2, 0, 0, 0, 0, 0,
        key.ljust(s.LMCACHE_MAX_KEY_LENGTH, b"\0"),
    )

    assert s.kv_key_hash(header) == s.hashlib.sha256(key).hexdigest()


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
    assert len({r["connection_id"] for r in rows}) == 1
    assert len(rows) <= 2
    connections = list(csv.DictReader((tmp_path / "proxy_connections.csv").open()))
    assert len(connections) == 1
    assert int(connections[0]["client_to_target_bytes"]) == 512
    assert connections[0]["key_hash"] == ""



def test_resp_proxy_attributes_and_shapes_each_returned_body(tmp_path):
    async def run():
        async def target(reader, writer):
            for body in (b"a" * 32, b"b" * 48):
                await s.read_resp(reader)
                writer.write(b"$" + str(len(body)).encode() + b"\r\n" + body + b"\r\n")
                await writer.drain()
            writer.close()
            await writer.wait_closed()

        target_server = await asyncio.start_server(target, "127.0.0.1", 0)
        target_port = target_server.sockets[0].getsockname()[1]
        log = tmp_path / "proxy.csv"
        servers, byte_log = await s.start_proxy(
            [s.Route("kv", "127.0.0.1", 0, "127.0.0.1", target_port, "resp")],
            rate_bps=320.0,
            log=log,
        )
        proxy_port = servers[0].sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", proxy_port)
        t0 = time.monotonic()
        for key in (b"session-a", b"session-b"):
            writer.write(b"*2\r\n$3\r\nGET\r\n$" + str(len(key)).encode() + b"\r\n" + key + b"\r\n")
        await writer.drain()
        assert (await s.read_resp(reader)).payload == b"a" * 32
        assert (await s.read_resp(reader)).payload == b"b" * 48
        elapsed = time.monotonic() - t0
        writer.close()
        await writer.wait_closed()
        for server in servers:
            server.close()
            await server.wait_closed()
        target_server.close()
        await target_server.wait_closed()
        await byte_log.close()
        return elapsed

    assert asyncio.run(run()) >= .2
    rows = list(csv.DictReader((tmp_path / "resp_transfers.csv").open()))
    assert [row["command"] for row in rows] == ["GET", "GET"]
    assert [int(row["payload_bytes"]) for row in rows] == [32, 48]
    assert [row["key_hashes"] for row in rows] == [
        s.hashlib.sha256(key).hexdigest() for key in (b"session-a", b"session-b")
    ]
    assert int(rows[0]["start_ns"]) < int(rows[1]["end_ns"])

def test_smoke2_live_cli_is_wired_with_1gbps_default():
    args = s.parse_args(["smoke2-live", "--run-root", "/tmp/live-proof"])

    assert args.cmd == "smoke2-live"
    assert args.mbps == 1000.0
    assert str(args.run_root) == "/tmp/live-proof"
