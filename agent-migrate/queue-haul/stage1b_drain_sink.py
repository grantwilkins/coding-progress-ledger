from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
from collections import OrderedDict
import http.client
import json
import os
import shlex
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from stage1_curves import shell

MODEL = "openai/gpt-oss-20b"
SANDBOX = Path("/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox")
RUNTIME_VERSIONS = ("0.10.1.1", "0.3.3")
LMCACHE_CLEAR_MARKER = '"operation":"clear"'


def apptainer_image_default() -> Path:
    return Path(os.environ.get("QH_APPTAINER_IMAGE", SANDBOX))


def port_offset() -> int:
    offset = int(os.environ.get("QH_PORT_OFFSET", "0"))
    if offset < 0 or offset > 50000:
        raise ValueError(f"invalid QH_PORT_OFFSET: {offset}")
    return offset


def port_default(base: int) -> int:
    return base + port_offset()


HF_HOME = Path("/scratch/users/gfw/ptsim/hf")
SCRATCH_BIND = Path("/scratch/users/gfw")
CACHE_ROOT = Path("/scratch/users/gfw/ptsim/cache")
LMCACHE_COMPAT = Path(__file__).with_name("lmcache_compat").resolve()
CHUNK = 65536
LMCACHE_MAX_LOCAL_CPU_GB = "4"
TYPED_VLLM_FLAGS = {
    "--host",
    "--port",
    "--served-model-name",
    "--tensor-parallel-size",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--kv-cache-dtype",
    "--enable-chunked-prefill",
    "--enable-sleep-mode",
    "--enforce-eager",
    "--kv-transfer-config",
}
BILLED_DIRECTIONS = {("api", "client_to_target"), ("kv", "target_to_client")}
LMCACHE_MAX_KEY_LENGTH = 150
LMCACHE_CLIENT_META = struct.Struct(f"iiiiiiiii{LMCACHE_MAX_KEY_LENGTH}s")
LMCACHE_SERVER_META = struct.Struct("iiiiiiiii")
LMCACHE_CLIENT_PUT = 1
LMCACHE_CLIENT_GET = 2
LMCACHE_CLIENT_EXIST = 3
LMCACHE_CLIENT_HEALTH = 5
LMCACHE_SERVER_SUCCESS = 200
LMCACHE_SERVER_FAIL = 400
LMCACHE_FAIL_PAYLOAD = (LMCACHE_SERVER_FAIL, 0, 1, 2, 0, 0, 0, 0, 0)
LMCACHE_OK_PAYLOAD = (LMCACHE_SERVER_SUCCESS, 0, 1, 2, 0, 0, 0, 0, 0)
LMCACHE_SERVER_MAX_BYTES = int(os.environ.get("QH_LMCACHE_SERVER_MAX_BYTES", "0"))


@dataclass(frozen=True)
class Config:
    model: str = MODEL
    sandbox: Path = field(default_factory=apptainer_image_default)
    hf_home: Path = HF_HOME
    scratch_bind: Path = SCRATCH_BIND
    cache_root: Path = CACHE_ROOT
    host: str = "127.0.0.1"
    src_port: int = field(default_factory=lambda: port_default(8100))
    sink_port: int = field(default_factory=lambda: port_default(8200))
    lmc_port: int = field(default_factory=lambda: port_default(5655))
    kv_proxy_port: int = field(default_factory=lambda: port_default(8300))
    api_proxy_port: int = field(default_factory=lambda: port_default(8400))
    smoke_port: int = field(default_factory=lambda: port_default(8120))
    max_model_len: int = 32768
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192


@dataclass(frozen=True)
class Route:
    name: str
    listen_host: str
    listen_port: int
    target_host: str
    target_port: int


@dataclass
class Stack:
    lmcache: subprocess.Popen
    proxy: subprocess.Popen
    source: subprocess.Popen | None
    sink: subprocess.Popen | None
    run_root: Path


def reject_duplicate_extra(extra: list[str]) -> None:
    for tok in extra:
        if tok.split("=", 1)[0] in TYPED_VLLM_FLAGS:
            raise ValueError(f"extra vLLM arg duplicates typed flag: {tok}")


def parse_addr(text: str) -> tuple[str, int]:
    host, port = text.rsplit(":", 1)
    if not host or not port.isdigit():
        raise ValueError(f"address must be host:port: {text}")
    return host, int(port)


def validate_ports(cfg: Config) -> None:
    ports = [cfg.src_port, cfg.sink_port, cfg.lmc_port, cfg.kv_proxy_port, cfg.api_proxy_port, cfg.smoke_port]
    bad = [p for p in ports if p <= 0 or p > 65535]
    if bad:
        raise ValueError(f"invalid ports: {bad}")
    dupes = sorted({p for p in ports if ports.count(p) > 1})
    if dupes:
        raise ValueError(f"duplicate ports: {dupes}")


def model_snapshot_dir(hf_home: Path, model: str) -> Path:
    return hf_home / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"


def cache_dirs(cfg: Config, role: str) -> dict[str, Path]:
    root = cfg.cache_root / f"stage1b-{role}"
    return {
        "XDG_CACHE_HOME": root / "xdg",
        "TORCH_EXTENSIONS_DIR": root / "torch_extensions",
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
        "TRITON_CACHE_DIR": root / "triton",
    }


def tmpdir(role: str) -> Path:
    tag = {"smoke1": "smk"}.get(role, role)
    return Path(f"/tmp/qh-{tag}-{os.getpid()}")


def kv_config(engine_id: str, kv_role: str, kv_port: int, rpc_port: str) -> str:
    return json.dumps(
        {
            "kv_connector": "LMCacheConnectorV1",
            "engine_id": engine_id,
            "kv_role": kv_role,
            "kv_port": kv_port,
            "kv_connector_extra_config": {"discard_partial_chunks": False, "lmcache_rpc_port": rpc_port},
        },
        separators=(",", ":"),
    )


def vllm_exports(cfg: Config, role: str, remote_url: str) -> list[str]:
    env = {
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(LMCACHE_COMPAT),
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "TMPDIR": str(tmpdir(role)),
        "VLLM_RPC_BASE_PATH": str(tmpdir(role)),
        "HF_HOME": str(cfg.hf_home),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LMCACHE_REMOTE_URL": remote_url,
        "LMCACHE_REMOTE_SERDE": "naive",
        "LMCACHE_LMCACHE_INSTANCE_ID": f"stage1b_{role}",
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_LOCAL_CPU": "False",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": LMCACHE_MAX_LOCAL_CPU_GB,
        **{k: str(v) for k, v in cache_dirs(cfg, role).items()},
    }
    exports = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    exports.append("export LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:/usr/local/cuda-12.9/targets/x86_64-linux/lib:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}")
    return exports


def apptainer_cmd(cfg: Config, script: str, gpu: int | None = None, nv: bool = True) -> list[str]:
    cmd = ["apptainer", "exec", "--bind", f"{cfg.scratch_bind}:{cfg.scratch_bind}", cfg.sandbox, "bash", "-lc", script]
    if nv:
        mode = os.environ.get("QH_APPTAINER_GPU_MODE", "nv")
        if mode not in {"nv", "nvccli"}:
            raise ValueError(f"unknown QH_APPTAINER_GPU_MODE: {mode}")
        cmd.insert(2, f"--{mode}")
    if gpu is None:
        return cmd
    return ["env", f"CUDA_VISIBLE_DEVICES={gpu}", f"APPTAINERENV_CUDA_VISIBLE_DEVICES={gpu}", f"NVIDIA_VISIBLE_DEVICES={gpu}", *cmd]


def lmcache_cmd(cfg: Config) -> list[str]:
    return [sys.executable, "queue-haul/stage1b_drain_sink.py", "lmcache-server", "--host", cfg.host, "--port", str(cfg.lmc_port), "--max-bytes", str(LMCACHE_SERVER_MAX_BYTES)]


def vllm_cmd(cfg: Config, role: str, extra: list[str] | None = None) -> list[str]:
    reject_duplicate_extra(extra or [])
    if role == "source":
        port, gpu, engine_id, kv_role, kv_port, rpc_port = cfg.src_port, 0, "s0", "kv_producer", 14579, "src"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.lmc_port}", "src"
    elif role == "sink":
        port, gpu, engine_id, kv_role, kv_port, rpc_port = cfg.sink_port, 1, "d0", "kv_consumer", 14580, "sink"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.kv_proxy_port}", "sink"
    elif role == "smoke1":
        port, gpu, engine_id, kv_role, kv_port, rpc_port = cfg.smoke_port, 0, "e0", "kv_both", 14579, "smk"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.lmc_port}", "smoke1"
    else:
        raise ValueError(f"unknown role: {role}")

    dirs = " ".join(shlex.quote(str(p)) for p in [tmpdir(cache_role), *cache_dirs(cfg, cache_role).values()])
    serve = [
        "vllm",
        "serve",
        cfg.model,
        "--host",
        cfg.host,
        "--port",
        port,
        "--served-model-name",
        cfg.model,
        "--tensor-parallel-size",
        1,
        "--max-model-len",
        cfg.max_model_len,
        "--max-num-seqs",
        cfg.max_num_seqs,
        "--max-num-batched-tokens",
        cfg.max_num_batched_tokens,
        "--kv-cache-dtype",
        "auto",
        "--enable-chunked-prefill",
        "--enforce-eager",
        *(["--enable-sleep-mode"] if role == "source" else []),
        "--kv-transfer-config",
        kv_config(engine_id, kv_role, kv_port, rpc_port),
        *(extra or []),
    ]
    script = "\n".join([f"mkdir -p {dirs}", *vllm_exports(cfg, cache_role, remote_url), shell(serve)])
    return apptainer_cmd(cfg, script, gpu)


def proxy_routes(cfg: Config) -> list[Route]:
    return [
        Route("kv", cfg.host, cfg.kv_proxy_port, cfg.host, cfg.lmc_port),
        Route("api", cfg.host, cfg.api_proxy_port, cfg.host, cfg.sink_port),
    ]


def proxy_cmd(cfg: Config, mbps: float = 1000.0, log: Path | None = None) -> list[str]:
    cmd = [
        sys.executable,
        "queue-haul/stage1b_drain_sink.py",
        "proxy",
        "--kv-listen",
        f"{cfg.host}:{cfg.kv_proxy_port}",
        "--kv-target",
        f"{cfg.host}:{cfg.lmc_port}",
        "--api-listen",
        f"{cfg.host}:{cfg.api_proxy_port}",
        "--api-target",
        f"{cfg.host}:{cfg.sink_port}",
        "--mbps",
        str(mbps),
    ]
    if log:
        cmd += ["--log", log]
    return cmd


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found")
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def runtime_versions(cfg: Config) -> tuple[str, str]:
    check = "from importlib.metadata import version; from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector; from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl; assert LMCServerConnector._qh_patched and LMCacheConnectorV1Impl._qh_bypass_patched; print(version('vllm'), version('lmcache'))"
    script = "\n".join([f"export PYTHONPATH={shlex.quote(str(LMCACHE_COMPAT))}", shell(["/usr/bin/python3", "-c", check])])
    vllm, lmcache = subprocess.check_output(apptainer_cmd(cfg, script, nv=False), text=True).split()
    return vllm, lmcache


def preflight(cfg: Config, required_gpus: int = 1) -> list[str]:
    validate_ports(cfg)
    failures = []
    if not shutil.which("apptainer"):
        failures.append("apptainer not found")
    if not cfg.sandbox.exists():
        failures.append(f"sandbox missing: {cfg.sandbox}")
    elif shutil.which("apptainer"):
        versions = runtime_versions(cfg)
        if versions != RUNTIME_VERSIONS:
            failures.append(f"need vLLM/LMCache {RUNTIME_VERSIONS}, saw {versions}")
    snapshots = model_snapshot_dir(cfg.hf_home, cfg.model)
    if not snapshots.exists() or not any(snapshots.iterdir()):
        failures.append(f"model snapshot missing: {snapshots}")
    for path in [tmpdir("src"), tmpdir("sink"), tmpdir("smoke1")]:
        if len(str(path)) > 20:
            failures.append(f"TMPDIR too long for LMCache IPC: {path}")
    ports = [cfg.lmc_port, cfg.smoke_port] if required_gpus == 1 else [cfg.src_port, cfg.sink_port, cfg.lmc_port, cfg.kv_proxy_port, cfg.api_proxy_port]
    for port in ports:
        if not port_free(cfg.host, port):
            failures.append(f"port busy: {cfg.host}:{port}")
    seen_gpus = gpu_count()
    if seen_gpus < required_gpus:
        failures.append(f"need {required_gpus} GPU(s), saw {seen_gpus}")
    if failures:
        raise RuntimeError("\n".join(failures))
    return [
        f"apptainer={shutil.which('apptainer')}",
        f"gpus={seen_gpus}",
        f"docker_present={bool(shutil.which('docker'))}",
        "real_tc=disabled_no_cap_net_admin",
        f"sandbox={cfg.sandbox}",
        f"vllm={RUNTIME_VERSIONS[0]}",
        f"lmcache={RUNTIME_VERSIONS[1]}",
        f"model_snapshots={snapshots}",
    ]


class TokenBucket:
    def __init__(self, rate_bps: float, burst_s: float = 0.0):
        if rate_bps <= 0:
            raise ValueError("rate_bps must be positive")
        self.rate = rate_bps
        self.capacity = rate_bps * burst_s
        self.tokens = 0.0
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    def reserve(self, nbytes: int, now: float) -> float:
        if nbytes < 0:
            raise ValueError("nbytes must be nonnegative")
        if now > self.updated:
            self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
        base = self.updated
        if self.tokens >= nbytes:
            self.tokens -= nbytes
            return 0.0
        delay = (nbytes - self.tokens) / self.rate
        self.tokens = 0.0
        self.updated = base + delay
        return delay

    async def wait(self, nbytes: int) -> None:
        async with self.lock:
            delay = self.reserve(nbytes, time.monotonic())
        if delay:
            await asyncio.sleep(delay)


class ByteLog:
    interval_ns = 250_000_000

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(["monotonic_ns", "wall_ns", "interval_ns", "route", "direction", "bytes", "billed"])
        connections = path.with_name("proxy_connections.csv")
        self.connections = connections.open("w", newline="", buffering=1)
        self.connection_writer = csv.writer(self.connections)
        self.connection_writer.writerow(["connection_id", "route", "start_ns", "end_ns", "client_to_target_bytes", "target_to_client_bytes"])
        self.buckets: dict[tuple[int, str, str, bool], int] = {}
        self.lock = asyncio.Lock()
        self.task: asyncio.Task | None = None
        self.active = 0
        self.idle = asyncio.Event()
        self.idle.set()

    async def start(self) -> None:
        self.task = asyncio.create_task(self._flush_loop())

    async def add(self, route: str, direction: str, nbytes: int, billed: bool) -> None:
        async with self.lock:
            bucket = time.monotonic_ns() // self.interval_ns * self.interval_ns
            key = bucket, route, direction, billed
            self.buckets[key] = self.buckets.get(key, 0) + nbytes

    async def opened(self) -> None:
        async with self.lock:
            self.active += 1
            self.idle.clear()

    async def connection(self, connection_id: str, route: str, start_ns: int, counts: tuple[int, int]) -> None:
        async with self.lock:
            self.connection_writer.writerow([connection_id, route, start_ns, time.monotonic_ns(), *counts])
            self.active -= 1
            if not self.active:
                self.idle.set()

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_ns / 1e9)
            await self.flush()

    async def flush(self, all_rows: bool = False) -> None:
        cutoff = time.monotonic_ns() // self.interval_ns * self.interval_ns
        async with self.lock:
            keys = [key for key in self.buckets if all_rows or key[0] < cutoff]
            for bucket, route, direction, billed in sorted(keys):
                self.writer.writerow([bucket, time.time_ns(), self.interval_ns, route, direction, self.buckets.pop((bucket, route, direction, billed)), int(billed)])

    async def close(self) -> None:
        await self.idle.wait()
        if self.task:
            self.task.cancel()
        await self.flush(all_rows=True)
        self.file.close()
        self.connections.close()


def billable(route: str, direction: str) -> bool:
    return (route, direction) in BILLED_DIRECTIONS


@dataclass
class LMCacheEntry:
    data: bytes
    length: int
    fmt: int
    dtype: int
    shape: tuple[int, int, int, int]
    location: int


class LiteLMCache:
    def __init__(self, max_bytes: int = 0):
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self.items: OrderedDict[str, LMCacheEntry] = OrderedDict()
        self.lock = threading.Lock()

    def put(self, key: str, entry: LMCacheEntry) -> None:
        with self.lock:
            old = self.items.pop(key, None)
            self.total_bytes -= old.length if old else 0
            self.items[key] = entry
            self.total_bytes += entry.length
            while self.max_bytes and self.total_bytes > self.max_bytes and len(self.items) > 1:
                _key, victim = self.items.popitem(last=False)
                self.total_bytes -= victim.length

    def get(self, key: str) -> LMCacheEntry | None:
        with self.lock:
            entry = self.items.get(key)
            if entry:
                self.items.move_to_end(key)
            return entry

    def clear(self) -> None:
        with self.lock:
            self.items.clear()
            self.total_bytes = 0

    def stats(self) -> tuple[int, int]:
        with self.lock:
            return len(self.items), self.total_bytes


def cache_event(operation: str, key: str = "", **fields) -> None:
    print(json.dumps({
        "monotonic_ns": time.monotonic_ns(),
        "wall_ns": time.time_ns(),
        "operation": operation,
        "key_hash": hashlib.sha256(key.encode()).hexdigest() if key else "",
        **fields,
    }, separators=(",", ":")), flush=True)


def _recv_all(sock: socket.socket, nbytes: int) -> bytes | None:
    data = bytearray()
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _send_meta(sock: socket.socket, payload: tuple[int, int, int, int, int, int, int, int, int]) -> None:
    sock.sendall(LMCACHE_SERVER_META.pack(*payload))


def handle_lmcache_client(sock: socket.socket, store: LiteLMCache) -> None:
    with sock:
        while header := _recv_all(sock, LMCACHE_CLIENT_META.size):
            command, length, fmt, dtype, location, s0, s1, s2, s3, raw_key = LMCACHE_CLIENT_META.unpack(header)
            key = raw_key.decode().strip(" \0")
            if command == LMCACHE_CLIENT_PUT:
                start = time.monotonic_ns()
                data = _recv_all(sock, length)
                if data is None:
                    return
                store.put(key, LMCacheEntry(data, length, fmt, dtype, (s0, s1, s2, s3), location))
                cache_event("source_write", key, bytes=length, start_ns=start, end_ns=time.monotonic_ns(), format=fmt, dtype=dtype, shape=[s0, s1, s2, s3])
            elif command == LMCACHE_CLIENT_GET:
                start = time.monotonic_ns()
                entry = store.get(key)
                if entry is None:
                    _send_meta(sock, LMCACHE_FAIL_PAYLOAD)
                else:
                    _send_meta(sock, (LMCACHE_SERVER_SUCCESS, entry.length, entry.fmt, entry.dtype, *entry.shape, entry.location))
                    sock.sendall(entry.data)
                cache_event("destination_read", key, bytes=entry.length if entry else 0, start_ns=start, end_ns=time.monotonic_ns(), hit=bool(entry))
            elif command == LMCACHE_CLIENT_EXIST:
                _send_meta(sock, LMCACHE_OK_PAYLOAD if store.get(key) else LMCACHE_FAIL_PAYLOAD)
            elif command == LMCACHE_CLIENT_HEALTH:
                _send_meta(sock, LMCACHE_OK_PAYLOAD)
            else:
                raise ValueError(f"unsupported LMCache command: {command}")


def run_lmcache_server(host: str, port: int, max_bytes: int = 0) -> None:
    store = LiteLMCache(max_bytes)

    def clear(_signum, _frame) -> None:
        store.clear()
        entries, nbytes = store.stats()
        cache_event("clear", entries=entries, bytes=nbytes)

    signal.signal(signal.SIGUSR1, clear)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen()
        print(f"LMCache lite server started at {host}:{port} max_bytes={max_bytes}", flush=True)
        while True:
            client, _addr = server.accept()
            threading.Thread(target=handle_lmcache_client, args=(client, store), daemon=True).start()


async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, bucket: TokenBucket, log: ByteLog | None, route: str, direction: str) -> int:
    total = 0
    try:
        charged = billable(route, direction)
        while data := await reader.read(CHUNK):
            if charged:
                await bucket.wait(len(data))
            writer.write(data)
            await writer.drain()
            total += len(data)
            if log:
                await log.add(route, direction, len(data), charged)
    finally:
        writer.close()
        await writer.wait_closed()
    return total


async def handle_proxy(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter, route: Route, bucket: TokenBucket, log: ByteLog | None) -> None:
    start = time.monotonic_ns()
    connection_id = hashlib.sha256(f"{route.name}:{start}".encode()).hexdigest()[:16]
    target_r, target_w = await asyncio.open_connection(route.target_host, route.target_port)
    if log:
        await log.opened()
    counts = (0, 0)
    try:
        counts = tuple(await asyncio.gather(
            relay(client_r, target_w, bucket, log, route.name, "client_to_target"),
            relay(target_r, client_w, bucket, log, route.name, "target_to_client"),
        ))
    finally:
        if log:
            await log.connection(connection_id, route.name, start, counts)


async def start_proxy(routes: list[Route], rate_bps: float, log: Path | None = None) -> tuple[list[asyncio.AbstractServer], ByteLog | None]:
    bucket = TokenBucket(rate_bps)
    byte_log = ByteLog(log) if log else None
    if byte_log:
        await byte_log.start()
    servers = []
    for route in routes:
        servers.append(
            await asyncio.start_server(
                lambda r, w, route=route: handle_proxy(r, w, route, bucket, byte_log),
                route.listen_host,
                route.listen_port,
            )
        )
    return servers, byte_log


async def run_proxy(routes: list[Route], rate_bps: float, log: Path | None = None) -> None:
    servers, byte_log = await start_proxy(routes, rate_bps, log)
    try:
        await asyncio.gather(*(server.serve_forever() for server in servers))
    finally:
        for server in servers:
            server.close()
            await server.wait_closed()
        if byte_log:
            await byte_log.close()


def wait_tcp(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {host}:{port}")


def tail(path: Path, lines: int = 40) -> str:
    text = read_text(path).splitlines()
    return "\n".join(text[-lines:])


def wait_tcp_process(host: str, port: int, timeout_s: float, proc: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited while waiting for {host}:{port}; log tail:\n{tail(log)}")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {host}:{port}; process still running; log tail:\n{tail(log)}")


def wait_health(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
            time.sleep(5)
        except OSError:
            time.sleep(5)
        finally:
            conn.close()
    raise TimeoutError(f"timed out waiting for http://{host}:{port}/health")


def wait_health_process(host: str, port: int, timeout_s: float, proc: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"process exited while waiting for http://{host}:{port}/health; log tail:\n{tail(log)}")
        conn = http.client.HTTPConnection(host, port, timeout=5)
        try:
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
            time.sleep(5)
        except OSError:
            time.sleep(5)
        finally:
            conn.close()
    raise TimeoutError(f"timed out waiting for http://{host}:{port}/health; process still running; log tail:\n{tail(log)}")


def prompt_text(session_id: str, words: int = 4096) -> str:
    body = " ".join(f"{session_id}_{i % 97}" for i in range(words))
    return f"Session {session_id}. {body}. Reply with exactly OK."


def chat_payload(cfg: Config, prompt: str, max_tokens: int = 4) -> str:
    return json.dumps({"model": cfg.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0})


def post_chat(cfg: Config, port: int, prompt: str, max_tokens: int = 4) -> dict:
    body = chat_payload(cfg, prompt, max_tokens)
    t0 = time.time()
    conn = http.client.HTTPConnection(cfg.host, port, timeout=600)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    text = resp.read().decode()
    conn.close()
    t1 = time.time()
    content = ""
    if resp.status == 200:
        content = json.loads(text)["choices"][0]["message"].get("content") or ""
    return {
        "status": resp.status,
        "content": content,
        "elapsed_s": t1 - t0,
        "start_ts": t0,
        "end_ts": t1,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "request_bytes": len(body.encode()),
        "response_text": text[:500] if resp.status != 200 else "",
    }


def chat_once(cfg: Config, port: int) -> str:
    result = post_chat(cfg, port, "Reply with exactly: OK", 128)
    if result["status"] != 200:
        raise RuntimeError(f"chat failed {result['status']}: {result['response_text']}")
    if "OK" not in result["content"]:
        raise RuntimeError(f"unexpected chat content: {result['content']}")
    return result["content"]


def start_logged(cmd: list[str], log: Path) -> subprocess.Popen:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("w")
    return subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)


def stop_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def stop_stack(stack: Stack) -> None:
    for proc in (stack.sink, stack.source, stack.proxy, stack.lmcache):
        if proc:
            stop_proc(proc)


def flush_lmcache(stack: Stack) -> None:
    if stack.lmcache.poll() is not None:
        raise RuntimeError("LMCache server is not running")
    log = stack.run_root / "lmcache.log"
    cleared = count_needle(log, LMCACHE_CLEAR_MARKER)
    os.kill(stack.lmcache.pid, signal.SIGUSR1)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if count_needle(log, LMCACHE_CLEAR_MARKER) > cleared:
            event = next(json.loads(line) for line in reversed(read_text(log).splitlines()) if LMCACHE_CLEAR_MARKER in line)
            if event["entries"] or event["bytes"]:
                raise RuntimeError(f"LMCache clear left data: {event}")
            return
        if stack.lmcache.poll() is not None:
            raise RuntimeError("LMCache server exited while clearing")
        time.sleep(0.05)
    raise TimeoutError("LMCache clear was not acknowledged")


def http_text(host: str, port: int, method: str, path: str) -> str:
    conn = http.client.HTTPConnection(host, port, timeout=30)
    try:
        conn.request(method, path)
        response = conn.getresponse()
        body = response.read().decode(errors="ignore")
    finally:
        conn.close()
    if response.status != 200:
        raise RuntimeError(f"{method} http://{host}:{port}{path} failed {response.status}: {body[:500]}")
    return body


def set_source_sleep(cfg: Config, sleeping: bool) -> None:
    def state() -> bool:
        return json.loads(http_text(cfg.host, cfg.src_port, "GET", "/is_sleeping"))["is_sleeping"]

    if state() == sleeping:
        return
    http_text(cfg.host, cfg.src_port, "POST", "/sleep?level=1" if sleeping else "/wake_up")
    if state() != sleeping:
        raise RuntimeError(f"source failed to become {'sleeping' if sleeping else 'awake'}")


def reset_result(text: str) -> bool | None:
    if "Failed to reset prefix cache" in text:
        return False
    if "Successfully reset prefix cache" in text:
        return True
    return None


def reset_vllm_caches(cfg: Config, logs: tuple[Path, Path]) -> None:
    for port, log in zip((cfg.src_port, cfg.sink_port), logs):
        offset = log.stat().st_size
        http_text(cfg.host, port, "POST", "/reset_prefix_cache")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            with log.open(errors="ignore") as handle:
                handle.seek(offset)
                result = reset_result(handle.read())
            if result is not None:
                if not result:
                    raise RuntimeError(f"vLLM prefix cache reset failed on port {port}")
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(f"vLLM prefix cache reset was not logged on port {port}")


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def count_needle(path: Path, needle: str) -> int:
    return read_text(path).count(needle)


def proxy_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["ts"] = str(int(row["wall_ns"]) / 1e9) if "wall_ns" in row else row["ts"]
    return rows


def proxy_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in proxy_rows(path):
        key = f"{row['route']}/{row['direction']}"
        counts[key] = counts.get(key, 0) + int(row["bytes"])
    return counts


def count_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {k: after.get(k, 0) - before.get(k, 0) for k in sorted(set(before) | set(after))}


def billed_window(rows: list[dict], route: str, direction: str) -> dict:
    xs = [r for r in rows if r["route"] == route and r["direction"] == direction and r.get("billed") == "1"]
    if not xs:
        return {"bytes": 0, "window_s": 0.0, "bytes_per_s": 0.0}
    ts = [float(r["ts"]) for r in xs]
    window = max(ts) - min(ts)
    nbytes = sum(int(r["bytes"]) for r in xs)
    return {"bytes": nbytes, "window_s": window, "bytes_per_s": nbytes / window if window else 0.0}


def start_stack(cfg: Config, run_root: Path, mbps: float, extra: list[str] | None = None) -> Stack:
    preflight(cfg, required_gpus=2)
    run_root.mkdir(parents=True, exist_ok=True)
    lmc = start_logged(lmcache_cmd(cfg), run_root / "lmcache.log")
    proxy = source = None
    try:
        wait_tcp_process(cfg.host, cfg.lmc_port, 60, lmc, run_root / "lmcache.log")
        proxy = start_logged(proxy_cmd(cfg, mbps, run_root / "proxy_bytes.csv"), run_root / "proxy.log")
        wait_tcp(cfg.host, cfg.kv_proxy_port, 60)
        wait_tcp(cfg.host, cfg.api_proxy_port, 60)
        source = start_logged(vllm_cmd(cfg, "source", extra or []), run_root / "source.log")
        wait_health_process(cfg.host, cfg.src_port, 1800, source, run_root / "source.log")
        return Stack(lmc, proxy, source, None, run_root)
    except Exception:
        for proc in (source, proxy, lmc):
            if proc:
                stop_proc(proc)
        raise


def start_sink(stack: Stack, cfg: Config, extra: list[str] | None = None) -> None:
    if stack.sink:
        return
    stack.sink = start_logged(vllm_cmd(cfg, "sink", extra or []), stack.run_root / "sink.log")
    wait_health_process(cfg.host, cfg.sink_port, 1800, stack.sink, stack.run_root / "sink.log")


def check_chat(result: dict, label: str) -> None:
    if result["status"] != 200:
        raise RuntimeError(f"{label} failed {result['status']}: {result['response_text']}")


def warm_source(cfg: Config, run_root: Path, prompt: str, label: str = "source warm") -> tuple[dict, int]:
    source_log = run_root / "source.log"
    stored0 = count_needle(source_log, "Stored")
    source = post_chat(cfg, cfg.src_port, prompt, 4)
    check_chat(source, label)
    time.sleep(2)
    if count_needle(source_log, "Stored") <= stored0:
        raise RuntimeError(f"{label} did not store KV")
    return source, stored0


def run_smoke2_probe(cfg: Config, run_root: Path, mbps: float, words: int = 4096, prompt: str | None = None, prewarmed: tuple[dict, int] | None = None) -> dict:
    proxy_log = run_root / "proxy_bytes.csv"
    source_log = run_root / "source.log"
    sink_log = run_root / "sink.log"
    prompt = prompt or prompt_text("smoke2-kv", words)
    replay_prompt = prompt_text(f"smoke2-replay-{int(time.time())}", min(words, 1024))

    retrieved0 = count_needle(sink_log, "Retrieved")
    before = proxy_counts(proxy_log)
    if prewarmed:
        source, stored0 = prewarmed
    else:
        source, stored0 = warm_source(cfg, run_root, prompt)

    before_sink = proxy_counts(proxy_log)
    sink = post_chat(cfg, cfg.api_proxy_port, prompt, 4)
    check_chat(sink, "sink kv resume")
    time.sleep(3)
    kv_rows = proxy_rows(proxy_log)
    kv_delta = count_delta(before_sink, proxy_counts(proxy_log))
    kv_link = billed_window(kv_rows, "kv", "target_to_client")
    if count_needle(sink_log, "Retrieved") <= retrieved0:
        raise RuntimeError("sink did not retrieve KV")
    if kv_delta.get("kv/target_to_client", 0) <= 0:
        raise RuntimeError("KV route had no source-to-sink bytes")
    link_bps = mbps * 1_000_000 / 8
    expected = kv_delta["kv/target_to_client"] / link_bps
    if expected > 0.5 and sink["elapsed_s"] + 0.5 < 0.2 * expected:
        raise RuntimeError("sink KV elapsed time is implausibly short for throttled bytes")
    if kv_link["window_s"] >= 0.5 and kv_link["bytes_per_s"] > 1.25 * link_bps:
        raise RuntimeError("KV proxy exceeded 1Gbps envelope")

    before_replay = proxy_counts(proxy_log)
    replay = post_chat(cfg, cfg.api_proxy_port, replay_prompt, 4)
    check_chat(replay, "sink replay")
    replay_delta = count_delta(before_replay, proxy_counts(proxy_log))
    if replay_delta.get("api/client_to_target", 0) <= 0:
        raise RuntimeError("replay route had no API request bytes")
    if replay_delta.get("kv/target_to_client", 0) > max(1_000_000, 0.25 * kv_delta["kv/target_to_client"]):
        raise RuntimeError("replay unexpectedly pulled large KV bytes")

    manifest = {
        "schema": "queue-haul-stage1b-smoke2-v1",
        "mbps": mbps,
        "lambda_src_bytes_per_s": link_bps,
        "endpoints": {"source": cfg.src_port, "sink": cfg.sink_port, "api_proxy": cfg.api_proxy_port, "kv_proxy": cfg.kv_proxy_port},
        "source": source,
        "sink_kv": sink,
        "sink_replay": replay,
        "proxy_total_delta": count_delta(before, proxy_counts(proxy_log)),
        "kv_delta": kv_delta,
        "replay_delta": replay_delta,
        "kv_link": kv_link,
        "evidence": {
            "source_connector": "engine_id: s0" in read_text(source_log) or "engine_id=s0" in read_text(source_log),
            "sink_connector": "engine_id: d0" in read_text(sink_log) or "engine_id=d0" in read_text(sink_log),
            "source_stored": count_needle(source_log, "Stored") > stored0,
            "sink_retrieved": count_needle(sink_log, "Retrieved") > retrieved0,
            "kv_proxy_bytes": kv_delta.get("kv/target_to_client", 0),
            "api_proxy_bytes": replay_delta.get("api/client_to_target", 0),
        },
    }
    manifest["acceptance"] = {
        "ok": all(manifest["evidence"].values()) and source["status"] == sink["status"] == replay["status"] == 200,
        "kv_expected_link_s": expected,
        "kv_elapsed_s": sink["elapsed_s"],
        "kv_observed_bytes_per_s": kv_link["bytes_per_s"],
        "source_warmed_before_sink": prewarmed is not None,
    }
    (run_root / "smoke2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    if not manifest["acceptance"]["ok"]:
        raise RuntimeError("smoke2 acceptance failed")
    return manifest


def smoke2_live(cfg: Config, run_root: Path, mbps: float, extra: list[str]) -> Path:
    stack = start_stack(cfg, run_root, mbps, extra)
    try:
        start_sink(stack, cfg, extra)
        manifest = run_smoke2_probe(cfg, run_root, mbps)
        source_after = post_chat(cfg, cfg.src_port, "Reply with exactly: OK", 4)
        check_chat(source_after, "source after live sink")
        manifest["live"] = {"source_after_sink": source_after, "source_poll": stack.source.poll(), "sink_poll": stack.sink.poll()}
        manifest["acceptance"]["ok"] = manifest["acceptance"]["ok"] and source_after["status"] == 200 and stack.source.poll() is None and stack.sink.poll() is None
        (run_root / "smoke2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if not manifest["acceptance"]["ok"]:
            raise RuntimeError("smoke2-live acceptance failed")
    finally:
        stop_stack(stack)
    return run_root


def smoke2(cfg: Config, run_root: Path, mbps: float, extra: list[str]) -> Path:
    stack = start_stack(cfg, run_root, mbps, extra)
    try:
        prompt = prompt_text("smoke2-kv")
        prewarmed = warm_source(cfg, run_root, prompt)
        stop_proc(stack.source)
        stack.source = None
        start_sink(stack, cfg, extra)
        run_smoke2_probe(cfg, run_root, mbps, prompt=prompt, prewarmed=prewarmed)
    finally:
        stop_stack(stack)
    return run_root

def smoke1(cfg: Config, run_root: Path, extra: list[str]) -> Path:
    preflight(cfg, required_gpus=1)
    run_root.mkdir(parents=True, exist_ok=True)
    lmc = start_logged(lmcache_cmd(cfg), run_root / "lmcache.log")
    vllm = None
    try:
        wait_tcp_process(cfg.host, cfg.lmc_port, 60, lmc, run_root / "lmcache.log")
        vllm = start_logged(vllm_cmd(cfg, "smoke1", extra), run_root / "vllm.log")
        wait_health_process(cfg.host, cfg.smoke_port, 1800, vllm, run_root / "vllm.log")
        chat_once(cfg, cfg.smoke_port)
        chat_once(cfg, cfg.smoke_port)
        time.sleep(3)
        text = read_text(run_root / "vllm.log")
        for needle in ["Stored", "Retrieved"]:
            if needle not in text:
                raise RuntimeError(f"missing LMCache proof in {run_root / 'vllm.log'}: {needle}")
    finally:
        if vllm:
            stop_proc(vllm)
        stop_proc(lmc)
    return run_root



def config_from_args(args) -> Config:
    return Config(
        model=args.model,
        sandbox=args.sandbox,
        hf_home=args.hf_home,
        scratch_bind=args.scratch_bind,
        cache_root=args.cache_root,
        host=args.host,
        src_port=args.src_port,
        sink_port=args.sink_port,
        lmc_port=args.lmc_port,
        kv_proxy_port=args.kv_proxy_port,
        api_proxy_port=args.api_proxy_port,
        smoke_port=args.smoke_port,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=MODEL)
    p.add_argument("--sandbox", type=Path, default=apptainer_image_default())
    p.add_argument("--hf-home", type=Path, default=HF_HOME)
    p.add_argument("--scratch-bind", type=Path, default=SCRATCH_BIND)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--src-port", type=int, default=port_default(8100))
    p.add_argument("--sink-port", type=int, default=port_default(8200))
    p.add_argument("--lmc-port", type=int, default=port_default(5655))
    p.add_argument("--kv-proxy-port", type=int, default=port_default(8300))
    p.add_argument("--api-proxy-port", type=int, default=port_default(8400))
    p.add_argument("--smoke-port", type=int, default=port_default(8120))
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Queue-Haul Stage 1b source/sink LMCache smoke tooling")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "smoke1", "smoke2", "smoke2-live"):
        sp = sub.add_parser(name)
        add_common(sp)
        if name == "preflight":
            sp.add_argument("--required-gpus", type=int, default=1)
        if name in ("smoke1", "smoke2", "smoke2-live"):
            sp.add_argument("--run-root", type=Path, default=Path(f"queue-haul/runs/stage1b/{name}"))
            if name.startswith("smoke2"):
                sp.add_argument("--mbps", type=float, default=1000.0)
            sp.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    sp = sub.add_parser("lmcache-server")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=5655)
    sp.add_argument("--max-bytes", type=int, default=LMCACHE_SERVER_MAX_BYTES)
    sp = sub.add_parser("proxy")
    sp.add_argument("--kv-listen", default="127.0.0.1:8300")
    sp.add_argument("--kv-target", default="127.0.0.1:5655")
    sp.add_argument("--api-listen", default="127.0.0.1:8400")
    sp.add_argument("--api-target", default="127.0.0.1:8200")
    sp.add_argument("--mbps", type=float, required=True)
    sp.add_argument("--log", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cmd == "lmcache-server":
        run_lmcache_server(args.host, args.port, args.max_bytes)
        return
    if args.cmd == "proxy":
        kv_listen = parse_addr(args.kv_listen)
        kv_target = parse_addr(args.kv_target)
        api_listen = parse_addr(args.api_listen)
        api_target = parse_addr(args.api_target)
        routes = [Route("kv", *kv_listen, *kv_target), Route("api", *api_listen, *api_target)]
        asyncio.run(run_proxy(routes, args.mbps * 1_000_000 / 8, args.log))
        return

    cfg = config_from_args(args)
    if args.cmd == "preflight":
        print("\n".join(preflight(cfg, args.required_gpus)))
    elif args.cmd == "smoke1":
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(smoke1(cfg, args.run_root, extra))
    elif args.cmd == "smoke2":
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(smoke2(cfg, args.run_root, args.mbps, extra))
    elif args.cmd == "smoke2-live":
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(smoke2_live(cfg, args.run_root, args.mbps, extra))


if __name__ == "__main__":
    main()
