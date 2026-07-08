from __future__ import annotations

import argparse
import asyncio
import csv
import http.client
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from stage1_curves import shell

MODEL = "openai/gpt-oss-20b"
SANDBOX = Path("/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox")
HF_HOME = Path("/scratch/users/gfw/ptsim/hf")
SCRATCH_BIND = Path("/scratch/users/gfw")
CACHE_ROOT = Path("/scratch/users/gfw/ptsim/cache")
TMPDIR = Path("/tmp/t")
CHUNK = 65536
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
    "--enforce-eager",
    "--kv-transfer-config",
}


@dataclass(frozen=True)
class Config:
    model: str = MODEL
    sandbox: Path = SANDBOX
    hf_home: Path = HF_HOME
    scratch_bind: Path = SCRATCH_BIND
    cache_root: Path = CACHE_ROOT
    host: str = "127.0.0.1"
    src_port: int = 8100
    sink_port: int = 8200
    lmc_port: int = 5655
    kv_proxy_port: int = 8300
    api_proxy_port: int = 8400
    smoke_port: int = 8120
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


def kv_config(engine_id: str, kv_role: str) -> str:
    return json.dumps(
        {
            "kv_connector": "LMCacheConnectorV1",
            "engine_id": engine_id,
            "kv_role": kv_role,
            "kv_connector_extra_config": {"discard_partial_chunks": False},
        },
        separators=(",", ":"),
    )


def vllm_exports(cfg: Config, role: str, remote_url: str) -> list[str]:
    env = {
        "PYTHONHASHSEED": "0",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TORCH_CUDA_ARCH_LIST": "8.0",
        "TMPDIR": str(TMPDIR),
        "HF_HOME": str(cfg.hf_home),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LMCACHE_REMOTE_URL": remote_url,
        "LMCACHE_REMOTE_SERDE": "naive",
        "LMCACHE_CHUNK_SIZE": "256",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "0.25",
        **{k: str(v) for k, v in cache_dirs(cfg, role).items()},
    }
    return [f"export {k}={shlex.quote(v)}" for k, v in env.items()]


def apptainer_cmd(cfg: Config, script: str, gpu: int | None = None) -> list[str]:
    cmd = ["apptainer", "exec", "--nv", "--bind", f"{cfg.scratch_bind}:{cfg.scratch_bind}", cfg.sandbox, "bash", "-lc", script]
    if gpu is None:
        return cmd
    return ["env", f"APPTAINERENV_CUDA_VISIBLE_DEVICES={gpu}", *cmd]


def lmcache_cmd(cfg: Config) -> list[str]:
    return apptainer_cmd(cfg, f"python3 -m lmcache.v1.server {cfg.host} {cfg.lmc_port} cpu")


def vllm_cmd(cfg: Config, role: str, extra: list[str] | None = None) -> list[str]:
    reject_duplicate_extra(extra or [])
    if role == "source":
        port, gpu, engine_id, kv_role = cfg.src_port, 0, "s0", "kv_producer"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.lmc_port}", "src"
    elif role == "sink":
        port, gpu, engine_id, kv_role = cfg.sink_port, 1, "d0", "kv_consumer"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.kv_proxy_port}", "sink"
    elif role == "smoke1":
        port, gpu, engine_id, kv_role = cfg.smoke_port, 0, "e0", "kv_both"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.lmc_port}", "smoke1"
    else:
        raise ValueError(f"unknown role: {role}")

    dirs = " ".join(shlex.quote(str(p)) for p in [TMPDIR, *cache_dirs(cfg, cache_role).values()])
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
        "--kv-transfer-config",
        kv_config(engine_id, kv_role),
        *(extra or []),
    ]
    script = "\n".join([f"mkdir -p {dirs}", *vllm_exports(cfg, cache_role, remote_url), shell(serve)])
    return apptainer_cmd(cfg, script, gpu)


def proxy_routes(cfg: Config) -> list[Route]:
    return [
        Route("kv", cfg.host, cfg.kv_proxy_port, cfg.host, cfg.lmc_port),
        Route("api", cfg.host, cfg.api_proxy_port, cfg.host, cfg.sink_port),
    ]


def proxy_cmd(cfg: Config, mbps: float = 100.0, log: Path | None = None) -> list[str]:
    cmd = [
        "python3",
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
        mbps,
    ]
    if log:
        cmd += ["--log", log]
    return cmd


def plan_text(cfg: Config, mode: str, mbps: float, extra: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"# stage1b {mode}: host orchestrator, Apptainer vLLM/LMCache children, user-space proxy",
        "mkdir -p logs",
        'start_bg() { local name="$1"; shift; "$@" > "logs/${name}.log" 2>&1 & echo $! > "logs/${name}.pid"; }',
        f"start_bg lmcache {shell(lmcache_cmd(cfg))}",
        f"start_bg proxy {shell(proxy_cmd(cfg, mbps, Path('proxy_bytes.csv')))}",
        f"start_bg source {shell(vllm_cmd(cfg, 'source', extra))}",
        f"curl --retry 360 --retry-delay 5 --retry-connrefused --max-time 5 -fsS http://{cfg.host}:{cfg.src_port}/health",
        f"start_bg sink {shell(vllm_cmd(cfg, 'sink', extra))}",
        f"curl --retry 360 --retry-delay 5 --retry-connrefused --max-time 5 -fsS http://{cfg.host}:{cfg.sink_port}/health",
    ]
    if mode == "drain":
        lines.append("# Run the Stage 1b drain driver here after smoke2 passes on a two-GPU allocation.")
    return "\n".join(lines) + "\n"


def port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex((host, port)) != 0


def gpu_count() -> int:
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found")
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def preflight(cfg: Config, required_gpus: int = 1) -> list[str]:
    validate_ports(cfg)
    failures = []
    if not shutil.which("apptainer"):
        failures.append("apptainer not found")
    if not cfg.sandbox.exists():
        failures.append(f"sandbox missing: {cfg.sandbox}")
    snapshots = model_snapshot_dir(cfg.hf_home, cfg.model)
    if not snapshots.exists() or not any(snapshots.iterdir()):
        failures.append(f"model snapshot missing: {snapshots}")
    if len(str(TMPDIR)) > 20:
        failures.append(f"TMPDIR too long for LMCache IPC: {TMPDIR}")
    for port in [cfg.lmc_port, cfg.smoke_port] if required_gpus == 1 else [cfg.src_port, cfg.sink_port, cfg.lmc_port, cfg.kv_proxy_port, cfg.api_proxy_port]:
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
    def __init__(self, path: Path):
        self.file = path.open("w", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(["ts", "route", "direction", "bytes"])
        self.lock = asyncio.Lock()

    async def write(self, route: str, direction: str, nbytes: int) -> None:
        async with self.lock:
            self.writer.writerow([f"{time.time():.6f}", route, direction, nbytes])

    def close(self) -> None:
        self.file.close()


async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, bucket: TokenBucket, log: ByteLog | None, route: str, direction: str) -> None:
    try:
        while data := await reader.read(CHUNK):
            await bucket.wait(len(data))
            writer.write(data)
            await writer.drain()
            if log:
                await log.write(route, direction, len(data))
    finally:
        writer.close()
        await writer.wait_closed()


async def handle_proxy(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter, route: Route, bucket: TokenBucket, log: ByteLog | None) -> None:
    target_r, target_w = await asyncio.open_connection(route.target_host, route.target_port)
    await asyncio.gather(
        relay(client_r, target_w, bucket, log, route.name, "client_to_target"),
        relay(target_r, client_w, bucket, log, route.name, "target_to_client"),
    )


async def start_proxy(routes: list[Route], rate_bps: float, log: Path | None = None) -> tuple[list[asyncio.AbstractServer], ByteLog | None]:
    bucket = TokenBucket(rate_bps)
    byte_log = ByteLog(log) if log else None
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
            byte_log.close()


def wait_tcp(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(1)
    raise TimeoutError(f"timed out waiting for {host}:{port}")


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


def chat_once(cfg: Config, port: int) -> str:
    body = json.dumps(
        {
            "model": cfg.model,
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "max_tokens": 128,
            "temperature": 0,
        }
    )
    conn = http.client.HTTPConnection(cfg.host, port, timeout=120)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    text = resp.read().decode()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"chat failed {resp.status}: {text}")
    content = json.loads(text)["choices"][0]["message"]["content"]
    if "OK" not in content:
        raise RuntimeError(f"unexpected chat content: {content}")
    return content


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


def smoke1(cfg: Config, run_root: Path, extra: list[str]) -> Path:
    preflight(cfg, required_gpus=1)
    run_root.mkdir(parents=True, exist_ok=True)
    lmc_log = run_root / "lmcache.log"
    vllm_log = run_root / "vllm.log"
    lmc = start_logged(lmcache_cmd(cfg), lmc_log)
    vllm = None
    try:
        wait_tcp(cfg.host, cfg.lmc_port, 60)
        vllm = start_logged(vllm_cmd(cfg, "smoke1", extra), vllm_log)
        wait_health(cfg.host, cfg.smoke_port, 1800)
        chat_once(cfg, cfg.smoke_port)
        chat_once(cfg, cfg.smoke_port)
        time.sleep(3)
        text = vllm_log.read_text(errors="ignore")
        for needle in ["Stored", "Retrieved"]:
            if needle not in text:
                raise RuntimeError(f"missing LMCache proof in {vllm_log}: {needle}")
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
    p.add_argument("--sandbox", type=Path, default=SANDBOX)
    p.add_argument("--hf-home", type=Path, default=HF_HOME)
    p.add_argument("--scratch-bind", type=Path, default=SCRATCH_BIND)
    p.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--src-port", type=int, default=8100)
    p.add_argument("--sink-port", type=int, default=8200)
    p.add_argument("--lmc-port", type=int, default=5655)
    p.add_argument("--kv-proxy-port", type=int, default=8300)
    p.add_argument("--api-proxy-port", type=int, default=8400)
    p.add_argument("--smoke-port", type=int, default=8120)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Queue-Haul Stage 1b source/sink LMCache smoke tooling")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "smoke1", "smoke2-plan", "drain-plan"):
        sp = sub.add_parser(name)
        add_common(sp)
        if name == "preflight":
            sp.add_argument("--required-gpus", type=int, default=1)
        if name == "smoke1":
            sp.add_argument("--run-root", type=Path, default=Path("queue-haul/runs/stage1b/smoke1"))
            sp.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
        if name.endswith("plan"):
            sp.add_argument("--mbps", type=float, default=100.0)
            sp.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
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
    if args.cmd == "proxy":
        kv_listen = parse_addr(args.kv_listen)
        kv_target = parse_addr(args.kv_target)
        api_listen = parse_addr(args.api_listen)
        api_target = parse_addr(args.api_target)
        routes = [
            Route("kv", *kv_listen, *kv_target),
            Route("api", *api_listen, *api_target),
        ]
        asyncio.run(run_proxy(routes, args.mbps * 1_000_000 / 8, args.log))
        return

    cfg = config_from_args(args)
    if args.cmd == "preflight":
        print("\n".join(preflight(cfg, args.required_gpus)))
    elif args.cmd == "smoke1":
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(smoke1(cfg, args.run_root, extra))
    elif args.cmd in {"smoke2-plan", "drain-plan"}:
        extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        print(plan_text(cfg, args.cmd.removesuffix("-plan"), args.mbps, extra), end="")


if __name__ == "__main__":
    main()
