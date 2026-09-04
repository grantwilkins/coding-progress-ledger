from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
from collections import OrderedDict
import http.client
import io
import json
import os
import re
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

from service_curve_runner import shell

MODEL = "openai/gpt-oss-20b"
MODEL_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
SANDBOX = Path("/scratch/users/gfw/ptsim/vllm-openai-v0.10.1.1.sandbox")
MP_IMAGE = Path("/scratch/users/gfw/ptsim/lmcache-v0.5.1-vllm0.22.0-cu129-primary.sif")
REDIS_IMAGE = Path("/scratch/users/gfw/ptsim/redis-7.4.2-bookworm.sif")
RUNTIME_VERSIONS = ("0.10.1.1", "0.3.3")
MP_RUNTIME_VERSIONS = ("0.22.0+cu129", "0.5.1")
NATIVE_RUNTIME_VERSIONS = ("0.22.0", "0.5.1")
LMCACHE_CLEAR_MARKER = '"operation":"clear"'


def lmcache_mode() -> str:
    mode = os.environ.get("QH_LMCACHE_MODE", "legacy")
    if mode not in {"legacy", "mp"}:
        raise ValueError(f"unknown QH_LMCACHE_MODE: {mode}")
    return mode


def prefix_caching() -> bool:
    mode = os.environ.get("QH_PREFIX_CACHING", "on")
    if mode not in {"on", "off"}:
        raise ValueError(f"unknown QH_PREFIX_CACHING: {mode}")
    return mode == "on"


def runtime_mode() -> str:
    mode = os.environ.get("QH_RUNTIME", "apptainer")
    if mode not in {"apptainer", "native"}:
        raise ValueError(f"unknown QH_RUNTIME: {mode}")
    if mode == "native" and lmcache_mode() != "mp":
        raise ValueError("native runtime requires QH_LMCACHE_MODE=mp")
    return mode


def expected_runtime_versions() -> tuple[str, str]:
    if runtime_mode() == "native":
        versions = os.environ.get("QH_NATIVE_RUNTIME_VERSIONS")
        if versions:
            parsed = tuple(versions.split(","))
            if len(parsed) != 2 or not all(parsed):
                raise ValueError("QH_NATIVE_RUNTIME_VERSIONS must be vllm,lmcache")
            return parsed
        return NATIVE_RUNTIME_VERSIONS
    return MP_RUNTIME_VERSIONS if lmcache_mode() == "mp" else RUNTIME_VERSIONS


def apptainer_image_default() -> Path:
    default = MP_IMAGE if lmcache_mode() == "mp" else SANDBOX
    return Path(os.environ.get("QH_APPTAINER_IMAGE", default))


def port_offset() -> int:
    offset = int(os.environ.get("QH_PORT_OFFSET", "0"))
    if offset < 0 or offset > 50000:
        raise ValueError(f"invalid QH_PORT_OFFSET: {offset}")
    return offset


def port_default(base: int) -> int:
    return base + port_offset()


HF_HOME = Path(os.environ.get("HF_HOME", "/scratch/users/gfw/ptsim/hf"))
SCRATCH_BIND = Path("/scratch/users/gfw")
CACHE_ROOT = Path(os.environ.get("QH_CACHE_ROOT", "/scratch/users/gfw/ptsim/cache"))
LMCACHE_COMPAT = Path(__file__).with_name("lmcache_compat").resolve()
CHUNK = 65536
LMCACHE_MAX_LOCAL_CPU_GB = "4"
DEFAULT_KV_ROLES = {"source": "kv_producer", "sink": "kv_consumer", "smoke1": "kv_both"}
TYPED_VLLM_FLAGS = {
    "--host",
    "--port",
    "--served-model-name",
    "--tensor-parallel-size",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--dtype",
    "--kv-cache-dtype",
    "--enable-chunked-prefill",
    "--enable-sleep-mode",
    "--enforce-eager",
    "--gpu-memory-utilization",
    "--block-size",
    "--disable-hybrid-kv-cache-manager",
    "--enable-prompt-tokens-details",
    "--async-scheduling",
    "--no-async-scheduling",
    "--stream-interval",
    "--language-model-only",
    "--mamba-cache-mode",
    "--limit-mm-per-prompt",
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
MP_CONTEXT = re.compile(r"Registered non-GPU context.*model=([^,]+), world_size=(\d+)")
MP_REQUEST = re.compile(r"(\d+)/(\d+) retained keys \((\d+) L1, (\d+) L2\).*external_request_id=([^,\)]+)")
MP_STORED = re.compile(r"Stored (\d+) tokens")


@dataclass(frozen=True)
class ModelSpec:
    revision: str
    batched_tokens: int = 8192
    chunk_tokens: int = 256
    vllm_args: tuple[str, ...] = ()
    unified_block_tokens: int | None = None
    separate_object_groups: bool = False
    hybrid_cache_groups: bool = False


MODEL_SPECS = {
    MODEL: ModelSpec(MODEL_REVISION, vllm_args=(
        "--hf-overrides", '{"allow_global_per_layer_attribute_access":true}')),
    "Qwen/Qwen3.8-27B": ModelSpec(
        "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", 1567, 784,
        ("--language-model-only", "--mamba-cache-mode", "align"), 784,
        True, True),
    "google/gemma-4-26B-A4B-it": ModelSpec(
        "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
        vllm_args=("--limit-mm-per-prompt", '{"image":0,"audio":0}'),
        hybrid_cache_groups=True),
}


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
    src_lmc_port: int = field(default_factory=lambda: port_default(5557))
    sink_lmc_port: int = field(default_factory=lambda: port_default(5556))
    src_lmc_http_port: int = field(default_factory=lambda: port_default(8080))
    sink_lmc_http_port: int = field(default_factory=lambda: port_default(8081))
    max_model_len: int = 32768
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    architecture_campaign: bool = False
    capacity_discovery: bool = False
    matched_prefill: bool = False
    literal_token_timing: bool = False
    enforce_eager: bool = True


@dataclass(frozen=True)
class Route:
    name: str
    listen_host: str
    listen_port: int
    target_host: str
    target_port: int
    protocol: str = "raw"


@dataclass
class Stack:
    lmcache: subprocess.Popen
    proxy: subprocess.Popen
    source: subprocess.Popen | None
    sink: subprocess.Popen | None
    run_root: Path
    cache_services: list[subprocess.Popen] = field(default_factory=list)
    bandwidth_mbps: float = 0


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
    if lmcache_mode() == "mp":
        ports += [cfg.src_lmc_port, cfg.sink_lmc_port, cfg.src_lmc_http_port, cfg.sink_lmc_http_port]
    bad = [p for p in ports if p <= 0 or p > 65535]
    if bad:
        raise ValueError(f"invalid ports: {bad}")
    dupes = sorted({p for p in ports if ports.count(p) > 1})
    if dupes:
        raise ValueError(f"duplicate ports: {dupes}")


def model_snapshot_dir(hf_home: Path, model: str) -> Path:
    return hf_home / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"


def model_spec(model: str) -> ModelSpec:
    if model not in MODEL_SPECS:
        raise ValueError(f"unsupported model: {model}")
    return MODEL_SPECS[model]


def model_vllm_args(cfg: Config) -> tuple[str, ...]:
    args = model_spec(cfg.model).vllm_args
    if cfg.model == "google/gemma-4-26B-A4B-it" \
            and expected_runtime_versions()[0] == "0.24.0":
        args += ("--hf-overrides", json.dumps({"text_config": {
            "allow_global_per_layer_attribute_access": True,
            "global_head_dim": 512,
            "num_global_key_value_heads": 2}}))
    return args


def model_path(cfg: Config) -> Path:
    return model_snapshot_dir(cfg.hf_home, cfg.model) / model_spec(cfg.model).revision


def model_campaign_config(model: str, *,
                          literal_token_timing: bool = False) -> Config:
    spec = model_spec(model)
    return Config(model=model, max_num_seqs=8,
                  max_num_batched_tokens=spec.batched_tokens,
                  architecture_campaign=True,
                  literal_token_timing=literal_token_timing)


def model_chunk_tokens(cfg: Config) -> int:
    return model_spec(cfg.model).chunk_tokens \
        if (getattr(cfg, "architecture_campaign", False)
            or getattr(cfg, "capacity_discovery", False)) else 256


def validate_model_runtime(cfg: Config) -> None:
    if cfg.literal_token_timing and not cfg.architecture_campaign:
        raise ValueError(
            "literal token timing requires architecture_campaign")
    if cfg.matched_prefill:
        if cfg.max_model_len != 32768 or cfg.max_num_seqs != 256 \
                or cfg.max_num_batched_tokens != 8192 \
                or lmcache_mode() != "mp":
            raise ValueError("matched-prefill runtime geometry changed")
        return
    if cfg.architecture_campaign and cfg.capacity_discovery:
        raise ValueError("model runtime modes are mutually exclusive")
    if cfg.capacity_discovery:
        spec = model_spec(cfg.model)
        if cfg.max_model_len != 32768 or cfg.max_num_seqs != 256 \
                or cfg.max_num_batched_tokens != spec.batched_tokens:
            raise ValueError("capacity-discovery runtime geometry changed")
        if lmcache_mode() != "mp":
            raise ValueError("capacity discovery requires QH_LMCACHE_MODE=mp")
        return
    if not cfg.architecture_campaign:
        if cfg.model != MODEL:
            raise ValueError("additional models require architecture_campaign")
        return
    spec = model_spec(cfg.model)
    if (cfg.max_model_len, cfg.max_num_seqs, cfg.max_num_batched_tokens) != (
            32768, 8, spec.batched_tokens):
        raise ValueError("architecture-campaign runtime geometry changed")
    if lmcache_mode() != "mp":
        raise ValueError("architecture campaign requires QH_LMCACHE_MODE=mp")


def effective_kv_cache_dtype(server_info: dict) -> str:
    config = server_info.get("vllm_config", {})
    cache_dtype = str(config.get("cache_config", {}).get("cache_dtype"))
    if cache_dtype.lower() == "auto":
        return str(config.get("model_config", {}).get("dtype"))
    return cache_dtype


def validate_model_runtime_log(cfg: Config, text: str,
                               server_info: dict | None = None) -> None:
    if not (cfg.architecture_campaign or cfg.capacity_discovery):
        return
    if cfg.capacity_discovery:
        effective_dtype = effective_kv_cache_dtype(server_info or {})
        if effective_dtype.lower() not in {"bfloat16", "torch.bfloat16"}:
            raise RuntimeError(
                "capacity discovery did not read back resolved BF16 KV cache")
    elif not re.search(
            r"(?:bfloat16|bf16).{0,80}kv.?cache|"
            r"kv.?cache.{0,80}(?:bfloat16|bf16)", text, re.IGNORECASE):
        raise RuntimeError("architecture campaign did not prove BF16 KV cache")
    expected = model_spec(cfg.model).unified_block_tokens
    if expected is not None and set(map(int, re.findall(
            r"attention block size to (\d+) tokens", text, re.IGNORECASE))) != {expected}:
        raise RuntimeError(f"runtime did not prove the {expected}-token unified block")
    if cfg.literal_token_timing and not re.search(
            r"Asynchronous scheduling is disabled", text, re.IGNORECASE):
        raise RuntimeError(
            "timing runtime did not prove asynchronous scheduling disabled")


def validate_optimized_runtime(command: str, text: str) -> None:
    if "--enforce-eager" in command:
        raise RuntimeError("optimized runtime cannot force eager execution")
    if not re.search(r"(?:torch[ ._-]?compile|compil(?:e|ation|ing))", text,
                     re.IGNORECASE) \
            or not re.search(r"cuda[ _-]?graphs?", text, re.IGNORECASE):
        raise RuntimeError("optimized runtime did not prove compilation and CUDA graphs")


def validate_h100_optimized_runtime(command: str, text: str) -> None:
    validate_optimized_runtime(command, text)


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


def kv_role_for(role: str) -> str:
    """vLLM kv_role per engine; QH_KV_ROLE_{SOURCE,SINK} override the defaults."""
    return os.environ.get(f"QH_KV_ROLE_{role.upper()}", DEFAULT_KV_ROLES[role])


def lmcache_l1_gb() -> int:
    return int(os.environ.get("QH_LMCACHE_L1_GB", "16"))


def mp_transfer_mode(cfg: Config) -> str:
    return ("lmcache_driven" if model_spec(cfg.model).hybrid_cache_groups
            else "engine_driven")


def kv_config(cfg: Config, engine_id: str, kv_role: str, kv_port: int,
              rpc_port: str) -> str:
    if lmcache_mode() == "mp":
        return json.dumps({
            "kv_connector": "LMCacheMPConnector",
            "kv_connector_module_path": "connector_patch",
            "engine_id": engine_id,
            "kv_role": kv_role,
            "kv_connector_extra_config": {
                "lmcache.mp.host": "tcp://127.0.0.1",
                "lmcache.mp.port": port_default(5557 if engine_id != "d0" else 5556),
                "lmcache.mp.mp_transfer_mode": mp_transfer_mode(cfg),
            },
        }, separators=(",", ":"))
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
    spec = model_spec(cfg.model)
    env = {
        "QH_MODEL": cfg.model,
        "PYTHONHASHSEED": "0",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "TMPDIR": str(tmpdir(role)),
        "VLLM_RPC_BASE_PATH": str(tmpdir(role)),
        "HF_HOME": str(cfg.hf_home),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "LMCACHE_REMOTE_URL": remote_url,
        "LMCACHE_REMOTE_SERDE": "naive",
        "LMCACHE_LMCACHE_INSTANCE_ID": f"stage1b_{role}",
        "LMCACHE_CHUNK_SIZE": str(model_chunk_tokens(cfg)),
        "LMCACHE_LOCAL_CPU": "False",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": LMCACHE_MAX_LOCAL_CPU_GB,
        "QH_KV_GEOMETRY_EVIDENCE": "1" if cfg.architecture_campaign else "0",
        "QH_LMCACHE_SEPARATE_OBJECT_GROUPS": "1" if (
            (cfg.architecture_campaign or cfg.capacity_discovery)
            and spec.separate_object_groups
        ) else "0",
        **{k: str(v) for k, v in cache_dirs(cfg, role).items()},
    }
    env["PYTHONPATH"] = str(LMCACHE_COMPAT)
    exports = [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    library_path = (
        "/usr/local/cuda-12.9/compat:/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
        "/usr/local/cuda-12.9/targets/x86_64-linux/lib"
        if lmcache_mode() == "mp" else
        "/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:"
        "/opt/venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:"
        "/usr/local/cuda-12.9/targets/x86_64-linux/lib:"
        "/usr/local/cuda/targets/x86_64-linux/lib"
    )
    if runtime_mode() == "apptainer":
        exports.append(f"export LD_LIBRARY_PATH={library_path}:${{LD_LIBRARY_PATH:-}}")
    return exports


def allocated_gpu_ids() -> list[str]:
    value = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS", "")
    return [item.strip() for item in value.split(",") if item.strip()]


def apptainer_cmd(cfg: Config, script: str, gpu: int | None = None, nv: bool = True) -> list[str]:
    if runtime_mode() == "native":
        script = f"export PATH={shlex.quote(str(Path(sys.executable).parent))}:$PATH\n{script}"
        if gpu is None:
            return ["bash", "-lc", script]
        devices = allocated_gpu_ids()
        device = devices[gpu] if devices else str(gpu)
        return ["env", f"CUDA_VISIBLE_DEVICES={device}", "bash", "-lc", script]
    cmd = ["apptainer", "exec", "--bind", f"{cfg.scratch_bind}:{cfg.scratch_bind}", cfg.sandbox, "bash", "-lc", script]
    if nv:
        mode = os.environ.get("QH_APPTAINER_GPU_MODE", "nv")
        if mode not in {"nv", "nvccli"}:
            raise ValueError(f"unknown QH_APPTAINER_GPU_MODE: {mode}")
        cmd.insert(2, f"--{mode}")
    if gpu is None:
        return cmd
    devices = allocated_gpu_ids()
    device = devices[gpu] if devices else str(gpu)
    return ["env", f"CUDA_VISIBLE_DEVICES={device}", f"APPTAINERENV_CUDA_VISIBLE_DEVICES={device}", f"NVIDIA_VISIBLE_DEVICES={device}", *cmd]


def lmcache_cmd(cfg: Config) -> list[str]:
    return [sys.executable, "queue-haul/migration_testbed.py", "lmcache-server", "--host", cfg.host, "--port", str(cfg.lmc_port), "--max-bytes", str(LMCACHE_SERVER_MAX_BYTES)]


def redis_maxmemory_gb() -> int:
    return int(os.environ.get("QH_REDIS_MAXMEMORY_GB", "0"))


def redis_cmd(cfg: Config) -> list[str]:
    cap = redis_maxmemory_gb()
    command = (["valkey-server", "--bind", cfg.host, "--port", str(cfg.lmc_port)]
               if runtime_mode() == "native" else
               ["apptainer", "exec", REDIS_IMAGE, "redis-server", "--bind",
                cfg.host, "--port", str(cfg.lmc_port)])
    return [*command, "--save", "", "--appendonly", "no",
            *(["--maxmemory", f"{cap}gb", "--maxmemory-policy", "allkeys-lru"]
              if cap else [])]


def mp_server_cmd(cfg: Config, role: str, *, bind_host: str | None = None,
                  http_host: str | None = None, l2_host: str | None = None,
                  l2_port: int | None = None) -> list[str]:
    validate_model_runtime(cfg)
    spec = model_spec(cfg.model)
    if role == "source":
        port, http_port, default_l2_port = (
            cfg.src_lmc_port, cfg.src_lmc_http_port, cfg.kv_proxy_port)
    elif role == "sink":
        port, http_port, default_l2_port = (
            cfg.sink_lmc_port, cfg.sink_lmc_http_port, cfg.kv_proxy_port)
    else:
        raise ValueError(f"unknown MP server role: {role}")
    l2_port = l2_port or default_l2_port
    adapter = json.dumps({"type": "resp", "host": l2_host or cfg.host,
                          "port": l2_port,
                          "num_workers": 8}, separators=(",", ":"))
    bind_host, http_host = bind_host or cfg.host, http_host or cfg.host
    transfer_mode = mp_transfer_mode(cfg)
    serve = [
        "lmcache", "server", "--instance-id", f"queue-haul-{role}",
        "--host", bind_host, "--port", port, "--http-host", http_host,
        "--http-port", http_port, "--l1-size-gb", lmcache_l1_gb(),
        "--eviction-policy", "LRU",
        "--chunk-size", model_chunk_tokens(cfg),
        *(["--separate-object-groups"] if (
            cfg.architecture_campaign or cfg.capacity_discovery)
          and spec.separate_object_groups else []),
        "--max-workers", 8,
        "--supported-transfer-mode", transfer_mode, "--l2-adapter", adapter,
    ]
    script = "\n".join([
        *(["export CUDA_VISIBLE_DEVICES="]
          if transfer_mode == "engine_driven" else []),
        shell(serve),
    ])
    return apptainer_cmd(cfg, script, nv=transfer_mode == "lmcache_driven")


def vllm_cmd(cfg: Config, role: str, extra: list[str] | None = None, *,
             gpu_index: int | None = None,
             bind_host: str | None = None,
             sleep_mode: bool | None = None,
             kv_transfer: bool = True) -> list[str]:
    validate_model_runtime(cfg)
    reject_duplicate_extra(extra or [])
    if role == "source":
        port, gpu, engine_id, kv_port, rpc_port = cfg.src_port, 0, "s0", port_default(14579), "src"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.lmc_port}", "src"
    elif role == "sink":
        port, gpu, engine_id, kv_port, rpc_port = cfg.sink_port, 1, "d0", port_default(14580), "sink"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.kv_proxy_port}", "sink"
    elif role == "smoke1":
        port, gpu, engine_id, kv_port, rpc_port = cfg.smoke_port, 0, "e0", port_default(14579), "smk"
        remote_url, cache_role = f"lm://{cfg.host}:{cfg.lmc_port}", "smoke1"
    else:
        raise ValueError(f"unknown role: {role}")
    kv_role = kv_role_for(role)
    spec = model_spec(cfg.model)

    dirs = " ".join(shlex.quote(str(p)) for p in [tmpdir(cache_role), *cache_dirs(cfg, cache_role).values()])
    serve = [
        "vllm",
        "serve",
        model_path(cfg),
        "--host",
        bind_host or cfg.host,
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
        *(["--dtype", "bfloat16", *model_vllm_args(cfg)]
          if (cfg.architecture_campaign or cfg.capacity_discovery) else []),
        "--kv-cache-dtype",
        "auto",
        "--block-size",
        16,
        "--enable-chunked-prefill",
        "--enable-prefix-caching" if prefix_caching() else "--no-enable-prefix-caching",
        *(["--no-async-scheduling", "--stream-interval", 1]
          if cfg.literal_token_timing else []),
        *(["--enforce-eager"] if cfg.enforce_eager else []),
        *(["--enable-sleep-mode"] if role == "source" and (
            sleep_mode if sleep_mode is not None else lmcache_mode() == "legacy"
        ) else []),
        *(["--gpu-memory-utilization", 0.9 if (
            cfg.architecture_campaign or cfg.capacity_discovery) else 0.75,
           *([] if (cfg.architecture_campaign or cfg.capacity_discovery)
             else ["--disable-hybrid-kv-cache-manager"]),
           "--enable-prompt-tokens-details"] if lmcache_mode() == "mp" else []),
        *(["--kv-transfer-config", kv_config(
            cfg, engine_id, kv_role, kv_port, rpc_port)] if kv_transfer else []),
        *(extra or []),
    ]
    script = "\n".join([f"mkdir -p {dirs}", *vllm_exports(cfg, cache_role, remote_url), shell(serve)])
    return apptainer_cmd(cfg, script, gpu if gpu_index is None else gpu_index)


def proxy_routes(cfg: Config) -> list[Route]:
    return [
        Route("kv", cfg.host, cfg.kv_proxy_port, cfg.host, cfg.lmc_port,
              "resp" if lmcache_mode() == "mp" else "lmcache"),
        Route("api", cfg.host, cfg.api_proxy_port, cfg.host, cfg.sink_port),
    ]


def proxy_cmd(cfg: Config, mbps: float = 1000.0, log: Path | None = None) -> list[str]:
    cmd = [
        sys.executable,
        "queue-haul/migration_testbed.py",
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
    devices = allocated_gpu_ids()
    if devices:
        return len(devices)
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found")
    out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    return sum(1 for line in out.splitlines() if line.startswith("GPU "))


def runtime_versions(cfg: Config) -> tuple[str, str]:
    if lmcache_mode() == "mp":
        check = (
            "from importlib.metadata import version; "
            "from connector_patch import LMCacheMPConnector; "
            "from lmcache.integration.vllm.kv_cache_group_edits import "
            "_SubpagedAttentionViewEdit; "
            "from vllm.v1.worker.gpu_worker import Worker; "
            "assert LMCacheMPConnector._qh_bypass_patched; "
            "assert LMCacheMPConnector._qh_kv_dtype_registration_patched; "
            "assert _SubpagedAttentionViewEdit._qh_kv_first_patched; "
            "assert Worker._qh_ipc_safe_kv_allocator_patched; "
            "print('QH_RUNTIME_VERSIONS', version('vllm'), "
            "version('lmcache'))"
        )
        script = "\n".join([
            f"export PYTHONPATH={shlex.quote(str(LMCACHE_COMPAT))}",
            shell(["python", "-c", check]),
        ])
    else:
        check = "from importlib.metadata import version; from lmcache.v1.storage_backend.connector.lm_connector import LMCServerConnector; from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl; assert LMCServerConnector._qh_patched and LMCacheConnectorV1Impl._qh_bypass_patched; print('QH_RUNTIME_VERSIONS', version('vllm'), version('lmcache'))"
        script = "\n".join([f"export PYTHONPATH={shlex.quote(str(LMCACHE_COMPAT))}", shell(["/usr/bin/python3", "-c", check])])
    output = subprocess.check_output(apptainer_cmd(cfg, script, nv=False), text=True)
    _tag, vllm, lmcache = next(line for line in reversed(output.splitlines()) if line.startswith("QH_RUNTIME_VERSIONS ")).split()
    return vllm, lmcache


def preflight(cfg: Config, required_gpus: int = 1) -> list[str]:
    validate_model_runtime(cfg)
    validate_ports(cfg)
    failures = []
    native = runtime_mode() == "native"
    if native:
        runtime_bin = str(Path(sys.executable).parent)
        for executable in ("vllm", "lmcache"):
            if not shutil.which(executable, path=runtime_bin):
                failures.append(f"{executable} not found in {runtime_bin}")
        if not shutil.which("valkey-server"):
            failures.append("valkey-server not found")
    else:
        if not shutil.which("apptainer"):
            failures.append("apptainer not found")
        if not cfg.sandbox.exists():
            failures.append(f"sandbox missing: {cfg.sandbox}")
        if lmcache_mode() == "mp" and not REDIS_IMAGE.exists():
            failures.append(f"Redis image missing: {REDIS_IMAGE}")
    if not failures:
        versions = runtime_versions(cfg)
        expected = expected_runtime_versions()
        if versions != expected:
            failures.append(f"need vLLM/LMCache {expected}, saw {versions}")
    snapshot = model_path(cfg)
    if not snapshot.is_dir() or not all((snapshot / name).exists() for name in (
            "config.json", "model.safetensors.index.json", "tokenizer.json")):
        failures.append(f"model snapshot missing: {snapshot}")
    for path in [tmpdir("src"), tmpdir("sink"), tmpdir("smoke1")]:
        if len(str(path)) > 20:
            failures.append(f"TMPDIR too long for LMCache IPC: {path}")
    ports = [cfg.lmc_port, cfg.smoke_port] if required_gpus == 1 else [cfg.src_port, cfg.sink_port, cfg.lmc_port, cfg.kv_proxy_port, cfg.api_proxy_port]
    if lmcache_mode() == "mp":
        ports += [cfg.src_lmc_port, cfg.sink_lmc_port, cfg.src_lmc_http_port, cfg.sink_lmc_http_port]
    for port in ports:
        if not port_free(cfg.host, port):
            failures.append(f"port busy: {cfg.host}:{port}")
    seen_gpus = gpu_count()
    if seen_gpus < required_gpus:
        failures.append(f"need {required_gpus} GPU(s), saw {seen_gpus}")
    if failures:
        raise RuntimeError("\n".join(failures))
    return [
        f"runtime={runtime_mode()}",
        f"apptainer={shutil.which('apptainer')}",
        f"gpus={seen_gpus}",
        f"docker_present={bool(shutil.which('docker'))}",
        "real_tc=disabled_no_cap_net_admin",
        f"runtime_path={Path(sys.executable).parent}" if native else f"sandbox={cfg.sandbox}",
        f"vllm={expected_runtime_versions()[0]}",
        f"lmcache={expected_runtime_versions()[1]}",
        f"model_snapshot={snapshot}",
    ]


class TokenBucket:
    def __init__(self, rate_bps: float, burst_s: float = 0.01):
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
        return self.updated - now

    async def wait(self, nbytes: int) -> None:
        async with self.lock:
            delay = self.reserve(nbytes, time.monotonic())
        if delay:
            await asyncio.sleep(delay)

    def set_rate(self, rate_bps: float, now: float) -> None:
        if rate_bps <= 0:
            raise ValueError("rate_bps must be positive")
        self.reserve(0, now)
        self.rate = rate_bps
        self.capacity = rate_bps * 0.01
        self.tokens = min(self.tokens, self.capacity)
        self.updated = now


class BandwidthLimiter:
    def __init__(self, aggregate_bps: float | None,
                 route_bps: dict[str, float] | None = None):
        self.aggregate = TokenBucket(aggregate_bps) if aggregate_bps else None
        self.routes = {
            route: TokenBucket(rate) for route, rate in (route_bps or {}).items()
        }
        self.lock = asyncio.Lock()

    def reserve(self, route: str, direction: str, nbytes: int,
                now: float) -> float:
        if not billable(route, direction):
            return 0.0
        buckets = [bucket for bucket in (
            self.aggregate,
            self.routes.get(route) or self.routes.get(route.rsplit("/", 1)[-1]),
        ) if bucket]
        return max((bucket.reserve(nbytes, now) for bucket in buckets),
                   default=0.0)

    async def wait(self, route: str, direction: str, nbytes: int) -> None:
        async with self.lock:
            delay = self.reserve(route, direction, nbytes, time.monotonic())
        if delay:
            await asyncio.sleep(delay)

    async def update(self, aggregate_bps: float | None,
                     route_bps: dict[str, float]) -> dict:
        if aggregate_bps is not None and aggregate_bps <= 0 \
                or any(rate <= 0 for rate in route_bps.values()):
            raise ValueError("bandwidth update rates must be positive")
        async with self.lock:
            now = time.monotonic()
            if aggregate_bps is None:
                self.aggregate = None
            elif self.aggregate is None:
                self.aggregate = TokenBucket(aggregate_bps)
            else:
                self.aggregate.set_rate(aggregate_bps, now)
            updated = {}
            for route, rate in route_bps.items():
                bucket = self.routes.get(route)
                if bucket is None:
                    bucket = TokenBucket(rate)
                else:
                    bucket.set_rate(rate, now)
                updated[route] = bucket
            self.routes = updated
            return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "aggregate_bps": None if self.aggregate is None else self.aggregate.rate,
            "route_bps": {route: bucket.rate
                          for route, bucket in sorted(self.routes.items())},
        }


class ByteLog:
    interval_ns = 250_000_000

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", buffering=1)
        self.writer = csv.writer(self.file)
        self.writer.writerow(["monotonic_ns", "wall_ns", "interval_ns", "connection_id", "route", "direction", "bytes", "billed"])
        connections = path.with_name("proxy_connections.csv")
        self.connections = connections.open("w", newline="", buffering=1)
        self.connection_writer = csv.writer(self.connections)
        self.connection_writer.writerow([
            "connection_id", "route", "key_hash", "start_ns", "end_ns",
            "client_to_target_bytes", "target_to_client_bytes",
            "client_first_byte_ns", "client_last_byte_ns",
            "target_first_byte_ns", "target_last_byte_ns",
            "client_rtt_us", "client_rttvar_us", "client_snd_cwnd",
            "client_total_retrans", "target_rtt_us", "target_rttvar_us",
            "target_snd_cwnd", "target_total_retrans",
        ])
        self.transfers = path.with_name("resp_transfers.csv").open("w", newline="", buffering=1)
        self.transfer_writer = csv.writer(self.transfers)
        self.transfer_writer.writerow(["connection_id", "command", "key_hashes", "start_ns", "end_ns", "request_wire_bytes", "response_wire_bytes", "request_body_bytes", "payload_bytes"])
        self.buckets: dict[tuple[int, str, str, str, bool], int] = {}
        self.lock = asyncio.Lock()
        self.task: asyncio.Task | None = None
        self.active = 0
        self.idle = asyncio.Event()
        self.idle.set()

    async def start(self) -> None:
        self.task = asyncio.create_task(self._flush_loop())

    async def add(self, connection_id: str, route: str, direction: str,
                  nbytes: int, billed: bool) -> None:
        async with self.lock:
            bucket = time.monotonic_ns() // self.interval_ns * self.interval_ns
            key = bucket, connection_id, route, direction, billed
            self.buckets[key] = self.buckets.get(key, 0) + nbytes

    async def opened(self) -> None:
        async with self.lock:
            self.active += 1
            self.idle.clear()

    async def connection(self, connection_id: str, route: str, key_hash: str,
                         start_ns: int, counts: tuple[int, int],
                         timing: tuple = ("", "", "", ""),
                         tcp: tuple[dict, dict] = ({}, {})) -> None:
        async with self.lock:
            self.connection_writer.writerow([
                connection_id, route, key_hash, start_ns, time.monotonic_ns(),
                *counts, *timing,
                *(tcp[0].get(key, "") for key in (
                    "rtt_us", "rttvar_us", "snd_cwnd", "total_retrans")),
                *(tcp[1].get(key, "") for key in (
                    "rtt_us", "rttvar_us", "snd_cwnd", "total_retrans")),
            ])
            self.active -= 1
            if not self.active:
                self.idle.set()

    async def resp_transfer(self, row: list) -> None:
        async with self.lock:
            self.transfer_writer.writerow(row)
            self.transfers.flush()

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self.interval_ns / 1e9)
            await self.flush()

    async def flush(self, all_rows: bool = False) -> None:
        cutoff = time.monotonic_ns() // self.interval_ns * self.interval_ns
        async with self.lock:
            keys = [key for key in self.buckets if all_rows or key[0] < cutoff]
            for bucket, connection_id, route, direction, billed in sorted(keys):
                self.writer.writerow([bucket, time.time_ns(), self.interval_ns,
                                      connection_id, route, direction,
                                      self.buckets.pop((bucket, connection_id, route, direction, billed)),
                                      int(billed)])

    async def close(self) -> None:
        await self.idle.wait()
        if self.task:
            self.task.cancel()
        await self.flush(all_rows=True)
        self.file.close()
        self.connections.close()
        self.transfers.close()


def billable(route: str, direction: str) -> bool:
    return (route.split("/", 1)[0], direction) in BILLED_DIRECTIONS


def parse_tcp_info(blob: bytes) -> dict[str, int]:
    if len(blob) < 104:
        return {}
    value = lambda offset: int.from_bytes(blob[offset:offset + 4], "little")
    return {
        "rtt_us": value(68), "rttvar_us": value(72),
        "snd_cwnd": value(80), "total_retrans": value(100),
    }


def stream_tcp_info(writer: asyncio.StreamWriter) -> dict[str, int]:
    sock = writer.get_extra_info("socket")
    if not sock or sock.fileno() < 0 or not hasattr(socket, "TCP_INFO"):
        return {}
    return parse_tcp_info(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104))


def kv_key_hash(header: bytes) -> str:
    raw_key = LMCACHE_CLIENT_META.unpack(header)[-1].rstrip(b" \0")
    return hashlib.sha256(raw_key).hexdigest()


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



@dataclass(frozen=True)
class RespFrame:
    raw: bytes
    payload: bytes | None = None
    items: tuple[RespFrame, ...] = ()

    def body_bytes(self) -> int:
        return len(self.payload) if self.payload is not None else sum(item.body_bytes() for item in self.items)


async def read_resp(reader: asyncio.StreamReader) -> RespFrame:
    prefix = await reader.readexactly(1)
    line = await reader.readline()
    if not line.endswith(b"\r\n"):
        raise ValueError("invalid RESP line")
    raw = prefix + line
    if prefix == b"$":
        length = int(line[:-2])
        if length < 0:
            return RespFrame(raw)
        body = await reader.readexactly(length + 2)
        if not body.endswith(b"\r\n"):
            raise ValueError("invalid RESP bulk trailer")
        return RespFrame(raw + body, body[:-2])
    if prefix == b"*":
        count = int(line[:-2])
        items = tuple([await read_resp(reader) for _ in range(max(0, count))])
        return RespFrame(raw + b"".join(item.raw for item in items), items=items)
    if prefix not in b"+-:":
        raise ValueError(f"unsupported RESP prefix: {prefix!r}")
    return RespFrame(raw, line[:-2])


def resp_request(frame: RespFrame) -> tuple[str, list[str]]:
    if not frame.items or frame.items[0].payload is None:
        raise ValueError("RESP request is not an array command")
    command = frame.items[0].payload.decode().upper()
    key = frame.items[1].payload if len(frame.items) > 1 else None
    return command, [hashlib.sha256(key).hexdigest()] if key is not None and command in {"GET", "SET", "EXISTS"} else []


async def relay_resp(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter,
                     target_r: asyncio.StreamReader, target_w: asyncio.StreamWriter,
                     limiter: BandwidthLimiter, log: ByteLog | None,
                     connection_id: str, route: str) -> tuple[tuple[int, int], tuple]:
    pending: asyncio.Queue = asyncio.Queue()
    counts, timing = [0, 0], [None] * 4

    async def requests() -> None:
        try:
            while True:
                start = time.monotonic_ns()
                frame = await read_resp(client_r)
                command, keys = resp_request(frame)
                target_w.write(frame.raw)
                await target_w.drain()
                now = time.monotonic_ns()
                timing[0] = timing[0] or now
                timing[1] = now
                counts[0] += len(frame.raw)
                if log:
                    await log.add(connection_id, route, "client_to_target", len(frame.raw), False)
                await pending.put((command, keys, start, len(frame.raw), frame.body_bytes()))
        except asyncio.IncompleteReadError:
            if target_w.can_write_eof():
                target_w.write_eof()
        finally:
            await pending.put(None)

    async def responses() -> None:
        try:
            while True:
                request = await pending.get()
                if request is None:
                    return
                command, keys, start, request_wire, request_body = request
                frame = await read_resp(target_r)
                await limiter.wait(route, "target_to_client", len(frame.raw))
                client_w.write(frame.raw)
                await client_w.drain()
                end = time.monotonic_ns()
                timing[2] = timing[2] or end
                timing[3] = end
                counts[1] += len(frame.raw)
                if log:
                    await log.add(connection_id, route, "target_to_client", len(frame.raw), True)
                    await log.resp_transfer([connection_id, command, ";".join(keys), start, end,
                                             request_wire, len(frame.raw), request_body, frame.body_bytes()])
                pending.task_done()
        except asyncio.IncompleteReadError:
            return

    await asyncio.gather(requests(), responses())
    return tuple(counts), tuple(value or "" for value in timing)

async def relay(reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                limiter: BandwidthLimiter, log: ByteLog | None, connection_id: str,
                route: str, direction: str, initial: bytes = b"") -> tuple:
    total, first, last = 0, None, None
    try:
        charged = billable(route, direction)
        data = initial or await reader.read(CHUNK)
        while data:
            if charged:
                await limiter.wait(route, direction, len(data))
            writer.write(data)
            await writer.drain()
            last = time.monotonic_ns()
            first = first or last
            total += len(data)
            if log:
                await log.add(connection_id, route, direction, len(data), charged)
            data = await reader.read(CHUNK)
    finally:
        writer.close()
        await writer.wait_closed()
    return total, first or "", last or ""


async def handle_proxy(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter,
                       route: Route, limiter: BandwidthLimiter,
                       log: ByteLog | None) -> None:
    start = time.monotonic_ns()
    connection_id = hashlib.sha256(f"{route.name}:{start}".encode()).hexdigest()[:16]
    target_r, target_w = await asyncio.open_connection(route.target_host, route.target_port)
    initial, key_hash = b"", ""
    if route.protocol == "lmcache":
        initial = await client_r.readexactly(LMCACHE_CLIENT_META.size)
        key_hash = kv_key_hash(initial)
    if log:
        await log.opened()
    counts, timing = (0, 0), ("", "", "", "")
    try:
        if route.protocol == "resp":
            counts, timing = await relay_resp(
                client_r, client_w, target_r, target_w, limiter, log,
                connection_id, route.name)
        else:
            flows = await asyncio.gather(
                relay(client_r, target_w, limiter, log, connection_id,
                      route.name, "client_to_target", initial),
                relay(target_r, client_w, limiter, log, connection_id,
                      route.name, "target_to_client"))
            counts = tuple(row[0] for row in flows)
            timing = tuple(value for row in flows for value in row[1:])
    finally:
        tcp = stream_tcp_info(client_w), stream_tcp_info(target_w)
        target_w.close()
        client_w.close()
        if log:
            await log.connection(
                connection_id, route.name, key_hash, start, counts, timing, tcp)


async def start_proxy(routes: list[Route], rate_bps: float | None,
                      log: Path | None = None,
                      route_bps: dict[str, float] | None = None,
                      control_socket: Path | None = None,
                      ) -> tuple[list[asyncio.AbstractServer], ByteLog | None]:
    limiter = BandwidthLimiter(rate_bps, route_bps)
    byte_log = ByteLog(log) if log else None
    if byte_log:
        await byte_log.start()
    servers = []
    for route in routes:
        servers.append(
            await asyncio.start_server(
                lambda r, w, route=route: handle_proxy(
                    r, w, route, limiter, byte_log),
                route.listen_host,
                route.listen_port,
            )
        )
    if control_socket is not None:
        control_socket.parent.mkdir(parents=True, exist_ok=True)
        if control_socket.exists():
            control_socket.unlink()
        control_log = control_socket.with_suffix(".jsonl").open(
            "w", buffering=1)

        async def control(reader, writer):
            try:
                raw = json.loads((await reader.readline()).decode())
                aggregate = raw.get("aggregate_bps")
                rates = {str(key): float(value)
                         for key, value in raw.get("route_bps", {}).items()}
                valid = {route.name for route in routes} | {
                    route.name.rsplit("/", 1)[-1] for route in routes}
                if set(rates) - valid:
                    raise ValueError("bandwidth update contains an unknown route")
                snapshot = await limiter.update(
                    None if aggregate is None else float(aggregate), rates)
                record = {"monotonic_ns": time.monotonic_ns(),
                          "wall_ns": time.time_ns(), **snapshot}
                control_log.write(json.dumps(record, separators=(",", ":")) + "\n")
                response = {"ok": True, **snapshot}
            except Exception as exc:
                response = {"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}
            writer.write((json.dumps(response, separators=(",", ":")) + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        server = await asyncio.start_unix_server(control, path=control_socket)
        server.qh_control_log = control_log
        server.qh_control_socket = control_socket
        servers.append(server)
    return servers, byte_log


async def run_proxy(routes: list[Route], rate_bps: float | None,
                    log: Path | None = None,
                    route_bps: dict[str, float] | None = None,
                    control_socket: Path | None = None) -> None:
    servers, byte_log = await start_proxy(
        routes, rate_bps, log, route_bps, control_socket)
    try:
        await asyncio.gather(*(server.serve_forever() for server in servers))
    finally:
        for server in servers:
            server.close()
            await server.wait_closed()
            if hasattr(server, "qh_control_log"):
                server.qh_control_log.close()
                if server.qh_control_socket.exists():
                    server.qh_control_socket.unlink()
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


def health_timeout() -> float:
    return max(60, float(os.environ.get("QH_HEALTH_TIMEOUT_S", "3600")))


def prompt_text(session_id: str, words: int = 4096) -> str:
    body = " ".join(f"{session_id}_{i % 97}" for i in range(words))
    return f"Session {session_id}. {body}. Reply with exactly OK."


def chat_payload(cfg: Config, prompt: str, max_tokens: int = 4) -> str:
    payload = {"model": cfg.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": 0}
    if lmcache_mode() == "mp":
        payload["reasoning_effort"] = "low"
    return json.dumps(payload)


def post_chat(cfg: Config, port: int, prompt: str, max_tokens: int = 4) -> dict:
    body = chat_payload(cfg, prompt, max_tokens)
    t0 = time.time()
    conn = http.client.HTTPConnection(cfg.host, port, timeout=600)
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    text = resp.read().decode()
    conn.close()
    t1 = time.time()
    parsed = json.loads(text) if resp.status == 200 else {}
    content = parsed.get("choices", [{}])[0].get("message", {}).get("content") or ""
    return {
        "status": resp.status,
        "id": parsed.get("id", ""),
        "usage": parsed.get("usage", {}),
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
    for proc in (stack.sink, stack.source, stack.proxy, *stack.cache_services, stack.lmcache):
        if proc:
            stop_proc(proc)


def flush_lmcache(stack: Stack, cfg: Config | None = None) -> None:
    cfg = cfg or Config()
    if stack.lmcache.poll() is not None:
        raise RuntimeError("LMCache server is not running")
    if lmcache_mode() == "mp":
        with socket.create_connection((cfg.host, cfg.lmc_port)) as sock:
            sock.sendall(b"*1\r\n$8\r\nFLUSHALL\r\n")
            if not sock.recv(64).startswith(b"+OK"):
                raise RuntimeError("Redis FLUSHALL failed")
        for port in (cfg.src_lmc_http_port, cfg.sink_lmc_http_port):
            http_text(cfg.host, port, "POST", "/cache/clear")
        return
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


def http_text(host: str, port: int, method: str, path: str, *,
              timeout_s: float = 30) -> str:
    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    try:
        conn.request(method, path)
        response = conn.getresponse()
        body = response.read().decode(errors="ignore")
    finally:
        conn.close()
    if response.status != 200:
        raise RuntimeError(f"{method} http://{host}:{port}{path} failed {response.status}: {body[:500]}")
    return body


def http_json(host: str, port: int, method: str, path: str,
              payload=None, statuses=(200,)) -> dict:
    conn = http.client.HTTPConnection(host, port, timeout=600)
    try:
        conn.request(method, path, json.dumps(payload) if payload is not None else None,
                     {"Content-Type": "application/json"})
        response, body = conn.getresponse(), None
        body = response.read().decode()
    finally:
        conn.close()
    if response.status not in statuses:
        raise RuntimeError(f"{method} {path} failed {response.status}: {body[:500]}")
    return json.loads(body)


def mp_chat_tokens(cfg: Config, messages: list[dict],
                   max_tokens: int = 512) -> list[int]:
    result = http_json(cfg.host, cfg.src_port, "POST",
                       "/v1/chat/completions/render", {
        "model": cfg.model, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0, "reasoning_effort": "low", "stream": True,
        "stream_options": {"include_usage": True},
    })
    tokens = result.get("token_ids")
    if not tokens:
        raise RuntimeError("vLLM did not render exact chat token IDs")
    return tokens


def mp_model_layout(log: Path) -> tuple[str, int]:
    match = MP_CONTEXT.search(read_text(log))
    if not match:
        raise RuntimeError("LMCache did not report its model layout")
    return match.group(1), int(match.group(2))


def mp_warm_prefetch(cfg: Config, tokens: list[int], model: str,
                     world_size: int) -> dict:
    result = http_json(cfg.host, cfg.sink_lmc_http_port, "POST",
                       "/cache/prefetches", {
                           "model_name": model, "world_size": world_size,
                           "token_ids": tokens,
                       }, (202,))
    request_id = result.get("request_id")
    if not request_id:
        raise RuntimeError("warm prefetch was not submitted")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        status = http_json(
            cfg.host, cfg.sink_lmc_http_port, "GET",
            f"/cache/prefetches/{request_id}",
        )
        if status["status"] == "completed":
            return status
        time.sleep(.05)
    raise TimeoutError(f"warm prefetch {request_id} did not complete")


def resp_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def mp_source_keys(path: Path, offset: int) -> set[str]:
    with path.open() as handle:
        fields = next(csv.reader(handle))
    with path.open("rb") as handle:
        handle.seek(offset)
        rows = csv.DictReader(io.StringIO(handle.read().decode()), fieldnames=fields)
        return {row["key_hashes"] for row in rows if row["command"] == "SET"}


def mp_wait_idle(path: Path, idle_s: float = 2) -> None:
    deadline = time.monotonic() + 600
    size, since = path.stat().st_size, time.monotonic()
    while time.monotonic() < deadline:
        current = path.stat().st_size
        if current != size:
            size, since = current, time.monotonic()
        elif time.monotonic() - since >= idle_s:
            return
        time.sleep(.05)
    raise TimeoutError("LMCache RESP transfers did not become idle")


def mp_wait_stored(log: Path, offset: int, tokens: int) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if sum(map(int, MP_STORED.findall(read_after(log, offset)))) >= tokens:
            return
        time.sleep(.05)
    raise TimeoutError(f"LMCache stored fewer than {tokens} tokens")


def mp_wait_source_keys(log: Path, offset: int, transfers: Path,
                        transfer_offset: int, tokens: int,
                        known_keys: set[str] | None = None,
                        chunk_tokens: int = 256) -> set[str]:
    expected = tokens // chunk_tokens
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        keys = mp_source_keys(transfers, transfer_offset) - (known_keys or set())
        if len(keys) >= expected:
            return keys
        time.sleep(.05)
    raise TimeoutError(f"LMCache stored fewer than {expected} unique keys")


def mp_request_hit(log: Path, offset: int, request_id: str,
                   require_all: bool = True, chunk_tokens: int = 256) -> int:
    matches = [
        tuple(map(int, match.groups()[:4]))
        for match in MP_REQUEST.finditer(read_after(log, offset))
        if match.group(5) == request_id
        or match.group(5).startswith(request_id + "-")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"LMCache reported {len(matches)} records for {request_id}")
    retained, queried, l1, l2 = matches[0]
    if retained != l1 or l2 or require_all and retained != queried:
        raise RuntimeError(f"request {request_id} was not L1-only")
    return retained * chunk_tokens


def set_source_sleep(cfg: Config, sleeping: bool) -> None:
    def state() -> bool:
        return json.loads(http_text(cfg.host, cfg.src_port, "GET", "/is_sleeping"))["is_sleeping"]

    if state() == sleeping:
        return
    http_text(
        cfg.host, cfg.src_port, "POST",
        "/sleep?level=1" if sleeping else "/wake_up", timeout_s=600,
    )
    if state() != sleeping:
        raise RuntimeError(f"source failed to become {'sleeping' if sleeping else 'awake'}")


def reset_result(text: str) -> bool | None:
    if "Failed to reset prefix cache" in text:
        return False
    if "Successfully reset prefix cache" in text:
        return True
    return None


def reset_vllm_caches(cfg: Config, logs: tuple[Path, ...],
                      ports: tuple[int, ...] | None = None) -> None:
    """Reset prefix caches; `ports` narrows the reset to a subset of engines."""
    for port, log in zip(ports or (cfg.src_port, cfg.sink_port), logs):
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


def read_after(path: Path, offset: int) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(offset)
        return handle.read().decode(errors="ignore")


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
    lmc = start_logged(redis_cmd(cfg) if lmcache_mode() == "mp" else lmcache_cmd(cfg), run_root / "lmcache.log")
    proxy = source = None
    services = []
    try:
        wait_tcp_process(cfg.host, cfg.lmc_port, 60, lmc, run_root / "lmcache.log")
        proxy = start_logged(proxy_cmd(cfg, mbps, run_root / "proxy_bytes.csv"), run_root / "proxy.log")
        wait_tcp(cfg.host, cfg.kv_proxy_port, 60)
        wait_tcp(cfg.host, cfg.api_proxy_port, 60)
        if lmcache_mode() == "mp":
            for role, port in (("source", cfg.src_lmc_port), ("sink", cfg.sink_lmc_port)):
                log = run_root / f"lmcache-{role}.log"
                service = start_logged(mp_server_cmd(cfg, role), log)
                services.append(service)
                wait_tcp_process(cfg.host, port, 300, service, log)
        source = start_logged(vllm_cmd(cfg, "source", extra or []), run_root / "source.log")
        wait_health_process(cfg.host, cfg.src_port, health_timeout(), source, run_root / "source.log")
        validate_model_runtime_log(cfg, read_text(run_root / "source.log"))
        return Stack(lmc, proxy, source, None, run_root, services, mbps)
    except BaseException:
        for proc in (source, proxy, *services, lmc):
            if proc:
                stop_proc(proc)
        raise


def start_sink(stack: Stack, cfg: Config, extra: list[str] | None = None) -> None:
    if stack.sink:
        return
    stack.sink = start_logged(vllm_cmd(cfg, "sink", extra or []), stack.run_root / "sink.log")
    wait_health_process(cfg.host, cfg.sink_port, health_timeout(), stack.sink, stack.run_root / "sink.log")
    validate_model_runtime_log(cfg, read_text(stack.run_root / "sink.log"))


def check_chat(result: dict, label: str) -> None:
    if result["status"] != 200:
        raise RuntimeError(f"{label} failed {result['status']}: {result['response_text']}")


def warm_source(cfg: Config, run_root: Path, prompt: str, label: str = "source warm") -> tuple[dict, int]:
    source_log = run_root / ("lmcache-source.log" if lmcache_mode() == "mp" else "source.log")
    stored0 = count_needle(source_log, "Stored")
    source = post_chat(cfg, cfg.src_port, prompt, 4)
    check_chat(source, label)
    time.sleep(2)
    if count_needle(source_log, "Stored") <= stored0:
        raise RuntimeError(f"{label} did not store KV")
    return source, stored0


def run_smoke2_probe(cfg: Config, run_root: Path, mbps: float, words: int = 4096, prompt: str | None = None, prewarmed: tuple[dict, int] | None = None) -> dict:
    proxy_log = run_root / "proxy_bytes.csv"
    source_log = run_root / ("lmcache-source.log" if lmcache_mode() == "mp" else "source.log")
    sink_log = run_root / ("lmcache-sink.log" if lmcache_mode() == "mp" else "sink.log")
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
            "source_connector": "Registered non-GPU context" in read_text(source_log) if lmcache_mode() == "mp" else "engine_id: s0" in read_text(source_log) or "engine_id=s0" in read_text(source_log),
            "sink_connector": "Registered non-GPU context" in read_text(sink_log) if lmcache_mode() == "mp" else "engine_id: d0" in read_text(sink_log) or "engine_id=d0" in read_text(sink_log),
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
        validate_model_runtime_log(cfg, read_text(run_root / "vllm.log"))
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
    p = argparse.ArgumentParser(description="Queue-Haul source/sink migration testbed")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "smoke1", "smoke2", "smoke2-live"):
        sp = sub.add_parser(name)
        add_common(sp)
        if name == "preflight":
            sp.add_argument("--required-gpus", type=int, default=1)
        if name in ("smoke1", "smoke2", "smoke2-live"):
            sp.add_argument("--run-root", type=Path, default=Path(f"queue-haul/runs/migration_testbed/{name}"))
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
    sp.add_argument("--mbps", type=float)
    sp.add_argument("--routes-json")
    sp.add_argument("--aggregate-mbps", type=float)
    sp.add_argument("--route-mbps-json")
    sp.add_argument("--log", type=Path)
    sp.add_argument("--control-socket", type=Path)
    return p.parse_args(argv)


def proxy_config(args) -> tuple[list[Route], float | None, dict[str, float]]:
    if args.routes_json:
        routes = [Route(**raw) for raw in json.loads(args.routes_json)]
        if args.mbps is not None:
            raise ValueError("network routes use --aggregate-mbps, not --mbps")
        aggregate = args.aggregate_mbps
        rates = json.loads(args.route_mbps_json or "{}")
    else:
        if args.mbps is None:
            raise ValueError("legacy proxy requires --mbps")
        kv_listen = parse_addr(args.kv_listen)
        kv_target = parse_addr(args.kv_target)
        api_listen = parse_addr(args.api_listen)
        api_target = parse_addr(args.api_target)
        routes = [Route("kv", *kv_listen, *kv_target,
                        "resp" if lmcache_mode() == "mp" else "lmcache"),
                  Route("api", *api_listen, *api_target)]
        aggregate, rates = args.mbps, {}
    route_keys = {route.name for route in routes}
    route_keys |= {route.rsplit("/", 1)[-1] for route in route_keys}
    if aggregate is not None and aggregate <= 0 \
            or any(float(rate) <= 0 for rate in rates.values()) \
            or set(rates) - route_keys:
        raise ValueError("invalid proxy bandwidth contract")
    return routes, (aggregate * 1_000_000 / 8 if aggregate else None), {
        route: float(rate) * 1_000_000 / 8 for route, rate in rates.items()
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.cmd == "lmcache-server":
        run_lmcache_server(args.host, args.port, args.max_bytes)
        return
    if args.cmd == "proxy":
        routes, aggregate, rates = proxy_config(args)
        asyncio.run(run_proxy(
            routes, aggregate, args.log, rates, args.control_socket))
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
