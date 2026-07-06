from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

from stage1_curves import reject_duplicate_extra, shell

DEFAULT_PREFILL_LENS = (256, 1024, 4096, 8192, 16384, 32767)
DEFAULT_DECODE_PROMPTS = (256, 4096, 8192, 16384, 28672)
DEFAULT_MIXED_PREFILL = (256, 16384)


def pruned_lens(max_model_len: int, explicit: list[int] | None, defaults=DEFAULT_PREFILL_LENS) -> list[int]:
    limit = max_model_len - 1
    lens = list(defaults if explicit is None else explicit)
    bad = [n for n in lens if n > limit]
    if explicit and bad:
        raise ValueError(f"requested lengths exceed max_model_len: {bad}")
    kept = [n for n in lens if n <= limit]
    if not kept:
        raise ValueError("no lengths fit max_model_len")
    return kept


def decode_prompts(max_model_len: int, output_len: int, explicit: list[int] | None) -> list[int]:
    limit = max_model_len - output_len - 64
    if limit <= 0:
        raise ValueError("decode output_len leaves no prompt context")
    lens = list(DEFAULT_DECODE_PROMPTS if explicit is None else explicit)
    bad = [n for n in lens if n > limit]
    if explicit and bad:
        raise ValueError(f"decode prompt lengths exceed served window: {bad}")
    kept = [n for n in lens if n <= limit]
    if not kept:
        raise ValueError("no decode prompt lengths fit served window")
    return kept


def mixed_prefill_range(max_model_len: int, output_len: int, lo: int, hi: int) -> tuple[int, int]:
    limit = max_model_len - output_len - 64
    if lo <= 0 or hi < lo or hi > limit:
        raise ValueError("mixed prefill range must be positive, ordered, and fit served window")
    return int(lo), int(hi)


def serve_cmd(args, chunked: bool, extra: list[str]) -> str:
    reject_duplicate_extra(extra)
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name or args.model,
        "--tensor-parallel-size",
        args.tp,
        "--max-num-seqs",
        args.max_num_seqs,
        "--max-num-batched-tokens",
        args.max_num_batched_tokens,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--max-model-len",
        args.max_model_len,
    ]
    if chunked:
        cmd.append("--enable-chunked-prefill")
    return shell(cmd + extra)


def probe_common(args, out_root: Path, probe: str) -> list:
    return [
        args.python,
        args.powertrace_root / "profiling" / "probes" / f"{probe}.py",
        "--model",
        args.model,
        "--hardware",
        args.hardware,
        "--tp",
        args.tp,
        "--gpus-per-node",
        args.gpus_per_node,
        "--max-num-seqs",
        args.max_num_seqs,
        "--max-num-batched-tokens",
        args.max_num_batched_tokens,
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--max-model-len",
        args.max_model_len,
        "--hold-s",
        args.hold_s,
        "--out-root",
        out_root,
    ]


def prefill_cmd(args, out_root: Path, lens: list[int]) -> str:
    return shell(probe_common(args, out_root, "prefill_staircase") + ["--input-lens", *lens])


def decode_cmd(args, out_root: Path, prompt_len: int) -> str:
    return shell(
        probe_common(args, out_root, "decode_staircase")
        + ["--prompt-len", prompt_len, "--output-len", args.decode_output_len]
    )


def mixed_cmd(args, out_root: Path, prefill_range: tuple[int, int]) -> str:
    return shell(
        probe_common(args, out_root, "mixed_grid")
        + [
            "--n-points",
            args.mixed_points,
            "--seed",
            args.mixed_seed,
            "--prefill-min",
            prefill_range[0],
            "--prefill-max",
            prefill_range[1],
            "--output-len",
            args.mixed_output_len,
        ]
    )


def runbook(args, extra: list[str]) -> str:
    prefill = pruned_lens(args.max_model_len, args.prefill_lens)
    decode = decode_prompts(args.max_model_len, args.decode_output_len, args.decode_prompt_lens)
    mixed_prefill = mixed_prefill_range(
        args.max_model_len, args.mixed_output_len, args.mixed_prefill_min, args.mixed_prefill_max
    )
    run_root = args.run_root / (args.run_id or time.strftime("%Y%m%d-%H%M%S"))
    out_root = run_root / "bundles"
    logs = run_root / "logs"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(args.powertrace_root))}",
        f"mkdir -p {shlex.quote(str(out_root))} {shlex.quote(str(logs))}",
        f"export RUNS={shlex.quote(str(out_root))}",
        f"export LOGS={shlex.quote(str(logs))}",
        'APP="${APP:-}"',
        "source profiling/jobs/server_lifecycle.sh",
        "",
        "run_probe() {",
        "  local name=\"$1\" serve=\"$2\" probe=\"$3\"",
        "  trap stop_server EXIT",
        "  SERVER_LOG=\"$LOGS/server-${name}.log\" start_server \"$APP $serve\"",
        "  eval \"$APP $probe\"",
        "  stop_server",
        "  trap - EXIT",
        "}",
        "",
        f"run_probe prefill_rho {shlex.quote(serve_cmd(args, False, extra))} {shlex.quote(prefill_cmd(args, out_root, prefill))}",
    ]
    for prompt_len in decode:
        lines.append(
            f"run_probe decode_g_t{prompt_len} "
            f"{shlex.quote(serve_cmd(args, True, extra))} "
            f"{shlex.quote(decode_cmd(args, out_root, prompt_len))}"
        )
    lines.append(
        f"run_probe mixed_surface {shlex.quote(serve_cmd(args, True, extra))} "
        f"{shlex.quote(mixed_cmd(args, out_root, mixed_prefill))}"
    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None):
    root = Path(__file__).resolve().parents[3] / "powertrace-sim"
    p = argparse.ArgumentParser(description="Queue-Haul Stage 1a service-surface runbook")
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--served-model-name")
    p.add_argument("--hardware", choices=("A100", "H100"), default="A100")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpus-per-node", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--hold-s", type=float, default=45.0)
    p.add_argument("--prefill-lens", nargs="*", type=int)
    p.add_argument("--decode-prompt-lens", nargs="*", type=int)
    p.add_argument("--decode-output-len", type=int, default=512)
    p.add_argument("--mixed-output-len", type=int, default=512)
    p.add_argument("--mixed-prefill-min", type=int, default=DEFAULT_MIXED_PREFILL[0])
    p.add_argument("--mixed-prefill-max", type=int, default=DEFAULT_MIXED_PREFILL[1])
    p.add_argument("--mixed-points", type=int, default=16)
    p.add_argument("--mixed-seed", type=int, default=0)
    p.add_argument("--powertrace-root", type=Path, default=root)
    p.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent / "runs" / "stage1_service_surface")
    p.add_argument("--run-id")
    p.add_argument("--python", default="python3")
    p.add_argument("--execute", action="store_true")
    p.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    extra = args.extra_vllm_args[1:] if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
    script = runbook(args, extra)
    out = args.run_root / args.run_id / "commands.sh"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script)
    out.chmod(0o755)
    print(out)
    if args.execute:
        subprocess.run(["bash", str(out)], check=True)


if __name__ == "__main__":
    main()
