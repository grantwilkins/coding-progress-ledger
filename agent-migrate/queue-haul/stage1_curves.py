from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

DEFAULT_PREFILL_LENS = (256, 1024, 4096, 16384, 65536)
PROBES = ("decode_staircase", "prefill_staircase", "mixed_grid")
TYPED_VLLM_FLAGS = {
    "--served-model-name",
    "--tensor-parallel-size",
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--kv-cache-dtype",
    "--max-model-len",
    "--enable-chunked-prefill",
}


def shell(argv: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(x)) for x in argv)


def reject_duplicate_extra(extra: list[str]) -> None:
    for tok in extra:
        if tok.split("=", 1)[0] in TYPED_VLLM_FLAGS:
            raise ValueError(f"extra vLLM arg duplicates typed flag: {tok}")


def pruned_prefill_lens(max_model_len: int, explicit: list[int] | None) -> list[int]:
    limit = max_model_len - 1
    lens = list(DEFAULT_PREFILL_LENS if explicit is None else explicit)
    bad = [n for n in lens if n > limit]
    if explicit and bad:
        raise ValueError(f"requested prefill lengths exceed max_model_len: {bad}")
    kept = [n for n in lens if n <= limit]
    if not kept:
        raise ValueError("no prefill lengths fit max_model_len")
    return kept


def serve_cmd(args, probe: str, extra: list[str]) -> str:
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
    if probe != "prefill_staircase":
        cmd.append("--enable-chunked-prefill")
    return shell(cmd + extra)


def probe_cmd(args, probe: str, out_root: Path, lens: list[int]) -> str:
    common = [
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
    if probe == "prefill_staircase":
        common += ["--input-lens", *lens]
    if probe == "decode_staircase":
        common += ["--output-len", args.decode_output_len]
    if probe == "mixed_grid":
        common += [
            "--n-points",
            args.mixed_points,
            "--prefill-min",
            min(lens),
            "--prefill-max",
            max(lens),
            "--output-len",
            args.mixed_output_len,
        ]
    return shell(common)


def runbook(args, extra: list[str]) -> str:
    lens = pruned_prefill_lens(args.max_model_len, args.prefill_lens)
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
    ]
    for probe in args.probes:
        lines.append(
            f"run_probe {shlex.quote(probe)} "
            f"{shlex.quote(serve_cmd(args, probe, extra))} "
            f"{shlex.quote(probe_cmd(args, probe, out_root, lens))}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None):
    root = Path(__file__).resolve().parents[3] / "powertrace-sim"
    p = argparse.ArgumentParser(description="Queue-Haul Stage 1a powertrace-sim runbook")
    p.add_argument("--model", required=True)
    p.add_argument("--served-model-name")
    p.add_argument("--hardware", choices=("A100", "H100"), default="H100")
    p.add_argument("--tp", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--hold-s", type=float, default=45.0)
    p.add_argument("--prefill-lens", nargs="*", type=int)
    p.add_argument("--decode-output-len", type=int, default=2048)
    p.add_argument("--mixed-output-len", type=int, default=512)
    p.add_argument("--mixed-points", type=int, default=16)
    p.add_argument("--probes", nargs="+", choices=PROBES, default=list(PROBES))
    p.add_argument("--powertrace-root", type=Path, default=root)
    p.add_argument("--run-root", type=Path, default=Path(__file__).resolve().parent / "runs" / "stage1")
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
