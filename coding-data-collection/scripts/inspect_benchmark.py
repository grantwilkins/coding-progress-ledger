from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_data_collection.benchmarks import (
    HarborTerminalBenchAdapter,
    SWEBenchProAdapter,
    TerminalBenchHFAdapter,
)
from coding_data_collection.benchmarks.adapters import write_registry_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect benchmark metadata rows.")
    parser.add_argument("--source", required=True, choices=["terminal_bench_hf", "terminal_bench_harbor", "swe_bench_pro"])
    parser.add_argument("--input-jsonl", help="Local JSONL rows for HF/SWE inspect mode.")
    parser.add_argument("--task-id", action="append", help="Harbor task id; can be repeated.")
    parser.add_argument("--out", default="manifests/benchmark_registry_manifest.csv")
    args = parser.parse_args(argv)

    if args.source == "terminal_bench_harbor":
        if not args.task_id:
            parser.error("--task-id is required for terminal_bench_harbor")
        tasks = HarborTerminalBenchAdapter().inspect_task_ids(args.task_id)
    else:
        if not args.input_jsonl:
            parser.error("--input-jsonl is required for this source")
        rows = [
            json.loads(line)
            for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        adapter = TerminalBenchHFAdapter() if args.source == "terminal_bench_hf" else SWEBenchProAdapter()
        tasks = adapter.inspect_rows(rows)

    write_registry_manifest(tasks, Path(args.out))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

