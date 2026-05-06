#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from agent_migrate_agent.cut_points import find_cut_points, load_trace_jsonl
from agent_migrate_agent.resume_ablation import run_resume_ablation, write_ablation_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build C4 static cut-and-resume ablation table")
    parser.add_argument("trace", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--trace-id", default=None)
    parser.add_argument("--cwd", default="/workspace")
    parser.add_argument("--open-file", default="")
    args = parser.parse_args()

    events = load_trace_jsonl(args.trace)
    trace_id = args.trace_id or args.trace.stem
    cuts = find_cut_points(events, trace_id=trace_id)
    harness = {"cwd": args.cwd, "open_file": args.open_file, "env": {}}
    rows = run_resume_ablation(events, cuts, harness_config=harness)
    write_ablation_csv(rows, args.out_dir / "ablation.csv")


if __name__ == "__main__":
    main()
