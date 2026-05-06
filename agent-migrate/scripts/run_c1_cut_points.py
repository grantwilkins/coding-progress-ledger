"""C1 — find cut points in a agent_migrate trace and write CSV.

Usage:
    uv run python scripts/run_c1_cut_points.py <trace.jsonl> <out_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

from agent_migrate_agent.cut_points import find_cut_points, load_trace_jsonl, write_cut_points_csv


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: run_c1_cut_points.py <trace.jsonl> <out_dir>", file=sys.stderr)
        raise SystemExit(2)
    trace_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    trace_id = trace_path.stem
    events = load_trace_jsonl(trace_path)
    cut_points = find_cut_points(events, trace_id=trace_id)
    out_csv = out_dir / f"{trace_id}.csv"
    write_cut_points_csv(cut_points, out_csv)

    phases = sorted({cp.phase for cp in cut_points})
    total = cut_points[0].total_llm_calls if cut_points else 0
    print(
        f"trace_id={trace_id}: {len(cut_points)} cut points; "
        f"total_llm_calls={total}; phases={phases}; out={out_csv}"
    )


if __name__ == "__main__":
    main()
