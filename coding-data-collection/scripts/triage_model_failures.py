#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coding_data_collection.failure_triage import (
    triage_corpus,
    write_tool_gap_markdown,
    write_triage_csv,
    write_triage_markdown,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Triage provider-backed model-agent failure runs.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--csv", type=Path, default=Path("reports/REAL_MODEL_MINI3_FAILURE_TRIAGE.csv"))
    parser.add_argument("--md", type=Path, default=Path("reports/REAL_MODEL_MINI3_FAILURE_TRIAGE.md"))
    parser.add_argument("--tool-gaps", type=Path, default=Path("reports/TOOL_AFFORDANCE_GAPS.md"))
    args = parser.parse_args(argv)

    rows = triage_corpus(args.run_root)
    write_triage_csv(rows, args.csv)
    write_triage_markdown(rows, args.md)
    write_tool_gap_markdown(rows, args.tool_gaps)
    print(json.dumps({"run_count": len(rows), "csv": str(args.csv), "md": str(args.md), "tool_gaps": str(args.tool_gaps)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
