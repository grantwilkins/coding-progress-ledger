#!/usr/bin/env python
"""Build a checkpoint dataset for one source.

Usage:
    uv run python scripts/build_checkpoints.py --source tb_live \\
        --out datasets/checkpoints_tb_live.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_estimator.checkpoints.build import write_source_checkpoints
from coding_estimator.ingest.sources import SOURCES


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, choices=sorted(SOURCES.keys()))
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output parquet path (e.g. datasets/checkpoints_tb_live.parquet)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    path, df = write_source_checkpoints(args.source, args.out)
    print(
        f"wrote {len(df)} checkpoints over {df['run_id'].nunique()} runs "
        f"to {path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
