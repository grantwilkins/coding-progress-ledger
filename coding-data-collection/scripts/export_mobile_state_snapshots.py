#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from coding_data_collection.mobile_state import snapshot_run_roots, write_snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description="Export measured post-run mobile-state snapshots")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("run_roots", nargs="+", type=Path)
    args = parser.parse_args()

    snapshots = snapshot_run_roots(args.run_roots)
    write_snapshots(snapshots, args.out_dir)


if __name__ == "__main__":
    main()
