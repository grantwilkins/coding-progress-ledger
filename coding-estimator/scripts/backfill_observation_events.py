#!/usr/bin/env python
"""Backfill observation_events.jsonl for a tb_live-style source."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from coding_estimator.runner.observation_events import (
    build_observation_events,
    write_observation_events,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    count = 0
    for run_dir in sorted(path for path in args.runs_root.iterdir() if path.is_dir()):
        out_path = run_dir / "observation_events.jsonl"
        if out_path.exists() and not args.force:
            continue
        events = build_observation_events(
            run_dir=run_dir,
            run_id=run_dir.name,
            repo_root=args.repo_root,
        )
        write_observation_events(events, out_path)
        count += 1
    print(count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
