#!/usr/bin/env python
"""H1 driver — emit canonical split JSONs from per-source checkpoint
frames into `datasets/splits/`.

Usage:
    uv run python scripts/build_splits.py \\
        --checkpoints-dir datasets \\
        --out-dir datasets/splits
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from coding_estimator.splits.builder import build_all


def _load(checkpoints_dir: Path) -> pd.DataFrame:
    paths = sorted(checkpoints_dir.glob("checkpoints_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints_*.parquet in {checkpoints_dir}")
    frames = [pd.read_parquet(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoints-dir", type=Path, default=Path("datasets"))
    p.add_argument("--out-dir", type=Path, default=Path("datasets/splits"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    df = _load(args.checkpoints_dir)
    written = build_all(df, args.out_dir)
    for p in written:
        print(p)
    print(f"wrote {len(written)} split file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
