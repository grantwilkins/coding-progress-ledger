"""Reproducibility primitives: seeded RNG and byte-stable writers."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def set_global_seed(seed: int) -> None:
    """Seed all RNGs used by sklearn / numpy / random."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _ordered(df: pd.DataFrame, sort_by: Iterable[str] | None) -> pd.DataFrame:
    out = df.copy()
    out = out.reindex(sorted(out.columns), axis=1)
    keys = list(sort_by) if sort_by is not None else [c for c in out.columns]
    keys = [k for k in keys if k in out.columns]
    if keys and not out.empty:
        out = out.sort_values(by=keys, kind="mergesort").reset_index(drop=True)
    return out


def write_parquet(
    df: pd.DataFrame, path: Path | str, sort_by: Iterable[str] | None = None
) -> Path:
    out = _ordered(df, sort_by)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_table(
        table,
        target,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
        write_statistics=False,
    )
    return target


def write_csv(df: pd.DataFrame, path: Path | str, sort_by: Iterable[str] | None = None) -> Path:
    out = _ordered(df, sort_by)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False, encoding="utf-8", lineterminator="\n")
    return target


def write_json(obj: Any, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2, ensure_ascii=False)
        fh.write("\n")
    return target
