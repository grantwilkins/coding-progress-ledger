"""Writers must produce byte-identical files on repeated runs.

Claim:
    write_parquet sorts columns alphabetically inside the parquet schema
    AND sorts rows by `sort_by` with a STABLE sort. Two input frames that
    differ only in column-insertion order must produce byte-identical
    parquet files.

Plausible wrong implementations:
    - sorts column NAMES in a list but does not reindex the DataFrame
      before writing -> parquet schema preserves insertion order
    - uses an unstable sort (e.g. quicksort) -> rows with equal sort keys
      can swap, breaking byte equality
    - relies on dict iteration order and silently varies between runs
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from coding_estimator.io import write_csv, write_json, write_parquet


def _hash(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _frame() -> pd.DataFrame:
    # columns intentionally out of alphabetical order to test the sort.
    return pd.DataFrame(
        {
            "z": [3, 1, 2],
            "run_id": ["r2", "r1", "r1"],
            "checkpoint_id": [0, 1, 0],
            "value": [0.5, 0.25, 0.125],
        }
    )


def test_parquet_byte_stable(tmp_path: Path) -> None:
    df = _frame()
    a = write_parquet(df, tmp_path / "a.parquet", sort_by=["run_id", "checkpoint_id"])
    b = write_parquet(df, tmp_path / "b.parquet", sort_by=["run_id", "checkpoint_id"])
    assert _hash(a) == _hash(b)


def test_csv_byte_stable(tmp_path: Path) -> None:
    df = _frame()
    a = write_csv(df, tmp_path / "a.csv", sort_by=["run_id", "checkpoint_id"])
    b = write_csv(df, tmp_path / "b.csv", sort_by=["run_id", "checkpoint_id"])
    assert _hash(a) == _hash(b)
    text = a.read_text(encoding="utf-8")
    assert text.startswith("checkpoint_id,run_id,value,z")
    assert "\r" not in text


def test_parquet_column_order_independent(tmp_path: Path) -> None:
    # Build the same logical frame two ways with deliberately different
    # column-insertion orders; the parquet schema must reflect alphabetical
    # column order, and bytes must be identical.
    import pyarrow.parquet as pq

    df1 = pd.DataFrame(
        {"z": [1, 2], "a": [3, 4], "m": [5, 6], "run_id": ["r1", "r2"]}
    )
    df2 = pd.DataFrame(
        {"run_id": ["r1", "r2"], "m": [5, 6], "a": [3, 4], "z": [1, 2]}
    )
    p1 = write_parquet(df1, tmp_path / "p1.parquet", sort_by=["run_id"])
    p2 = write_parquet(df2, tmp_path / "p2.parquet", sort_by=["run_id"])
    assert _hash(p1) == _hash(p2)
    schema = pq.read_schema(p1)
    assert schema.names == sorted(schema.names) == ["a", "m", "run_id", "z"]


def test_parquet_row_sort_is_stable(tmp_path: Path) -> None:
    # Two input orderings that share a sort key but differ in the secondary
    # column must produce DIFFERENT outputs once we sort by `key` alone --
    # but if we sort by both, the outputs must agree. A stable sort is
    # required so that ties on `key` preserve the reordering imposed by the
    # secondary sort key during multi-key sort.
    a = pd.DataFrame({"key": [1, 1, 2, 2], "tag": ["x", "y", "x", "y"]})
    b = pd.DataFrame({"key": [1, 1, 2, 2], "tag": ["y", "x", "y", "x"]})
    pa1 = write_parquet(a, tmp_path / "a.parquet", sort_by=["key"])
    pb1 = write_parquet(b, tmp_path / "b.parquet", sort_by=["key"])
    # Same key, different tag order in input => different bytes when sorting
    # by key alone (stable sort preserves input tag order within ties).
    assert _hash(pa1) != _hash(pb1)
    # Sorting by (key, tag) is total -> bytes converge regardless of input
    # ordering.
    pa2 = write_parquet(a, tmp_path / "a2.parquet", sort_by=["key", "tag"])
    pb2 = write_parquet(b, tmp_path / "b2.parquet", sort_by=["key", "tag"])
    assert _hash(pa2) == _hash(pb2)


def test_json_byte_stable_and_sorted(tmp_path: Path) -> None:
    obj = {"b": 2, "a": {"y": 1, "x": [3, 2, 1]}}
    a = write_json(obj, tmp_path / "a.json")
    b = write_json(obj, tmp_path / "b.json")
    assert _hash(a) == _hash(b)
    text = a.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')
