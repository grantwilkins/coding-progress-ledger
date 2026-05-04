"""Writers must produce byte-identical files on repeated runs."""

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


def test_json_byte_stable_and_sorted(tmp_path: Path) -> None:
    obj = {"b": 2, "a": {"y": 1, "x": [3, 2, 1]}}
    a = write_json(obj, tmp_path / "a.json")
    b = write_json(obj, tmp_path / "b.json")
    assert _hash(a) == _hash(b)
    text = a.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert text.index('"a"') < text.index('"b"')
