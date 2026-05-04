"""Forbidden-column guard.

The guard fails loud when a checkpoint frame contains a column that
matches the forbidden list (exact, prefix, or suffix). It is the last
line of defense before any feature frame is written to disk.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "forbidden_columns.json"


@dataclass(frozen=True)
class ForbiddenSpec:
    exact: frozenset[str]
    prefixes: tuple[str, ...]
    suffixes: tuple[str, ...]


def load_forbidden_spec(path: Path | None = None) -> ForbiddenSpec:
    raw = json.loads((path or SCHEMA_PATH).read_text(encoding="utf-8"))
    data = raw["data"]
    return ForbiddenSpec(
        exact=frozenset(data["exact"]),
        prefixes=tuple(sorted(data["prefixes"])),
        suffixes=tuple(sorted(data["suffixes"])),
    )


def find_forbidden(columns: Iterable[str], spec: ForbiddenSpec | None = None) -> list[str]:
    spec = spec or load_forbidden_spec()
    bad: list[str] = []
    for col in columns:
        if col in spec.exact:
            bad.append(col)
            continue
        if any(col.startswith(p) for p in spec.prefixes):
            bad.append(col)
            continue
        if any(col.endswith(s) for s in spec.suffixes):
            bad.append(col)
    return sorted(set(bad))


def assert_no_forbidden(df: pd.DataFrame, spec: ForbiddenSpec | None = None) -> None:
    bad = find_forbidden(df.columns, spec=spec)
    if bad:
        raise ValueError(f"forbidden columns present in checkpoint frame: {bad}")
