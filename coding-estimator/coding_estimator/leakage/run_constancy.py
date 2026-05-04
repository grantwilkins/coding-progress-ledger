"""Run-constancy audit.

Run-constant features are not forbidden — but pairing one with a
run-constant target on a tiny dataset produces a perfect classifier that
learns nothing transferable. The audit refuses to ship that pairing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "run_constant_features.json"


@dataclass(frozen=True)
class RunConstantSpec:
    declared: frozenset[str]
    fail_on_run_constant_pair: bool


def load_spec(path: Path | None = None) -> RunConstantSpec:
    raw = json.loads((path or SCHEMA_PATH).read_text(encoding="utf-8"))["data"]
    return RunConstantSpec(
        declared=frozenset(raw["features"]),
        fail_on_run_constant_pair=bool(raw["policy"]["fail_on_run_constant_pair"]),
    )


def is_run_constant(df: pd.DataFrame, column: str, run_id_col: str = "run_id") -> bool:
    if column not in df.columns:
        raise KeyError(column)
    if run_id_col not in df.columns:
        raise KeyError(run_id_col)
    grouped = df.groupby(run_id_col)[column].nunique(dropna=False)
    return bool((grouped <= 1).all())


def audit(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    target_columns: Iterable[str],
    run_id_col: str = "run_id",
    spec: RunConstantSpec | None = None,
) -> list[tuple[str, str]]:
    """Return offending (feature, target) pairs. Both empirically run-constant
    AND the feature is in the declared run-constant register."""
    spec = spec or load_spec()
    offenders: list[tuple[str, str]] = []
    for tgt in target_columns:
        if tgt not in df.columns:
            continue
        if not is_run_constant(df, tgt, run_id_col=run_id_col):
            continue
        for feat in feature_columns:
            if feat not in df.columns:
                continue
            if feat not in spec.declared:
                continue
            if is_run_constant(df, feat, run_id_col=run_id_col):
                offenders.append((feat, tgt))
    return sorted(offenders)


def assert_clean(
    df: pd.DataFrame,
    *,
    feature_columns: Iterable[str],
    target_columns: Iterable[str],
    run_id_col: str = "run_id",
    spec: RunConstantSpec | None = None,
) -> None:
    spec = spec or load_spec()
    offenders = audit(
        df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        run_id_col=run_id_col,
        spec=spec,
    )
    if offenders and spec.fail_on_run_constant_pair:
        raise ValueError(
            "run-constant feature paired with run-constant target: "
            f"{offenders}; this would fit perfectly under loro and learn nothing."
        )
