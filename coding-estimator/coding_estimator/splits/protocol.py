"""Canonical split schemes. Run-level disjointness is the invariant."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Scheme = Literal["loro", "ltfo", "loso", "holdout", "temporal"]
SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train_run_ids: tuple[str, ...]
    test_run_ids: tuple[str, ...]
    val_run_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Split:
    scheme: str
    seed: int
    folds: tuple[Fold, ...]

    def to_dict(self) -> dict:
        return {
            "scheme": self.scheme,
            "seed": self.seed,
            "schema_version": SCHEMA_VERSION,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_run_ids": list(f.train_run_ids),
                    "test_run_ids": list(f.test_run_ids),
                    **({"val_run_ids": list(f.val_run_ids)} if f.val_run_ids else {}),
                }
                for f in self.folds
            ],
        }


def _runs(df: pd.DataFrame) -> list[str]:
    return sorted(df["run_id"].unique().tolist())


def loro(df: pd.DataFrame, seed: int = 0) -> Split:
    runs = _runs(df)
    folds = tuple(
        Fold(
            fold_id=f"loro::{r}",
            train_run_ids=tuple(x for x in runs if x != r),
            test_run_ids=(r,),
        )
        for r in runs
    )
    return Split("loro", seed, folds)


def ltfo(df: pd.DataFrame, seed: int = 0, family_col: str = "task_family") -> Split:
    if family_col not in df.columns:
        raise KeyError(family_col)
    fam_to_runs: dict[str, list[str]] = {}
    for fam, sub in df[["run_id", family_col]].drop_duplicates().groupby(family_col):
        fam_to_runs[str(fam)] = sorted(sub["run_id"].tolist())
    families = sorted(fam_to_runs)
    folds: list[Fold] = []
    for fam in families:
        held = tuple(fam_to_runs[fam])
        train = tuple(r for f in families if f != fam for r in fam_to_runs[f])
        folds.append(Fold(f"ltfo::{fam}", train, held))
    return Split("ltfo", seed, tuple(folds))


def loso(df: pd.DataFrame, seed: int = 0) -> Split:
    src_to_runs: dict[str, list[str]] = {}
    for src, sub in df[["run_id", "source"]].drop_duplicates().groupby("source"):
        src_to_runs[str(src)] = sorted(sub["run_id"].tolist())
    sources = sorted(src_to_runs)
    folds: list[Fold] = []
    for s in sources:
        held = tuple(src_to_runs[s])
        train = tuple(r for o in sources if o != s for r in src_to_runs[o])
        folds.append(Fold(f"loso::{s}", train, held))
    return Split("loso", seed, tuple(folds))


def holdout(df: pd.DataFrame, *, seed: int = 0, test_frac: float = 0.2) -> Split:
    if not 0 < test_frac < 1:
        raise ValueError(f"test_frac must be in (0,1), got {test_frac}")
    runs = _runs(df)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(runs))
    n_test = max(1, int(round(len(runs) * test_frac)))
    test = tuple(sorted(runs[i] for i in order[:n_test]))
    train = tuple(sorted(r for r in runs if r not in test))
    return Split("holdout", seed, (Fold("holdout", train, test),))


def temporal(
    df: pd.DataFrame,
    *,
    seed: int = 0,
    train_frac: float = 0.6,
    timestamp_col: str = "checkpoint_wall_time",
    quality_col: str = "timestamp_quality",
) -> Split:
    """Train on earliest k% of runs by start_time. Warns when timestamps
    aren't real."""
    if not 0 < train_frac < 1:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")
    if quality_col in df.columns:
        qualities = df[quality_col].dropna().unique().tolist()
        if any(q != "real" for q in qualities):
            warnings.warn(
                f"temporal split with non-real timestamps: {qualities}",
                stacklevel=2,
            )
    starts = (
        df[df[timestamp_col].notna()]
        .groupby("run_id")[timestamp_col]
        .min()
        .sort_values()
    )
    if starts.empty:
        raise ValueError("temporal split requires populated timestamps")
    runs = starts.index.tolist()
    n_train = max(1, int(round(len(runs) * train_frac)))
    train = tuple(runs[:n_train])
    test = tuple(runs[n_train:])
    if not test:
        raise ValueError("temporal split produced empty test partition")
    return Split("temporal", seed, (Fold("temporal", train, test),))


def assert_disjoint(split: Split) -> None:
    """No run_id in more than one partition of any fold."""
    for f in split.folds:
        partitions: list[Iterable[str]] = [f.train_run_ids, f.test_run_ids]
        if f.val_run_ids:
            partitions.append(f.val_run_ids)
        seen: dict[str, str] = {}
        for label, part in zip(("train", "test", "val"), partitions, strict=False):
            for r in part:
                if r in seen:
                    raise ValueError(
                        f"run {r} in both '{seen[r]}' and '{label}' of fold {f.fold_id}"
                    )
                seen[r] = label
