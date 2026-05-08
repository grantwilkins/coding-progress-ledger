"""H1 — Split builder. Materialize JSON split files at
`datasets/splits/<scheme>_<source>.json` from a checkpoint frame.

Delegates the per-scheme construction to `protocol.py`; this module only
wires together (a) source slicing, (b) per-source `task_family`
enrichment for LTFO, (c) JSON serialization, and (d) the disjointness
invariant from B5.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from coding_estimator.ingest.paths import runs_root
from coding_estimator.ingest.run_record import load_run
from coding_estimator.splits.protocol import (
    Split,
    assert_disjoint,
    holdout,
    loro,
    loso,
    ltfo,
    temporal,
)

SCHEMES: tuple[str, ...] = ("loro", "ltfo", "loso", "holdout", "temporal")
COMBINED_TAG = "all"


EXACT_TASK_LTFO_SOURCES: frozenset[str] = frozenset({"tb_live_v2"})


def task_family_map(source: str) -> dict[str, str | None]:
    """Run-id → generalization group for `ltfo`.

    Most sources group by coarse `task_family`. For `tb_live_v2`, exact
    `task_id` is the safer unit because the corpus deliberately contains
    same-task multi-arm replications; leaving out only the coarse shape
    family would let the model train on task X / arm A and test on task
    X / arm B, which is the wrong generalization claim.

    Runs whose ledger fails to load are skipped (the upstream tree may
    have helper directories like `plots/` next to real run dirs).
    """
    root = runs_root(source)
    out: dict[str, str | None] = {}
    for p in sorted(root.iterdir()):
        if not p.is_dir() or not (p / "ledger.jsonl").is_file():
            continue
        rec = load_run(source, p.name)
        if source in EXACT_TASK_LTFO_SOURCES and rec.task_id is not None:
            out[rec.run_id] = rec.task_id
        else:
            out[rec.run_id] = rec.task_family
    return out


def attach_task_family(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    out = df.copy()
    out["task_family"] = out["run_id"].map(mapping)
    return out


def _build_one(scheme: str, df: pd.DataFrame) -> Split | None:
    if scheme == "loro":
        return loro(df) if df["run_id"].nunique() >= 2 else None
    if scheme == "ltfo":
        if "task_family" not in df.columns:
            return None
        fams = df.loc[df["task_family"].notna(), "task_family"].unique()
        if len(fams) < 2:
            return None
        return ltfo(df.dropna(subset=["task_family"]))
    if scheme == "loso":
        if df["source"].nunique() < 2:
            return None
        return loso(df)
    if scheme == "holdout":
        return holdout(df) if df["run_id"].nunique() >= 2 else None
    if scheme == "temporal":
        if "checkpoint_wall_time" not in df.columns:
            return None
        if df["checkpoint_wall_time"].notna().any() and df["run_id"].nunique() >= 2:
            return temporal(df)
        return None
    raise KeyError(f"unknown scheme: {scheme}")


def build_split(
    scheme: str,
    df: pd.DataFrame,
    *,
    task_families: dict[str, str | None] | None = None,
) -> Split | None:
    """Build one Split for `scheme` over `df`. Returns None if the
    scheme is structurally infeasible on this slice (e.g. ltfo with no
    families, loso with one source)."""
    if scheme not in SCHEMES:
        raise KeyError(f"unknown scheme: {scheme}")
    work = df
    if task_families is not None:
        work = attach_task_family(work, task_families)
    s = _build_one(scheme, work)
    if s is not None:
        assert_disjoint(s)
    return s


def write_split_json(split: Split, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split.to_dict(), indent=2, sort_keys=True) + "\n")
    return out_path


def _slice_filename(scheme: str, source_tag: str) -> str:
    return f"{scheme}_{source_tag}.json"


def build_all(
    checkpoints_df: pd.DataFrame,
    out_dir: Path,
    *,
    sources: Iterable[str] | None = None,
    schemes: Iterable[str] = SCHEMES,
) -> list[Path]:
    """Per-source: emit loro/ltfo/holdout/temporal. Across sources:
    emit loso. Skips schemes that are infeasible for a slice and warns
    so the caller knows why no file landed."""
    written: list[Path] = []
    src_iter = sorted(sources) if sources is not None else sorted(checkpoints_df["source"].unique())

    family_cache: dict[str, dict[str, str | None]] = {}
    for src in src_iter:
        family_cache[src] = task_family_map(src)

    for src in src_iter:
        sub = checkpoints_df[checkpoints_df["source"] == src]
        if sub.empty:
            continue
        for scheme in schemes:
            if scheme == "loso":
                continue
            s = build_split(scheme, sub, task_families=family_cache.get(src))
            if s is None:
                warnings.warn(f"skip {scheme} for {src}: infeasible", stacklevel=2)
                continue
            written.append(write_split_json(s, out_dir / _slice_filename(scheme, src)))

    if "loso" in schemes:
        s = build_split("loso", checkpoints_df)
        if s is None:
            warnings.warn("skip loso: <2 sources in frame", stacklevel=2)
        else:
            written.append(write_split_json(s, out_dir / _slice_filename("loso", COMBINED_TAG)))

    return written
