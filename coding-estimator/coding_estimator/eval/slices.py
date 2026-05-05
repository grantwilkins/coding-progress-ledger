"""H4 — slice-specific evaluation.

For one (target, model, scheme, source_slice) cell with concatenated
test-fold predictions, slice the rows by:
    phase  -- early/middle/late thirds of (checkpoint_step / max_step) per run
    shape  -- per-run boolean shape tag from `coding_estimator.labels.shapes`

Slices with < 5 positives or < 5 negatives in the test set emit
`n/a (insufficient data)` rather than computed metrics, matching the
data-budget gate used everywhere else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from coding_estimator.eval.metrics import auroc, brier, ece

PHASE_BINS: tuple[str, str, str] = ("early", "middle", "late")
MIN_PER_SLICE: int = 5


@dataclass(frozen=True)
class SliceCell:
    target: str
    model: str
    scheme: str
    source_slice: str
    slice_kind: str  # "phase" | "shape"
    slice_value: str
    feasible: bool
    n_runs: int | None
    n_checkpoints: int | None
    positives: int | None
    negatives: int | None
    auroc: float | None
    brier: float | None
    ece: float | None
    note: str | None = None


def assign_phase(predictions_df: pd.DataFrame) -> pd.Series:
    """Per-run, bin checkpoint_step into thirds. Runs with a single
    checkpoint land in 'early' (the only valid phase for them)."""
    g = predictions_df.groupby("run_id")["checkpoint_step"]
    max_step = g.transform("max")
    min_step = g.transform("min")
    span = (max_step - min_step).where(max_step > min_step, other=1)
    frac = (predictions_df["checkpoint_step"] - min_step) / span
    out = pd.Series(np.full(len(predictions_df), "early", dtype=object),
                    index=predictions_df.index)
    out[(frac > 1 / 3) & (frac <= 2 / 3)] = "middle"
    out[frac > 2 / 3] = "late"
    out[max_step == min_step] = "early"
    return out


def _evaluate_slice(
    sub: pd.DataFrame,
    *,
    target: str,
    model: str,
    scheme: str,
    source_slice: str,
    slice_kind: str,
    slice_value: str,
) -> SliceCell:
    pos = int(sub["_y"].sum())
    neg = int(len(sub) - pos)
    if pos < MIN_PER_SLICE or neg < MIN_PER_SLICE:
        return SliceCell(
            target=target, model=model, scheme=scheme, source_slice=source_slice,
            slice_kind=slice_kind, slice_value=slice_value,
            feasible=False,
            n_runs=int(sub["run_id"].nunique()),
            n_checkpoints=int(len(sub)),
            positives=pos, negatives=neg,
            auroc=None, brier=None, ece=None,
            note="insufficient data",
        )
    y = sub["_y"].astype(int).to_numpy()
    p = sub["_p"].to_numpy()
    return SliceCell(
        target=target, model=model, scheme=scheme, source_slice=source_slice,
        slice_kind=slice_kind, slice_value=slice_value,
        feasible=True,
        n_runs=int(sub["run_id"].nunique()),
        n_checkpoints=int(len(sub)),
        positives=pos, negatives=neg,
        auroc=auroc(y, p), brier=brier(y, p), ece=ece(y, p),
        note=None,
    )


def evaluate_phase_slices(
    predictions_df: pd.DataFrame,
    *,
    target: str,
    model: str,
    scheme: str,
    source_slice: str,
) -> list[SliceCell]:
    if predictions_df.empty:
        return []
    df = predictions_df.assign(_phase=assign_phase(predictions_df))
    cells: list[SliceCell] = []
    for phase in PHASE_BINS:
        sub = df[df["_phase"] == phase]
        if sub.empty:
            cells.append(SliceCell(
                target=target, model=model, scheme=scheme, source_slice=source_slice,
                slice_kind="phase", slice_value=phase,
                feasible=False, n_runs=0, n_checkpoints=0, positives=0, negatives=0,
                auroc=None, brier=None, ece=None, note="empty slice",
            ))
            continue
        cells.append(_evaluate_slice(
            sub, target=target, model=model, scheme=scheme, source_slice=source_slice,
            slice_kind="phase", slice_value=phase,
        ))
    return cells


def evaluate_shape_slices(
    predictions_df: pd.DataFrame,
    shapes_df: pd.DataFrame,
    *,
    target: str,
    model: str,
    scheme: str,
    source_slice: str,
) -> list[SliceCell]:
    """`shapes_df` has columns run_id + shape_<tag> booleans (see
    `coding_estimator.labels.shapes`). One slice per shape tag — runs
    can belong to multiple shape classes."""
    if predictions_df.empty or shapes_df.empty:
        return []
    shape_cols = [c for c in shapes_df.columns if c.startswith("shape_")]
    cells: list[SliceCell] = []
    sj = predictions_df.merge(shapes_df[["run_id", *shape_cols]], on="run_id", how="left")
    for col in shape_cols:
        tag = col[len("shape_"):]
        sub = sj[sj[col].astype("boolean").fillna(False).astype(bool)]
        if sub.empty:
            cells.append(SliceCell(
                target=target, model=model, scheme=scheme, source_slice=source_slice,
                slice_kind="shape", slice_value=tag,
                feasible=False, n_runs=0, n_checkpoints=0, positives=0, negatives=0,
                auroc=None, brier=None, ece=None, note="no runs in shape",
            ))
            continue
        cells.append(_evaluate_slice(
            sub, target=target, model=model, scheme=scheme, source_slice=source_slice,
            slice_kind="shape", slice_value=tag,
        ))
    return cells


def slice_cells_to_frame(cells: list[SliceCell]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in cells])
