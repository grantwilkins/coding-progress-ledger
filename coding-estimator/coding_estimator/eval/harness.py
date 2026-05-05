"""G7 — Cross-validated baseline evaluation harness.

For one (target, baseline, split, source slice) cell, run every fold of
the split, concatenate test predictions, compute the standard binary
metrics, and bootstrap the Brier 95% CI at the RUN level.

The harness consumes already-built artifacts:
- a checkpoint feature frame (`checkpoints_df`)
- a long-form labels frame (`labels_df` — one row per (run, ckpt, target))
- a `Split` from `coding_estimator.splits.protocol`

Cells flagged not-feasible by `coding_estimator.profile.budget` skip
training and emit `n/a` metrics (see EvalCell with `feasible=False`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from coding_estimator.baselines.base import BaselineSpec, fit_binary
from coding_estimator.eval.bootstrap import bootstrap_brier_ci, brier_per_run
from coding_estimator.eval.metrics import (
    OUTPUT_CLIP,
    auroc,
    brier,
    ece,
    log_loss,
)
from coding_estimator.splits.protocol import Split


@dataclass(frozen=True)
class EvalCell:
    target: str
    model: str
    scheme: str
    source_slice: str
    feasible: bool
    n_runs_train: int | None
    n_runs_test: int | None
    n_checkpoints_test: int | None
    positive_rate_data: float | None
    predicted_positive_rate: float | None
    auroc: float | None
    brier: float | None
    log_loss: float | None
    ece: float | None
    brier_ci_low: float | None
    brier_ci_high: float | None
    note: str | None = None


def _join(checkpoints_df: pd.DataFrame, labels_df: pd.DataFrame, target: str) -> pd.DataFrame:
    lab = labels_df[
        (labels_df["target_name"] == target)
        & (~labels_df["is_masked"].astype(bool))
    ][["run_id", "checkpoint_id", "label_value"]]
    if lab.empty:
        return lab
    j = checkpoints_df.merge(lab, on=["run_id", "checkpoint_id"], how="inner")
    j = j.rename(columns={"label_value": "_y"})
    j["_y"] = j["_y"].astype(float)
    return j


def _na_cell(target: str, spec_name: str, scheme: str, source_slice: str, note: str) -> EvalCell:
    return EvalCell(
        target=target,
        model=spec_name,
        scheme=scheme,
        source_slice=source_slice,
        feasible=False,
        n_runs_train=None,
        n_runs_test=None,
        n_checkpoints_test=None,
        positive_rate_data=None,
        predicted_positive_rate=None,
        auroc=None,
        brier=None,
        log_loss=None,
        ece=None,
        brier_ci_low=None,
        brier_ci_high=None,
        note=note,
    )


def evaluate_cell(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str,
    spec: BaselineSpec,
    split: Split,
    source_slice: str,
    sources_in_train: tuple[str, ...],
    feasible: bool = True,
    bootstrap_b: int = 1000,
    bootstrap_seed: int = 0,
) -> EvalCell:
    if not feasible:
        return _na_cell(target, spec.name, split.scheme, source_slice, "insufficient data")
    j = _join(checkpoints_df, labels_df, target)
    if j.empty:
        return _na_cell(target, spec.name, split.scheme, source_slice, "no joined rows")

    y_by_run: dict[str, np.ndarray] = {}
    p_by_run: dict[str, np.ndarray] = {}
    train_run_total: set[str] = set()
    test_run_total: set[str] = set()

    for fold in split.folds:
        train_ids = set(fold.train_run_ids)
        test_ids = set(fold.test_run_ids)
        train = j[j["run_id"].isin(train_ids)]
        test = j[j["run_id"].isin(test_ids)]
        if train.empty or test.empty:
            continue
        y_train = train["_y"].astype(int).to_numpy()
        fitted = fit_binary(spec, train, y_train, sources_in_train)
        probs = fitted.predict_proba(test)
        lo, hi = OUTPUT_CLIP
        probs = np.clip(probs, lo, hi)
        scored = test.assign(_p=probs)
        for rid, sub in scored.groupby("run_id", sort=True):
            if str(rid) in y_by_run:
                raise ValueError(f"run {rid} appears in two test folds; splits must be disjoint")
            y_by_run[str(rid)] = sub["_y"].astype(int).to_numpy()
            p_by_run[str(rid)] = sub["_p"].to_numpy()
        train_run_total.update(str(r) for r in train["run_id"].unique())
        test_run_total.update(str(r) for r in test["run_id"].unique())

    if not y_by_run:
        return _na_cell(target, spec.name, split.scheme, source_slice, "no test predictions")

    y_all = np.concatenate([y_by_run[r] for r in sorted(y_by_run)])
    p_all = np.concatenate([p_by_run[r] for r in sorted(y_by_run)])
    bundle = brier_per_run(y_by_run, p_by_run)
    lo, hi = bootstrap_brier_ci(bundle, b=bootstrap_b, seed=bootstrap_seed)

    return EvalCell(
        target=target,
        model=spec.name,
        scheme=split.scheme,
        source_slice=source_slice,
        feasible=True,
        n_runs_train=len(train_run_total),
        n_runs_test=len(test_run_total),
        n_checkpoints_test=int(len(y_all)),
        positive_rate_data=float(y_all.mean()),
        predicted_positive_rate=float(p_all.mean()),
        auroc=auroc(y_all, p_all),
        brier=brier(y_all, p_all),
        log_loss=log_loss(y_all, p_all),
        ece=ece(y_all, p_all),
        brier_ci_low=lo,
        brier_ci_high=hi,
        note=None,
    )


def cells_to_frame(cells: list[EvalCell]) -> pd.DataFrame:
    return pd.DataFrame([asdict(c) for c in cells])


def predict_cell(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str,
    spec: BaselineSpec,
    split: Split,
    sources_in_train: tuple[str, ...],
) -> pd.DataFrame:
    """Return concatenated test-fold predictions as a long-form frame
    with columns (run_id, source, checkpoint_id, checkpoint_step, _y, _p).
    Run-disjoint (a run never appears across two test folds) is enforced
    by the same guard `evaluate_cell` uses."""
    j = _join(checkpoints_df, labels_df, target)
    if j.empty:
        return j

    pieces: list[pd.DataFrame] = []
    seen: set[str] = set()
    keep_cols = ["run_id", "source", "checkpoint_id", "checkpoint_step", "_y"]
    for fold in split.folds:
        train = j[j["run_id"].isin(set(fold.train_run_ids))]
        test = j[j["run_id"].isin(set(fold.test_run_ids))]
        if train.empty or test.empty:
            continue
        y_train = train["_y"].astype(int).to_numpy()
        fitted = fit_binary(spec, train, y_train, sources_in_train)
        probs = fitted.predict_proba(test)
        lo, hi = OUTPUT_CLIP
        probs = np.clip(probs, lo, hi)
        for rid in test["run_id"].unique():
            if str(rid) in seen:
                raise ValueError(f"run {rid} appears in two test folds; splits must be disjoint")
            seen.add(str(rid))
        pieces.append(test[keep_cols].assign(_p=probs))

    if not pieces:
        return pd.DataFrame(columns=[*keep_cols, "_p"])
    return pd.concat(pieces, ignore_index=True)
