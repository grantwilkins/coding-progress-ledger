"""Shared Workstream I utilities for binary models."""

from __future__ import annotations

import pickle
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from coding_estimator.eval.bootstrap import bootstrap_brier_ci, brier_per_run
from coding_estimator.eval.harness import EvalCell, cells_to_frame
from coding_estimator.eval.metrics import OUTPUT_CLIP, auroc, brier, ece, log_loss
from coding_estimator.io import write_csv, write_json
from coding_estimator.labels.registry import V0_TARGETS
from coding_estimator.leakage.guard import assert_no_forbidden
from coding_estimator.leakage.run_constancy import assert_clean as assert_run_constancy_clean
from coding_estimator.splits.protocol import Split, holdout, loro

MODEL_VERSION = "0.1.0"
HEADLINE_BINARY_TARGETS: tuple[str, ...] = (
    "y_success_eventual",
    "y_future_progress_drop_h5",
    "y_validation_new_work_h5",
    "y_submit_without_validation",
)
RUN_CONSTANT_TARGETS: frozenset[str] = frozenset(
    name for name, target in V0_TARGETS.items() if target.run_constant_flag
)


class BinaryPredictor(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


BinaryFitFn = Callable[[pd.DataFrame, np.ndarray, tuple[str, ...]], BinaryPredictor]


def repo_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()[:7]
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "0000000"


def join_binary_target(
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str,
) -> pd.DataFrame:
    assert_no_forbidden(checkpoints_df)
    lab = labels_df[
        (labels_df["target_name"] == target)
        & (~labels_df["is_masked"].astype(bool))
    ][["run_id", "checkpoint_id", "label_value"]]
    if lab.empty:
        return pd.DataFrame()
    joined = checkpoints_df.merge(lab, on=["run_id", "checkpoint_id"], how="inner")
    joined = joined.rename(columns={"label_value": "_y"})
    joined["_y"] = joined["_y"].astype(float)
    joined.attrs["target_name"] = target
    return joined


def assert_model_frame_clean(
    df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_name: str,
) -> None:
    assert_no_forbidden(df)
    audit_df = df.copy()
    audit_df["__target__"] = audit_df["_y"].astype(float)
    assert_run_constancy_clean(
        audit_df,
        feature_columns=feature_columns,
        target_columns=("__target__",),
    )
    if target_name not in HEADLINE_BINARY_TARGETS:
        raise KeyError(f"unsupported Workstream I target: {target_name}")


def evaluate_model_cell(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    target: str,
    model_name: str,
    split: Split,
    source_slice: str,
    fit_model: BinaryFitFn,
    sources_in_train: tuple[str, ...],
    bootstrap_b: int = 1000,
    bootstrap_seed: int = 0,
) -> EvalCell:
    joined = join_binary_target(checkpoints_df, labels_df, target)
    if joined.empty:
        return EvalCell(
            target=target,
            model=model_name,
            scheme=split.scheme,
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
            note="no joined rows",
        )

    y_by_run: dict[str, np.ndarray] = {}
    p_by_run: dict[str, np.ndarray] = {}
    train_run_total: set[str] = set()
    test_run_total: set[str] = set()
    n_single_class_folds = 0
    n_active_folds = 0

    for fold in split.folds:
        train_ids = set(fold.train_run_ids)
        test_ids = set(fold.test_run_ids)
        train = joined[joined["run_id"].isin(train_ids)]
        test = joined[joined["run_id"].isin(test_ids)]
        if train.empty or test.empty:
            continue
        n_active_folds += 1
        y_train = train["_y"].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            n_single_class_folds += 1
        model = fit_model(train, y_train, sources_in_train)
        probs = np.clip(model.predict_proba(test), *OUTPUT_CLIP)
        scored = test.assign(_p=probs)
        for run_id, sub in scored.groupby("run_id", sort=True):
            key = str(run_id)
            if key in y_by_run:
                raise ValueError(f"run {run_id} appears in two test folds")
            y_by_run[key] = sub["_y"].astype(int).to_numpy()
            p_by_run[key] = sub["_p"].to_numpy()
        train_run_total.update(str(v) for v in train["run_id"].unique())
        test_run_total.update(str(v) for v in test["run_id"].unique())

    if not y_by_run:
        return EvalCell(
            target=target,
            model=model_name,
            scheme=split.scheme,
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
            note="no test predictions",
        )

    y_all = np.concatenate([y_by_run[r] for r in sorted(y_by_run)])
    p_all = np.concatenate([p_by_run[r] for r in sorted(y_by_run)])
    lo, hi = bootstrap_brier_ci(
        brier_per_run(y_by_run, p_by_run),
        b=bootstrap_b,
        seed=bootstrap_seed,
    )
    notes: list[str] = []
    if target in RUN_CONSTANT_TARGETS:
        notes.append("run_constant_target")
    if n_single_class_folds == n_active_folds and n_active_folds > 0:
        notes.append("all_train_folds_single_class")
    elif n_single_class_folds > 0:
        notes.append(f"{n_single_class_folds}_of_{n_active_folds}_train_folds_single_class")
    return EvalCell(
        target=target,
        model=model_name,
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
        note=";".join(notes) if notes else None,
    )


def evaluate_headline_model_suite(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    model_name: str,
    fit_model: BinaryFitFn,
) -> list[EvalCell]:
    cells: list[EvalCell] = []
    holdout_split = holdout(checkpoints_df, seed=0)
    train_sources = tuple(sorted(checkpoints_df["source"].unique()))
    for target in HEADLINE_BINARY_TARGETS:
        cells.append(
            evaluate_model_cell(
                checkpoints_df=checkpoints_df,
                labels_df=labels_df,
                target=target,
                model_name=model_name,
                split=holdout_split,
                source_slice="holdout->all",
                fit_model=fit_model,
                sources_in_train=train_sources,
            )
        )
    for source in sorted(checkpoints_df["source"].unique()):
        sub = checkpoints_df[checkpoints_df["source"] == source]
        if sub["run_id"].nunique() < 2:
            continue
        split = loro(sub)
        for target in HEADLINE_BINARY_TARGETS:
            cells.append(
                evaluate_model_cell(
                    checkpoints_df=sub,
                    labels_df=labels_df,
                    target=target,
                    model_name=model_name,
                    split=split,
                    source_slice=source,
                    fit_model=fit_model,
                    sources_in_train=(source,),
                )
            )
    return cells


def fit_full_models(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    fit_model: BinaryFitFn,
) -> dict[str, BinaryPredictor]:
    models: dict[str, BinaryPredictor] = {}
    train_sources = tuple(sorted(checkpoints_df["source"].unique()))
    for target in HEADLINE_BINARY_TARGETS:
        joined = join_binary_target(checkpoints_df, labels_df, target)
        if joined.empty:
            continue
        y = joined["_y"].astype(int).to_numpy()
        models[target] = fit_model(joined, y, train_sources)
    return models


def target_horizon_json(target: str) -> dict[str, int | str | None]:
    meta = V0_TARGETS[target]
    return {"units": meta.horizon_units, "value": meta.horizon_value}


def save_model_bundle(
    *,
    out_dir: Path,
    model_id: str,
    fitted_models: dict[str, BinaryPredictor],
    eval_cells: list[EvalCell],
    model_card_text: str,
    calibration_source: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.pkl").write_bytes(pickle.dumps(fitted_models))
    fitted_targets = set(fitted_models)
    write_json(
        {
            "model_id": model_id,
            "model_version": MODEL_VERSION,
            "calibration_source": calibration_source,
            "targets": {
                cell.target: {
                    "holdout_brier": cell.brier,
                    "holdout_ece": cell.ece,
                    "holdout_auroc": cell.auroc,
                    "holdout_positive_rate": cell.positive_rate_data,
                    "target_horizon": target_horizon_json(cell.target),
                    "note": cell.note,
                }
                for cell in eval_cells
                if cell.scheme == "holdout" and cell.target in fitted_targets
            },
        },
        out_dir / "calibration.json",
    )
    (out_dir / "model_card.md").write_text(
        model_card_text,
        encoding="utf-8",
        newline="\n",
    )
    write_csv(
        cells_to_frame(eval_cells),
        out_dir / "metrics.csv",
        sort_by=["scheme", "source_slice", "target", "model"],
    )


def render_model_card(
    *,
    model_id: str,
    feature_groups: tuple[str, ...],
    holdout_cells: list[EvalCell],
    known_limits: list[str],
) -> str:
    lines = [
        f"# {model_id}",
        "",
        "## Intended use",
        "",
        "Offline estimation of coding-run belief-state targets from prefix-only ledger features.",
        "",
        "## Not safe for control",
        "",
        "- `true`",
        "",
        "## Training data",
        "",
        "- canonical sources: `swe_agent_pilot`, `hermes_pilot_h5_v2`, `tb_live`",
        "- inputs: `datasets/checkpoints_all.parquet`, `datasets/labels_all.parquet`",
        f"- commit_sha: `{repo_commit_sha()}`",
        "",
        "## Features",
        "",
        f"- groups: {', '.join(feature_groups)}",
        "",
        "## Split protocol",
        "",
        "- headline metrics: combined `holdout` split, seed=0",
        "- diagnostics: per-source `loro`",
        "",
        "## Calibration status",
        "",
    ]
    for cell in holdout_cells:
        brier = "n/a" if cell.brier is None else f"{cell.brier:.3f}"
        ece = "n/a" if cell.ece is None else f"{cell.ece:.3f}"
        auroc = "n/a" if cell.auroc is None else f"{cell.auroc:.3f}"
        lines.append(
            f"- `{cell.target}`: Brier={brier}, ECE={ece}, AUROC={auroc}"
        )
    lines.extend(["", "## Known limits", ""])
    lines.extend(f"- {item}" for item in known_limits)
    bad_ece = [cell for cell in holdout_cells if cell.ece is not None and cell.ece > 0.1]
    if bad_ece:
        lines.extend(["", "## Downstream readiness", ""])
        lines.append(
            "- Not ready for downstream probability consumption on these headline holdout cells:"
        )
        lines.extend(
            f"  - `{cell.target}` (ECE={cell.ece:.3f})"
            for cell in bad_ece
        )
    return "\n".join(lines) + "\n"


def holdout_cells_only(cells: list[EvalCell]) -> list[EvalCell]:
    return [cell for cell in cells if cell.scheme == "holdout"]


def calibration_source_for(cells: list[EvalCell]) -> str:
    """Holdout `calibration_source` for the saved bundle.

    `raw`      — the model was fit on every holdout cell and its
                 raw probability output is used uncalibrated.
    `constant` — at least one holdout cell had every active train
                 fold collapse to a single class, so the model
                 falls back to a constant base-rate predictor.
    """
    holdout = holdout_cells_only(cells)
    if not holdout:
        return "constant"
    for cell in holdout:
        if cell.note and "single_class" in cell.note:
            return "constant"
    return "raw"


def eval_cells_json(cells: list[EvalCell]) -> list[dict]:
    return [asdict(cell) for cell in cells]
