"""I1 — logistic regression on G4 features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from coding_estimator.baselines.ledger_basic import LEDGER_BASIC
from coding_estimator.eval.metrics import OUTPUT_CLIP
from coding_estimator.models import assert_audit_clean
from coding_estimator.models.common import (
    BinaryPredictor,
    assert_model_frame_clean,
    calibration_source_for,
    evaluate_headline_model_suite,
    fit_full_models,
    holdout_cells_only,
    render_model_card,
    save_model_bundle,
)

FEATURE_GROUPS = ("closure", "frontier", "instability", "discovery")
MODEL_ID = "logreg_v0"


def g4_feature_columns(sources: tuple[str, ...]) -> tuple[str, ...]:
    return LEDGER_BASIC.feature_cols_for(sources)


def _feature_matrix(X: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    sub = X[list(cols)]
    arr = sub.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).all():
        bad = sub.columns[np.isnan(arr).any(axis=0) | ~np.isfinite(arr).all(axis=0)].tolist()
        raise ValueError(f"non-finite feature values in {bad}")
    return arr


@dataclass
class FittedLogReg(BinaryPredictor):
    feature_columns: tuple[str, ...]
    base_rate: float
    model: LogisticRegression | None

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lo, hi = OUTPUT_CLIP
        if self.model is None:
            return np.full(len(X), float(np.clip(self.base_rate, lo, hi)), dtype=float)
        feat = _feature_matrix(X, self.feature_columns)
        return np.clip(self.model.predict_proba(feat)[:, 1], lo, hi)

    def coefficient_map(self) -> dict[str, float]:
        if self.model is None:
            return {col: 0.0 for col in self.feature_columns}
        coef = self.model.coef_.ravel()
        return {
            column: float(value)
            for column, value in zip(self.feature_columns, coef, strict=True)
        }


def fit_logreg(
    X: pd.DataFrame,
    y: np.ndarray,
    sources: tuple[str, ...],
) -> FittedLogReg:
    feature_columns = g4_feature_columns(sources)
    train = X.copy()
    train["_y"] = y
    assert_model_frame_clean(
        train,
        feature_columns=feature_columns,
        target_name=train.attrs.get("target_name", "y_success_eventual"),
    )
    base_rate = float(np.mean(y)) if len(y) else 0.0
    if len(np.unique(y)) < 2:
        return FittedLogReg(feature_columns=feature_columns, base_rate=base_rate, model=None)
    feat = _feature_matrix(train, feature_columns)
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(feat, y)
    return FittedLogReg(feature_columns=feature_columns, base_rate=base_rate, model=model)


def train_logreg_bundle(
    *,
    checkpoints_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    out_dir: Path,
    audit_path: Path | None = None,
) -> list:
    assert_audit_clean(audit_path)
    cells = evaluate_headline_model_suite(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        model_name=MODEL_ID,
        fit_model=fit_logreg,
    )
    models = fit_full_models(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        fit_model=fit_logreg,
    )
    save_model_bundle(
        out_dir=out_dir,
        model_id=MODEL_ID,
        fitted_models=models,
        eval_cells=cells,
        model_card_text=render_model_card(
            model_id=MODEL_ID,
            feature_groups=FEATURE_GROUPS,
            holdout_cells=holdout_cells_only(cells),
            known_limits=[
                "raw probabilities are un-recalibrated (`calibration_source=raw`)",
                "retrospective sources carry outcome-aware annotation caveats",
                (
                    "`y_submit_without_validation` is reported as a "
                    "run-constant sanity target, not a headline control target"
                ),
            ],
        ),
        calibration_source=calibration_source_for(cells),
    )
    return cells
