"""I0 — empirical-bin model.

Predict the training-fold empirical positive rate in the
`coding_progress quartile x elapsed-signal quartile` cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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

FEATURE_GROUPS = ("closure", "time_budget")
MODEL_ID = "empirical_bin_v0"


def _quantile_edges(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return (0.25, 0.5, 0.75)
    q = np.quantile(values, [0.25, 0.5, 0.75])
    return (float(q[0]), float(q[1]), float(q[2]))


def _bucketize(values: np.ndarray, edges: tuple[float, float, float]) -> np.ndarray:
    out = np.full(len(values), -1, dtype=int)
    finite = np.isfinite(values)
    out[finite] = np.searchsorted(np.array(edges, dtype=float), values[finite], side="right")
    return out


def _normalized_elapsed_steps(
    values: np.ndarray,
    min_elapsed: float,
    max_elapsed: float,
) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("elapsed_steps must be finite for empirical binning")
    if max_elapsed <= min_elapsed:
        return np.zeros(len(values), dtype=float)
    scaled = (values - min_elapsed) / (max_elapsed - min_elapsed)
    return np.clip(scaled, 0.0, 1.0)


@dataclass(frozen=True)
class EmpiricalBinModel(BinaryPredictor):
    progress_edges: tuple[float, float, float]
    elapsed_edges: tuple[float, float, float]
    elapsed_min: float
    elapsed_max: float
    base_rate: float
    bin_rates: dict[tuple[int, int], float]

    def _elapsed_signal(self, X: pd.DataFrame) -> np.ndarray:
        if "checkpoint_fraction_timeout" in X.columns:
            timeout_frac = X["checkpoint_fraction_timeout"].to_numpy(dtype=float, copy=False)
            if np.isfinite(timeout_frac).all():
                return timeout_frac
        elapsed_steps = X["elapsed_steps"].to_numpy(dtype=float, copy=False)
        fallback = _normalized_elapsed_steps(
            elapsed_steps,
            self.elapsed_min,
            self.elapsed_max,
        )
        if "checkpoint_fraction_timeout" not in X.columns:
            return fallback
        timeout_frac = X["checkpoint_fraction_timeout"].to_numpy(dtype=float, copy=False)
        out = fallback.copy()
        mask = np.isfinite(timeout_frac)
        out[mask] = timeout_frac[mask]
        return out

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        progress = X["coding_progress"].to_numpy(dtype=float, copy=False)
        if not np.isfinite(progress).all():
            raise ValueError("coding_progress must be finite for empirical binning")
        elapsed = self._elapsed_signal(X)
        p_bucket = _bucketize(progress, self.progress_edges)
        e_bucket = _bucketize(elapsed, self.elapsed_edges)
        probs = np.empty(len(X), dtype=float)
        for i, key in enumerate(zip(p_bucket, e_bucket, strict=True)):
            probs[i] = self.bin_rates.get(key, self.base_rate)
        lo, hi = OUTPUT_CLIP
        return np.clip(probs, lo, hi)


def fit_empirical_bin(
    X: pd.DataFrame,
    y: np.ndarray,
    _sources: tuple[str, ...],
) -> EmpiricalBinModel:
    feature_columns = ("coding_progress", "elapsed_steps", "checkpoint_fraction_timeout")
    train = X.copy()
    train["_y"] = y
    assert_model_frame_clean(
        train,
        feature_columns=feature_columns,
        target_name=train.attrs.get("target_name", "y_success_eventual"),
    )

    progress = train["coding_progress"].to_numpy(dtype=float, copy=False)
    if not np.isfinite(progress).all():
        raise ValueError("coding_progress must be finite for empirical binning")
    elapsed_steps = train["elapsed_steps"].to_numpy(dtype=float, copy=False)
    elapsed_min = float(elapsed_steps.min())
    elapsed_max = float(elapsed_steps.max())
    fallback = _normalized_elapsed_steps(elapsed_steps, elapsed_min, elapsed_max)
    if "checkpoint_fraction_timeout" in train.columns:
        timeout_frac = train["checkpoint_fraction_timeout"].to_numpy(dtype=float, copy=False)
        elapsed = fallback.copy()
        mask = np.isfinite(timeout_frac)
        elapsed[mask] = timeout_frac[mask]
    else:
        elapsed = fallback

    progress_edges = _quantile_edges(progress)
    elapsed_edges = _quantile_edges(elapsed)
    p_bucket = _bucketize(progress, progress_edges)
    e_bucket = _bucketize(elapsed, elapsed_edges)
    base_rate = float(np.mean(y)) if len(y) else 0.0

    bucket_df = pd.DataFrame(
        {"p_bucket": p_bucket, "e_bucket": e_bucket, "_y": y.astype(float)}
    )
    rates = (
        bucket_df.groupby(["p_bucket", "e_bucket"])["_y"]
        .mean()
        .to_dict()
    )
    return EmpiricalBinModel(
        progress_edges=progress_edges,
        elapsed_edges=elapsed_edges,
        elapsed_min=elapsed_min,
        elapsed_max=elapsed_max,
        base_rate=base_rate,
        bin_rates={(int(k0), int(k1)): float(v) for (k0, k1), v in rates.items()},
    )


def train_empirical_bin_bundle(
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
        fit_model=fit_empirical_bin,
    )
    models = fit_full_models(
        checkpoints_df=checkpoints_df,
        labels_df=labels_df,
        fit_model=fit_empirical_bin,
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
                "coarse binning loses within-cell ordering information",
                (
                    "elapsed signal falls back to normalized elapsed_steps "
                    "when timeout fraction is unavailable"
                ),
                (
                    "`y_submit_without_validation` remains a run-constant "
                    "sanity target, not a headline control target"
                ),
            ],
        ),
        calibration_source=calibration_source_for(cells),
    )
    return cells
