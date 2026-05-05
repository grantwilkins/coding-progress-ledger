"""Shared baseline machinery.

A `BaselineSpec` is a name + a callable that, given the tuple of source
ids covered by the slice being trained on, returns the feature columns
to consume. A fitted baseline either holds a sklearn LogisticRegression
or a constant base-rate (when the spec is featureless or the training
labels are single-class).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    feature_cols_for: Callable[[tuple[str, ...]], tuple[str, ...]]


@dataclass
class FittedBinary:
    name: str
    feature_cols: tuple[str, ...]
    base_rate: float
    model: LogisticRegression | None  # None = constant predictor

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        if self.model is None:
            return np.full(n, self.base_rate, dtype=float)
        feat = _features(X, self.feature_cols)
        return self.model.predict_proba(feat)[:, 1]


def _features(X: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    sub = X[list(cols)]
    arr = sub.to_numpy(dtype=float, copy=False)
    if not np.isfinite(arr).all():
        bad = sub.columns[np.isnan(arr).any(axis=0) | ~np.isfinite(arr).all(axis=0)].tolist()
        raise ValueError(f"non-finite feature values in {bad}")
    return arr


def fit_binary(
    spec: BaselineSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    sources: tuple[str, ...],
) -> FittedBinary:
    cols = spec.feature_cols_for(sources)
    base_rate = float(np.mean(y)) if len(y) else 0.0
    if len(cols) == 0 or len(np.unique(y)) < 2:
        return FittedBinary(spec.name, cols, base_rate, None)
    feat = _features(X, cols)
    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(feat, y)
    return FittedBinary(spec.name, cols, base_rate, model)
