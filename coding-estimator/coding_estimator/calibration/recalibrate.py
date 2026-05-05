"""J4 — post-hoc probability recalibration.

Three methods:
    PlattRecalibrator   — 1-D logistic on (logit(p_raw), y).
    IsotonicRecalibrator — pool-adjacent-violators isotonic regression
                          on (p_raw, y).
    SourceIsotonicRecalibrator — per-source isotonic with a global
                          fallback for unseen sources.

Each recalibrator is fitted on a held-out calibration set and exposes
`transform(p, source=None) -> np.ndarray`. Outputs are clipped to
`OUTPUT_CLIP` so downstream log-loss / Brier remains finite.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from coding_estimator.eval.metrics import OUTPUT_CLIP

_LOGIT_CLIP = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, _LOGIT_CLIP, 1.0 - _LOGIT_CLIP)
    return np.log(pc / (1.0 - pc))


def _clip(p: np.ndarray) -> np.ndarray:
    lo, hi = OUTPUT_CLIP
    return np.clip(p, lo, hi)


@dataclass
class PlattRecalibrator:
    name: str = "platt"
    _model: LogisticRegression | None = None
    _base_rate: float = 0.5

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattRecalibrator":
        p_arr = np.asarray(p, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if p_arr.shape != y_arr.shape:
            raise ValueError("p and y must align")
        self._base_rate = float(y_arr.mean()) if len(y_arr) else 0.5
        if len(np.unique(y_arr)) < 2:
            self._model = None
            return self
        x = _logit(p_arr).reshape(-1, 1)
        # Standard Platt scaling is the unregularized MLE on logit(p).
        # sklearn's default C=1.0 shrinks the slope on small calibration
        # sets; use C very large to approximate no penalty.
        self._model = LogisticRegression(
            max_iter=2000, random_state=0, C=1e10, solver="lbfgs"
        )
        self._model.fit(x, y_arr)
        return self

    def transform(self, p: np.ndarray, source: str | None = None) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if self._model is None:
            return _clip(np.full_like(p_arr, self._base_rate))
        x = _logit(p_arr).reshape(-1, 1)
        return _clip(self._model.predict_proba(x)[:, 1])


@dataclass
class IsotonicRecalibrator:
    name: str = "isotonic"
    _model: IsotonicRegression | None = None
    _base_rate: float = 0.5

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicRecalibrator":
        p_arr = np.asarray(p, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if p_arr.shape != y_arr.shape:
            raise ValueError("p and y must align")
        self._base_rate = float(y_arr.mean()) if len(y_arr) else 0.5
        if len(np.unique(y_arr)) < 2:
            self._model = None
            return self
        self._model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._model.fit(p_arr, y_arr.astype(float))
        return self

    def transform(self, p: np.ndarray, source: str | None = None) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if self._model is None:
            return _clip(np.full_like(p_arr, self._base_rate))
        return _clip(self._model.transform(p_arr))


@dataclass
class SourceIsotonicRecalibrator:
    """One isotonic recalibrator per source. Sources unseen at fit-time
    fall back to a global isotonic recalibrator trained on every row."""

    name: str = "source_isotonic"
    _by_source: dict[str, IsotonicRecalibrator] = field(default_factory=dict)
    _global: IsotonicRecalibrator | None = None

    def fit(
        self, p: np.ndarray, y: np.ndarray, source: np.ndarray
    ) -> "SourceIsotonicRecalibrator":
        p_arr = np.asarray(p, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        s_arr = np.asarray(source)
        if not (p_arr.shape == y_arr.shape == s_arr.shape):
            raise ValueError("p, y, source must align")
        self._by_source = {}
        for src in np.unique(s_arr):
            mask = s_arr == src
            self._by_source[str(src)] = IsotonicRecalibrator().fit(
                p_arr[mask], y_arr[mask]
            )
        self._global = IsotonicRecalibrator().fit(p_arr, y_arr)
        return self

    def transform(self, p: np.ndarray, source: str | None = None) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        if self._global is None:
            raise RuntimeError("SourceIsotonicRecalibrator must be fit before transform")
        if source is None:
            return self._global.transform(p_arr)
        if isinstance(source, (str, bytes)):
            cal = self._by_source.get(str(source), self._global)
            return cal.transform(p_arr)
        s_arr = np.asarray(source)
        if s_arr.shape != p_arr.shape:
            raise ValueError("source array must align with p")
        out = np.empty_like(p_arr)
        for src in np.unique(s_arr):
            mask = s_arr == src
            cal = self._by_source.get(str(src), self._global)
            out[mask] = cal.transform(p_arr[mask])
        return out


METHODS: dict[str, type] = {
    "platt": PlattRecalibrator,
    "isotonic": IsotonicRecalibrator,
    "source_isotonic": SourceIsotonicRecalibrator,
}


def fit_method(
    method: str,
    p: np.ndarray,
    y: np.ndarray,
    source: np.ndarray | None = None,
):
    if method not in METHODS:
        raise KeyError(f"unknown recalibration method: {method}")
    cls = METHODS[method]
    if cls is SourceIsotonicRecalibrator:
        if source is None:
            raise ValueError("source_isotonic requires a `source` array")
        return cls().fit(p, y, source)
    return cls().fit(p, y)


def fit_from_predictions(
    method: str, predictions_df: pd.DataFrame
):
    """Convenience: fit a recalibrator from a long-form
    (run_id, source, _y, _p) predictions frame."""
    if predictions_df.empty:
        raise ValueError("cannot fit recalibrator on empty predictions frame")
    p = predictions_df["_p"].to_numpy(dtype=float)
    y = predictions_df["_y"].to_numpy(dtype=int)
    source = (
        predictions_df["source"].to_numpy()
        if "source" in predictions_df.columns
        else None
    )
    return fit_method(method, p, y, source)


__all__ = [
    "PlattRecalibrator",
    "IsotonicRecalibrator",
    "SourceIsotonicRecalibrator",
    "METHODS",
    "fit_method",
    "fit_from_predictions",
]
