"""J1 — calibration metrics.

Public API:
    brier(y, p)
    expected_calibration_error(y, p, n_bins)
    reliability_table(y, p, n_bins)

`brier` and ECE share their numeric definitions with
`coding_estimator.eval.metrics`; this module re-exports them so the
calibration workstream has one import point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from coding_estimator.eval.metrics import brier, ece as _ece_eval


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width-bin ECE. Identical to `eval.metrics.ece`."""
    return _ece_eval(np.asarray(y), np.asarray(p), n_bins=n_bins)


@dataclass(frozen=True)
class ReliabilityRow:
    bin_index: int
    bin_lower: float
    bin_upper: float
    count: int
    avg_predicted: float | None
    avg_observed: float | None
    gap: float | None


def reliability_table(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Equal-width bins on [0, 1]. One row per bin (empty bins emit n/a
    for the means)."""
    y_arr = np.asarray(y).astype(float)
    p_arr = np.asarray(p).astype(float)
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"y and p must align; got {y_arr.shape} vs {p_arr.shape}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p_arr, edges[1:-1], right=False), 0, n_bins - 1)
    rows: list[ReliabilityRow] = []
    for b in range(n_bins):
        mask = idx == b
        m = int(mask.sum())
        if m == 0:
            rows.append(
                ReliabilityRow(
                    bin_index=b,
                    bin_lower=float(edges[b]),
                    bin_upper=float(edges[b + 1]),
                    count=0,
                    avg_predicted=None,
                    avg_observed=None,
                    gap=None,
                )
            )
            continue
        ap = float(p_arr[mask].mean())
        ay = float(y_arr[mask].mean())
        rows.append(
            ReliabilityRow(
                bin_index=b,
                bin_lower=float(edges[b]),
                bin_upper=float(edges[b + 1]),
                count=m,
                avg_predicted=ap,
                avg_observed=ay,
                gap=ap - ay,
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


__all__ = [
    "brier",
    "expected_calibration_error",
    "reliability_table",
    "ReliabilityRow",
]
