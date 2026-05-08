"""Run-level bootstrap for Brier-score CIs.

The harness MUST resample at the run level: rows within a run are
correlated, so checkpoint-level resampling inflates power. See
TASKS § G7 (resampling rule).
"""

from __future__ import annotations

import numpy as np


def brier_per_run(
    y_by_run: dict[str, np.ndarray],
    p_by_run: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bundle (y, p) per run_id for bootstrap consumption."""
    if set(y_by_run) != set(p_by_run):
        raise ValueError("y_by_run and p_by_run must cover the same runs")
    return {r: (y_by_run[r], p_by_run[r]) for r in y_by_run}


def bootstrap_brier_ci(
    bundle: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    b: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Resample test runs with replacement; compute Brier from concatenated
    rows of the resampled runs; return percentile CI."""
    runs = sorted(bundle)
    if not runs:
        raise ValueError("bootstrap requires at least one run")
    rng = np.random.default_rng(seed)
    samples = np.empty(b, dtype=float)
    n = len(runs)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        ys: list[np.ndarray] = []
        ps: list[np.ndarray] = []
        for j in idx:
            y, p = bundle[runs[j]]
            ys.append(y)
            ps.append(p)
        y_all = np.concatenate(ys)
        p_all = np.concatenate(ps)
        samples[i] = float(np.mean((p_all - y_all) ** 2))
    lo = float(np.percentile(samples, 100.0 * alpha / 2))
    hi = float(np.percentile(samples, 100.0 * (1.0 - alpha / 2)))
    return lo, hi
