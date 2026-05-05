"""Binary-classification metrics. Defined here so the harness and tests
share a single numeric definition."""

from __future__ import annotations

import numpy as np

LOG_LOSS_CLIP = 1e-6
# Output clip for prediction probabilities: matches upstream
# `q_baselines.py::PROB_CLIP` so G2 Brier parity is meaningful.
OUTPUT_CLIP = (0.001, 0.999)


def auroc(y: np.ndarray, p: np.ndarray) -> float | None:
    """Mann-Whitney U formulation, ties broken by average rank."""
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    p_s = p[order]
    y_s = y[order]
    rank_sum = 0.0
    i = 0
    n = len(p_s)
    while i < n:
        j = i + 1
        while j < n and p_s[j] == p_s[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum += avg_rank * float(y_s[i:j].sum())
        i = j
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, clip: float = LOG_LOSS_CLIP) -> float:
    pc = np.clip(p, clip, 1.0 - clip)
    return float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1.0 - pc)))


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error, equal-width bins on [0, 1]."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1], right=False), 0, n_bins - 1)
    n = len(y)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        m = int(mask.sum())
        if m == 0:
            continue
        avg_p = float(p[mask].mean())
        avg_y = float(y[mask].mean())
        total += (m / n) * abs(avg_p - avg_y)
    return total
