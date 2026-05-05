"""G1 — Constant baseline. Predicts training-set positive rate."""

from __future__ import annotations

from coding_estimator.baselines.base import BaselineSpec


def _cols(_sources: tuple[str, ...]) -> tuple[str, ...]:
    return ()


CONSTANT = BaselineSpec(name="constant", feature_cols_for=_cols)
