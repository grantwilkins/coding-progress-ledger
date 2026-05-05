"""G5 — ledger-dynamics baseline.

Diagnostic feature group derived in
`coding_estimator.checkpoints.dynamics`. Available only on frames that
have been passed through `attach_g5_features` first.
"""

from __future__ import annotations

from coding_estimator.baselines.base import BaselineSpec
from coding_estimator.checkpoints.dynamics import G5_FEATURES


def _cols(_sources: tuple[str, ...]) -> tuple[str, ...]:
    return G5_FEATURES


LEDGER_DYNAMICS = BaselineSpec(name="ledger_dynamics", feature_cols_for=_cols)


__all__ = ["LEDGER_DYNAMICS"]
