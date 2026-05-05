"""G4 — Ledger-basic baseline.

Features = closure + frontier + instability + discovery feature groups
(no dynamics, no semantics, no source identifiers). The v0 headline:
its no-regression vs G2 is the gate in Workstream P.
"""

from __future__ import annotations

from coding_estimator.baselines.base import BaselineSpec
from coding_estimator.checkpoints.features.registry import GROUPS

_LEDGER_GROUPS = ("closure", "frontier", "instability", "discovery")


def _cols(_sources: tuple[str, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for g in _LEDGER_GROUPS:
        for f in GROUPS[g]:
            if f.dtype not in ("int", "float", "bool"):
                continue
            out.append(f.column_name)
    return tuple(out)


LEDGER_BASIC = BaselineSpec(name="ledger_basic", feature_cols_for=_cols)
