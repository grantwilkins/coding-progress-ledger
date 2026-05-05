"""v0 baseline ladder. G1 / G2 / G4 ship; G3 / G5 / G6 are deferred."""

from coding_estimator.baselines.base import (
    BaselineSpec,
    FittedBinary,
    fit_binary,
)
from coding_estimator.baselines.constant import CONSTANT
from coding_estimator.baselines.ledger_basic import LEDGER_BASIC
from coding_estimator.baselines.time_only import TIME_ONLY

V0_BASELINES: tuple[BaselineSpec, ...] = (CONSTANT, TIME_ONLY, LEDGER_BASIC)

__all__ = [
    "BaselineSpec",
    "FittedBinary",
    "fit_binary",
    "CONSTANT",
    "TIME_ONLY",
    "LEDGER_BASIC",
    "V0_BASELINES",
]
