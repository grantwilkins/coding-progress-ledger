"""Frontier features: active leaf counts at the checkpoint.

These come straight from the upstream `score()` observation: the count
of active leaves (status not in INVALIDATED/DELETED, no active children)
overall, in coding categories, and in validation specifically.

Prefix-only by construction: every value is a function of
ReplayState.ledger, which the replay engine has already filtered.
"""

from __future__ import annotations

from typing import Any

from ledger_progress.core import SubtaskCategory
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.scoring import score

from coding_estimator.checkpoints.replay import ReplayState

GROUP = "frontier"
COLUMNS: tuple[str, ...] = (
    "active_leaf_count",
    "active_coding_leaf_count",
    "active_validation_leaf_count",
)


def compute(state: ReplayState) -> dict[str, Any]:
    overall = score(state.ledger)
    coding = score(state.ledger, categories=CODING_CATEGORIES)
    validation = score(state.ledger, categories=[SubtaskCategory.VALIDATION])
    return {
        "active_leaf_count": overall.active_leaf_count,
        "active_coding_leaf_count": coding.active_leaf_count,
        "active_validation_leaf_count": validation.active_leaf_count,
    }
