"""Closure features: counts and progress fractions at the checkpoint.

Re-uses upstream `score()` so progress semantics are inherited from
the ledger, not redefined here.
"""

from __future__ import annotations

from typing import Any

from ledger_progress.core import SubtaskCategory
from ledger_progress.queries import CODING_CATEGORIES
from ledger_progress.scoring import score

from coding_estimator.checkpoints.replay import ReplayState

GROUP = "closure"
COLUMNS: tuple[str, ...] = (
    "completed_leaf_count",
    "coding_progress",
    "validation_progress",
    "product_progress",
    "investigation_progress",
)


def compute(state: ReplayState) -> dict[str, Any]:
    overall = score(state.ledger)
    coding = score(state.ledger, categories=CODING_CATEGORIES)
    validation = score(state.ledger, categories=[SubtaskCategory.VALIDATION])
    product = score(state.ledger, categories=[SubtaskCategory.PRODUCT])
    investigation = score(state.ledger, categories=[SubtaskCategory.INVESTIGATION])
    return {
        "completed_leaf_count": overall.complete_leaf_count,
        "coding_progress": coding.progress,
        "validation_progress": validation.progress,
        "product_progress": product.progress,
        "investigation_progress": investigation.progress,
    }
