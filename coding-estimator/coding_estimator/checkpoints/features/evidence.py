"""Evidence features: classify completion-evidence at the checkpoint.

Re-uses a snapshot of upstream `classify_evidence` (see
_upstream_evidence_snapshot.py) so the classification matches what the
ledger ecosystem already produces.

A "strong" completion is one where at least one evidence string
classifies into a STRONG_EVIDENCE_TYPES bucket (mechanical evidence:
test_output, diff, file_exists, command_output, tool_action).
A "manual_only" completion is one whose evidence classifies into
{manual_note} ONLY (annotator_judgment level). A "weak product
completion" is a PRODUCT-category completion that is manual-only.
"""

from __future__ import annotations

from typing import Any

from ledger_progress.core import EventType, SubtaskCategory

from coding_estimator.checkpoints.features._upstream_evidence_snapshot import (
    STRONG_EVIDENCE_TYPES,
    classify_evidence,
)
from coding_estimator.checkpoints.replay import ReplayState

GROUP = "evidence"
COLUMNS: tuple[str, ...] = (
    "strong_completion_count",
    "manual_only_completion_count",
    "weak_product_completion_count",
    "strong_evidence_fraction",
    "manual_only_evidence_fraction",
    "latest_completion_evidence_type",
)


def compute(state: ReplayState) -> dict[str, Any]:
    """For every COMPLETE update_status event seen in the prefix, look
    up its evidence list (carried in the event payload), classify, and
    aggregate."""
    strong = 0
    manual_only = 0
    weak_product = 0
    latest_type: str | None = None
    completions = 0

    for e in state.events_so_far:
        if e.event_type is not EventType.UPDATE_STATUS:
            continue
        if e.payload.get("status") != "complete":
            continue
        completions += 1
        evidence_list = e.payload.get("evidence") or []
        if not isinstance(evidence_list, list):
            continue
        types = classify_evidence([str(x) for x in evidence_list])
        is_strong = bool(types & STRONG_EVIDENCE_TYPES)
        is_manual_only = types == {"manual_note"}
        if is_strong:
            strong += 1
            # Pick a representative type for the latest_completion. If
            # multiple strong types fire, prefer test_output, then diff.
            for preferred in ("test_output", "diff", "file_exists",
                              "command_output", "tool_action"):
                if preferred in types:
                    latest_type = preferred
                    break
        if is_manual_only:
            manual_only += 1
            latest_type = "manual_note"
        if is_manual_only and e.subtask_id:
            subtask = state.ledger.subtasks.get(e.subtask_id)
            if subtask and subtask.category is SubtaskCategory.PRODUCT:
                weak_product += 1

    strong_frac = (strong / completions) if completions else 0.0
    manual_frac = (manual_only / completions) if completions else 0.0

    return {
        "strong_completion_count": strong,
        "manual_only_completion_count": manual_only,
        "weak_product_completion_count": weak_product,
        "strong_evidence_fraction": strong_frac,
        "manual_only_evidence_fraction": manual_frac,
        "latest_completion_evidence_type": latest_type,
    }
