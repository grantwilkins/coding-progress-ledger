"""Snapshot of upstream evidence-classification helpers.

Source: ../coding-progress-ledger/scripts/rescore_suite_by_category.py
Snapshot date: 2026-05-04
Snapshot SHA256 of upstream file at snapshot time:
  f2b9f9e5dc771cd83290576c5cfcd6b16a9c98d6e2da91540d49793d65574b3d

Reason: upstream `scripts/` is not packaged, so we cannot import
directly. Per AGENTS.md and TASKS.md D3g, we copy a snapshot. Do NOT
diverge from upstream without bumping the snapshot date AND the SHA
recorded above, and recording the change in the commit.
"""

from __future__ import annotations

import re

UPSTREAM_FILE_RELPATH = "../coding-progress-ledger/scripts/rescore_suite_by_category.py"
SNAPSHOT_SHA256 = "f2b9f9e5dc771cd83290576c5cfcd6b16a9c98d6e2da91540d49793d65574b3d"

STRONG_EVIDENCE_TYPES = frozenset(
    {"test_output", "diff", "file_exists", "command_output", "tool_action"}
)

EVIDENCE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("test_output", (
        "pytest", "unittest", "npm test", "node --test",
        "test_output.txt", "passed", "failed",
    )),
    ("diff", (
        "final_diff.patch", "diff", "patch", "changed", "modified",
        "implementation", "function", "class",
    )),
    ("file_exists", ("exists", "created", "wrote file", "artifact present")),
    ("command_output", (
        "stdout", "stderr", "command output", "cli invocation", "python -m",
    )),
    ("contract_text", (
        "task.md", "readme", "issue statement", "expected behavior", "contract",
    )),
    ("tool_action", (
        "edit ", "submit ", "goto ", "search_file ", "search_dir ", "tool ack",
    )),
)

EVIDENCE_LEVELS: dict[str, str] = {
    "test_output": "mechanical",
    "command_output": "mechanical",
    "diff": "mechanical",
    "file_exists": "mechanical",
    "tool_action": "mechanical",
    "contract_text": "trace_semantic",
    "manual_note": "annotator_judgment",
}

SOURCE_PATH_PATTERN = re.compile(r"(^|\s|`)[\w./-]+\.(py|js|ts|jsx|tsx)\b", re.IGNORECASE)


def classify_evidence(evidence: list[str]) -> set[str]:
    """Classify a list of evidence strings into evidence types.

    Mirrors upstream rescore_suite_by_category.classify_evidence verbatim
    as of the snapshot date.
    """
    evidence_types: set[str] = set()
    for item in evidence:
        text = item.lower()
        for evidence_type, terms in EVIDENCE_PATTERNS:
            if any(term in text for term in terms):
                evidence_types.add(evidence_type)
        if SOURCE_PATH_PATTERN.search(item):
            evidence_types.add("diff")
    if not evidence_types:
        evidence_types.add("manual_note")
    return evidence_types
