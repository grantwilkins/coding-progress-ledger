"""
Claim:
A realistic tiny coding task should produce a non-monotonic progress curve as
new required work is discovered.

Plausible wrong implementations:
- Treat progress as elapsed effort and force monotonicity.
- Forget to expand the denominator when hidden work is discovered.
- Complete bookkeeping without preserving numerator and denominator.
"""

from conftest import add_subtask, event, mark_complete
from ledger_progress import EventType, apply_event, new_ledger, score


def test_timezone_parser_toy_coding_run_curve_feels_right():
    ledger = new_ledger("Fix parse_offset(s) to accept +0530 as well as +05:30")
    for description in [
        "Understand expected offset behavior",
        "Locate parser",
        "Add regression test",
        "Patch parser",
        "Run targeted tests",
    ]:
        ledger = add_subtask(ledger, description, step=0)
    assert score(ledger).progress == 0.0

    ledger = mark_complete(ledger, "S1", "Expected +0530 means UTC+05:30", step=3)
    ledger = mark_complete(ledger, "S2", "Opened parse_offset implementation", step=3)
    assert score(ledger).progress == 0.4

    ledger = add_subtask(ledger, "Preserve existing +05:30 behavior", step=5)
    assert round(score(ledger).progress, 2) == 0.33

    ledger = mark_complete(ledger, "S3", "Regression test added", step=7)
    ledger = mark_complete(ledger, "S4", "Parser patched", step=7)
    assert round(score(ledger).progress, 2) == 0.67

    apply_event(ledger, event(9, EventType.ADD_EVIDENCE, "S5", {"evidence": ["Targeted test passes."]}))
    ledger = add_subtask(ledger, "Inspect serializer round-trip path", step=9)
    ledger = add_subtask(ledger, "Fix serializer regression if caused by parser change", step=9)
    assert score(ledger).progress == 0.5

    for subtask_id, evidence in [
        ("S5", "Targeted and broader tests pass"),
        ("S6", "Existing +05:30 behavior preserved"),
        ("S7", "Serializer round-trip inspected"),
        ("S8", "Serializer regression addressed"),
    ]:
        ledger = mark_complete(ledger, subtask_id, evidence, step=12)
    assert score(ledger).progress == 1.0
