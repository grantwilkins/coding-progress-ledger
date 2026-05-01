"""
Claim:
score_set is the weight-weighted mean of per-member coding-progress
(CODING_CATEGORIES = PRODUCT/VALIDATION/INVESTIGATION). Members with
status_override INVALIDATED or DELETED drop out of both numerator and
denominator. ENVIRONMENT/ARTIFACT/DOCUMENTATION subtasks inside a member
ledger do not contribute to the member's coding-progress.

Plausible wrong implementations:
- Sum unweighted progress instead of weighted.
- Forget to drop INVALIDATED/DELETED members from the denominator.
- Use score(ledger) (all categories) instead of CODING_CATEGORIES.
- Treat status_override BLOCKED as drop-out (the protocol explicitly
  keeps blocked members in the mean — see § 9 open question 2).
"""

from pathlib import Path

from ledger_progress import (
    EventType,
    LedgerEvent,
    LedgerSession,
    LedgerSet,
    LedgerSetMember,
    Status,
    SubtaskCategory,
    apply_event,
    new_ledger,
    score_set,
    to_jsonl,
)


def _write(ledger, path):
    to_jsonl(ledger, str(path))


def _build_progress_ledger(coding_progress: float, tmp_path: Path, name: str) -> str:
    """Build a 4-leaf coding ledger reaching the requested progress, write to disk, return path."""
    session = LedgerSession(f"task {name}", clock=lambda: None)
    ids = [session.add(f"leaf {i}", step=1) for i in range(4)]
    n_complete = round(coding_progress * 4)
    for i in range(n_complete):
        session.complete(ids[i], "done", step=2)
    path = tmp_path / f"{name}.jsonl"
    session.export_jsonl(str(path))
    return str(path)


def test_weighted_mean_three_members(tmp_path):
    a = _build_progress_ledger(1.00, tmp_path, "a")
    b = _build_progress_ledger(0.50, tmp_path, "b")
    c = _build_progress_ledger(0.25, tmp_path, "c")

    s = LedgerSet("trio", members=[
        LedgerSetMember("M1", a, weight=1.0),
        LedgerSetMember("M2", b, weight=2.0),
        LedgerSetMember("M3", c, weight=1.0),
    ])

    expected = (1.0 * 1.00 + 2.0 * 0.50 + 1.0 * 0.25) / (1.0 + 2.0 + 1.0)
    assert score_set(s) == expected


def test_invalidated_member_drops_from_numerator_and_denominator(tmp_path):
    a = _build_progress_ledger(1.00, tmp_path, "a")
    b = _build_progress_ledger(0.50, tmp_path, "b")
    c = _build_progress_ledger(0.00, tmp_path, "c")

    s = LedgerSet("trio", members=[
        LedgerSetMember("M1", a, weight=1.0),
        LedgerSetMember("M2", b, weight=2.0),
        LedgerSetMember("M3", c, weight=5.0, status_override=Status.INVALIDATED),
    ])

    expected = (1.0 * 1.00 + 2.0 * 0.50) / (1.0 + 2.0)
    assert score_set(s) == expected


def test_deleted_member_drops_too(tmp_path):
    a = _build_progress_ledger(1.00, tmp_path, "a")
    b = _build_progress_ledger(0.00, tmp_path, "b")

    s = LedgerSet("pair", members=[
        LedgerSetMember("M1", a),
        LedgerSetMember("M2", b, status_override=Status.DELETED),
    ])

    assert score_set(s) == 1.0


def test_blocked_member_kept_in_mean(tmp_path):
    a = _build_progress_ledger(1.00, tmp_path, "a")
    b = _build_progress_ledger(0.50, tmp_path, "b")

    s = LedgerSet("pair", members=[
        LedgerSetMember("M1", a),
        LedgerSetMember("M2", b, status_override=Status.BLOCKED),
    ])

    assert score_set(s) == 0.75


def test_empty_set_scores_zero():
    assert score_set(LedgerSet("empty")) == 0.0


def test_all_members_invalidated_scores_zero(tmp_path):
    a = _build_progress_ledger(1.00, tmp_path, "a")
    s = LedgerSet("x", members=[
        LedgerSetMember("M1", a, status_override=Status.INVALIDATED),
    ])

    assert score_set(s) == 0.0


def test_non_coding_categories_ignored_inside_member(tmp_path):
    """A member ledger with only ENVIRONMENT/ARTIFACT/DOCUMENTATION leaves
    contributes 0.0 to coding-progress (no coding leaves to score)."""
    session = LedgerSession("infra-only", clock=lambda: None)
    s_env = session.add("env setup", step=1, category=SubtaskCategory.ENVIRONMENT)
    session.complete(s_env, "configured", step=2)
    path = tmp_path / "infra.jsonl"
    session.export_jsonl(str(path))

    s = LedgerSet("x", members=[LedgerSetMember("M1", str(path))])

    assert score_set(s) == 0.0


def test_base_dir_resolves_relative_refs(tmp_path):
    a_path = _build_progress_ledger(1.00, tmp_path, "a")
    rel = Path(a_path).name

    s = LedgerSet("x", members=[LedgerSetMember("M1", rel)])

    assert score_set(s, base_dir=tmp_path) == 1.0
