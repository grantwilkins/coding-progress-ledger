"""
Claim:
score_set is the weight-weighted MEAN of PER-MEMBER coding-progress,
where coding-progress is itself a per-member fraction over the
CODING_CATEGORIES = (PRODUCT, VALIDATION, INVESTIGATION) slice. The
aggregation lives at the member level. Member order does not affect
the score. Multiplying every weight by a positive constant does not
affect the score.

Plausible wrong implementations:
- Pool every leaf across members and compute one big leaf fraction
  (a 1-leaf member + 100-leaf member would then score 1/101 instead
  of 0.5).
- Apply the CODING_CATEGORIES filter at the member level instead of
  inside each member (a member with mixed PRODUCT + ENVIRONMENT
  would silently use all-categories progress).
- Iterate members by member_id sort order (round-trips byte-equal but
  loses the protocol-required order preservation; would silently
  matter for any future order-dependent reductions and is the easy
  way to introduce a future regression).
- Cache the score on first call and ignore later mark_member calls.
- Normalize by len(members) instead of sum(weights), so equal
  weights (1,1,1) and (10,10,10) accidentally agree but mixed
  weights (1,2,3) vs (10,20,30) drift after a refactor.
"""

import random
from pathlib import Path

import pytest

from ledger_progress import (
    LedgerSession,
    LedgerSet,
    LedgerSetMember,
    LedgerSetSession,
    Status,
    SubtaskCategory,
    read_set_jsonl,
    score_set,
)


def _ledger_with_n_coding_leaves(tmp_path: Path, name: str, n_leaves: int, n_complete: int) -> str:
    assert 0 <= n_complete <= n_leaves
    session = LedgerSession(f"task {name}", clock=lambda: None)
    ids = [session.add(f"leaf {i}", step=1) for i in range(n_leaves)]
    for i in range(n_complete):
        session.complete(ids[i], "done", step=2)
    path = tmp_path / f"{name}.jsonl"
    session.export_jsonl(str(path))
    return str(path)


def test_per_member_aggregation_not_leaf_pooling(tmp_path):
    """A 1-leaf-complete member (progress 1.0) + a 100-leaf-zero-complete
    member (progress 0.0) at equal weight must score 0.5 — exactly the
    mean of (1.0, 0.0). A leaf-pooling impostor would return 1/101 ≈ 0.0099."""
    a = _ledger_with_n_coding_leaves(tmp_path, "small", n_leaves=1, n_complete=1)
    b = _ledger_with_n_coding_leaves(tmp_path, "huge", n_leaves=100, n_complete=0)

    s = LedgerSet("two", members=[
        LedgerSetMember("M1", a, weight=1.0),
        LedgerSetMember("M2", b, weight=1.0),
    ])

    actual = score_set(s)
    assert actual == pytest.approx(0.5)
    assert actual > 0.4  # rules out leaf-pooling = 1/101


def test_coding_filter_applied_inside_each_member(tmp_path):
    """A member whose ledger has 1 PRODUCT leaf (complete) + 1 ENVIRONMENT
    leaf (incomplete) has coding-progress 1.0 (ENVIRONMENT excluded), not
    0.5 (which an all-categories impostor would compute).
    Catches: filter-at-member-level vs filter-at-leaf-level confusion."""
    session = LedgerSession("mixed", clock=lambda: None)
    p = session.add("write fix", step=1, category=SubtaskCategory.PRODUCT)
    session.add("apt-get install gcc", step=1, category=SubtaskCategory.ENVIRONMENT)
    session.complete(p, "diff applied", step=2)
    path = tmp_path / "mixed.jsonl"
    session.export_jsonl(str(path))

    s = LedgerSet("x", members=[LedgerSetMember("M1", str(path))])

    assert score_set(s) == pytest.approx(1.0)


def test_score_invariant_under_member_permutation(tmp_path):
    """Aggregation must be permutation-invariant. We score a 5-member set,
    then shuffle the member list, write+reload, and score again. Any
    impostor that depends on iteration order (e.g. running max, last-wins
    dict folding) breaks here."""
    paths = [
        _ledger_with_n_coding_leaves(tmp_path, f"p{i}", n_leaves=4, n_complete=k)
        for i, k in enumerate([0, 1, 2, 3, 4])
    ]
    members = [LedgerSetMember(f"M{i}", paths[i], weight=float(1 + i)) for i in range(5)]

    canonical = LedgerSet("x", members=list(members))
    canonical_score = score_set(canonical)

    rng = random.Random(0)
    for _ in range(8):
        shuffled = list(members)
        rng.shuffle(shuffled)
        permuted = LedgerSet("x", members=shuffled)
        assert score_set(permuted) == pytest.approx(canonical_score), \
            "score_set must not depend on member order"


def test_score_invariant_under_uniform_weight_rescaling(tmp_path):
    """A weight-weighted mean is invariant under multiplying every weight
    by a positive constant. Catches normalization-by-len(members) and any
    accidental absolute-weight dependence."""
    a = _ledger_with_n_coding_leaves(tmp_path, "a", 4, 4)
    b = _ledger_with_n_coding_leaves(tmp_path, "b", 4, 2)
    c = _ledger_with_n_coding_leaves(tmp_path, "c", 4, 1)

    base = LedgerSet("x", members=[
        LedgerSetMember("M1", a, weight=1.0),
        LedgerSetMember("M2", b, weight=2.0),
        LedgerSetMember("M3", c, weight=3.0),
    ])
    scaled = LedgerSet("y", members=[
        LedgerSetMember("M1", a, weight=10.0),
        LedgerSetMember("M2", b, weight=20.0),
        LedgerSetMember("M3", c, weight=30.0),
    ])

    assert score_set(base) == pytest.approx(score_set(scaled))


def test_single_member_score_is_member_progress_regardless_of_weight(tmp_path):
    """For any single-member set, score == member's coding-progress for
    every positive weight. Catches normalization-by-len, normalization-
    by-fixed-constant, and weight-leakage-into-numerator bugs."""
    a = _ledger_with_n_coding_leaves(tmp_path, "a", 4, 3)  # progress 0.75

    for w in (0.001, 1.0, 7.5, 1000.0):
        s = LedgerSet("x", members=[LedgerSetMember("M1", a, weight=w)])
        assert score_set(s) == pytest.approx(0.75), f"weight {w} leaked into the score"


def test_mark_member_invalidates_on_subsequent_score_call(tmp_path):
    """LedgerSetSession.score() after a mark_member must reflect the new
    override — not a cached pre-mark value. The expected score is the
    mean over the surviving 2 members."""
    a = _ledger_with_n_coding_leaves(tmp_path, "a", 4, 4)  # 1.00
    b = _ledger_with_n_coding_leaves(tmp_path, "b", 4, 2)  # 0.50
    c = _ledger_with_n_coding_leaves(tmp_path, "c", 4, 0)  # 0.00

    session = LedgerSetSession("rollup")
    session.add_member(a)
    session.add_member(b)
    session.add_member(c)

    pre = session.score()
    assert pre == pytest.approx((1.00 + 0.50 + 0.00) / 3)

    session.mark_member("M3", Status.INVALIDATED)
    post = session.score()

    assert post == pytest.approx((1.00 + 0.50) / 2)
    assert post != pre  # rules out a stale cache


def test_read_set_jsonl_preserves_member_order(tmp_path):
    """Writing members in [B, A, C] then reading must yield [B, A, C],
    not the alphabetical [A, B, C]. Catches dict-keyed-by-id readers."""
    s = LedgerSet("x", members=[
        LedgerSetMember("B_member", "b.jsonl"),
        LedgerSetMember("A_member", "a.jsonl"),
        LedgerSetMember("C_member", "c.jsonl"),
    ])
    path = tmp_path / "set.jsonl"

    from ledger_progress.set_serialization import write_set_jsonl
    write_set_jsonl(s, str(path))
    loaded = read_set_jsonl(str(path))

    assert [m.member_id for m in loaded.members] == ["B_member", "A_member", "C_member"]


def test_invalidated_member_does_not_require_a_loadable_ledger(tmp_path):
    """An invalidated member is dropped before its ledger is read.
    A set containing an INVALIDATED member with a bogus ledger_ref must
    still score successfully. Catches an impostor that reads every
    ledger first and only then applies the override filter."""
    a = _ledger_with_n_coding_leaves(tmp_path, "a", 4, 4)

    s = LedgerSet("x", members=[
        LedgerSetMember("M1", a),
        LedgerSetMember("M2", "/no/such/file/should/exist.jsonl",
                        status_override=Status.INVALIDATED),
    ])

    assert score_set(s) == pytest.approx(1.0)


def test_blocked_override_does_not_drop_member_but_invalidated_does(tmp_path):
    """Companion to the protocol § 9 open question 2 lock-in: BLOCKED
    keeps the member in the mean, INVALIDATED removes it. Tested here as
    a single contrast on the same fixture so a future refactor that
    accidentally folds BLOCKED into the drop set fails immediately."""
    a = _ledger_with_n_coding_leaves(tmp_path, "a", 4, 4)  # 1.00
    b = _ledger_with_n_coding_leaves(tmp_path, "b", 4, 0)  # 0.00

    blocked = LedgerSet("x", members=[
        LedgerSetMember("M1", a),
        LedgerSetMember("M2", b, status_override=Status.BLOCKED),
    ])
    invalidated = LedgerSet("x", members=[
        LedgerSetMember("M1", a),
        LedgerSetMember("M2", b, status_override=Status.INVALIDATED),
    ])

    assert score_set(blocked) == pytest.approx(0.5)
    assert score_set(invalidated) == pytest.approx(1.0)
