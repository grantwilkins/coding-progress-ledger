"""
Claim:
compare_annotations.compare_pair computes per-pilot inter-annotator
agreement metrics. The metrics must be SYMMETRIC where the underlying
quantity is symmetric (category-vector L1) and SIGNED where direction
matters (progress delta is v2 - v1; sign matters).

Plausible wrong implementations:
- _category_l1_distance treats keys present only in one vector as 0
  rather than as the other vector's count, understating disagreement.
- progress delta is computed as v1 - v2 (sign reversed), so "v2
  scored higher" is reported as a negative delta.
- Two specs with the same pilot_id but different content (different
  leaves) get a "high agreement" verdict because the verdict only
  looks at progress and not at leaf count or category disagreement.
- compare_pair silently accepts pilot_id mismatch.
- _verdict thresholds are off — e.g. progress delta 0.30 returns
  "high" because of an open-vs-closed-interval bug.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_annotations import (
    _category_l1_distance,
    _verdict,
    compare_pair,
)


def _spec(events, *, pilot_id="test_pilot_x", root="root", quality=None):
    return {
        "pilot_id": pilot_id,
        "instance_id": "x__y-1",
        "root_task": root,
        "events": events,
        "quality": quality
        or {
            "annotator": "test",
            "annotation_time_minutes": 0,
            "number_of_uncertain_events": 0,
            "number_of_evidence_gaps": 0,
            "whether_final_success_used_only_at_end": True,
            "whether_progress_forced": False,
            "whether_schema_gap_found": False,
        },
    }


def _all_complete(n: int, category: str = "PRODUCT") -> list:
    """Spec with n leaves, all complete."""
    out = []
    for i in range(1, n + 1):
        out.append({"op": "add", "step": i, "id": f"S{i}", "category": category, "description": f"task {i}"})
        out.append({"op": "complete", "step": i, "id": f"S{i}", "evidence": f"step {i}: done"})
    return out


# ---------- _category_l1_distance ----------


def test_category_l1_zero_when_vectors_match():
    assert _category_l1_distance({"PRODUCT": 2, "VALIDATION": 1}, {"PRODUCT": 2, "VALIDATION": 1}) == 0


def test_category_l1_uses_union_of_keys_not_intersection():
    # If a category appears in only one vector, it must contribute its
    # full count to the distance — otherwise an annotator who omits a
    # category entirely looks identical to one who has it.
    assert _category_l1_distance({"PRODUCT": 3}, {"PRODUCT": 3, "VALIDATION": 2}) == 2
    assert _category_l1_distance({"VALIDATION": 5}, {"PRODUCT": 3}) == 8


def test_category_l1_is_symmetric():
    a = {"PRODUCT": 2, "INVESTIGATION": 4}
    b = {"PRODUCT": 5, "VALIDATION": 1}
    assert _category_l1_distance(a, b) == _category_l1_distance(b, a)


# ---------- _verdict thresholds ----------


def test_verdict_high_requires_all_three_close():
    assert _verdict(progress_delta=0.05, leaf_delta=0, cat_l1=0) == "high"
    # progress agreement alone is not enough
    assert _verdict(progress_delta=0.05, leaf_delta=5, cat_l1=8) == "low"
    # leaf agreement alone is not enough
    assert _verdict(progress_delta=0.4, leaf_delta=0, cat_l1=0) == "low"


def test_verdict_signs_are_handled_via_abs():
    # large positive vs large negative progress delta both fail
    assert _verdict(progress_delta=0.4, leaf_delta=0, cat_l1=0) == _verdict(progress_delta=-0.4, leaf_delta=0, cat_l1=0)


def test_verdict_moderate_band():
    assert _verdict(progress_delta=0.15, leaf_delta=2, cat_l1=2) == "moderate"
    # Just over the moderate band -> low
    assert _verdict(progress_delta=0.21, leaf_delta=2, cat_l1=2) == "low"


# ---------- compare_pair: signs and structure ----------


def test_compare_pair_progress_delta_is_v2_minus_v1():
    # v1: 1 leaf, complete -> progress 1.0
    # v2: 2 leaves, both complete -> progress 1.0 (same)
    # v3: 2 leaves, one complete -> progress 0.5
    spec_v1 = _spec(_all_complete(1))
    spec_v2_lower = _spec(_all_complete(1) + [
        {"op": "add", "step": 5, "id": "S2", "category": "PRODUCT", "description": "extra"},
    ])
    out = compare_pair(spec_v1, spec_v2_lower)
    # v1 progress 1.0, v2 progress 0.5 -> delta = -0.5 (v2 is lower)
    assert out["progress_delta"] == pytest.approx(-0.5)
    assert out["leaf_delta"] == 1  # v2 has one more leaf
    # And the reverse direction:
    out_rev = compare_pair(spec_v2_lower, spec_v1)
    assert out_rev["progress_delta"] == pytest.approx(0.5)


def test_compare_pair_raises_on_pilot_id_mismatch():
    s1 = _spec(_all_complete(1), pilot_id="A")
    s2 = _spec(_all_complete(1), pilot_id="B")
    with pytest.raises(ValueError, match="pilot_id mismatch"):
        compare_pair(s1, s2)


def test_compare_pair_counts_reopens_and_blocks_signed():
    s1 = _spec([
        {"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "x"},
        {"op": "complete", "step": 2, "id": "S1", "evidence": "step 2: ok"},
    ])
    s2 = _spec([
        {"op": "add", "step": 1, "id": "S1", "category": "PRODUCT", "description": "x"},
        {"op": "complete", "step": 2, "id": "S1", "evidence": "step 2: ok"},
        {"op": "reopen", "step": 5, "id": "S1", "reason": "patch wrong"},
    ])
    out = compare_pair(s1, s2)
    assert out["reopen_delta"] == 1
    assert out["block_delta"] == 0
    # And reverse
    out_rev = compare_pair(s2, s1)
    assert out_rev["reopen_delta"] == -1


def test_compare_pair_high_verdict_needs_progress_AND_leaves_AND_categories():
    """A pair that agrees on progress only (e.g. both 1.0) but disagrees
    sharply on leaf count and categories should not earn a 'high' verdict.
    Otherwise the verdict is just a thresholded progress check."""
    # v1: 5 INVESTIGATION leaves all complete -> progress=1.0
    # v2: 1 PRODUCT leaf complete -> progress=1.0
    s1 = _spec(_all_complete(5, category="INVESTIGATION"))
    s2 = _spec(_all_complete(1, category="PRODUCT"))
    out = compare_pair(s1, s2)
    # Progress matches exactly (both 1.0)
    assert out["progress_delta"] == pytest.approx(0.0)
    # But leaves and categories disagree dramatically
    assert out["leaf_delta"] == -4
    assert out["category_l1_distance"] >= 5  # 5 INV vs 1 PRODUCT -> at least |5|+|1|=6
    # So verdict is NOT high
    assert out["verdict"] != "high"
