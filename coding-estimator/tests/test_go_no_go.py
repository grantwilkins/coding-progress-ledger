"""
Claim:
- _g4_wins_or_ties: True iff Brier_G4 <= Brier_G2 + P1A_TIE_TOL.
  Lower Brier is better; equal Briers are ties; G4 worse by more than
  the tolerance is a loss.
- _decide_verdict: any required FAIL → 'fail'; all required PASS →
  'pass'; otherwise → 'indeterminate'. Non-required conditions never
  affect the verdict.
- evaluate_p1e: forbidden-column audit returns 'pass' iff no
  checkpoint frame column is in the forbidden list.

Plausible wrong implementations:
- Tie predicate uses `<` instead of `<=` — exact ties become losses.
- Tie tolerance applied symmetrically (also accepts G4 BETTER than G2 by
  -tol as not-a-tie) — usually fine but breaks at the boundary.
- _decide_verdict uses `any(pass)` instead of `all(pass)` — declares
  pass when only one condition passes.
- _decide_verdict counts non-required conditions toward the verdict.
- forbidden audit only checks `exact` matches and ignores prefix /
  suffix entries.
"""

from __future__ import annotations

import pandas as pd
import pytest

from coding_estimator.eval.go_no_go import (
    P1A_TIE_TOL,
    GateCondition,
    _decide_verdict,
    _g4_wins_or_ties,
    evaluate_p1e,
)


# ---------- _g4_wins_or_ties ------------------------------------------------


def test_g4_wins_or_ties_exact_tie_is_true():
    assert _g4_wins_or_ties(0.25, 0.25) is True


def test_g4_wins_or_ties_g4_strictly_better_is_true():
    assert _g4_wins_or_ties(0.30, 0.25) is True


def test_g4_wins_or_ties_g4_within_tolerance_is_true():
    """G4 worse by less than tolerance should still count as a tie."""
    assert _g4_wins_or_ties(0.25, 0.25 + P1A_TIE_TOL / 2) is True


def test_g4_wins_or_ties_g4_outside_tolerance_is_false():
    assert _g4_wins_or_ties(0.25, 0.25 + P1A_TIE_TOL * 10) is False


def test_g4_wins_or_ties_boundary_at_exact_tolerance_is_true():
    """At delta == tolerance, the predicate uses `<=` so the boundary
    is inclusive (still a tie)."""
    assert _g4_wins_or_ties(0.25, 0.25 + P1A_TIE_TOL) is True


# ---------- _decide_verdict -------------------------------------------------


def _cond(cid: str, outcome: str, required: bool = True) -> GateCondition:
    return GateCondition(
        condition_id=cid,
        name=cid,
        required=required,
        outcome=outcome,
        summary="",
    )


def test_decide_verdict_all_pass_required_means_pass():
    conds = [
        _cond("a", "pass", required=True),
        _cond("b", "pass", required=True),
    ]
    assert _decide_verdict(conds) == "pass"


def test_decide_verdict_any_required_fail_means_fail():
    conds = [
        _cond("a", "pass", required=True),
        _cond("b", "fail", required=True),
        _cond("c", "pass", required=True),
    ]
    assert _decide_verdict(conds) == "fail"


def test_decide_verdict_required_indeterminate_means_indeterminate():
    """One indeterminate, no fails — verdict must be `indeterminate`,
    not `pass`."""
    conds = [
        _cond("a", "pass", required=True),
        _cond("b", "indeterminate", required=True),
    ]
    assert _decide_verdict(conds) == "indeterminate"


def test_decide_verdict_non_required_failures_do_not_block_pass():
    """A non-required FAIL should not block PASS."""
    conds = [
        _cond("a", "pass", required=True),
        _cond("b", "pass", required=True),
        _cond("h", "fail", required=False),
    ]
    assert _decide_verdict(conds) == "pass"


def test_decide_verdict_no_required_conditions_is_indeterminate():
    """An empty required-set must NOT silently report pass — that
    would be vacuously-true and dangerous."""
    conds = [
        _cond("a", "pass", required=False),
        _cond("b", "fail", required=False),
    ]
    assert _decide_verdict(conds) == "indeterminate"


def test_decide_verdict_fail_takes_precedence_over_indeterminate():
    """If both fail and indeterminate are present in required, the
    overall verdict must be FAIL (not indeterminate)."""
    conds = [
        _cond("a", "fail", required=True),
        _cond("b", "indeterminate", required=True),
    ]
    assert _decide_verdict(conds) == "fail"


# ---------- evaluate_p1e ----------------------------------------------------


def test_p1e_clean_frame_passes():
    df = pd.DataFrame({"run_id": ["r0"], "source": ["swe_agent_pilot"]})
    out = evaluate_p1e(df)
    assert out.outcome == "pass"
    assert out.evidence["hits"] == []


def test_p1e_explicit_forbidden_column_fails():
    """Synthesize a frame with a column that is on the exact forbidden
    list. We don't hard-code a name (the spec is loaded from JSON);
    instead we read the spec and use one of its exact entries."""
    from coding_estimator.leakage.guard import load_forbidden_spec

    spec = load_forbidden_spec()
    if not spec.exact:
        pytest.skip("no exact forbidden columns in spec")
    bad = next(iter(spec.exact))
    df = pd.DataFrame({"run_id": ["r0"], "source": ["x"], bad: [1]})
    out = evaluate_p1e(df)
    assert out.outcome == "fail"
    assert bad in out.evidence["hits"]


def test_p1e_summary_lists_offenders_when_failing():
    from coding_estimator.leakage.guard import load_forbidden_spec

    spec = load_forbidden_spec()
    if not spec.exact:
        pytest.skip("no exact forbidden columns in spec")
    bad = next(iter(spec.exact))
    df = pd.DataFrame({"run_id": ["r0"], "source": ["x"], bad: [1]})
    out = evaluate_p1e(df)
    assert bad in out.summary
