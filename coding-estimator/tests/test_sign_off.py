"""
Claim:
- _decide_not_safe_for_control returns (True, reasons) iff any one of:
    gate.verdict != 'pass'
    OR any required GateCondition.outcome != 'pass'
    OR O1.outcome != 'pass'
    OR O5.outcome != 'pass'
    OR any per-source O7 result has outcome == 'fail'
  Returns (False, []) only when all of the above clear.
- write_p3_report writes READY_FOR_SCHEDULING.md iff gate.verdict ==
  'pass'; otherwise writes NOT_READY_FOR_SCHEDULING.md. The two file
  names are mutually exclusive — never both, never neither.

Plausible wrong implementations:
- Treats `indeterminate` failure-mode results as 'pass' (silently
  flips not_safe_for_control to False on incomplete data).
- Only checks gate.verdict and ignores individual condition outcomes
  (a `pass` verdict can mask required-condition issues if `_decide_verdict`
  is also wrong — defense in depth).
- write_p3_report writes BOTH files (or neither) for some inputs.
- write_p3_report uses gate verdict spelling like 'PASS' instead of
  'pass'.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from coding_estimator.eval.failure_modes import FailureModeResult
from coding_estimator.eval.go_no_go import GateCondition, GateReport
from coding_estimator.eval.sign_off import (
    _decide_not_safe_for_control,
    write_p3_report,
)


def _cond(cid: str, outcome: str, required: bool = True) -> GateCondition:
    return GateCondition(
        condition_id=cid, name=cid, required=required, outcome=outcome, summary=""
    )


def _result(test_id: str, outcome: str, source: str | None = None) -> FailureModeResult:
    detail = {"source": source} if source is not None else {}
    return FailureModeResult(
        test_id=test_id,
        outcome=outcome,
        metric_name="m",
        metric_value=0.0,
        threshold=0.02,
        note=None,
        detail=detail,
    )


def _all_pass_gate() -> GateReport:
    return GateReport(
        verdict="pass",
        conditions=[
            _cond("P1.a", "pass"),
            _cond("P1.b", "pass"),
            _cond("P1.c", "pass"),
            _cond("P1.d", "pass"),
            _cond("P1.e", "pass"),
            _cond("P1.f", "pass"),
            _cond("P1.g", "pass"),
            _cond("P1.h", "pass", required=False),
        ],
    )


def test_not_safe_returns_false_only_when_everything_passes():
    gate = _all_pass_gate()
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = [_result("O7", "pass", source="src1"), _result("O7", "pass", source="src2")]
    unsafe, reasons = _decide_not_safe_for_control(gate, o1, o5, o7)
    assert unsafe is False
    assert reasons == []


def test_not_safe_when_gate_verdict_is_fail():
    gate = GateReport(verdict="fail", conditions=_all_pass_gate().conditions)
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = [_result("O7", "pass", source="src1")]
    unsafe, reasons = _decide_not_safe_for_control(gate, o1, o5, o7)
    assert unsafe is True
    assert any("gate verdict" in r for r in reasons)


def test_not_safe_when_gate_verdict_is_indeterminate():
    gate = GateReport(
        verdict="indeterminate", conditions=_all_pass_gate().conditions
    )
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = []
    unsafe, _ = _decide_not_safe_for_control(gate, o1, o5, o7)
    assert unsafe is True


def test_not_safe_when_required_condition_indeterminate():
    """Even if verdict is somehow 'pass', a required-condition flag of
    indeterminate must trip the safety flag."""
    conds = [
        _cond("P1.a", "pass"),
        _cond("P1.b", "indeterminate"),  # required, indeterminate
    ]
    gate = GateReport(verdict="pass", conditions=conds)  # mismatch is intentional
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = []
    unsafe, reasons = _decide_not_safe_for_control(gate, o1, o5, o7)
    assert unsafe is True
    assert any("P1.b" in r for r in reasons)


def test_not_safe_when_o1_indeterminate_does_not_pass():
    """Indeterminate is NOT pass — it must propagate to unsafe."""
    gate = _all_pass_gate()
    o1 = _result("O1", "indeterminate")
    o5 = _result("O5", "pass")
    o7 = [_result("O7", "pass", source="src1")]
    unsafe, reasons = _decide_not_safe_for_control(gate, o1, o5, o7)
    assert unsafe is True
    assert any("O1" in r for r in reasons)


def test_not_safe_when_any_o7_per_source_fails():
    gate = _all_pass_gate()
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = [
        _result("O7", "pass", source="src1"),
        _result("O7", "fail", source="src2"),
    ]
    unsafe, reasons = _decide_not_safe_for_control(gate, o1, o5, o7)
    assert unsafe is True
    assert any("src2" in r for r in reasons)


def test_not_safe_o7_indeterminate_alone_does_not_trip_flag():
    """Per the contract, only `fail` on O7 trips the flag; an entirely
    `indeterminate` O7 (e.g., single-class y) does not by itself."""
    gate = _all_pass_gate()
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = [_result("O7", "indeterminate", source="src1")]
    unsafe, reasons = _decide_not_safe_for_control(gate, o1, o5, o7)
    # No O7 fail, no O1/O5 fail, all gate pass → must be safe
    assert unsafe is False
    assert reasons == []


def test_p3_writes_ready_when_gate_passes(tmp_path: Path):
    gate = _all_pass_gate()
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = [_result("O7", "pass", source="src1")]
    record = {
        "estimator_version": "0.1.0",
        "calibration_status": {},
        "not_safe_for_control": False,
        "estimator_id": "logreg_v0.1",
    }
    out = write_p3_report(
        tmp_path, estimator_id="logreg_v0.1",
        record=record, gate=gate, o1=o1, o5=o5, o7=o7,
    )
    assert out.name == "READY_FOR_SCHEDULING.md"
    assert (tmp_path / "READY_FOR_SCHEDULING.md").exists()
    assert not (tmp_path / "NOT_READY_FOR_SCHEDULING.md").exists()


@pytest.mark.parametrize("verdict", ["fail", "indeterminate"])
def test_p3_writes_not_ready_when_gate_does_not_pass(tmp_path: Path, verdict):
    gate = GateReport(verdict=verdict, conditions=_all_pass_gate().conditions)
    o1 = _result("O1", "pass")
    o5 = _result("O5", "pass")
    o7 = [_result("O7", "pass", source="src1")]
    record = {
        "estimator_version": "0.1.0",
        "calibration_status": {},
        "not_safe_for_control": True,
        "estimator_id": "logreg_v0.1",
        "known_limits": [],
    }
    out = write_p3_report(
        tmp_path, estimator_id="logreg_v0.1",
        record=record, gate=gate, o1=o1, o5=o5, o7=o7,
    )
    assert out.name == "NOT_READY_FOR_SCHEDULING.md"
    assert (tmp_path / "NOT_READY_FOR_SCHEDULING.md").exists()
    assert not (tmp_path / "READY_FOR_SCHEDULING.md").exists()
