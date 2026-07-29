"""
Claim:
Every met fixed-contract point has nonnegative planned slack for every enforced
physical capacity, and the plotted value is exactly that normalized slack.

Plausible wrong implementations:
- Use achieved rather than requested shed on the x-axis.
- Reverse the residual sign.
- Divide residual capacity by resource use.
- Hide unmet watts at infeasible targets.
- Mark a source/WAN-feasible point met despite violating destination capacity.
"""

import pytest

from plot_fixed_contract_residuals import (
    WORKLOADS,
    _fingerprint,
    minimum_slack,
    result_row,
    validate_slacks,
)


def test_result_row_preserves_requested_power_and_capacity_normalized_slack():
    row = result_row(10, 8, "route", 3, 4)

    assert row["requested_shed_w"] == 10
    assert row["achieved_shed_w"] == 8
    assert row["unmet_shed_w"] == 2
    assert row["normalized_slack"] == pytest.approx(.25)
    assert row["planned_slack"] == row["normalized_slack"]

    assert result_row(10, 10, "route", 5, 4)["normalized_slack"] == pytest.approx(-.25)


def test_cache_fingerprint_tracks_workload_inputs(tmp_path, monkeypatch):
    model, manifest, workload = (tmp_path / name for name in ("model", "manifest", "workload"))
    for path in (model, manifest, workload):
        path.write_text("first")
    monkeypatch.setitem(WORKLOADS, "interactive_coding", workload)
    first = _fingerprint(model, manifest)["input_sha256"]

    workload.write_text("second")

    assert _fingerprint(model, manifest)["input_sha256"] != first


def test_met_target_rejects_any_enforced_capacity_violation():
    validate_slacks(True, {"route": 0, "service": 1e-8})

    with pytest.raises(AssertionError, match="service"):
        validate_slacks(True, {"route": 0, "service": -1e-6})

    validate_slacks(False, {"service": -1})


def test_minimum_slack_tracks_the_binding_transition():
    rows = [
        {"target_fraction": .25, "resource": "source", "normalized_slack": .2},
        {"target_fraction": .25, "resource": "service", "normalized_slack": .8},
        {"target_fraction": .50, "resource": "source", "normalized_slack": .4},
        {"target_fraction": .50, "resource": "service", "normalized_slack": .1},
    ]

    assert minimum_slack(rows, ("source", "service")) == ("source", "service")
