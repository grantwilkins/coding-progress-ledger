"""
Claim:
The fixed-contract table preserves physical capacity accounting and the plot
normalizes only positive capacity overruns into binding shares.

Plausible wrong implementations:
- Use achieved rather than requested shed on the x-axis.
- Reverse the residual sign.
- Divide residual capacity by resource use.
- Hide unmet watts at infeasible targets.
- Normalize unused capacity into the binding mix.
"""

import pytest

from plot_fixed_contract_residuals import (
    WORKLOADS,
    _fingerprint,
    overrun_shares,
    result_row,
)


def test_result_row_preserves_requested_power_and_capacity_normalized_slack():
    row = result_row(10, 8, "route", 3, 4)

    assert row["requested_shed_w"] == 10
    assert row["achieved_shed_w"] == 8
    assert row["unmet_shed_w"] == 2
    assert row["normalized_slack"] == pytest.approx(.25)

    assert result_row(10, 10, "route", 5, 4)["normalized_slack"] == pytest.approx(-.25)


def test_cache_fingerprint_tracks_workload_inputs(tmp_path, monkeypatch):
    model, manifest, workload = (tmp_path / name for name in ("model", "manifest", "workload"))
    for path in (model, manifest, workload):
        path.write_text("first")
    monkeypatch.setitem(WORKLOADS, "interactive_coding", workload)
    first = _fingerprint(model, manifest)["input_sha256"]

    workload.write_text("second")

    assert _fingerprint(model, manifest)["input_sha256"] != first


def test_binding_mix_normalizes_only_capacity_overruns():
    rows = [
        result_row(10, 10, "unused", 1, 2),
        result_row(10, 10, "small overrun", 3, 2),
        result_row(10, 10, "large overrun", 6, 2),
    ]
    shares = {row["resource"]: row["overrun_share"] for row in overrun_shares(rows)}

    assert shares == pytest.approx({
        "unused": 0,
        "small overrun": .2,
        "large overrun": .8,
    })
