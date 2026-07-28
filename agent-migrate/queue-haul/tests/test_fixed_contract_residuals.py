"""
Claim:
The fixed-contract table plots requested source-power shed against residual
headroom normalized by advertised capacity while retaining unmet targets.

Plausible wrong implementations:
- Use achieved rather than requested shed on the x-axis.
- Reverse the residual sign.
- Divide residual capacity by resource use.
- Hide unmet watts at infeasible targets.
"""

import pytest

from plot_fixed_contract_residuals import WORKLOADS, _fingerprint, result_row


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
