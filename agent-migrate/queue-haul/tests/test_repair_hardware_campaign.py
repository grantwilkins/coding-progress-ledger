"""
Claim:
The live repair bundle is an exact 16-cell, three-repeat schedule, preceded by
regional 0.1x timing calibration and guarded by pending-only shadow validation.

Plausible wrong implementations:
- Launch the main grid without measuring the out-of-support 0.1x timing point.
- Drop one East/Germany impairment combination or silently change the cut.
- Replan active attempts instead of limiting applied changes to pending work.
"""

import json

import pytest

import repair_hardware_campaign as campaign


def test_ttft_uses_request_start_and_first_content_token():
    assert campaign._ttft_s({
        "start_ns": 1_000_000_000,
        "first_byte_ns": 1_250_000_000,
        "end_ns": 2_000_000_000,
    }) == .25
    assert campaign._ttft_s({"start_ns": 1, "first_byte_ns": None}) is None


def test_hardware_plan_has_calibration_gate_and_exact_repair_grid(tmp_path):
    out = tmp_path / "bundle"
    plan = campaign.prepare(
        campaign.DEFAULT_PARENT,
        campaign.ROOT / "azure_network_cluster_east_germany.json",
        campaign.ROOT / (
            "outputs/east-germany-frontier-20260808/control/"
            "calibration-east-germany-frontier-001.json"),
        out,
    )

    campaign.validate_plan(plan)
    assert len(plan["calibration_cells"]) == 36
    assert len(plan["episodes"]) == 48
    assert {row["node"] for row in plan["calibration_cells"]} == {
        "east", "germany"}
    assert {row["method"] for row in plan["calibration_cells"]} == {
        "replay", "kv_transfer"}
    assert {row["context_tokens"] for row in plan["calibration_cells"]} == {
        1536, 7680, 32256}
    assert {row["cut_scale"] for row in plan["episodes"]} == {.1}
    assert {row["trigger_work_fraction"] for row in plan["episodes"]} == {.25}
    assert plan["apply_policy"] == "shadow_validate_then_apply_pending_only"
    assert plan["calibration_gate"] == {
        "relative_error": .15,
        "absolute_error_s": 1.0,
        "contexts": [1536, 7680, 32256],
        "repeats": 3,
    }
    assert json.loads((out / "plan.json").read_text()) == plan
    script = (out / "run.sh").read_text()
    assert "QH_AZURE_SSH_KEY" in script
    assert "QH_REPAIR_RUN_ROOT" in script
    assert "repair_hardware_campaign.py run" in script


def test_timing_promotion_only_expands_the_measured_lower_bound():
    parent = json.loads(campaign.DEFAULT_PARENT.read_text())
    template = campaign._template(parent)
    contract = json.loads((campaign.ROOT / (
        "outputs/east-germany-frontier-20260808/control/"
        "frontier-pilot-002.json")).read_text())["network_contract"]

    promoted = campaign._promote_components(template, contract)

    for node in ("east", "germany"):
        expected = contract["paths"][node]["natural_mbps"] * .1 * 125_000
        for component in promoted["migration_components"][node].values():
            assert component["bandwidth_range_bytes_per_s"][0] <= expected
            assert component["allow_extrapolation"] is False
            assert "pending live 0.1x route calibration" in component["provenance"]


def _synthetic_timing_rows():
    rows = []
    parameters = {
        ("east", "replay"): .6,
        ("east", "kv_transfer"): 1.8,
        ("germany", "replay"): .5,
        ("germany", "kv_transfer"): .9,
    }
    for node in ("east", "germany"):
        for method in campaign.CALIBRATION_METHODS:
            for context in campaign.CALIBRATION_CONTEXTS:
                for repeat in range(campaign.REPEATS):
                    base = context / 100_000
                    scale = context / 10_000 if method == "replay" else 1.0
                    observed = base + parameters[node, method] * scale
                    rows.append({
                        "cell_id": f"{node}-{method}-{context}-{repeat}",
                        "node": node, "method": method,
                        "context_tokens": context, "repeat": repeat,
                        "bandwidth_mbps": 100.0,
                        "status": "complete", "observed_s": observed,
                        "fit_base_s": base, "fit_scale_s": scale,
                        "ttft_s": observed, "kv_verified": True,
                    })
    return rows


def test_timing_fit_trains_small_contexts_and_gates_on_32k_holdout():
    template = campaign._template(json.loads(campaign.DEFAULT_PARENT.read_text()))
    gate = {
        "relative_error": .15, "absolute_error_s": 1.0,
        "contexts": list(campaign.CALIBRATION_CONTEXTS),
        "repeats": campaign.REPEATS,
    }

    summary = campaign._timing_summary(
        _synthetic_timing_rows(), gate, template)

    assert summary["passed"] is True
    assert summary["training_contexts"] == [1536, 7680]
    assert summary["holdout_context"] == 32256
    assert summary["held_out_rows"] == 12
    assert summary["ttft_rows"] == 36
    assert summary["fits"]["east"]["migration_components"][
        "replay"]["compute_completion_factor"] == pytest.approx(.6)
    assert summary["fits"]["germany"]["migration_components"][
        "kv_transfer"]["residual_s"] == pytest.approx(.9)

    missing_ttft = _synthetic_timing_rows()
    missing_ttft[0]["ttft_s"] = None
    assert campaign._timing_summary(
        missing_ttft, gate, template)["passed"] is False


def test_low_bandwidth_fit_is_applied_only_to_cut_nodes():
    scenario = {"migration_components": {
        "east": {"replay": {"compute_completion_factor": 1}},
        "germany": {"replay": {"compute_completion_factor": 2}},
    }}
    timing = {"fits": {
        "east": {"migration_components": {
            "replay": {"compute_completion_factor": .5}}},
        "germany": {"migration_components": {
            "replay": {"compute_completion_factor": .6}}},
    }}

    fitted = campaign._apply_timing_fit(scenario, timing, ("east",))

    assert fitted["migration_components"]["east"]["replay"][
        "compute_completion_factor"] == .5
    assert fitted["migration_components"]["germany"]["replay"][
        "compute_completion_factor"] == 2
    assert scenario["migration_components"]["east"]["replay"][
        "compute_completion_factor"] == 1
