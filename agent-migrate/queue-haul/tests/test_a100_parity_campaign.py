import json
from collections import Counter

import pytest

import a100_parity_campaign as campaign


def inputs(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "queue-haul-migration-manifest-v2", "workload": "coding",
        "sessions": [{"id": f"s{index}", "job_class": "coding"}
                     for index in range(8)],
    }))
    cluster = tmp_path / "cluster.json"
    node = {"ssh_user": "u", "repo_root": "/r", "run_root": "/d"}
    cluster.write_text(json.dumps({
        "schema": campaign.network.CLUSTER_SCHEMA,
        "source": {**node, "id": "source", "region": "swedencentral",
                   "host": "10.0.0.1"},
        "destinations": [
            {**node, "id": "east", "region": "eastus2", "host": "10.0.0.2"},
            {**node, "id": "germany", "region": "germanywestcentral",
             "host": "10.0.0.3"},
        ],
    }))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "schema": campaign.network.CALIBRATION_SCHEMA,
        "clock_uncertainty_ms": {"source": .1, "east": .1, "germany": .1},
        "aggregate_simultaneous_mbps": [4000],
        "paths": {
            "east": {"rtt_ms": [98], "simultaneous_mbps": [1200]},
            "germany": {"rtt_ms": [25], "simultaneous_mbps": [2800]},
        },
    }))
    model = tmp_path / "timing-model.json"
    components = {
        method: {"context_range": [8192, 31488],
                 "compute_completion_factor": 1,
                 "residual_s": 1 if method == "kv_transfer" else 0}
        for method in ("replay", "kv_transfer")
    }
    model.write_text(json.dumps({
        "schema": campaign.network.PLAN_SCHEMA,
        "model_profile": {"sha256": campaign.profiler.file_hash(campaign.PROFILE)},
        "calibration": {"sha256": campaign.profiler.file_hash(calibration)},
        "network_contract": {"paths": {
            "east": {"natural_mbps": 1000, "migration_components": components},
            "germany": {"natural_mbps": 3000, "migration_components": components},
        }},
    }))
    return manifest, cluster, calibration, model


def make_plan(tmp_path, per_action=4):
    manifest, cluster, calibration, model = inputs(tmp_path)
    return campaign.make_timing_plan(
        manifest, cluster, calibration, campaign.PROFILE, model,
        tmp_path / "plan.json", scenarios_per_action=per_action)


def test_timing_plan_freezes_balanced_diverse_operational_predictions(tmp_path):
    plan = make_plan(tmp_path)
    rows = plan["scenarios"]

    assert Counter(row["parity_prediction"]["action"] for row in rows) == {
        "replay": 4, "kv_transfer": 4, "mixed": 4}
    assert len({round(row["parity_prediction"]["predicted_s"], 9)
                for row in rows}) >= 11
    assert all(4 <= len(row["moves"]) <= 16 for row in rows)
    assert all({move["destination_instance"] for move in row["moves"]}
               == {"east", "germany"} for row in rows)
    assert all(session["initial_tokens"] % 256 == 0
               for row in rows for session in row["sessions"])
    assert plan["parity"]["prediction"].startswith("sum fitted isolated times")


def test_timing_plan_rejects_binned_predictions(tmp_path):
    plan = make_plan(tmp_path)
    for row in plan["scenarios"]:
        row["parity_prediction"]["predicted_s"] = 1

    with pytest.raises(ValueError, match="tightly binned"):
        campaign.validate_timing_plan(plan)


def test_timing_reduction_requires_complete_exact_moves_and_reports_gates(
        tmp_path, monkeypatch):
    plan = make_plan(tmp_path, 2)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(json.dumps(plan))
    for scenario in plan["scenarios"]:
        predicted = scenario["parity_prediction"]["predicted_s"]
        requests = [{**move, "request": {"start_ns": 0,
                     "stream_chunks": [{"monotonic_ns": round(predicted * 1e9)}]}}
                    for move in scenario["moves"]]
        root = run_root / "scenarios" / scenario["scenario_id"] / "attempt-0001"
        root.mkdir(parents=True)
        (root / "result.json").write_text(json.dumps({
            "status": "complete", "request_failures": 0, "requests": requests}))
    monkeypatch.setattr(campaign, "live_measurements", lambda _scenario, result: [
        {"row": row, "request": row["request"]} for row in result["requests"]])

    rows, summary = campaign.timing_rows(run_root)

    assert len(rows) == 6
    assert summary["action_counts"] == {"replay": 2, "kv_transfer": 2, "mixed": 2}
    assert summary["mae_s"] < 1e-9
    assert summary["r2"] == pytest.approx(1)
    assert summary["passed"]


def test_power_plot_requires_direct_a100_campaign(tmp_path, monkeypatch):
    root = tmp_path / "power"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps({
        "hardware": "a100", "gpu": {
            "name": "NVIDIA A100 80GB PCIe", "power_limit_w": 300.0}}))
    monkeypatch.setattr(campaign, "load_power", lambda run, history: [run, history])
    written = []
    monkeypatch.setattr(campaign, "write_power", lambda rows, out: written.append((rows, out)))

    campaign.plot_power(root, [], tmp_path / "plot")

    assert written == [([root, []], tmp_path / "plot")]

    (root / "metadata.json").write_text(json.dumps({
        "hardware": "h100", "gpu": {
            "name": "NVIDIA H100 NVL", "power_limit_w": 400.0}}))
    with pytest.raises(RuntimeError, match="300 W A100"):
        campaign.plot_power(root, [], tmp_path / "plot")
