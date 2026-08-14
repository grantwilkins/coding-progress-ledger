import json
import math

import live_timing_campaign as timing
import pytest

H100_PROFILE = timing.network.ROOT / "profiles/gpt_oss_20b_h100_tp1.json"


def cluster(tmp_path):
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps({
        "schema": "queue-haul-azure-cluster-v1",
        "source": {"id": "source", "region": "westus3", "host": "10.0.0.1",
                   "ssh_user": "u", "repo_root": "/r", "run_root": "/d"},
        "destinations": [{
            "id": name, "region": region, "host": host, "ssh_user": "u",
            "repo_root": "/r", "run_root": "/d",
        } for name, region, host in (
            ("east", "australiaeast", "10.0.0.2"),
            ("germany", "southcentralus", "10.0.0.3"))],
    }))
    return path


def calibration(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "schema": timing.network.CALIBRATION_SCHEMA,
        "clock_uncertainty_ms": {"source": .1, "east": .1, "germany": .1},
        "aggregate_simultaneous_mbps": [3000],
        "paths": {name: {"rtt_ms": [10], "simultaneous_mbps": [rate]}
                  for name, rate in (("east", 1000), ("germany", 8000))},
    }))
    return path


def test_pilot_plan_is_live_holdout_across_both_real_regions(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sessions": [
        {"id": f"s{index}", "rank": index} for index in range(8)]}))
    plan = timing.make_plan(
        manifest, cluster(tmp_path), tmp_path / "plan.json", "pilot")

    timing.validate_plan(plan)
    assert len(plan["scenarios"]) == 8
    assert {row["region"] for row in plan["scenarios"]} == {
        "australiaeast", "southcentralus"}
    assert {row["context_tokens"] for row in plan["scenarios"]} == {
        8192, 31488}
    assert all(row["design"] == "timing_live"
               and row["split"] == "holdout" for row in plan["scenarios"])


def test_targeted_plan_separates_all_path_calibration_from_unseen_holdout(
        monkeypatch, tmp_path):
    monkeypatch.setattr(timing.network, "MODEL_PATH", H100_PROFILE)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sessions": [
        {"id": "s", "rank": 0}]}))

    plan = timing.make_plan(manifest, cluster(tmp_path), tmp_path / "plan.json",
                            "targeted", calibration(tmp_path))
    timing.validate_plan(plan)
    train = [row for row in plan["scenarios"]
             if row["split"] == "calibration"]
    holdout = [row for row in plan["scenarios"] if row["split"] == "holdout"]

    assert len(train) == 32 and len(holdout) == 16
    assert {(row["region"], row["method"]) for row in train} == {
        (region, method) for region in ("australiaeast", "southcentralus")
        for method in timing.METHODS}
    assert {row["context_tokens"] for row in train} == set(timing.CONTEXTS)
    assert {row["context_tokens"] for row in holdout} == set(timing.CONTEXTS)
    assert {(row["region"], row["method"]) for row in holdout} == {
        (region, method) for region in ("australiaeast", "southcentralus")
        for method in timing.METHODS}
    blocks = [[row for row in plan["scenarios"] if row["pair_id"] == pair]
              for pair in dict.fromkeys(row["pair_id"]
                                        for row in plan["scenarios"])]
    assert all(len({row["sessions"][0]["session_id"] for row in block}) == 1
               for block in blocks)
    assert len({block[0]["sessions"][0]["session_id"]
                for block in blocks}) == 12
    assert len({tuple((row["region"], row["method"]) for row in block)
                for block in blocks}) > 1


def test_service_load_uses_only_the_aligned_window(tmp_path):
    path = tmp_path / "load.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in (
        {"start_ns": 5_000_000_000, "prompt_tokens": 100,
         "output_tokens": 10},
        {"start_ns": 9_000_000_000, "prompt_tokens": 200,
         "output_tokens": 20},
    )))

    assert timing.service_load(path, 10_000_000_000, 4, 100, 10) == 1


def test_live_integrity_uses_get_bytes_and_rejects_cache_mismatch(monkeypatch):
    monkeypatch.setattr(timing.network, "MODEL_PATH", H100_PROFILE)
    scenario = {"sessions": [{"session_id": "s", "initial_tokens": 256}]}
    move = {"session_id": "s", "destination_instance": "east",
            "method": "kv_transfer", "request": {
                "start_ns": 10, "end_ns": 100, "status_code": 200,
                "state_code_verified": True, "cached_tokens": 256,
                "stream_chunks": [{"monotonic_ns": 70}]}}
    result = {"status": "complete", "requests": [move], "connections": [{
        "route": "api/east", "start_ns": "11",
        "client_first_byte_ns": "12", "client_last_byte_ns": "15",
        "target_first_byte_ns": "60", "target_last_byte_ns": "80"}],
        "resp_transfers": [{
            "command": "GET", "start_ns": "20", "end_ns": "40",
            "payload_bytes": "12582912", "response_wire_bytes": "12582925"}]}

    row = timing.live_measurements(scenario, result)[0]

    assert row["get_payload_bytes"] == 12582912
    assert row["get_window_s"] == 2e-8
    move["method"] = "replay"
    with pytest.raises(RuntimeError, match="cached KV"):
        timing.live_measurements(scenario, result)


def test_live_fit_uses_condition_holdouts_and_passes_exact_data(monkeypatch,
                                                               tmp_path):
    paths = [f"{region}:{method}" for region in (
        "australiaeast", "southcentralus") for method in timing.METHODS]
    rows = []
    for condition in range(40):
        for path in paths:
            row = {
                "scenario_id": str(condition), "condition_index": condition,
                "split": "prior_live", "destination": path.split(":")[0],
                "method": path.split(":")[1], "path": path,
                "context_tokens": timing.CONTEXTS[condition % 4],
                "width": 4, "same_path_width": 2,
                "destination_width": 4, "order_fraction": .5,
                "destination_load": (condition % 3) * .25,
                "initial_time_to_first_response_s": 1, "decode_tail_s": 1,
                "api_upload_s": None, "remote_response_start_s": None,
                "response_header_to_first_response_s": None,
                "response_stream_s": None, "client_residual_s": None,
                "kv_ingest_envelope_s": None,
            }
            row["observed_s"] = math.exp(sum(timing.features(row, paths)))
            rows.append(row)
    monkeypatch.setattr(timing, "collect", lambda _root: rows)

    model = timing.fit(tmp_path, tmp_path / "fit")

    assert model["gate"]["passed"]
    assert model["source"] == "measured_live_transfers"
    assert len(model["cross_validation"]) == 5


def test_targeted_refit_uses_only_calibration_split(monkeypatch, tmp_path):
    paths = [f"{region}:{method}" for region in (
        "australiaeast", "southcentralus") for method in timing.METHODS]
    profile_path = tmp_path / "profile.json"
    profile_path.write_bytes(H100_PROFILE.read_bytes())
    profile = json.loads(profile_path.read_text())
    run_root = tmp_path / "run"
    run_root.mkdir()
    calibration_path = calibration(tmp_path)
    raw_calibration = json.loads(calibration_path.read_text())
    raw_calibration["paths"]["east"]["simultaneous_mbps"] = [2000]
    calibration_path.write_text(json.dumps(raw_calibration))
    contract = timing.network.freeze_contract(json.loads(
        calibration_path.read_text()))
    (run_root / "plan.json").write_text(json.dumps({
        "git_sha": "12345678", "network_contract": contract,
        "cluster": {"destinations": [
            {"id": "east", "region": "australiaeast"},
            {"id": "germany", "region": "southcentralus"}]},
        "calibration": {
            "path": str(calibration_path),
            "sha256": timing.profiler.file_hash(calibration_path)},
        "model_profile": {"sha256": timing.profiler.file_hash(profile_path),
                          "profile_id": profile["profile_id"]}}))

    def row(path, context, repeat):
        method = path.split(":")[1]
        size = context * 49152 if method == "kv_transfer" else 0
        bandwidth = 2000 if path.startswith("australiaeast") else 8000
        effective = 1000 if path.startswith("australiaeast") else bandwidth
        observed = (max(size / (effective * 1e6 / 8), size / 6e8) + 1
                    if size else context / 5000)
        value = {"path": path, "context_tokens": context, "width": 1,
                 "same_path_width": 1, "destination_width": 1,
                 "destination_load": 0, "order_fraction": 0,
                 "method": method, "repeat": repeat,
                 "split": "holdout" if repeat == 2 else "calibration",
                 "measured_processed_tokens": context,
                 "measured_kv_bytes": size, "bandwidth_mbps": bandwidth}
        return {**value, "observed_s": observed}

    rows = [row(path, context, repeat) for path in paths
            for context in timing.CONTEXTS for repeat in range(3)]
    monkeypatch.setattr(timing, "collect", lambda _root: rows)

    report = timing.refit_targeted(
        profile_path, run_root, tmp_path / "out")

    assert report["passed"]
    assert report["overall"]["median_absolute_percentage_error"] < .01
    assert (tmp_path / "out/profile.json").exists()
    fitted = json.loads((tmp_path / "out/calibration.json").read_text())
    assert 900 < fitted["paths"]["east"]["simultaneous_mbps"][0] < 1100
    assert fitted["paths"]["germany"]["simultaneous_mbps"] == [8000]
    assert fitted["aggregate_simultaneous_mbps"] == [3000]
