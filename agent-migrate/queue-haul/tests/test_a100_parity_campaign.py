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


def test_timing_reduction_requires_complete_exact_moves_and_reports_gates(tmp_path):
    plan = make_plan(tmp_path, 2)
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "plan.json").write_text(json.dumps(plan))
    for scenario in plan["scenarios"]:
        predicted = scenario["parity_prediction"]["predicted_s"]
        contexts = {row["session_id"]: row["initial_tokens"]
                    for row in scenario["sessions"]}
        end = round(predicted * 1e9) + 1
        requests = [{**move, "request": {"start_ns": 0, "end_ns": end,
                     "status_code": 200, "state_code_verified": True,
                     "prompt_tokens": contexts[move["session_id"]] + 128,
                     "output_tokens": 32, "probe_max_tokens": 512,
                     "cached_tokens": contexts[move["session_id"]]
                     if move["method"] == "kv_transfer" else 0,
                     "stream_chunks": [{"monotonic_ns": end - 1}]}}
                    for move in scenario["moves"]]
        root = run_root / "scenarios" / scenario["scenario_id"] / "attempt-0001"
        root.mkdir(parents=True)
        (root / "result.json").write_text(json.dumps({
            "status": "complete", "started_ns": 0, "ended_ns": end,
            "request_failures": 0, "requests": requests}))

    rows, summary = campaign.timing_rows(run_root)

    assert len(rows) == 6
    assert summary["action_counts"] == {"replay": 2, "kv_transfer": 2, "mixed": 2}
    assert summary["mae_s"] < 1e-9
    assert summary["r2"] == pytest.approx(1)
    assert summary["passed"]
    assert summary["probe_max_tokens"] == 512

    result_path = root / "result.json"
    result = json.loads(result_path.read_text())
    result["requests"][0]["request"]["probe_max_tokens"] = 128
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="mixes state-probe"):
        campaign.timing_rows(run_root)


@pytest.fixture
def queue_result():
    moves = [{"session_id": str(i), "destination_instance": destination,
              "method": method, "order": i}
             for i, (destination, method) in enumerate(
                 [("east", "kv_transfer"), ("germany", "replay")])]
    scenario = {"moves": moves, "sessions": [
        {"session_id": str(i), "initial_tokens": 256} for i in range(2)]}
    result = {"status": "complete", "started_ns": 100, "ended_ns": 1000,
              "requests": [{**move, "request": {
                  "status_code": 200, "state_code_verified": True,
                  "start_ns": start, "end_ns": 900, "prompt_tokens": 300,
                  "output_tokens": 32, "cached_tokens": cached,
                  "probe_max_tokens": 512,
                  "stream_chunks": [{"monotonic_ns": first}]}}
                  for move, start, first, cached in zip(
                      moves, [110, 400], [300, 800], [256, 0])]}
    return scenario, result


def test_queue_makespan_includes_dispatch_delay(queue_result):
    assert campaign.queue_makespan(*queue_result) == pytest.approx(700e-9)


@pytest.mark.parametrize("field,value", [
    ("status_code", 500), ("state_code_verified", False),
    ("cached_tokens", 128), ("prompt_tokens", 128), ("output_tokens", 0),
    ("output_tokens", 513), ("start_ns", 99), ("end_ns", 299),
    ("stream_chunks", []),
])
def test_queue_makespan_rejects_invalid_request_evidence(queue_result, field, value):
    scenario, result = queue_result
    result["requests"][0]["request"][field] = value
    with pytest.raises(RuntimeError):
        campaign.queue_makespan(scenario, result)


def test_queue_makespan_rejects_cached_replay_and_changed_moves(queue_result):
    scenario, result = queue_result
    result["requests"][1]["request"]["cached_tokens"] = 256
    with pytest.raises(RuntimeError):
        campaign.queue_makespan(scenario, result)
    result["requests"][1]["request"]["cached_tokens"] = 0
    result["requests"].reverse()
    with pytest.raises(RuntimeError):
        campaign.queue_makespan(scenario, result)


def test_queue_makespan_accepts_archived_concurrent_kv_requests():
    root = campaign.ROOT / (
        "outputs/a100-parity-20260905/timing-v022/scenarios/"
        "5e19c10f47fa0204/attempt-0001")
    scenario = json.loads((root / "scenario.json").read_text())
    result = json.loads((root / "result.json").read_text())
    assert len(result["connections"]) > len(result["requests"])
    expected = (max(campaign.profiler.first_stream_ns(row["request"])
                    for row in result["requests"]) - result["started_ns"]) / 1e9
    assert campaign.queue_makespan(scenario, result) == expected


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


def test_timing_only_requires_matching_protocol_and_preserves_cache_checks(queue_result):
    scenario, result = queue_result
    for row in result['requests']:
        row['request'].update(timing_only=True, state_code_verified=False)
    assert campaign.queue_makespan(scenario, result, True) == pytest.approx(700e-9)
    with pytest.raises(RuntimeError):
        campaign.queue_makespan(scenario, result)
    result['requests'][0]['request']['cached_tokens'] = 0
    with pytest.raises(RuntimeError):
        campaign.queue_makespan(scenario, result, True)


def test_replay_allows_only_verified_framing_prefix(queue_result):
    scenario, result = queue_result
    request = result['requests'][1]['request']
    request.update(prompt_tokens=384, cached_tokens=64)
    assert campaign.queue_makespan(scenario, result, replay_prefix_tokens=64) > 0
    with pytest.raises(RuntimeError):
        campaign.queue_makespan(scenario, result)
    for cached in (65, 80, 256):
        request['cached_tokens'] = cached
        with pytest.raises(RuntimeError):
            campaign.queue_makespan(scenario, result, replay_prefix_tokens=64)
    request.update(prompt_tokens=300, cached_tokens=64)
    with pytest.raises(RuntimeError):
        campaign.queue_makespan(scenario, result, replay_prefix_tokens=64)


@pytest.fixture
def archived_timing(monkeypatch):
    validate = campaign.validate_timing_plan
    monkeypatch.setattr(campaign, 'validate_timing_plan',
                        lambda plan: validate(plan, check_files=False))
    return campaign.ROOT / 'outputs/a100-parity-20260907/timing'


def test_archived_prospective_scale_retains_balanced_unseen_validation(archived_timing):
    rows, summary = campaign.timing_rows(archived_timing, prospective_scale=True)
    assert len(rows) == 24
    assert summary['action_counts'] == dict.fromkeys(campaign.ACTIONS, 8)
    assert summary['prediction'] == 'prospective_scale'
    assert summary['replay_prefix_tokens'] == 64
    assert summary['passed']


@pytest.mark.parametrize('change', ['chronology', 'scale', 'prediction'])
def test_prospective_scale_rejects_leakage_and_changed_predictions(
        archived_timing, monkeypatch, change):
    fit_path = archived_timing / 'scale-fit.json'
    fit = json.loads(fit_path.read_text())
    if change == 'chronology':
        fit['frozen_wall_ns'] = 10**30
    elif change == 'scale':
        fit['scales']['replay'] *= 1.01
    else:
        fit['predictions'][0]['predicted_s'] *= 1.01
    original = type(fit_path).read_text
    monkeypatch.setattr(type(fit_path), 'read_text',
                        lambda path, *args, **kwargs: json.dumps(fit)
                        if path == fit_path else original(path, *args, **kwargs))
    with pytest.raises(RuntimeError):
        campaign.timing_rows(archived_timing, prospective_scale=True)
