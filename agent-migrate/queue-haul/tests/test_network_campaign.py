"""
Claim:
The Azure campaign freezes simultaneous source-egress capacity at the route and
source-NIC levels, rejects unsafe clock or allocation drift, and builds the
agreed seven-cell matched policy design without a Cartesian explosion.

Plausible wrong implementations:
- Derive controlled rates from isolated-path rather than simultaneous goodput.
- Apply 40/80 as percentages twice or forget the aggregate source cap.
- Accept a clock just outside the formal 2 ms bound.
- Compare resumed calibration in only one direction or silently change caps.
- Recreate the 648-run factorial instead of the seven targeted conditions.
"""

import csv
import json
from pathlib import Path

import pytest

import network_campaign as n
import migration_testbed as testbed


def cluster(tmp_path):
    path = tmp_path / "cluster.json"
    path.write_text(json.dumps({
        "schema": n.CLUSTER_SCHEMA,
        "source": {
            "id": "sweden", "region": "swedencentral",
            "host": "10.0.0.4", "ssh_user": "azureuser",
            "repo_root": "/home/azureuser/coding-progress-ledger/agent-migrate",
            "run_root": "/datadrive/queue-haul-network",
        },
        "destinations": [
            {"id": "east", "region": "eastus2", "host": "10.1.0.4",
             "ssh_user": "azureuser",
             "repo_root": "/home/azureuser/coding-progress-ledger/agent-migrate",
             "run_root": "/datadrive/queue-haul-network"},
            {"id": "west", "region": "westeurope", "host": "10.2.0.4",
             "ssh_user": "azureuser",
             "repo_root": "/home/azureuser/coding-progress-ledger/agent-migrate",
             "run_root": "/datadrive/queue-haul-network"},
        ],
    }))
    return n.Cluster.load(path)


def calibration():
    return {
        "schema": n.CALIBRATION_SCHEMA,
        "clock_uncertainty_ms": {"sweden": .2, "east": 1.5, "west": 2.0},
        "paths": {
            "east": {
                "rtt_ms": [80, 100, 90],
                "isolated_mbps": [18_000, 17_000, 19_000],
                "simultaneous_mbps": [7_550, 7_450, 7_500],
            },
            "west": {
                "rtt_ms": [30, 40, 35],
                "isolated_mbps": [19_000, 18_000, 18_500],
                "simultaneous_mbps": [9_050, 8_950, 9_000],
            },
        },
        "aggregate_simultaneous_mbps": [16_600, 16_400, 16_500],
    }


def test_cluster_pins_actual_roles_and_rejects_ambiguous_hosts(tmp_path):
    value = cluster(tmp_path)
    assert (value.source.region, value.source.host) == (
        "swedencentral", "10.0.0.4")
    assert {(node.region, node.host) for node in value.destinations} == {
        ("eastus2", "10.1.0.4"), ("westeurope", "10.2.0.4")}

    raw = value.as_dict()
    raw["destinations"][0]["host"] = "10.0.0.4"
    with pytest.raises(ValueError, match="unique"):
        n.Cluster.parse(raw)

    raw = value.as_dict()
    raw["destinations"] = raw["destinations"][:1]
    assert [node.id for node in n.Cluster.parse(raw).destinations] == ["east"]


def test_contract_uses_simultaneous_route_and_aggregate_goodput():
    contract = n.freeze_contract(calibration())

    assert contract["paths"]["east"] == {
        "rtt_ms": 90.0, "natural_mbps": 7500.0,
        "controlled_mbps": {"40": 3000, "80": 6000},
    }
    assert contract["paths"]["west"]["natural_mbps"] == 9000
    assert contract["aggregate"] == {
        "natural_mbps": 16500.0,
        "controlled_mbps": {"40": 6600, "80": 13200},
    }


def test_clock_and_resume_drift_are_hard_boundaries():
    n.validate_calibration(calibration())
    bad = calibration()
    bad["clock_uncertainty_ms"]["west"] = 2.001
    with pytest.raises(ValueError, match="clock"):
        n.validate_calibration(bad)

    original = freeze = n.freeze_contract(calibration())
    within = json.loads(json.dumps(freeze))
    within["paths"]["east"]["natural_mbps"] *= .9
    n.validate_resume(original, within)
    outside = json.loads(json.dumps(freeze))
    outside["paths"]["east"]["natural_mbps"] *= .899
    with pytest.raises(ValueError, match="drift"):
        n.validate_resume(original, outside)


def test_targeted_design_has_seven_cells_and_126_policy_migrations():
    cells = n.target_conditions()
    assert len(cells) == 7
    assert len({tuple(sorted(cell.items())) for cell in cells}) == 7
    assert {cell["workload"] for cell in cells} == {
        "interactive_coding", "coding", "agentic_tool_loop"}
    assert {cell["bandwidth"] for cell in cells} == {
        "natural", "controlled_40", "controlled_80"}
    assert {cell["sink_load"] for cell in cells} == {"idle", "rho_0.8"}
    assert {cell["deadline_s"] for cell in cells} == {19, 30}
    assert n.POLICIES[-1] == "random"
    assert len(cells) * n.REPEATS * len(n.POLICIES) == 126


def test_hierarchical_limiter_enforces_route_and_source_caps():
    limiter = testbed.BandwidthLimiter(100, {"east": 60, "west": 60})
    for bucket in (limiter.aggregate, *limiter.routes.values()):
        bucket.updated = 0

    assert limiter.reserve("kv/east", "target_to_client", 60, 0) == 1
    assert limiter.reserve("kv/west", "target_to_client", 60, 0) == 1.2
    assert limiter.reserve("kv/east", "client_to_target", 10_000, 0) == 0


def test_linux_tcp_info_parser_uses_microseconds_and_total_retransmissions():
    blob = bytearray(104)
    blob[68:72] = (12_345).to_bytes(4, "little")
    blob[72:76] = (2_000).to_bytes(4, "little")
    blob[80:84] = (17).to_bytes(4, "little")
    blob[100:104] = (9).to_bytes(4, "little")

    assert testbed.parse_tcp_info(bytes(blob)) == {
        "rtt_us": 12_345, "rttvar_us": 2_000,
        "snd_cwnd": 17, "total_retrans": 9,
    }


def test_chrony_uncertainty_includes_offset_and_dispersion():
    tracking = """
Leap status     : Normal
Last offset     : -0.000400000 seconds
Root dispersion: 0.000700000 seconds
"""
    assert n.chrony_uncertainty_ms(tracking) == pytest.approx(1.1)
    with pytest.raises(ValueError, match="Leap status"):
        n.chrony_uncertainty_ms(tracking.replace("Normal", "Not synchronised"))


def test_iperf_uses_receiver_goodput_and_rejects_partial_runs():
    raw = {
        "error": "",
        "end": {
            "sum_sent": {"bits_per_second": 9_000_000_000},
            "sum_received": {"bits_per_second": 8_000_000_000},
        },
    }
    assert n.iperf_mbps(raw) == 8000
    raw["error"] = "unable to send control message"
    with pytest.raises(RuntimeError, match="iperf3"):
        n.iperf_mbps(raw)


def test_iperf_waits_for_remote_listener(monkeypatch, tmp_path):
    node = cluster(tmp_path).destinations[0]
    calls = []
    monkeypatch.setattr(n.subprocess, "run", lambda command, check: calls.append(
        (command, check)))

    n._wait_iperf_server(node, Path("key"), 5201)

    assert calls[0][1] is True
    assert "until ss -ltnH" in calls[0][0][-1]


def test_host_reports_must_match_commit_runtime_and_expected_regions(tmp_path):
    base = {
        "git_sha": "abc", "dirty": False, "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_memory_mib": 81920, "vllm": "0.22.0", "lmcache": "0.5.1",
        "vm_size": "Standard_NC24ads_A100_v4", "clock_uncertainty_ms": 1,
        "ptp": "/dev/ptp0", "datadrive": True,
    }
    reports = {
        "sweden": {**base, "region": "SwedenCentral", "private_ip": "10.0.0.4"},
        "east": {**base, "region": "EastUS2", "private_ip": "10.1.0.4"},
        "west": {**base, "region": "WestEurope", "private_ip": "10.2.0.4"},
    }
    n.validate_hosts(cluster(tmp_path), reports)

    reports["west"]["vllm"] = "0.22.1"
    with pytest.raises(ValueError, match="runtime"):
        n.validate_hosts(cluster(tmp_path), reports)


def test_remote_sink_uses_gpu_zero_private_api_and_source_l2(monkeypatch):
    monkeypatch.setenv("QH_RUNTIME", "native")
    monkeypatch.setenv("QH_LMCACHE_MODE", "mp")
    cfg = testbed.Config()

    cache = testbed.shell(testbed.mp_server_cmd(
        cfg, "sink", bind_host="127.0.0.1", http_host="10.1.0.4",
        l2_host="10.0.0.4", l2_port=8301,
    ))
    vllm = testbed.shell(testbed.vllm_cmd(
        cfg, "sink", gpu_index=0, bind_host="10.1.0.4"))
    source = testbed.shell(testbed.vllm_cmd(cfg, "source", sleep_mode=True))

    assert '"host":"10.0.0.4","port":8301' in cache
    assert '"lmcache.mp.port":5557' in source
    assert "--http-host 10.1.0.4" in cache
    assert "CUDA_VISIBLE_DEVICES=0" in vllm
    assert "--host 10.1.0.4" in vllm
    assert "--enable-sleep-mode" in source


def test_network_proxy_cli_preserves_named_routes_and_caps():
    routes = [
        testbed.Route("kv/east", "10.0.0.4", 8301, "127.0.0.1", 5655, "resp"),
        testbed.Route("api/east", "127.0.0.1", 8401, "10.1.0.4", 8200),
    ]
    args = testbed.parse_args([
        "proxy", "--routes-json", json.dumps([n.asdict(route) for route in routes]),
        "--aggregate-mbps", "6000", "--route-mbps-json",
        json.dumps({"kv/east": 3000, "api/east": 3000}),
    ])
    parsed, aggregate, rates = testbed.proxy_config(args)

    assert parsed == routes
    assert aggregate == 750_000_000
    assert rates == {"kv/east": 375_000_000, "api/east": 375_000_000}

    args.route_mbps_json = json.dumps({"east": 3000})
    assert testbed.proxy_config(args)[2] == {"east": 375_000_000}


def test_network_smoke_prompt_fits_model_context():
    assert n.parse_args([
        "smoke", "--cluster", "cluster.json", "--calibration", "c.json",
        "--run-root", "run",
    ]).words == 1024


def test_cluster_routes_keep_data_private_and_share_destination_caps(tmp_path):
    value = cluster(tmp_path)
    routes, ports = n.cluster_routes(value)

    assert routes == [
        testbed.Route("kv/east", "10.0.0.4", 8301, "127.0.0.1", 5655, "resp"),
        testbed.Route("api/east", "127.0.0.1", 8401, "10.1.0.4", 8200),
        testbed.Route("kv/west", "10.0.0.4", 8302, "127.0.0.1", 5655, "resp"),
        testbed.Route("api/west", "127.0.0.1", 8402, "10.2.0.4", 8200),
    ]
    assert ports == {"east": {"kv": 8301, "api": 8401},
                     "west": {"kv": 8302, "api": 8402}}

    contract = n.freeze_contract(calibration())
    aggregate, rates = n.bandwidth_limits(contract, "controlled_40")
    assert aggregate == 6600
    assert rates == {"east": 3000, "west": 3600}
    assert n.bandwidth_limits(contract, "natural") == (None, {})


def test_network_plan_is_matched_balanced_and_exactly_126(monkeypatch, tmp_path):
    manifest = {
        "schema": "queue-haul-migration-manifest-v2",
        "source": {"path": "trace.json", "sha256": "0" * 64},
        "seed": 1, "workload": "coding",
        "classification": {"turn_rate_q25_hz": .1,
                           "turn_rate_q75_hz": 1},
        "message_generator": "deterministic_trace_tokens_v2",
        "sessions": [{
            "id": f"s{i}", "job_class": "coding", "state_code": f"C{i}",
            "rank": i, "turn_rate_hz": 1, "human_fraction": 0,
            "tool_fraction": 0, "turns": [{"time_s": 0,
                "input_tokens": 4096, "append_tokens": 0,
                "output_tokens": 4, "reset": False}],
        } for i in range(12)],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(n, "WORKLOAD_PATHS", {
        name: Path(__file__).parents[1] / "profiles" / f"{name}.json"
        for name in ("coding", "interactive_coding", "agentic_tool_loop")
    })

    plan = n.make_plan(path, n.freeze_contract(calibration()), seed=7)

    assert len(plan["scenarios"]) == 126
    assert {row["policy"] for row in plan["scenarios"]} == set(n.POLICIES)
    assert all({row["destination"] for row in plan["scenarios"]
                if row["policy"] == policy} == {"east", "west"}
               for policy in n.POLICIES)
    for condition in range(7):
        for repeat in range(3):
            rows = [row for row in plan["scenarios"]
                    if (row["condition_index"], row["repeat"])
                    == (condition, repeat)]
            signatures = {tuple((item["session_id"], item["initial_tokens"])
                                for item in row["sessions"]) for row in rows}
            assert len(rows) == 6 and len(signatures) == 1
            assert all(row["bandwidth_mbps"] ==
                       plan["network_contract"]["paths"]
                       [row["destination"]]["controlled_mbps"]["80"]
                       for row in rows if row["bandwidth"] == "controlled_80")

    contract = n.freeze_contract(calibration())
    contract["paths"] = {"east": contract["paths"]["east"]}
    contract["aggregate"] = {
        "natural_mbps": contract["paths"]["east"]["natural_mbps"],
        "controlled_mbps": contract["paths"]["east"]["controlled_mbps"],
    }
    east = n.make_plan(path, contract, seed=7)
    assert {row["destination"] for row in east["scenarios"]} == {"east"}


def test_scheduled_events_reject_active_spot_notice():
    assert n.active_scheduled_events({"Events": []}) == []
    event = {"EventId": "e", "EventType": "Preempt", "Resources": ["vm"]}
    assert n.active_scheduled_events({"Events": [event]}) == [event]
    with pytest.raises(ValueError, match="Scheduled Events"):
        n.active_scheduled_events({})


def test_scheduled_event_monitor_allows_azure_first_call_delay(tmp_path):
    monitor = n.ScheduledEventMonitor(tmp_path / "events.jsonl")
    timeouts = []
    monitor._get = lambda timeout: timeouts.append(timeout) or {
        "DocumentIncarnation": "1", "Events": []}
    monitor.start()
    while len(timeouts) < 2:
        n.time.sleep(.01)
    monitor.close()

    assert timeouts[:2] == [125, 5]


def test_remote_stop_signals_recorded_node_serve_pid(monkeypatch, tmp_path):
    node = cluster(tmp_path).destinations[0]
    calls = []
    process = type("Process", (), {
        "poll": lambda self: None,
        "wait": lambda self, timeout: calls.append(("wait", timeout)),
    })()
    monkeypatch.setattr(n.subprocess, "run", lambda command, check: calls.append(
        (command, check)))

    n._stop_remote(node, Path("key"), Path("run/east"), process)

    assert "node-serve.pid" in calls[0][0][-1]
    assert calls[-1] == ("wait", 30)


def test_reducer_keeps_failed_attempts_and_uses_latest(tmp_path):
    scenario = {
        "scenario_id": "s", "condition_index": 0, "repeat": 0,
        "policy": "queue_haul", "destination": "west",
        "workload": "coding", "bandwidth": "natural",
        "sink_load": "idle", "deadline_s": 30,
    }
    root = tmp_path / "run"
    first = root / "scenarios/s/attempt-0001"
    second = root / "scenarios/s/attempt-0002"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "result.json").write_text(json.dumps({
        "status": "failed", "error": "Spot eviction"}))
    (second / "result.json").write_text(json.dumps({
        "status": "complete", "migration_s": 2, "deadline_met": True,
        "wire_bytes": {}, "connections": []}))

    summary = n.reduce_run({"scenarios": [scenario]}, root)

    assert summary == {"schema": "queue-haul-network-summary-v1",
                       "expected": 1, "completed": 1, "failed": 0,
                       "missing": 0, "valid": True}
    assert (root / "artifacts.sha256").is_file()
    assert (first / "result.json").is_file()
    with (root / "results.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert (row["scenario_id"], row["attempt"]) == ("s", "2")


def test_resume_skips_complete_and_advances_past_interrupted_attempt(tmp_path):
    root = tmp_path / "scenarios/s"
    interrupted = root / "attempt-0002"
    interrupted.mkdir(parents=True)
    first = root / "attempt-0001"
    first.mkdir()
    n.write_checkpoint(first / "result.json", {"status": "complete"})

    assert n._latest_result(root)[0] == 1
    assert n._next_attempt(root) == 3


def test_progress_checkpoint_is_atomic_and_counts_latest_results(tmp_path):
    plan = {"scenarios": [{"scenario_id": "done"}, {"scenario_id": "todo"}]}
    result = tmp_path / "scenarios/done/attempt-0001/result.json"
    n.write_checkpoint(result, {"status": "complete"})

    progress = n.checkpoint_progress(plan, tmp_path)

    assert progress["completed_scenario_ids"] == ["done"]
    assert (progress["completed"], progress["missing"]) == (1, 1)
    assert json.loads((tmp_path / "progress.json").read_text()) == progress
    assert not (tmp_path / "progress.json.tmp").exists()


def test_chat_explicitly_probes_state_code(monkeypatch):
    seen = {}

    def stream(_cfg, _port, messages, tokens, _hash, _timeout):
        seen.update(messages=messages, tokens=tokens)
        now = n.time.monotonic_ns()
        return n.profiler.RequestResult("r", 200, "", now, now), "CODE"

    monkeypatch.setattr(n.profiler, "stream_chat", stream)
    n._chat(object(), 1, [{"role": "user", "content": "context"}],
            "CODE", 1)

    assert seen == {"messages": [
        {"role": "user", "content": "context"},
        {"role": "user", "content":
         "Reply only with session state code CODE."}], "tokens": 32}


def test_resume_metadata_appends_dynamic_check_but_pins_identity():
    first = {"git_sha": "abc", "plan_sha256": "p", "checks": [{"clock": 1}]}
    current = {"git_sha": "abc", "plan_sha256": "p", "checks": [{"clock": 2}]}

    assert n.merge_metadata(current, first)["checks"] == [
        {"clock": 1}, {"clock": 2}]
    current["git_sha"] = "changed"
    with pytest.raises(RuntimeError, match="metadata changed"):
        n.merge_metadata(current, first)
