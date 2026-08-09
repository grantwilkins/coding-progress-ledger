"""
Claim:
The Azure campaign freezes simultaneous source-egress capacity at the route and
source-NIC levels, rejects unsafe clock or allocation drift, and builds the
agreed seven-cell matched policy design without a Cartesian explosion. The
constraint design freezes four measured-path operating points, exact recorded
context packs, six matched policies, and one method-specific replay quota.
The separation design uses measured load/bandwidth support to require both
actions and separates two joint planners from five baselines by at least 10%.

Plausible wrong implementations:
- Derive controlled rates from isolated-path rather than simultaneous goodput.
- Apply 40/80 as percentages twice or forget the aggregate source cap.
- Accept a clock just outside the formal 2 ms bound.
- Compare resumed calibration in only one direction or silently change caps.
- Recreate the 648-run factorial instead of the seven targeted conditions.
- Round the recorded constraint contexts or fail to reuse the quota pack.
- Apply the Germany replay quota to East, KV transfer, or the WAN route.
- Accept completed constraint evidence with a missed target or load warning.
- Normalize destination load with source power throughput instead of destination
  prefill/decode service, or serialize migrations that should run in parallel.
- Admit a separation cell whose winner or loser is within 10% of the target.
- Let a restricted baseline use both actions or let deadline-blind planning use
  the physical deadline.
- Call a greedy restriction an oracle, evaluate a frozen plan against its
  assumed state, or credit eventual nonlinear bundle shed at the deadline.
- Treat a capacity-invalid stale plan as a slow plan, or call the worst-corner
  plan robust without checking every monotone constraint release.
- Credit a queued request by its own latency instead of the shared campaign
  deadline, use the last request as an all-or-nothing deadline, or mix seconds
  and nanoseconds at the exact boundary.
"""

import csv
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import network_campaign as n
import migration_testbed as testbed


def test_campaign_scopes_300w_profile_to_azure_nodes():
    generic = n.ModelProfile.load(n.MODEL_PATH.with_name("gpt_oss_20b_a100_tp1.json"))
    azure = n.ModelProfile.load(n.MODEL_PATH)

    assert n.MODEL_PATH.name == "gpt_oss_20b_a100_tp1_azure_300w.json"
    assert generic.max_ell == pytest.approx(.531357714017)
    assert azure.max_ell == pytest.approx(10.054248624043707)
    assert azure.hardware == "NVIDIA A100 80GB PCIe (Azure 300 W)"


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


def campaign_manifest(tmp_path, count=12):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": "queue-haul-migration-manifest-v2",
        "source": {"path": "trace.json", "sha256": "0" * 64},
        "seed": 1, "workload": "coding", "classification": {},
        "message_generator": "deterministic_trace_tokens_v2",
        "sessions": [{
            "id": f"s{i}", "job_class": "coding", "state_code": f"C{i}",
            "rank": i, "turn_rate_hz": 1, "human_fraction": 0,
            "tool_fraction": 0, "turns": [{"time_s": 0,
                "input_tokens": 4096, "append_tokens": 0,
                "output_tokens": 4, "reset": False}],
        } for i in range(count)],
    }))
    return path


def constraint_contract():
    return json.loads((n.ROOT / "outputs/east-germany-frontier-20260808/control/"
                       "frontier-pilot-002.json").read_text())["network_contract"]


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

    germany = n.Cluster.load(n.ROOT / "azure_network_cluster_germany.json")
    assert [(node.id, node.region, node.host) for node in germany.destinations] == [
        ("germany", "germanywestcentral", "10.3.0.4")]

    east_germany = n.Cluster.load(
        n.ROOT / "azure_network_cluster_east_germany.json")
    assert [(node.id, node.region, node.host)
            for node in east_germany.destinations] == [
        ("east", "eastus2", "10.1.0.4"),
        ("germany", "germanywestcentral", "10.3.0.4")]


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
    assert len({json.dumps(cell, sort_keys=True) for cell in cells}) == 7
    assert {cell["workload"] for cell in cells} == {"agentic_tool_loop"}
    assert {cell["bandwidth"] for cell in cells} == {
        "natural", "controlled_40", "controlled_80"}
    assert {tuple(sorted(cell["background"].items())) for cell in cells} == {
        (("east", (0, 0)), ("west", (0, 0))),
        (("east", (.2, .2)), ("west", (.2, .2))),
        (("east", (.2, .4)), ("west", (.4, .2))),
        (("east", (.4, .2)), ("west", (.2, .4))),
    }
    assert {cell["deadline_s"] for cell in cells} == {19, 30}
    assert n.POLICIES[-1] == "random"
    assert len(cells) * n.REPEATS * len(n.POLICIES) == 126
    assert n.REQUEST_TIMEOUT_S > max(cell["deadline_s"] for cell in cells)
    assert {node for cell in n.target_conditions(("east", "germany"))
            for node in cell["background"]} == {"east", "germany"}


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

    handoff = n.parse_args([
        "handoff", "--cluster", "cluster.json", "--calibration", "c.json",
        "--plan", "p.json", "--manifest", "m.json", "--run-root", "run",
        "--policy", "kv_only", "--repeat", "2",
    ])
    assert (handoff.policy, handoff.repeat) == ("kv_only", 2)


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
    path = campaign_manifest(tmp_path)
    monkeypatch.setattr(n, "WORKLOAD_PATHS", {
        name: Path(__file__).parents[1] / "profiles" / f"{name}.json"
        for name in ("coding", "interactive_coding", "agentic_tool_loop")
    })

    plan = n.make_plan(path, n.freeze_contract(calibration()), seed=7)

    assert len(plan["scenarios"]) == 126
    assert plan["design"] == "joint"
    assert {row["policy"] for row in plan["scenarios"]} == set(n.POLICIES)
    assert all("destination" not in row and "moves" not in row
               for row in plan["scenarios"])
    for condition in range(7):
        for repeat in range(3):
            rows = [row for row in plan["scenarios"]
                    if (row["condition_index"], row["repeat"])
                    == (condition, repeat)]
            signatures = {tuple((item["session_id"], item["initial_tokens"])
                                for item in row["sessions"]) for row in rows}
            assert len(rows) == 6 and len(signatures) == 1
            assert all(row["bandwidth_mbps"] == {
                node: path["controlled_mbps"]["80"]
                for node, path in plan["network_contract"]["paths"].items()
            } for row in rows if row["bandwidth"] == "controlled_80")

    contract = n.freeze_contract(calibration())
    contract["paths"] = {"east": contract["paths"]["east"]}
    contract["aggregate"] = {
        "natural_mbps": contract["paths"]["east"]["natural_mbps"],
        "controlled_mbps": contract["paths"]["east"]["controlled_mbps"],
    }
    with pytest.raises(ValueError, match="two destinations"):
        n.make_plan(path, contract, seed=7)


def test_frontier_plan_is_the_matched_185_episode_natural_bandwidth_pilot(tmp_path):
    plan = n.make_plan(
        campaign_manifest(tmp_path, 4), n.freeze_contract(calibration()),
        seed=7, design="frontier")

    assert len(plan["scenarios"]) == 185
    assert plan["policies"] == list(n.FRONTIER_POLICIES)
    assert {row["bandwidth"] for row in plan["scenarios"]} == {"natural"}
    assert {row["deadline_s"] for row in plan["scenarios"]} == {30}
    assert {row["source_load"] for row in plan["scenarios"]} == {.8}
    assert {row["requested_shed_fraction"] for row in plan["scenarios"]} == {.8}
    blocks = {}
    for row in plan["scenarios"]:
        blocks.setdefault(row["condition_index"], []).append(row)
    assert len(blocks) == 37
    assert all(len(rows) == 5 for rows in blocks.values())
    assert {len(rows[0]["sessions"]) for rows in blocks.values()} == {4, 8, 16}
    canonical = [rows[0] for rows in blocks.values() if rows[0]["pack"] == "8x16k"]
    pairs = {tuple(value[0] for value in row["background"].values())
             for row in canonical}
    asymmetric = (.5, .85, .9, .95)
    assert pairs == {(load, load) for load in n.FRONTIER_LOADS} | {
        (.5, load) for load in asymmetric
    } | {(load, .5) for load in asymmetric}


def test_constraint_plan_freezes_four_single_block_stress_cases(tmp_path):
    plan = n.make_plan(
        campaign_manifest(tmp_path, 8), constraint_contract(), seed=7,
        design="constraint")

    assert len(plan["scenarios"]) == 24
    assert plan["policies"] == list(n.CONSTRAINT_POLICIES)
    assert [row["condition_index"] for row in plan["scenarios"]] == [
        index for index in range(4) for _ in n.CONSTRAINT_POLICIES]
    expected = {
        "window-19": (19, 22, 513_650, 1.0),
        "window-30": (30, 28, 648_131, 1.0),
        "window-60": (60, 64, 898_688, 1.0),
        "quota-30": (30, 28, 648_131, 1.0),
    }
    signatures = {}
    for condition, (deadline, count, tokens, target) in expected.items():
        rows = [row for row in plan["scenarios"]
                if row["condition_id"] == condition]
        signatures[condition] = {tuple(
            (item["template_id"], item["initial_tokens"])
            for item in row["sessions"]
        ) for row in rows}
        assert len(rows) == 6 and len(signatures[condition]) == 1
        assert {row["policy"] for row in rows} == set(n.CONSTRAINT_POLICIES)
        assert all(row["scenario_id"] == n._hash([
            "constraint", row["condition_index"], row["policy"],
            row["sessions"], row["migration_headroom"], deadline, target,
            "max_shed",
        ])[:16] for row in rows)
        assert all(row["deadline_s"] == deadline
                   and len(row["sessions"]) == count
                   and sum(item["initial_tokens"] for item in row["sessions"])
                   == tokens
                   and row["objective"] == "max_shed"
                   and row["requested_shed_fraction"] == target for row in rows)
    assert signatures["window-30"] == signatures["quota-30"]
    assert all(row["migration_headroom"] == {"germany": {"replay": .25}}
               for row in plan["scenarios"] if row["condition_id"] == "quota-30")
    assert all(row["migration_headroom"] == {} for row in plan["scenarios"]
               if row["condition_id"] != "quota-30")
    assert n.parse_args([
        "prepare", "--cluster", "c", "--calibration", "k",
        "--manifest", "m", "--out", "o", "--design", "constraint",
    ]).design == "constraint"
    changed = json.loads(json.dumps(plan))
    changed["scenarios"][0]["sessions"][0]["initial_tokens"] += 1
    changed["scenarios"][0]["sessions"][1]["initial_tokens"] -= 1
    with pytest.raises(ValueError, match="policy block"):
        n.validate_plan(changed)


def test_constraint_simulation_separates_joint_policies_and_exports_duals(tmp_path):
    plan = n.make_plan(
        n.ROOT / "outputs/coding-manifest.json", constraint_contract(), seed=1,
        design="constraint")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    summary = n.simulate_constraint(plan_path, tmp_path / "simulation")

    assert summary == {"scenarios": 24, "conditions": 4,
                       "out": str(tmp_path / "simulation"), "valid": True}
    out = tmp_path / "simulation"
    assert all((out / f"constraint_{name}.{suffix}").is_file()
               for name in ("attainment", "actions", "duals")
               for suffix in ("png", "pdf"))
    with (out / "constraint_predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    assert not any(row["target_met"] == "True" for row in predictions)
    by_cell = {(row["condition_id"], row["policy"]): row
               for row in predictions}
    assert [float(by_cell[cell, "queue_haul"]["attained_shed_w"])
            for cell in ("window-19", "window-30", "window-60", "quota-30")] \
        == pytest.approx((49.245505, 55.918628, 60.480647, 51.688888))
    assert all(float(by_cell[cell, "queue_haul"]["attained_shed_w"])
               >= max(float(by_cell[cell, policy]["attained_shed_w"])
                      for policy in n.CONSTRAINT_POLICIES[1:]) - 1e-8
               for cell, *_ in n.CONSTRAINT_CELLS)
    quota = next(row for row in predictions
                 if (row["condition_id"], row["policy"])
                 == ("quota-30", "queue_haul"))
    assert (int(quota["germany_kv_transfer"]) + int(quota["east_replay"])) \
        / int(quota["selected_sessions"]) >= .70
    with (out / "constraint_duals.csv").open() as handle:
        duals = [row for row in csv.DictReader(handle)
                 if row["resource"].startswith("migration:")]
    assert len(duals) == 16
    assert all(float(row["shadow_w_per_full_capacity"]) > 0 for row in duals)


def test_separation_plan_freezes_three_robust_matched_regimes(tmp_path):
    plan = n.make_plan(
        campaign_manifest(tmp_path, 8), constraint_contract(), seed=7,
        design="separation")

    assert len(plan["scenarios"]) == 63
    assert plan["policies"] == list(n.SEPARATION_POLICIES)
    expected = {
        "germany-service": ("natural", .25, .95, .60),
        "east-service-slow-path": ("natural", .90, .25, .77),
        "joint-shaped": ("controlled_40", .50, .85, .54),
    }
    signatures = set()
    for condition, (bandwidth, east, germany, target) in expected.items():
        for repeat in range(3):
            rows = [row for row in plan["scenarios"]
                    if (row["condition_id"], row["repeat"])
                    == (condition, repeat)]
            signatures |= {tuple(
                (item["template_id"], item["initial_tokens"])
                for item in row["sessions"]
            ) for row in rows}
            assert len(rows) == 7
            assert {row["policy"] for row in rows} == set(n.SEPARATION_POLICIES)
            assert all(
                row["bandwidth"] == bandwidth
                and tuple(row["background"]["east"]) == (east, 0)
                and tuple(row["background"]["germany"]) == (germany, 0)
                and row["deadline_s"] == 45
                and row["planning_deadline_s"] == 30
                and row["load_warmup_s"] == 30
                and row["load_normalization"] == "destination_service"
                and row["requested_shed_fraction"] == target
                and row["planner_seed"] == n.profiler.stable_seed(
                    plan["seed"], row["condition_index"], repeat,
                    row["policy"])
                and len(row["sessions"]) == 28
                and sum(item["initial_tokens"] for item in row["sessions"])
                == 648_131 for row in rows)
    assert len(signatures) == 1
    assert n.parse_args([
        "prepare", "--cluster", "c", "--calibration", "k",
        "--manifest", "m", "--out", "o", "--design", "separation",
    ]).design == "separation"


def test_destination_load_normalization_uses_both_service_rates(tmp_path):
    profile = n.ModelProfile.load(n.MODEL_PATH)
    dtype = n.dedicated_sink_architecture(
        profile, "sink", ("link",)).types[0]

    work = n.destination_background_work(dtype, .95)
    load = n.SinkLoad(
        SimpleNamespace(), 1, 100, .5, tmp_path / "load.jsonl", 50)

    assert sum(work) == pytest.approx(.95)
    assert work[0] > work[1] > 0
    assert load.interval_s == pytest.approx(
        (n.SINK_LOAD_PREFILL_TOKENS / 100
         + n.SINK_LOAD_DECODE_TOKENS / 50) / .5)


def test_deadline_credit_uses_the_shared_campaign_epoch():
    second = 10**9
    results = [
        {"session_id": "before", "request": {
            "start_ns": 9 * second, "end_ns": 10 * second}},
        {"session_id": "boundary", "request": {
            "start_ns": 10 * second, "end_ns": 11 * second}},
        {"session_id": "queued-late", "request": {
            "start_ns": 10 * second, "end_ns": 12 * second}},
        {"session_id": "failed", "error": "request failed"},
    ]

    assert n.deadline_credited_sessions(results, second, 10) == {
        "before", "boundary"}


def test_separation_reducer_repairs_stale_per_request_deadline_credit(
        monkeypatch, tmp_path):
    monkeypatch.setattr(n, "plot_separation", lambda *_args: None)
    plan = n.make_plan(
        n.ROOT / "outputs/coding-manifest.json", constraint_contract(), seed=1,
        design="separation")
    scenario = next(row for row in plan["scenarios"]
                    if row["policy"] == n.DEADLINE_BLIND_POLICY)
    plan["scenarios"] = [scenario]
    result_path = tmp_path / "scenarios" / scenario["scenario_id"] \
        / "attempt-0001" / "result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({
        "status": "complete", "started_ns": 10**9, "deadline_met": False,
        "requested_shed_w": 1, "realized_shed_w": 1, "target_met": True,
        "request_failures": 0, "kv_evidence_warnings": 0,
        "load_warnings": [], "background": {
            "east": {"warning": False}, "germany": {"warning": False}},
        "requests": [{
            "session_id": scenario["sessions"][0]["session_id"],
            "destination_instance": "east", "method": "replay",
            "request": {"start_ns": 44 * 10**9, "end_ns": 47 * 10**9},
        }],
    }))

    summary = n.reduce_run(plan, tmp_path)

    assert summary["valid"] and summary["invalid_evidence"] == 0
    with (tmp_path / "results.csv").open() as handle:
        row = next(csv.DictReader(handle))
    assert float(row["requested_shed_w"]) > 0
    assert float(row["realized_shed_w"]) == pytest.approx(0)
    assert row["target_met"] == "False"


def test_separation_simulation_has_wide_joint_action_margins(tmp_path):
    plan = n.make_plan(
        n.ROOT / "outputs/coding-manifest.json", constraint_contract(), seed=1,
        design="separation")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    summary = n.simulate_separation(plan_path, tmp_path / "simulation")

    assert summary == {"scenarios": 63, "conditions": 3,
                       "out": str(tmp_path / "simulation"), "valid": True}
    out = tmp_path / "simulation"
    with (out / "separation_predictions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        ratio = float(row["attainment_fraction"])
        if row["policy"] in n.SEPARATION_POLICIES[:2]:
            assert ratio >= 1.1
            assert row["deadline_met"] == "True"
            assert int(row["east_replay"]) + int(row["germany_replay"]) >= 2
            assert int(row["east_kv_transfer"]) \
                + int(row["germany_kv_transfer"]) >= 2
            assert int(row["east_replay"]) + int(row["east_kv_transfer"]) > 0
            assert int(row["germany_replay"]) \
                + int(row["germany_kv_transfer"]) > 0
        else:
            assert ratio <= .9
    deadline_blind = [row for row in rows
                      if row["policy"] == n.DEADLINE_BLIND_POLICY]
    assert all(row["planner_feasible"] == "True"
               and row["deadline_met"] == "False" for row in deadline_blind)
    with (out / "separation_resources.csv").open() as handle:
        resources = list(csv.DictReader(handle))
    expected_conditions = {row[0] for row in n.SEPARATION_CELLS}
    utilization = {(row["condition_id"], row["resource"]):
                   float(row["utilization"]) for row in resources}
    assert all(utilization[condition, resource] >= minimum
               for condition, bindings in n.SEPARATION_BINDINGS.items()
               for resource, minimum in bindings.items())
    assert expected_conditions == {
        "germany-service", "east-service-slow-path", "joint-shaped"}
    assert all((out / f"separation_{name}.{suffix}").is_file()
               for name in ("campaign", "resources")
               for suffix in ("png", "pdf"))


def test_oracle_stale_simulation_certifies_capacity_and_deadline_traps(tmp_path):
    plan = n.make_plan(
        n.ROOT / "outputs/coding-manifest.json", constraint_contract(), seed=1,
        design="separation")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))

    summary = n.simulate_oracle_stale(plan_path, tmp_path / "simulation")

    assert summary == {"oracle_conditions": 3, "toggle_states": 8,
                       "out": str(tmp_path / "simulation"), "valid": True}
    out = tmp_path / "simulation"
    with (out / "restricted_oracles.csv").open() as handle:
        oracles = list(csv.DictReader(handle))
    assert {row["admission_mode"] for row in oracles} == {"normal"}
    for condition in {row[0] for row in n.SEPARATION_CELLS}:
        rows = [row for row in oracles if row["condition_id"] == condition]
        joint = next(float(row["shed_w"]) for row in rows
                     if row["restriction"] == "joint")
        restricted = max(float(row["shed_w"]) for row in rows
                         if row["restriction"] != "joint")
        assert joint - restricted >= 12

    with (out / "toggle_predictions.csv").open() as handle:
        toggles = list(csv.DictReader(handle))
    selected = {(row["state"], row["plan"]): row for row in toggles}
    target = float(selected["all-bind", "adaptive"]["requested_shed_w"])
    assert float(selected["all-bind", "adaptive"]["shed_by_deadline_w"]) \
        >= 1.2 * target
    assert all(selected[state, "robust"]["target_by_deadline"] == "True"
               for state, *_ in n.ORACLE_STALE_STATES)
    assert selected["all-bind", "stale-optimistic"]["status"] \
        == "capacity_infeasible"
    aware = selected["all-bind", "deadline-aware"]
    blind = selected["all-bind", "deadline-blind"]
    assert float(aware["time_to_target_s"]) < n.SEPARATION_DEADLINE_S
    assert float(blind["shed_by_deadline_w"]) <= .85 * target
    assert float(blind["eventual_shed_w"]) >= 1.2 * target
    assert float(blind["time_to_target_s"]) >= 55

    with (out / "toggle_resources.csv").open() as handle:
        resources = {(row["state"], row["resource"]): float(row["utilization"])
                     for row in csv.DictReader(handle)}
    assert resources["all-bind", "kv:pool/east"] >= .94
    assert resources["all-bind", "service:pool/germany:0"] >= .97
    assert all((out / f"oracle_stale_{name}.{suffix}").is_file()
               for name in ("summary", "resources")
               for suffix in ("png", "pdf"))


def test_frontier_refinement_adds_boundary_midpoint_and_five_repeats(tmp_path):
    plan = n.make_plan(
        campaign_manifest(tmp_path, 4), n.freeze_contract(calibration()),
        seed=7, design="frontier")
    root = tmp_path / "run"
    for scenario in plan["scenarios"]:
        loads = tuple(value[0] for value in scenario["background"].values())
        changed = scenario["pack"] == "4x16k" and loads[0] == loads[1] \
            and loads[0] >= .9
        result = root / "scenarios" / scenario["scenario_id"] \
            / "attempt-0001" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({
            "status": "complete", "target_met": True, "realized_shed_w": 50,
            "requests": [{"destination_instance": "east",
                          "method": "kv_transfer" if changed else "replay",
                          "request": {}}],
        }))

    refined = n.frontier_refinement(plan, root)

    assert refined["phase"] == "refinement"
    assert len(refined["scenarios"]) == 65
    midpoint = [row for row in refined["scenarios"]
                if {value[0] for value in row["background"].values()} == {.875}]
    assert {row["repeat"] for row in midpoint} == set(range(5))
    assert {row["policy"] for row in midpoint} == set(n.FRONTIER_POLICIES)


def test_frontier_refinement_caps_many_boundaries_in_matched_blocks(tmp_path):
    plan = n.make_plan(
        campaign_manifest(tmp_path, 4), n.freeze_contract(calibration()),
        seed=7, design="frontier")
    root = tmp_path / "run"
    for scenario in plan["scenarios"]:
        load = next(iter(scenario["background"].values()))[0]
        result = root / "scenarios" / scenario["scenario_id"] \
            / "attempt-0001" / "result.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({
            "status": "complete", "target_met": load < .9,
            "realized_shed_w": 50,
            "requests": [{"destination_instance": "east",
                          "method": "kv_transfer" if load >= .9 else "replay",
                          "request": {}}],
        }))

    refined = n.frontier_refinement(plan, root)
    blocks = {}
    for row in refined["scenarios"]:
        blocks.setdefault((row["condition_index"], row["repeat"]), set()) \
            .add(row["policy"])

    assert len(refined["scenarios"]) == n.FRONTIER_REFINEMENT_EPISODES
    assert all(policies == set(n.FRONTIER_POLICIES)
               for policies in blocks.values())
    assert any(row["condition_index"] >= 37 for row in refined["scenarios"])


def test_frontier_reduction_plots_mechanism_and_attainment(tmp_path):
    plan = n.make_plan(
        campaign_manifest(tmp_path, 4), n.freeze_contract(calibration()),
        seed=7, design="frontier")
    scenario = next(row for row in plan["scenarios"]
                    if row["policy"] == "queue_haul")
    result = {"target_met": True, "requests": [{
        "destination_instance": "east", "method": "replay", "request": {},
    }]}

    n.plot_frontier(plan, [(scenario, result)], tmp_path)

    assert all((tmp_path / f"{name}.{suffix}").is_file()
               for name in ("prefill_network_mechanism",
                            "frontier_target_attainment")
               for suffix in ("png", "pdf"))


def test_isolated_plan_is_54_paired_route_relative_migrations(tmp_path):
    path = campaign_manifest(tmp_path, 4)
    contract = n.freeze_contract(calibration())
    contract["paths"] = {"east": contract["paths"]["east"]}
    contract["aggregate"] = {
        "natural_mbps": contract["paths"]["east"]["natural_mbps"],
        "controlled_mbps": contract["paths"]["east"]["controlled_mbps"],
    }

    plan = n.make_plan(path, contract, seed=7, design="isolated")

    assert plan["design"] == "isolated" and len(plan["scenarios"]) == 54
    assert all(row["sessions"][0]["initial_tokens"] ==
               row["context_size"] - n.ISOLATED_PROMPT_HEADROOM_TOKENS
               for row in plan["scenarios"])
    assert {(row["context_size"], row["bandwidth"], row["repeat"], row["method"])
            for row in plan["scenarios"]} == {
        (size, bandwidth, repeat, method)
        for size in (2048, 8192, 32768)
        for bandwidth in ("controlled_40", "controlled_80", "natural")
        for repeat in range(3) for method in ("replay", "kv_transfer")
    }


def test_prometheus_snapshot_reads_live_kv_and_warns_on_drift():
    samples = ["""
vllm:kv_cache_usage_perc 0.31
vllm:num_requests_running 2
vllm:num_requests_waiting 1
"""]
    snapshot = n.summarize_metrics(samples, .2)

    assert snapshot["kv_fraction"] == pytest.approx(.31)
    assert snapshot["warning"]


def test_joint_planner_preserves_dynamic_destinations(monkeypatch):
    moves = [SimpleNamespace(
        session_id="s0", destination_instance="west",
        destination_pool="pool/west", method="replay", order=0,
        path=("link/west",), rate_limit_bytes_per_s=None, quiesce_s=None,
    )]
    seen = {}
    monkeypatch.setattr(n, "solve", lambda problem, profile, routes, solver,
                        seed, destination: seen.update(
                            routes=routes, architecture=destination) or
                        SimpleNamespace(moves=moves))
    scenario = {
        "policy": "queue_haul", "deadline_s": 30,
        "sessions": [{"session_id": "s0", "initial_tokens": 8192}],
        "bandwidth_mbps": {"east": 1000, "west": 2000},
        "background": {"east": (.2, .4), "west": (.4, .2)},
        "migration_headroom": {"west": {"replay": .25}},
    }
    profile = n.ModelProfile.load(n.MODEL_PATH)

    result = n.plan_joint_scenario(
        scenario, {"east": {"kv_fraction": .4},
                   "west": {"kv_fraction": .2}}, profile, 1)

    assert result[0]["destination_instance"] == "west"
    assert seen["routes"] == {("source", "east"): ("link/east",),
                              ("source", "west"): ("link/west",)}
    assert {pool.replicas[0].baseline_kv_tokens
            for pool in seen["architecture"].pools} == {
                round(.4 * profile.kv_capacity_tokens),
                round(.2 * profile.kv_capacity_tokens),
            }
    dtype = seen["architecture"].types[0]
    assert {pool.replicas[0].baseline_work
            for pool in seen["architecture"].pools} == {
                tuple(dtype.work(profile.case().F * rho, 0, 512))
                for rho in (.2, .4)}
    assert {pool.pool_id: pool.migration_headroom
            for pool in seen["architecture"].pools} == {
                "pool/east": None, "pool/west": {"replay": .25}}
    assert n.joint_solver("isolated_fastest") == "isolated_fastest"
    assert n.joint_solver("queue_haul", "max_shed") == "max_shed"


def test_frontier_planner_targets_power_and_does_not_force_evacuation(monkeypatch):
    move = SimpleNamespace(
        session_id="s0", destination_instance="east",
        destination_pool="pool/east", method="replay", order=0,
        path=("link/east",), rate_limit_bytes_per_s=None, quiesce_s=None)
    seen = {}
    monkeypatch.setattr(n, "solve", lambda problem, *_args, **_kwargs:
                        seen.update(problem=problem) or SimpleNamespace(moves=[move]))
    scenario = {
        "design": "frontier", "policy": "queue_haul", "deadline_s": 30,
        "requested_shed_fraction": .8,
        "sessions": [{"session_id": session, "initial_tokens": 8192}
                     for session in ("s0", "s1")],
        "bandwidth_mbps": {"east": 1000, "west": 2000},
        "background": {"east": (0, 0), "west": (0, 0)},
    }

    result = n.plan_joint_scenario(
        scenario, {node: {"kv_fraction": 0} for node in ("east", "west")},
        n.ModelProfile.load(n.MODEL_PATH), 1)

    assert [row["session_id"] for row in result] == ["s0"]
    assert seen["problem"].power_limit_w > 0


def test_deadline_blind_planner_uses_nonbinding_horizon(monkeypatch):
    seen = {}
    monkeypatch.setattr(n, "solve", lambda problem, *_args, **_kwargs:
                        seen.update(problem=problem) or SimpleNamespace(moves=[]))
    scenario = {
        "design": "frontier", "policy": n.DEADLINE_BLIND_POLICY,
        "deadline_s": 30, "requested_shed_fraction": .8,
        "sessions": [{"session_id": "s0", "initial_tokens": 8192}],
        "bandwidth_mbps": {"east": 1000, "west": 2000},
        "background": {"east": (0, 0), "west": (0, 0)},
    }

    n.plan_joint_scenario(
        scenario, {node: {"kv_fraction": 0} for node in ("east", "west")},
        n.ModelProfile.load(n.MODEL_PATH), 1)

    assert scenario["deadline_s"] == 30
    assert seen["problem"].deadline_s == n.DEADLINE_BLIND_HORIZON_S


def test_deadline_blind_plan_matches_each_input_block(tmp_path):
    pilot = n.make_plan(
        campaign_manifest(tmp_path, 4), n.freeze_contract(calibration()),
        seed=7, design="frontier")
    template = next(row for row in pilot["scenarios"]
                    if row["policy"] == "queue_haul")
    refinement = {**pilot, "phase": "refinement", "scenarios": [
        {**template, "policy": policy, "repeat": 1,
         "scenario_id": f"refined-{policy}"}
        for policy in n.FRONTIER_POLICIES
    ]}

    plan = n.deadline_blind_plan([pilot, refinement])

    assert plan["phase"] == "deadline_blind"
    assert len(plan["scenarios"]) == 38
    assert {row["policy"] for row in plan["scenarios"]} == {
        n.DEADLINE_BLIND_POLICY}


def test_observed_demand_uses_uncached_prefill_and_decode_tokens():
    rows = [
        {"session_id": "a", "prompt_tokens": 100, "cached_tokens": 80,
         "output_tokens": 10},
        {"session_id": "a", "prompt_tokens": 120, "cached_tokens": 100,
         "output_tokens": 14},
    ]

    assert n.observed_demand(rows, 2) == {"a": (20, 12)}


def test_agentic_demand_uses_trace_turn_rate_and_token_work():
    records = {"a": {"id": "a", "turn_rate_hz": .5, "turns": [
        {"append_tokens": 10, "output_tokens": 2},
        {"append_tokens": 30, "output_tokens": 6},
    ]}}

    profile = n.ModelProfile.load(n.MODEL_PATH)
    demand = n.agentic_demand(records, [{"session_id": "a"}], profile)

    assert demand["a"][0] / profile.case().F \
        + demand["a"][1] / profile.case().G == pytest.approx(.4)


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


def test_remote_readiness_waits_in_parallel(monkeypatch):
    started = []
    both_started = threading.Event()

    def ready(process, timeout):
        started.append((process, timeout))
        if len(started) == 2:
            both_started.set()
        assert both_started.wait(1)

    monkeypatch.setattr(n, "_remote_ready", ready)
    n._wait_remote_ready({"east": "east", "west": "west"}, 300)

    assert sorted(started) == [("east", 300), ("west", 300)]


def test_serve_window_routes_every_session(monkeypatch):
    ticks = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(n.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(n, "_chat", lambda _cfg, port, _messages, code,
                        _timeout: {"port": port, "code": code})
    stack = SimpleNamespace(cfg=object())

    rows = n._serve_window(
        stack, {"a": [], "b": []}, {"a": "A", "b": "B"},
        {"a": ("east", 1), "b": ("west", 2)}, 1, "pre")

    assert {(row["session_id"], row["node"], row["port"], row["code"])
            for row in rows} == {("a", "east", 1, "A"),
                                 ("b", "west", 2, "B")}


def test_handoff_uses_queue_haul_deadline_and_cache_isolated_load(monkeypatch, tmp_path):
    shared = {"bandwidth": "natural", "deadline_s": 30,
              "condition_index": 5, "repeat": 1, "sessions": [{"session_id": "s"}]}
    plan = {"scenarios": [{**shared, "policy": policy}
                           for policy in ("queue_haul", "kv_only", "replay_only")]}
    scenarios = [n.handoff_scenario(
        plan, cluster(tmp_path), .5, policy, 1)
        for policy in ("queue_haul", "kv_only", "replay_only")]
    assert [row["policy"] for row in scenarios] == [
        "queue_haul", "kv_only", "replay_only"]
    assert {tuple(row["sessions"][0].items()) for row in scenarios} == {
        (("session_id", "s"),)}
    assert all(row["deadline_s"] == 30 for row in scenarios)
    assert all(row["background"] == {"east": [.5, 0], "west": [.5, 0]}
               for row in scenarios)
    assert n.HANDOFF_ENV == {
        "QH_KV_ROLE_SOURCE": "kv_both", "QH_KV_ROLE_SINK": "kv_both",
        "QH_LMCACHE_L1_GB": "33", "QH_PREFIX_CACHING": "off",
        "QH_REDIS_MAXMEMORY_GB": "32",
    }

    seen = {}
    result = n.profiler.RequestResult("r", 200, "", 1, 2)
    monkeypatch.setattr(n.profiler, "stream_chat", lambda *args:
                        seen.update(args=args) or (result, ""))
    load = n.SinkLoad(SimpleNamespace(), 1, 1000, .5, tmp_path / "load.jsonl")
    load._request(7)
    assert seen["args"][6:] == (True, "load-7")
    assert seen["args"][2][0]["role"] == "system"
    load.stop_admissions()
    assert load.stop.is_set()


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


def test_constraint_reducer_hard_fails_semantically_invalid_evidence(
        monkeypatch, tmp_path):
    monkeypatch.setattr(n, "plot_constraint", lambda *_args: None)
    scenario = {
        "scenario_id": "s", "condition_index": 0,
        "condition_id": "window-19", "repeat": 0,
        "policy": "queue_haul", "workload": "agentic_tool_loop",
        "bandwidth": "natural", "background": {"east": [.5, 0]},
        "pack": "window-19", "deadline_s": 19, "sessions": [],
    }
    root = tmp_path / "run"
    result_path = root / "scenarios/s/attempt-0001/result.json"
    result_path.parent.mkdir(parents=True)
    result = {
        "status": "complete", "deadline_met": True, "target_met": False,
        "requested_shed_w": 10, "realized_shed_w": 11,
        "request_failures": 0, "kv_evidence_warnings": 0,
        "load_warnings": [], "background": {
            "east": {"warning": True}, "germany": {"warning": False}},
        "requests": [],
    }
    result_path.write_text(json.dumps(result))

    invalid = n.reduce_run({"design": "constraint", "scenarios": [scenario]}, root)
    assert not invalid["valid"] and invalid["invalid_evidence"] == 1

    result["background"]["east"]["warning"] = False
    result_path.write_text(json.dumps(result))
    valid = n.reduce_run({"design": "constraint", "scenarios": [scenario]}, root)
    assert valid["valid"] and valid["invalid_evidence"] == 0

    scenario["policy"] = "kv_only"
    result["target_met"] = False
    result_path.write_text(json.dumps(result))
    expected_miss = n.reduce_run(
        {"design": "constraint", "scenarios": [scenario]}, root)
    assert expected_miss["valid"] and expected_miss["invalid_evidence"] == 0

    result["target_met"] = True
    result_path.write_text(json.dumps(result))
    unexpected_pass = n.reduce_run(
        {"design": "constraint", "scenarios": [scenario]}, root)
    assert not unexpected_pass["valid"] \
        and unexpected_pass["invalid_evidence"] == 1


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

    def stream(_cfg, _port, messages, tokens, _hash, _timeout, bypass):
        seen.update(messages=messages, tokens=tokens, bypass=bypass)
        now = n.time.monotonic_ns()
        return n.profiler.RequestResult("r", 200, "", now, now), "CODE"

    monkeypatch.setattr(n.profiler, "stream_chat", stream)
    n._chat(object(), 1, [{"role": "user", "content": "context"}],
            "CODE", 1)

    assert seen == {"messages": [
        {"role": "user", "content": "context"},
        {"role": "user", "content":
            "Reply only with session state code CODE."}], "tokens": 128,
                    "bypass": False}

    n._chat(object(), 1, [], "CODE", 1, True)
    assert seen["bypass"] is True


def test_chat_retries_one_invalid_probe(monkeypatch):
    replies = iter(["invalid", "CODE"])
    result = n.profiler.RequestResult("r", 200, "", 1, 2)
    monkeypatch.setattr(n.profiler, "stream_chat",
                        lambda *_args: (result, next(replies)))

    assert n._chat(object(), 1, [], "CODE", 1)["probe_attempts"] == 2


def test_warm_waits_only_for_complete_lmcache_blocks(monkeypatch, tmp_path):
    log = tmp_path / "lmcache-source.log"
    log.touch()
    stack = SimpleNamespace(
        cfg=SimpleNamespace(src_port=1), run_root=tmp_path)
    seen = []
    monkeypatch.setattr(n, "_chat", lambda *_args: {"prompt_tokens": 19086})
    monkeypatch.setattr(n.testbed, "mp_wait_stored",
                        lambda *args: seen.append(args))

    n._warm(stack, [], "CODE", 1)

    assert seen == [(log, 0, 18944)]


def test_resume_metadata_allows_audited_commit_change_but_pins_identity():
    first = {"git_sha": "abc", "plan_sha256": "p",
             "hosts": {"east": {"git_sha": "abc", "gpu": "A100"}},
             "checks": [{"clock": 1}]}
    current = {"git_sha": "changed", "plan_sha256": "p",
               "hosts": {"east": {"git_sha": "changed", "gpu": "A100"}},
               "checks": [{"clock": 2}]}

    assert n.merge_metadata(current, first)["checks"] == [
        {"clock": 1}, {"clock": 2}]
    current["hosts"]["east"]["gpu"] = "H100"
    with pytest.raises(RuntimeError, match="metadata changed"):
        n.merge_metadata(current, first)
