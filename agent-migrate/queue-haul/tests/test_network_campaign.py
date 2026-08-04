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

import json

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
    limiter = testbed.BandwidthLimiter(100, {"kv/east": 60, "kv/west": 60})
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


def test_host_reports_must_match_commit_runtime_and_expected_regions(tmp_path):
    base = {
        "git_sha": "abc", "dirty": False, "gpu": "NVIDIA A100 80GB PCIe",
        "gpu_memory_mib": 81920, "vllm": "0.22.0", "lmcache": "0.5.1",
        "vm_size": "Standard_NC24ads_A100_v4", "clock_uncertainty_ms": 1,
        "ptp": "/dev/ptp0", "datadrive": True,
    }
    reports = {
        "sweden": {**base, "region": "swedencentral", "private_ip": "10.0.0.4"},
        "east": {**base, "region": "eastus2", "private_ip": "10.1.0.4"},
        "west": {**base, "region": "westeurope", "private_ip": "10.2.0.4"},
    }
    n.validate_hosts(cluster(tmp_path), reports)

    reports["west"]["vllm"] = "0.22.1"
    with pytest.raises(ValueError, match="runtime"):
        n.validate_hosts(cluster(tmp_path), reports)
