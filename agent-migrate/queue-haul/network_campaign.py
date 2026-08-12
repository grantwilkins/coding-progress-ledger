"""Run the private three-region Azure Queue-Haul campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import select
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import migration_testbed as testbed
import migration_profiler as profiler
import policy_hardware_campaign as policy_campaign
from destination_runner import MetricsSampler
from destination import (DestinationArchitecture, DestinationPool,
                         DestinationReplica, dedicated_sink_architecture)
from planner import _expected_scenario, plan as solve, source_power
from pool_planner import candidate_table, phase_one_capacity_duals
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile
from simulate import (NetworkLink, PlannedMove, PowerNode, ServingInstance,
                      predict)


CLUSTER_SCHEMA = "queue-haul-azure-cluster-v1"
CALIBRATION_SCHEMA = "queue-haul-network-calibration-v1"
PLAN_SCHEMA = "queue-haul-network-plan-v2"
RESULT_SCHEMA = "queue-haul-network-result-v2"
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "profiles" / os.environ.get(
    "QH_MODEL_PROFILE", "gpt_oss_20b_a100_tp1_azure_300w.json")
H100_CAMPAIGN = "H100" in ModelProfile.load(MODEL_PATH).hardware
CLOCK_LIMIT_MS = 2.0
RESUME_DRIFT = .10
REQUEST_TIMEOUT_S = 600.0
REPEATS = 3
ISOLATED_PROMPT_HEADROOM_TOKENS = 512
POLICIES = (
    "queue_haul", "greedy", "greedy_lagrangian", "kv_only", "replay_only",
    "random",
)
DEADLINE_BLIND_POLICY = "queue_haul_deadline_blind"


def frontier_policies(h100: bool):
    base = ("queue_haul", "greedy", "replay_only", "kv_only",
            "queue_haul_power_blind")
    return base + (("greedy_lagrangian", "isolated_fastest",
                    DEADLINE_BLIND_POLICY) if h100 else ())


FRONTIER_POLICIES = frontier_policies(H100_CAMPAIGN)
CONSTRAINT_POLICIES = (
    "queue_haul", "greedy", "kv_only", "replay_only", "isolated_fastest",
    "queue_haul_power_blind",
)
SEPARATION_POLICIES = (*CONSTRAINT_POLICIES, DEADLINE_BLIND_POLICY)
CONSTRAINT_CELLS = (
    ("window-19", 19, 22, 15, 1.0),
    ("window-30", 30, 28, 8, 1.0),
    ("window-60", 60, 64, None, 1.0),
    ("quota-30", 30, 28, 8, 1.0),
)
SEPARATION_CELLS = (
    ("germany-service", "natural", .25, .95, .60),
    ("east-service-slow-path", "natural", .90, .25, .77),
    ("joint-shaped", "controlled_40", .50, .85, .54),
)
SEPARATION_REPEATS = 3
SEPARATION_DEADLINE_S = 45
SEPARATION_PLANNING_DEADLINE_S = 30
SEPARATION_LOAD_WARMUP_S = 30
SEPARATION_MARGIN = .10
ORACLE_RESTRICTIONS = (
    "joint", "kv_only", "replay_only", "east_only", "germany_only",
)
KV_RESERVED_FRACTION = .96 if H100_CAMPAIGN else .90
ORACLE_STALE_STATES = (
    ("all-bind", .75, KV_RESERVED_FRACTION, "controlled_40"),
    ("free-kv", .75, 0, "controlled_40"),
    ("free-service", .25, KV_RESERVED_FRACTION, "controlled_40"),
    ("free-bandwidth", .75, KV_RESERVED_FRACTION, "natural"),
    ("free-kv-bandwidth", .75, 0, "natural"),
    ("free-service-bandwidth", .25, KV_RESERVED_FRACTION, "natural"),
    ("free-kv-service", .25, 0, "controlled_40"),
    ("all-release", .25, 0, "natural"),
)
ORACLE_STALE_TARGET_FRACTION = .414 if H100_CAMPAIGN else .65
ORACLE_STALE_HORIZON_S = 90
HARDWARE_GAP_TARGET_FRACTION = .414 if H100_CAMPAIGN else .72
HARDWARE_GAP_REPEATS = 3
HARDWARE_GAP_MATRIX = (
    ("all-bind", .75, KV_RESERVED_FRACTION, "controlled_40", (
        "queue_haul_robust", "greedy", "oracle_kv_only",
        "oracle_replay_only", "oracle_east_only", "oracle_germany_only",
        "isolated_fastest", "queue_haul_power_blind",
        DEADLINE_BLIND_POLICY, "queue_haul_stale",
    )),
    ("free-kv", .75, 0, "controlled_40", (
        "queue_haul", "queue_haul_robust", "greedy", "oracle_east_only",
    )),
    ("free-service", .25, KV_RESERVED_FRACTION, "controlled_40", (
        "queue_haul", "queue_haul_robust", "oracle_germany_only",
    )),
    ("free-bandwidth", .75, KV_RESERVED_FRACTION, "natural", (
        "queue_haul", "queue_haul_robust", "oracle_kv_only",
    )),
    ("all-release", .25, 0, "natural", (
        "queue_haul", "queue_haul_robust", "queue_haul_stale",
        "oracle_germany_only",
    )),
)
HARDWARE_GAP_POLICIES = tuple(dict.fromkeys(
    policy for *_, policies in HARDWARE_GAP_MATRIX for policy in policies))
KV_BLOCK_SIZE = 16
SEPARATION_BINDINGS = {
    "germany-service": {
        "service:pool/germany:0": .98,
        "migration:pool/east:replay": .89,
        "migration:pool/east:kv_transfer": .94,
    },
    "east-service-slow-path": {
        "migration:pool/east:replay": .98,
        "migration:pool/east:kv_transfer": .94,
        "migration:pool/germany:replay": .92,
        "migration:pool/germany:kv_transfer": .98,
    },
    "joint-shaped": {
        "service:pool/east:0": .92,
        "service:pool/germany:0": .89,
        "migration:pool/east:replay": .89,
        "migration:pool/east:kv_transfer": .97,
        "migration:pool/germany:replay": .92,
        "migration:pool/germany:kv_transfer": .96,
    },
}
SINK_LOAD_PREFILL_TOKENS = 604
SINK_LOAD_DECODE_TOKENS = 64
HARDWARE_GAP_BACKGROUND_KV_TOKENS = (
    SINK_LOAD_PREFILL_TOKENS + SINK_LOAD_DECODE_TOKENS)
CONSTRAINT_ACTIONS = (
    "germany_kv_transfer", "east_replay", "east_kv_transfer",
    "germany_replay",
)
TAB10_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
HARDWARE_GAP_COLORS = {
    "queue_haul": TAB10_COLORS[0], "queue_haul_robust": TAB10_COLORS[0],
    "greedy": TAB10_COLORS[1], "oracle_kv_only": TAB10_COLORS[2],
    "oracle_replay_only": TAB10_COLORS[3],
    "oracle_east_only": TAB10_COLORS[4],
    "oracle_germany_only": TAB10_COLORS[5],
    "isolated_fastest": TAB10_COLORS[6],
    "queue_haul_power_blind": TAB10_COLORS[7],
    DEADLINE_BLIND_POLICY: TAB10_COLORS[8],
    "queue_haul_stale": TAB10_COLORS[9],
}
def frontier_packs(h100: bool):
    base = (
        ("4x16k", 4, 16_384), ("8x16k", 8, 16_384),
        ("16x16k", 16, 16_384),
    )
    return base + ((
        ("16x8k", 16, 8_192), ("16x24k", 16, 24_576),
        ("16x31k", 16, 31_488), ("32x31k", 32, 31_488),
    ) if h100 else (
        ("8x8k", 8, 8_192), ("8x24k", 8, 24_576),
        ("8x31k", 8, 31_488),
    ))


FRONTIER_PACKS = frontier_packs(H100_CAMPAIGN)
FRONTIER_LOADS = (0, .5, .85, .9, .95)
FRONTIER_FAILURE_GATE = .5
FRONTIER_REFINEMENT_EPISODES = 64 if H100_CAMPAIGN else 65
DEADLINE_BLIND_HORIZON_S = 600
WORKLOAD_PATHS = {name: ROOT / f"profiles/{name}.json" for name in (
    "coding", "interactive_coding", "agentic_tool_loop",
)}
LOAD_SUPPORT_PATH = ROOT / "outputs/capacity-load-publication-20260807/live_plan.json"
EXPECTED_RUNTIME = {"vllm": "0.22.0", "lmcache": "0.5.1"}
HANDOFF_DEADLINE_S = 30
HANDOFF_POLICIES = ("queue_haul", "kv_only", "replay_only")
HANDOFF_ENV = {
    "QH_KV_ROLE_SOURCE": "kv_both", "QH_KV_ROLE_SINK": "kv_both",
    "QH_LMCACHE_L1_GB": "33", "QH_PREFIX_CACHING": "off",
    "QH_REDIS_MAXMEMORY_GB": "32",
}
RUNTIME_ENV = (*HANDOFF_ENV, "QH_MODEL_PROFILE")


def write_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class Node:
    id: str
    region: str
    host: str
    ssh_user: str
    repo_root: str
    run_root: str
    ssh_port: int = 22

    @classmethod
    def parse(cls, raw: dict) -> "Node":
        value = cls(**raw)
        if not all((value.id, value.region, value.host, value.ssh_user,
                    value.repo_root, value.run_root)) \
                or not 0 < value.ssh_port < 65536:
            raise ValueError("invalid cluster node")
        return value


@dataclass(frozen=True)
class Cluster:
    source: Node
    destinations: tuple[Node, ...]
    schema: str = CLUSTER_SCHEMA

    @classmethod
    def parse(cls, raw: dict) -> "Cluster":
        if raw.get("schema") != CLUSTER_SCHEMA:
            raise ValueError("invalid cluster schema")
        value = cls(
            Node.parse(raw["source"]),
            tuple(Node.parse(node) for node in raw["destinations"]),
        )
        nodes = (value.source, *value.destinations)
        if not 1 <= len(value.destinations) <= 2 \
                or len({n.id for n in nodes}) != len(nodes) \
                or len({n.host for n in nodes}) != len(nodes):
            raise ValueError("cluster node ids and hosts must be unique")
        regions = {node.region for node in value.destinations}
        if not (value.source.region == "swedencentral" and regions <= {
                "eastus2", "westeurope", "germanywestcentral"} or
                value.source.region == "westus3" and
                regions == {"australiaeast", "southcentralus"}):
            raise ValueError("cluster regions do not match the frozen topology")
        return value

    @classmethod
    def load(cls, path: Path) -> "Cluster":
        return cls.parse(json.loads(path.read_text()))

    def as_dict(self) -> dict:
        return {
            "schema": self.schema, "source": asdict(self.source),
            "destinations": [asdict(node) for node in self.destinations],
        }


def validate_calibration(raw: dict) -> None:
    if raw.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError("invalid calibration schema")
    if not raw.get("clock_uncertainty_ms") or max(
        map(float, raw["clock_uncertainty_ms"].values())
    ) > CLOCK_LIMIT_MS:
        raise ValueError("clock uncertainty exceeds 2 ms")
    if not 1 <= len(raw.get("paths", {})) <= 2 \
            or any(not row.get("rtt_ms") or not row.get("simultaneous_mbps")
                   or min(map(float, row["simultaneous_mbps"])) <= 0
                   for row in raw["paths"].values()) \
            or not raw.get("aggregate_simultaneous_mbps"):
        raise ValueError("calibration is incomplete")


def _rates(mbps: float) -> dict[str, int]:
    return {str(percent): int(mbps * percent / 100 // 100 * 100)
            for percent in (40, 80)}


def freeze_contract(raw: dict) -> dict:
    validate_calibration(raw)
    paths = {}
    for node, row in sorted(raw["paths"].items()):
        natural = float(statistics.median(map(float, row["simultaneous_mbps"])))
        paths[node] = {
            "rtt_ms": float(statistics.median(map(float, row["rtt_ms"]))),
            "natural_mbps": natural,
            "controlled_mbps": _rates(natural),
        }
    aggregate = float(statistics.median(
        map(float, raw["aggregate_simultaneous_mbps"])))
    return {
        "schema": "queue-haul-network-contract-v1", "paths": paths,
        "aggregate": {
            "natural_mbps": aggregate,
            "controlled_mbps": _rates(aggregate),
        },
    }


def validate_resume(original: dict, current: dict) -> None:
    values = [(original["aggregate"]["natural_mbps"],
               current["aggregate"]["natural_mbps"])]
    for node in original["paths"]:
        values.extend((
            (original["paths"][node]["natural_mbps"],
             current["paths"][node]["natural_mbps"]),
            (original["paths"][node]["rtt_ms"],
             current["paths"][node]["rtt_ms"]),
        ))
    if any(abs(float(new) - float(old)) / float(old) > RESUME_DRIFT + 1e-12
           for old, new in values):
        raise ValueError("resumed allocation network drift exceeds 10%")


def chrony_uncertainty_ms(text: str) -> float:
    if not re.search(r"Leap status\s*:\s*Normal", text):
        raise ValueError("chrony Leap status is not Normal")
    values = {}
    for key in ("Last offset", "Root dispersion"):
        match = re.search(rf"{key}\s*:\s*([+-]?[0-9.eE+-]+) seconds", text)
        if not match:
            raise ValueError(f"chrony omitted {key}")
        values[key] = float(match.group(1))
    return 1000 * (abs(values["Last offset"]) + values["Root dispersion"])


def iperf_mbps(raw: dict) -> float:
    if raw.get("error"):
        raise RuntimeError(f"iperf3 failed: {raw['error']}")
    try:
        value = float(raw["end"]["sum_received"]["bits_per_second"]) / 1e6
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("iperf3 omitted receiver goodput") from exc
    if value <= 0:
        raise RuntimeError("iperf3 reported nonpositive goodput")
    return value


def _output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _imds() -> dict:
    raw = json.loads(_output([
        "curl", "-fsS", "--noproxy", "*", "-H", "Metadata:true",
        "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
    ]))
    compute, addresses = raw["compute"], raw["network"]["interface"]
    private = [address["privateIpAddress"] for interface in addresses
               for address in interface["ipv4"]["ipAddress"]]
    return {
        "region": compute["location"], "vm_size": compute["vmSize"],
        "priority": compute.get("priority"), "private_ips": private,
    }


def node_report() -> dict:
    metadata = _imds()
    gpu = _output([
        "nvidia-smi", "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]).splitlines()
    if len(gpu) != 1:
        raise RuntimeError("each network node must expose exactly one GPU")
    name, memory = map(str.strip, gpu[0].split(","))
    tracking = _output(["chronyc", "tracking"])
    ptp = Path("/dev/ptp_hyperv")
    return {
        "git_sha": _output(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_output([
            "git", "status", "--porcelain", "--untracked-files=no"])),
        "gpu": name, "gpu_memory_mib": int(float(memory)),
        **EXPECTED_RUNTIME,
        "vllm": version("vllm"), "lmcache": version("lmcache"),
        "clock_uncertainty_ms": chrony_uncertainty_ms(tracking),
        "ptp": str(ptp.resolve()) if ptp.exists() else "",
        "datadrive": subprocess.run(
            ["mountpoint", "-q", "/datadrive"], check=False).returncode == 0,
        "private_ip": next((ip for ip in metadata.pop("private_ips")
                            if ip.startswith("10.")), ""),
        **metadata,
    }


def ssh_command(node: Node, key: Path, remote: list[str]) -> list[str]:
    script = f"cd {shlex.quote(node.repo_root)} && " \
        + " ".join(map(shlex.quote, remote))
    return [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ServerAliveInterval=15", "-i", str(key), "-p",
        str(node.ssh_port), f"{node.ssh_user}@{node.host}",
        "bash", "-lc", shlex.quote(script),
    ]


def remote_report(node: Node, key: Path) -> dict:
    command = ssh_command(node, key, [
        ".venv/bin/python", "queue-haul/network_campaign.py", "node-check",
    ])
    return json.loads(_output(command).splitlines()[-1])


def validate_hosts(cluster: Cluster | None, reports: dict[str, dict]) -> None:
    expected = ({cluster.source.id: cluster.source,
                 **{node.id: node for node in cluster.destinations}}
                if cluster else None)
    if expected and set(reports) != set(expected):
        raise ValueError("host reports do not cover the cluster")
    for node_id, report in reports.items():
        node = expected[node_id] if expected else None
        if node and (report["region"].lower() != node.region.lower()
                     or report["private_ip"] != node.host):
            raise ValueError(f"{node_id} region or private IP changed")
        h100 = "H100" in ModelProfile.load(MODEL_PATH).hardware
        if report.get("dirty") or report.get("vm_size") not in (
                {"Standard_NC40ads_H100_v5"} if h100 else
                {"Standard_NC24ads_A100_v4"}) \
                or ("H100" if h100 else "A100") not in report.get("gpu", "") \
                or report.get("gpu_memory_mib", 0) < 80_000 \
                or not report.get("ptp") or not report.get("datadrive") \
                or float(report.get("clock_uncertainty_ms", 1e9)) \
                > CLOCK_LIMIT_MS:
            raise ValueError(f"{node_id} host contract failed")
    signatures = {(row.get("git_sha"), row.get("vllm"), row.get("lmcache"))
                  for row in reports.values()}
    if len(signatures) != 1 or any(
        row.get(name) != expected_version for row in reports.values()
        for name, expected_version in EXPECTED_RUNTIME.items()
    ):
        raise ValueError("host commit or runtime mismatch")


def host_check(cluster: Cluster, key: Path) -> dict[str, dict]:
    reports = {cluster.source.id: node_report()}
    reports.update({node.id: remote_report(node, key)
                    for node in cluster.destinations})
    validate_hosts(cluster, reports)
    return reports


def ping_rtt_ms(host: str, count: int) -> list[float]:
    text = _output(["ping", "-n", "-c", str(count), "-i", ".1", host])
    values = [float(value) for value in re.findall(r"time=([0-9.]+) ms", text)]
    if len(values) != count:
        raise RuntimeError(f"ping lost {count - len(values)} of {count} packets")
    return values


def _iperf_server(node: Node, key: Path, port: int) -> subprocess.Popen:
    return subprocess.Popen(
        ssh_command(node, key, ["iperf3", "-s", "-1", "-J", "-p", str(port)]),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def _iperf_client(host: str, port: int, seconds: int,
                  streams: int) -> subprocess.Popen:
    return subprocess.Popen([
        "iperf3", "-c", host, "-p", str(port), "-t", str(seconds),
        "-O", "5", "-P", str(streams), "-J",
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _wait_iperf_server(node: Node, key: Path, port: int) -> None:
    subprocess.run(ssh_command(node, key, [
        "timeout", "10", "bash", "-c",
        f"until ss -ltnH 'sport = :{port}' | grep -q .; do sleep .1; done",
    ]), check=True)


def _finish_iperf(client: subprocess.Popen, server: subprocess.Popen) -> dict:
    output, error = client.communicate()
    server_output, server_error = server.communicate(timeout=10)
    if client.returncode or server.returncode:
        raise RuntimeError(f"iperf3 failed: {error} {server_error}")
    raw = json.loads(output)
    iperf_mbps(raw)
    return {"client": raw, "server": json.loads(server_output)}


def calibrate(cluster: Cluster, key: Path, out: Path, seconds: int = 60,
              repeats: int = 3, ping_count: int = 200) -> dict:
    reports = host_check(cluster, key)
    paths = {node.id: {
        "rtt_ms": ping_rtt_ms(node.host, ping_count),
        "isolated_mbps": [], "simultaneous_mbps": [], "raw": [],
    } for node in cluster.destinations}
    port = 5201
    for node in cluster.destinations:
        for repeat in range(repeats):
            for streams in (1, 8):
                server = _iperf_server(node, key, port)
                _wait_iperf_server(node, key, port)
                raw = _finish_iperf(
                    _iperf_client(node.host, port, seconds, streams), server)
                paths[node.id]["raw"].append({
                    "kind": "isolated", "repeat": repeat,
                    "streams": streams, **raw,
                })
                if streams == 8:
                    paths[node.id]["isolated_mbps"].append(
                        iperf_mbps(raw["client"]))
    aggregate = []
    for repeat in range(repeats):
        servers = [_iperf_server(node, key, port)
                   for node in cluster.destinations]
        for node in cluster.destinations:
            _wait_iperf_server(node, key, port)
        clients = [_iperf_client(node.host, port, seconds, 8)
                   for node in cluster.destinations]
        rows = [_finish_iperf(client, server)
                for client, server in zip(clients, servers)]
        values = []
        for node, raw in zip(cluster.destinations, rows):
            value = iperf_mbps(raw["client"])
            values.append(value)
            paths[node.id]["simultaneous_mbps"].append(value)
            paths[node.id]["raw"].append({
                "kind": "simultaneous", "repeat": repeat,
                "streams": 8, **raw,
            })
        aggregate.append(sum(values))
    formal = (seconds, repeats, ping_count) == (60, 3, 200)
    result = {
        "schema": CALIBRATION_SCHEMA, "formal": formal, "hosts": reports,
        "clock_uncertainty_ms": {
            node: report["clock_uncertainty_ms"]
            for node, report in reports.items()
        },
        "paths": paths, "aggregate_simultaneous_mbps": aggregate,
    }
    validate_calibration(result)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def target_conditions(destinations: tuple[str, str] = ("east", "west")
                      ) -> list[dict]:
    first, second = destinations
    anchor = {
        "workload": "agentic_tool_loop", "bandwidth": "controlled_80",
        "background": {first: (.2, .2), second: (.2, .2)},
        "deadline_s": 30,
    }
    changes = (
        {}, {"background": {first: (0, 0), second: (0, 0)}},
        {"background": {first: (.2, .4), second: (.4, .2)}},
        {"background": {first: (.4, .2), second: (.2, .4)}},
        {"bandwidth": "controlled_40"}, {"bandwidth": "natural"},
        {"deadline_s": 19},
    )
    return [{**anchor, **change} for change in changes]


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bandwidths(contract: dict, label: str) -> dict[str, float]:
    return {node: row["natural_mbps"] if label == "natural" else
            row["controlled_mbps"][label.rsplit("_", 1)[-1]]
            for node, row in contract["paths"].items()}


def make_plan(manifest_path: Path, contract: dict, seed: int = 1,
              sessions: int = 8, design: str = "joint") -> dict:
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_manifest(manifest)
    available = sorted(manifest["sessions"], key=lambda row: row["id"])
    if design == "joint" and not 0 < sessions <= len(available) \
            or design == "isolated" and not available:
        raise ValueError("invalid session count")
    scenarios = []
    destinations = tuple(sorted(contract["paths"]))
    if design not in {"joint", "isolated", "frontier", "constraint", "separation"} \
            or len(destinations) != (
        2 if design in {"joint", "frontier", "constraint", "separation"} else 1
    ):
        raise ValueError(f"{design} design requires "
                         f"{'two destinations' if design in {'joint', 'frontier', 'constraint', 'separation'} else 'one destination'}")
    if design == "frontier":
        if not available:
            raise ValueError("frontier design needs a manifest template")
        load_pairs = [(load, load) for load in FRONTIER_LOADS]
        asymmetric_loads = (.5, .85, .9, .95)
        asymmetric = [(a, b) for a, b in (
            *((.5, load) for load in asymmetric_loads),
            *((load, .5) for load in asymmetric_loads[1:]),
        )]
        conditions = [
            (pack, loads) for pack in FRONTIER_PACKS
            for loads in (load_pairs[:1] if pack[0] == "32x31k" else load_pairs)
        ] + [
            (next(pack for pack in FRONTIER_PACKS if pack[0] == "8x16k"), loads)
            for loads in asymmetric
        ]
        for condition_index, ((pack_id, count, context), loads) in enumerate(conditions):
            session_rows = [{
                "session_id": f"{available[index % len(available)]['id']}-f{index}",
                "template_id": available[index % len(available)]["id"],
                "job_class": available[index % len(available)]["job_class"],
                "turn_index": 0, "initial_tokens": context, "order": index,
            } for index in range(count)]
            background = dict(zip(destinations, ((loads[0], 0), (loads[1], 0))))
            for policy in FRONTIER_POLICIES:
                scenarios.append({
                    "scenario_id": _hash([
                        design, condition_index, policy, session_rows,
                    ])[:16],
                    "design": design, "condition_index": condition_index,
                    "repeat": 0, "pack": pack_id, "policy": policy,
                    "workload": "agentic_tool_loop", "bandwidth": "natural",
                    "bandwidth_mbps": _bandwidths(contract, "natural"),
                    "deadline_s": HANDOFF_DEADLINE_S, "background": background,
                    "source_load": .8, "requested_shed_fraction": .8,
                    "planner_seed": profiler.stable_seed(
                        seed, condition_index, policy),
                    "sessions": session_rows,
                })
    elif design == "constraint":
        if destinations != ("east", "germany") or len(available) != 8:
            raise ValueError("constraint design requires East, Germany, and eight traces")
        workload = WorkloadProfile.load(WORKLOAD_PATHS["agentic_tool_loop"])
        support = tuple(sorted({row.context_tokens for row in workload.records}))
        packs = {}
        for pack_id, _deadline, count, context_seed, _target in CONSTRAINT_CELLS[:3]:
            rng = random.Random(context_seed)
            contexts = ([support[0]] * count if context_seed is None else
                        [rng.choice(support) for _ in range(count)])
            packs[pack_id] = [{
                "session_id": f"{available[index % len(available)]['id']}-{pack_id}-{index}",
                "template_id": available[index % len(available)]["id"],
                "job_class": available[index % len(available)]["job_class"],
                "turn_index": 0, "initial_tokens": context, "order": index,
            } for index, context in enumerate(contexts)]
        packs["quota-30"] = packs["window-30"]
        for condition_index, (condition_id, deadline, _count, context_seed,
                              target) in enumerate(CONSTRAINT_CELLS):
            headroom = ({"germany": {"replay": .25}}
                        if condition_id == "quota-30" else {})
            for policy in CONSTRAINT_POLICIES:
                session_rows = packs[condition_id]
                scenarios.append({
                    "scenario_id": _hash([
                        design, condition_index, policy, session_rows, headroom,
                        deadline, target, "max_shed",
                    ])[:16],
                    "design": design, "condition_index": condition_index,
                    "condition_id": condition_id, "repeat": 0,
                    "pack": condition_id, "policy": policy,
                    "workload": "agentic_tool_loop", "bandwidth": "natural",
                    "objective": "max_shed",
                    "bandwidth_mbps": _bandwidths(contract, "natural"),
                    "deadline_s": deadline,
                    "background": {"east": (.5, 0), "germany": (.95, 0)},
                    "source_load": .8, "requested_shed_fraction": target,
                    "migration_headroom": headroom,
                    "context_seed": context_seed,
                    "planner_seed": profiler.stable_seed(
                        seed, condition_index, policy),
                    "sessions": session_rows,
                })
    elif design == "separation":
        if destinations != ("east", "germany") or len(available) != 8:
            raise ValueError("separation design requires East, Germany, and eight traces")
        support = tuple(sorted({row.context_tokens for row in
                               WorkloadProfile.load(
                                   WORKLOAD_PATHS["agentic_tool_loop"]).records}))
        rng = random.Random(8)
        contexts = [rng.choice(support) for _ in range(28)]
        session_rows = [{
            "session_id": f"{available[index % len(available)]['id']}-separation-{index}",
            "template_id": available[index % len(available)]["id"],
            "job_class": available[index % len(available)]["job_class"],
            "turn_index": 0, "initial_tokens": context, "order": index,
        } for index, context in enumerate(contexts)]
        for condition_index, (condition_id, bandwidth, east, germany,
                              target) in enumerate(SEPARATION_CELLS):
            for repeat in range(SEPARATION_REPEATS):
                for policy in SEPARATION_POLICIES:
                    scenarios.append({
                        "scenario_id": _hash([
                            design, condition_index, repeat, policy,
                            session_rows, bandwidth, east, germany, target,
                        ])[:16],
                        "design": design, "condition_index": condition_index,
                        "condition_id": condition_id, "repeat": repeat,
                        "pack": "recorded-28-seed-8", "policy": policy,
                        "workload": "agentic_tool_loop", "bandwidth": bandwidth,
                        "bandwidth_mbps": _bandwidths(contract, bandwidth),
                        "deadline_s": SEPARATION_DEADLINE_S,
                        "planning_deadline_s": SEPARATION_PLANNING_DEADLINE_S,
                        "load_warmup_s": SEPARATION_LOAD_WARMUP_S,
                        "load_normalization": "destination_service",
                        "background": {"east": (east, 0),
                                       "germany": (germany, 0)},
                        "source_load": .8,
                        "requested_shed_fraction": target,
                        "objective": "max_shed", "migration_headroom": {},
                        "context_seed": 8,
                        "planner_seed": profiler.stable_seed(
                            seed, condition_index, repeat, policy),
                        "sessions": session_rows,
                    })
    elif design == "isolated":
        destination = destinations[0]
        for bandwidth in ("controlled_80", "controlled_40", "natural"):
            for context in (2048, 8192, 32768):
                for repeat in range(REPEATS):
                    rng = random.Random(profiler.stable_seed(seed, context, repeat))
                    row = rng.choice(available)
                    session = {"session_id": row["id"], "job_class": row["job_class"],
                               "turn_index": 0,
                               "initial_tokens": context - ISOLATED_PROMPT_HEADROOM_TOKENS,
                               "order": 0}
                    for method in ("replay", "kv_transfer"):
                        scenario_id = _hash([
                            design, destination, bandwidth, context, repeat, method, session,
                        ])[:16]
                        scenarios.append({
                            "scenario_id": scenario_id, "design": design,
                            "condition_index": context, "repeat": repeat,
                            "policy": method, "method": method,
                            "destination": destination, "workload": "agentic_tool_loop",
                            "bandwidth": bandwidth,
                            "bandwidth_mbps": _bandwidths(contract, bandwidth)[destination],
                            "context_size": context, "deadline_s": 180,
                            "background": {destination: (0, 0)},
                            "sessions": [session],
                            "moves": [{"session_id": row["id"], "method": method,
                                       "destination_instance": destination, "order": 0,
                                       "deadline_admitted": True}],
                        })
    else:
        for condition_index, condition in enumerate(target_conditions(destinations)):
            workload = WorkloadProfile.load(WORKLOAD_PATHS[condition["workload"]])
            for repeat in range(REPEATS):
                rng = random.Random(profiler.stable_seed(
                    seed, condition_index, repeat))
                chosen = rng.sample(available, sessions)
                contexts = policy_campaign._context_tokens(
                    workload, "uniform_support", sessions, rng)
                session_rows = [{
                    "session_id": row["id"], "job_class": row["job_class"],
                    "turn_index": 0, "initial_tokens": contexts[index],
                    "order": index,
                } for index, row in enumerate(chosen)]
                for policy in POLICIES:
                    scenario_id = _hash([
                        design, condition_index, repeat, policy, session_rows,
                    ])[:16]
                    scenarios.append({
                        "scenario_id": scenario_id, "design": design,
                        "condition_index": condition_index, "repeat": repeat,
                        **condition, "policy": policy,
                        "planner_seed": profiler.stable_seed(
                            seed, condition_index, repeat, policy),
                        "bandwidth_mbps": _bandwidths(contract, condition["bandwidth"]),
                        "sessions": session_rows,
                    })
    rng = random.Random(seed)
    if design in {"constraint", "separation"}:
        cells = CONSTRAINT_CELLS if design == "constraint" else SEPARATION_CELLS
        blocks = [[row for row in scenarios if row["condition_index"] == index
                   and (design == "constraint" or row["repeat"] == repeat)]
                  for index in range(len(cells))
                  for repeat in range(1 if design == "constraint"
                                      else SEPARATION_REPEATS)]
        for block in blocks:
            rng.shuffle(block)
        scenarios = [row for block in blocks for row in block]
    else:
        for bandwidth in ("controlled_80", "controlled_40", "natural"):
            rows = [row for row in scenarios if row["bandwidth"] == bandwidth]
            rng.shuffle(rows)
            scenarios = [row for row in scenarios if row["bandwidth"] != bandwidth]
            scenarios.extend(rows)
    output = {
        "schema": PLAN_SCHEMA, "design": design, "seed": seed,
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "model_profile": {"path": str(MODEL_PATH),
                          "sha256": profiler.file_hash(MODEL_PATH)},
        "network_contract": contract,
        "policies": list(FRONTIER_POLICIES if design == "frontier" else
                         CONSTRAINT_POLICIES if design == "constraint" else
                         SEPARATION_POLICIES if design == "separation" else POLICIES),
        "conditions": target_conditions(destinations) if design == "joint" else
            [{"condition_id": row[0], "deadline_s": row[1],
              "sessions": row[2], "context_seed": row[3],
              "requested_shed_fraction": row[4]}
             for row in CONSTRAINT_CELLS] if design == "constraint" else
            [{"condition_id": row[0], "bandwidth": row[1],
              "background": {"east": row[2], "germany": row[3]},
              "deadline_s": SEPARATION_DEADLINE_S,
              "planning_deadline_s": SEPARATION_PLANNING_DEADLINE_S,
              "requested_shed_fraction": row[4]}
             for row in SEPARATION_CELLS] if design == "separation" else [],
        "repeats": 1 if design in {"frontier", "constraint"} else
            SEPARATION_REPEATS if design == "separation" else REPEATS,
        "sessions_per_scenario": None if design in {"constraint", "separation"}
            else sessions,
        "scenarios": scenarios,
    }
    if design == "separation":
        output["load_support"] = {
            "path": str(LOAD_SUPPORT_PATH.relative_to(ROOT)),
            "sha256": profiler.file_hash(LOAD_SUPPORT_PATH),
        }
    validate_plan(output)
    return output


def validate_plan(plan: dict) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid network plan schema")
    scenarios = plan.get("scenarios", [])
    design = plan.get("design")
    expected = 126 if design == "joint" else 54 if design == "isolated" \
        else ((sum(1 if pack[0] == "32x31k" else len(FRONTIER_LOADS)
                   for pack in FRONTIER_PACKS) + 7)
              * len(FRONTIER_POLICIES)
              if plan.get("phase", "pilot") == "pilot" else
              len(scenarios)) if design == "frontier" \
        else 24 if design == "constraint" \
        else len(SEPARATION_CELLS) * SEPARATION_REPEATS \
        * len(SEPARATION_POLICIES) if design == "separation" \
        else HARDWARE_GAP_REPEATS * sum(
            len(row[-1]) for row in HARDWARE_GAP_MATRIX
        ) if design == "hardware_gap" else 0
    if len(scenarios) != expected \
            or len({row["scenario_id"] for row in scenarios}) != len(scenarios):
        raise ValueError(f"network plan must contain exactly {expected} unique scenarios")
    if design == "frontier":
        contract = plan["network_contract"]
        blocks = {}
        for row in scenarios:
            key = (row["condition_index"], row["repeat"])
            blocks.setdefault(key, []).append(row)
            if row["bandwidth"] != "natural" \
                    or row["bandwidth_mbps"] != _bandwidths(contract, "natural") \
                    or row["deadline_s"] != 30 \
                    or row["requested_shed_fraction"] != .8:
                raise ValueError("frontier scenario contract changed")
        policies = ({DEADLINE_BLIND_POLICY}
                    if plan.get("phase") == "deadline_blind"
                    else set(FRONTIER_POLICIES))
        if any({row["policy"] for row in rows} != policies
               or len({tuple((item["session_id"], item["initial_tokens"])
                              for item in row["sessions"]) for row in rows}) != 1
               for rows in blocks.values()):
            raise ValueError("frontier policy block is incomplete or unmatched")
        return
    if design == "constraint":
        if set(plan["network_contract"]["paths"]) != {"east", "germany"}:
            raise ValueError("constraint plan requires East and Germany")
        manifest_path = Path(plan["manifest"]["path"])
        if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
            raise ValueError("constraint manifest changed")
        templates = tuple(row["id"] for row in sorted(
            json.loads(manifest_path.read_text())["sessions"],
            key=lambda row: row["id"],
        ))
        if len(templates) != 8:
            raise ValueError("constraint design requires eight traces")
        support = (14_042, 30_785, 31_547)
        contexts = {}
        for condition_id, _deadline, count, context_seed, _target \
                in CONSTRAINT_CELLS[:3]:
            rng = random.Random(context_seed)
            contexts[condition_id] = ([support[0]] * count
                                      if context_seed is None else
                                      [rng.choice(support) for _ in range(count)])
        contexts["quota-30"] = contexts["window-30"]
        blocks = {}
        for row in scenarios:
            blocks.setdefault(row["condition_index"], []).append(row)
            if row["bandwidth"] != "natural" \
                    or row["bandwidth_mbps"] != _bandwidths(
                        plan["network_contract"], "natural") \
                    or row["background"] != {"east": [.5, 0],
                                             "germany": [.95, 0]} \
                    and row["background"] != {"east": (.5, 0),
                                              "germany": (.95, 0)}:
                raise ValueError("constraint route or load contract changed")
        expected_cells = {
            0: ("window-19", 19, 22, 513_650, 15, 1.0, {}),
            1: ("window-30", 30, 28, 648_131, 8, 1.0, {}),
            2: ("window-60", 60, 64, 898_688, None, 1.0, {}),
            3: ("quota-30", 30, 28, 648_131, 8, 1.0,
                {"germany": {"replay": .25}}),
        }
        signatures = {}
        for index, (condition_id, deadline, count, tokens, context_seed, target,
                    headroom) in expected_cells.items():
            rows = blocks.get(index, [])
            signatures[index] = {tuple(
                (item["session_id"], item["template_id"], item["initial_tokens"])
                for item in row["sessions"]
            ) for row in rows}
            if len(rows) != len(CONSTRAINT_POLICIES) \
                    or {row["policy"] for row in rows} != set(CONSTRAINT_POLICIES) \
                    or len(signatures[index]) != 1 \
                    or any(row["condition_id"] != condition_id
                           or row["deadline_s"] != deadline
                           or row["workload"] != "agentic_tool_loop"
                           or row["objective"] != "max_shed"
                           or row["source_load"] != .8
                           or row["context_seed"] != context_seed
                           or len(row["sessions"]) != count
                           or sum(item["initial_tokens"] for item in row["sessions"])
                           != tokens
                           or [item["initial_tokens"] for item in row["sessions"]]
                           != contexts[condition_id]
                           or [item["template_id"] for item in row["sessions"]]
                           != [templates[i % len(templates)] for i in range(count)]
                           or [item["order"] for item in row["sessions"]]
                           != list(range(count))
                           or row["requested_shed_fraction"] != target
                           or row["migration_headroom"] != headroom
                           for row in rows):
                raise ValueError("constraint policy block changed")
        if {tuple((item[1], item[2]) for item in signature)
                for signature in signatures[1]} != {
                    tuple((item[1], item[2]) for item in signature)
                    for signature in signatures[3]}:
            raise ValueError("quota counterfactual must reuse the 30-second pack")
        return
    if design == "separation":
        if set(plan["network_contract"]["paths"]) != {"east", "germany"}:
            raise ValueError("separation plan requires East and Germany")
        if plan.get("load_support", {}).get("sha256") \
                != profiler.file_hash(LOAD_SUPPORT_PATH):
            raise ValueError("separation load support changed")
        manifest_path = Path(plan["manifest"]["path"])
        if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
            raise ValueError("separation manifest changed")
        templates = tuple(row["id"] for row in sorted(
            json.loads(manifest_path.read_text())["sessions"],
            key=lambda row: row["id"],
        ))
        if len(templates) != 8:
            raise ValueError("separation design requires eight traces")
        rng = random.Random(8)
        contexts = [rng.choice((14_042, 30_785, 31_547)) for _ in range(28)]
        signature = None
        for index, (condition_id, bandwidth, east, germany,
                    target) in enumerate(SEPARATION_CELLS):
            for repeat in range(SEPARATION_REPEATS):
                rows = [row for row in scenarios if (
                    row["condition_index"], row["repeat"]
                ) == (index, repeat)]
                signatures = {tuple(
                    (item["session_id"], item["template_id"],
                     item["initial_tokens"])
                    for item in row["sessions"]
                ) for row in rows}
                signature = signature or next(iter(signatures), None)
                if len(rows) != len(SEPARATION_POLICIES) \
                        or {row["policy"] for row in rows} \
                        != set(SEPARATION_POLICIES) \
                        or signatures != {signature} \
                        or any(
                            row["scenario_id"] != _hash([
                                design, index, repeat, row["policy"],
                                row["sessions"], bandwidth, east, germany,
                                target,
                            ])[:16]
                            or row["condition_id"] != condition_id
                            or row["bandwidth"] != bandwidth
                            or row["bandwidth_mbps"] != _bandwidths(
                                plan["network_contract"], bandwidth)
                            or tuple(row["background"]["east"]) != (east, 0)
                            or tuple(row["background"]["germany"]) != (germany, 0)
                            or row["deadline_s"] != SEPARATION_DEADLINE_S
                            or row["planning_deadline_s"]
                            != SEPARATION_PLANNING_DEADLINE_S
                            or row["load_warmup_s"] != SEPARATION_LOAD_WARMUP_S
                            or row["load_normalization"] != "destination_service"
                            or row["requested_shed_fraction"] != target
                            or row["objective"] != "max_shed"
                            or row["migration_headroom"] != {}
                            or row["context_seed"] != 8
                            or row["source_load"] != .8
                            or row["planner_seed"] != profiler.stable_seed(
                                plan["seed"], index, repeat, row["policy"])
                            or [item["initial_tokens"] for item in row["sessions"]]
                            != contexts
                            or [item["template_id"] for item in row["sessions"]]
                            != [templates[i % len(templates)] for i in range(28)]
                            for row in rows
                        ):
                    raise ValueError("separation policy block changed")
        return
    if design == "hardware_gap":
        if set(plan["network_contract"]["paths"]) != {"east", "germany"} \
                or plan.get("policies") != list(HARDWARE_GAP_POLICIES):
            raise ValueError("hardware gap plan requires East and Germany")
        for key in ("parent_plan", "oracle_plans"):
            pin = plan.get(key, {})
            if profiler.file_hash(Path(pin.get("path", ""))) \
                    != pin.get("sha256"):
                raise ValueError(f"hardware gap {key} changed")
        signature = None
        for index, (state, germany, kv, bandwidth, policies) \
                in enumerate(HARDWARE_GAP_MATRIX):
            capacity = {"east": round(1 - kv, 10), "germany": 1}
            for repeat in range(HARDWARE_GAP_REPEATS):
                rows = [row for row in scenarios if (
                    row["condition_index"], row["repeat"]
                ) == (index, repeat)]
                signatures = {tuple(
                    (item["session_id"], item["template_id"],
                     item["initial_tokens"]) for item in row["sessions"]
                ) for row in rows}
                signature = signature or next(iter(signatures), None)
                if len(rows) != len(policies) \
                        or {row["policy"] for row in rows} != set(policies) \
                        or signatures != {signature} \
                        or any(
                            row["scenario_id"] != _hash([
                                design, state, repeat, row["policy"],
                                row.get("moves", []), bandwidth, germany, kv,
                                HARDWARE_GAP_TARGET_FRACTION,
                            ])[:16]
                            or row["condition_id"] != state
                            or row["bandwidth"] != bandwidth
                            or row["bandwidth_mbps"] != _bandwidths(
                                plan["network_contract"], bandwidth)
                            or tuple(row["background"]["east"]) != (.25, kv)
                            or tuple(row["background"]["germany"]) \
                            != (germany, 0)
                            or row["kv_capacity_fraction"] != capacity
                            or row["deadline_s"] != SEPARATION_DEADLINE_S
                            or row["planning_deadline_s"] \
                            != SEPARATION_PLANNING_DEADLINE_S
                            or row["load_warmup_s"] != SEPARATION_LOAD_WARMUP_S
                            or row["load_normalization"] \
                            != "destination_service"
                            or row["requested_shed_fraction"] \
                            != HARDWARE_GAP_TARGET_FRACTION
                            or row["objective"] != "max_shed"
                            or row["source_load"] != .8
                            or row["admission_mode"] != "normal"
                            or row["full_horizon_s"] \
                            != ORACLE_STALE_HORIZON_S
                            or row["background_kv_headroom_tokens"] != {
                                "east": HARDWARE_GAP_BACKGROUND_KV_TOKENS,
                                "germany": HARDWARE_GAP_BACKGROUND_KV_TOKENS,
                            }
                            or row["planning_state"] != (
                                "all-bind" if row["policy"]
                                == "queue_haul_robust" else
                                "all-release" if row["policy"]
                                == "queue_haul_stale" else
                                "all-bind-90s" if row["policy"]
                                == DEADLINE_BLIND_POLICY else state)
                            or len(row["sessions"]) != 28
                            or sum(item["initial_tokens"]
                                   for item in row["sessions"]) != 648_131
                            or row["expected_admission"] != (not (
                                state == "all-bind"
                                and row["policy"] == "queue_haul_stale"))
                            or bool(row.get("moves")) != (
                                row["policy"].startswith("oracle_")
                                or row["policy"] in {
                                    "queue_haul_robust",
                                    "queue_haul_stale",
                                    DEADLINE_BLIND_POLICY,
                                })
                            for row in rows
                        ):
                    raise ValueError("hardware gap policy block changed")
        return
    if design == "isolated":
        if len(plan["network_contract"]["paths"]) != 1 or any(
            row.get("destination") not in plan["network_contract"]["paths"]
            or len(row.get("moves", ())) != 1 for row in scenarios
        ):
            raise ValueError("invalid isolated plan")
        return
    contract = plan["network_contract"]
    for condition_index in range(len(target_conditions())):
        for repeat in range(REPEATS):
            rows = [row for row in scenarios if (
                row["condition_index"], row["repeat"]
            ) == (condition_index, repeat)]
            signatures = {tuple((item["session_id"], item["initial_tokens"])
                                for item in row["sessions"]) for row in rows}
            if {row["policy"] for row in rows} != set(POLICIES) \
                    or len(signatures) != 1:
                raise ValueError("policy block is incomplete or unmatched")
            for row in rows:
                if row["bandwidth_mbps"] != _bandwidths(contract, row["bandwidth"]) \
                        or "destination" in row or "moves" in row:
                    raise ValueError("scenario route or move contract changed")


def active_scheduled_events(raw: dict) -> list[dict]:
    events = raw.get("Events")
    if not isinstance(events, list):
        raise ValueError("invalid Azure Scheduled Events response")
    return events


class ScheduledEventMonitor:
    url = "http://169.254.169.254/metadata/scheduledevents?api-version=2020-07-01"

    def __init__(self, path: Path, stop: threading.Event | None = None):
        self.path, self.stop = path, stop or threading.Event()
        self.events: list[dict] = []
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _get(self, timeout: int) -> dict:
        request = urllib.request.Request(self.url, headers={"Metadata": "true"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)

    def _run(self) -> None:
        incarnation = None
        try:
            with self.path.open("w", buffering=1) as handle:
                while not self.stop.wait(1):
                    raw = self._get(125 if incarnation is None else 5)
                    events = active_scheduled_events(raw)
                    current = raw.get("DocumentIncarnation")
                    if events or current != incarnation:
                        handle.write(json.dumps({
                            "monotonic_ns": time.monotonic_ns(),
                            "wall_ns": time.time_ns(), **raw,
                        }, separators=(",", ":")) + "\n")
                    incarnation = current
                    if events:
                        self.events.extend(events)
                        self.stop.set()
        except BaseException as exc:
            self.error = exc
            self.stop.set()

    def check(self) -> None:
        if self.error:
            raise RuntimeError("Azure Scheduled Events monitor failed") \
                from self.error
        if self.events:
            raise RuntimeError(f"Azure Spot event received: {self.events}")

    def close(self) -> None:
        self.stop.set()
        self.thread.join(130)
        self.check()


def cluster_routes(cluster: Cluster) -> tuple[list[testbed.Route], dict]:
    routes, ports = [], {}
    for index, node in enumerate(sorted(cluster.destinations,
                                        key=lambda value: value.id), start=1):
        kv, api = 8300 + index, 8400 + index
        ports[node.id] = {"kv": kv, "api": api}
        routes.extend((
            testbed.Route(
                f"kv/{node.id}", cluster.source.host, kv,
                "127.0.0.1", 5655, "resp"),
            testbed.Route(
                f"api/{node.id}", "127.0.0.1", api,
                node.host, 8200),
        ))
    return routes, ports


def bandwidth_limits(contract: dict, label: str
                     ) -> tuple[float | None, dict[str, float]]:
    if label == "natural":
        return None, {}
    if label not in {"controlled_40", "controlled_80"}:
        raise ValueError(f"unknown bandwidth condition: {label}")
    percent = label.rsplit("_", 1)[-1]
    return (
        float(contract["aggregate"]["controlled_mbps"][percent]),
        {node: float(row["controlled_mbps"][percent])
         for node, row in contract["paths"].items()},
    )


def proxy_command(routes: list[testbed.Route], aggregate_mbps: float | None,
                  route_mbps: dict[str, float], log: Path) -> list[str]:
    command = [
        sys.executable, "queue-haul/migration_testbed.py", "proxy",
        "--routes-json", json.dumps([asdict(route) for route in routes]),
        "--route-mbps-json", json.dumps(route_mbps), "--log", str(log),
    ]
    if aggregate_mbps:
        command += ["--aggregate-mbps", str(aggregate_mbps)]
    return command


def vllm_kv_capacity(path: Path) -> int:
    match = re.search(r"GPU KV cache size:\s+([\d,]+) tokens",
                      path.read_text(errors="ignore"))
    if not match:
        raise RuntimeError("vLLM did not report its GPU KV capacity")
    return int(match.group(1).replace(",", ""))


def node_serve(node_id: str, bind_host: str, source_host: str, kv_port: int,
               run_root: Path, power_interval_s: float = .25,
               kv_blocks: int | None = None) -> None:
    import migration_profiler

    cfg = testbed.Config(host="127.0.0.1")
    testbed.preflight(cfg, 1)
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "node-serve.pid").write_text(str(os.getpid()))
    cache = sink = None
    sampler = migration_profiler.PowerSampler(
        run_root / "power.csv", power_interval_s)
    stopped = threading.Event()
    spot = ScheduledEventMonitor(run_root / "scheduled_events.jsonl", stopped)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_args: stopped.set())
    try:
        spot.start()
        cache_log = run_root / "lmcache-sink.log"
        cache = testbed.start_logged(testbed.mp_server_cmd(
            cfg, "sink", bind_host="127.0.0.1", http_host=bind_host,
            l2_host=source_host, l2_port=kv_port,
        ), cache_log)
        testbed.wait_tcp_process(
            "127.0.0.1", cfg.sink_lmc_port, 300, cache, cache_log)
        sink_log = run_root / "sink.log"
        extra = (["--num-gpu-blocks-override", str(kv_blocks)]
                 if kv_blocks is not None else [])
        sink = testbed.start_logged(testbed.vllm_cmd(
            cfg, "sink", extra, gpu_index=0, bind_host=bind_host), sink_log)
        testbed.wait_health_process(
            bind_host, cfg.sink_port, testbed.health_timeout(), sink, sink_log)
        sampler.start()
        print(json.dumps({
            "status": "ready", "node_id": node_id, "host": bind_host,
            "vllm_port": cfg.sink_port, "kv_port": kv_port,
            "kv_capacity_tokens": vllm_kv_capacity(sink_log),
            "monotonic_ns": time.monotonic_ns(), "wall_ns": time.time_ns(),
        }, sort_keys=True), flush=True)
        stopped.wait()
    finally:
        if sampler.thread.is_alive():
            sampler.close()
        for process in (sink, cache):
            if process:
                testbed.stop_proc(process)
        spot.close()


@dataclass
class ClusterStack:
    cluster: Cluster
    cfg: testbed.Config
    local: testbed.Stack
    sampler: object
    remote: dict[str, subprocess.Popen]
    remote_roots: dict[str, Path]
    ports: dict
    run_root: Path
    key: Path
    spot: ScheduledEventMonitor
    node_reports: dict[str, dict]


def _remote_ready(process: subprocess.Popen, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("remote sink exited before readiness")
        ready, _, _ = select.select([process.stdout], [], [], 1)
        if ready:
            line = process.stdout.readline()
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            if report.get("status") == "ready":
                return report
    raise TimeoutError("remote sink readiness timed out")


def _wait_remote_ready(remote: dict[str, subprocess.Popen],
                       timeout_s: float) -> dict[str, dict]:
    with ThreadPoolExecutor(max_workers=len(remote)) as pool:
        futures = {node_id: pool.submit(_remote_ready, process, timeout_s)
                   for node_id, process in remote.items()}
        return {node_id: futures[node_id].result()
                for node_id in sorted(futures)}


def _stop_remote(node: Node, key: Path, root: Path,
                 process: subprocess.Popen) -> None:
    if process.poll() is None:
        subprocess.run(ssh_command(node, key, [
            "python3", "-c",
            "import os,sys; os.kill(int(open(sys.argv[1]).read()), 15)",
            str(root / "node-serve.pid"),
        ]), check=True)
        process.wait(timeout=30)


def start_cluster(cluster: Cluster, key: Path, contract: dict,
                  bandwidth: str, run_root: Path,
                  power_interval_s: float = .25,
                  kv_capacity_fraction: dict[str, float] | None = None
                  ) -> ClusterStack:
    import migration_profiler

    cfg = testbed.Config(host="127.0.0.1")
    testbed.preflight(cfg, 1)
    run_root.mkdir(parents=True, exist_ok=False)
    routes, ports = cluster_routes(cluster)
    aggregate, rates = bandwidth_limits(contract, bandwidth)
    lmc = proxy = source = None
    services, remote = [], {}
    sampler = migration_profiler.PowerSampler(
        run_root / "power.csv", power_interval_s)
    spot = ScheduledEventMonitor(run_root / "scheduled_events.jsonl")
    fractions = kv_capacity_fraction or {}
    if any(node not in {item.id for item in cluster.destinations}
           or not 0 < fraction <= 1 for node, fraction in fractions.items()):
        raise ValueError("invalid destination KV capacity fraction")
    profile_capacity = ModelProfile.load(MODEL_PATH).kv_capacity_tokens
    try:
        spot.start()
        lmc_log = run_root / "redis.log"
        lmc = testbed.start_logged(testbed.redis_cmd(cfg), lmc_log)
        testbed.wait_tcp_process("127.0.0.1", cfg.lmc_port, 60, lmc, lmc_log)
        proxy_log = run_root / "proxy.log"
        proxy = testbed.start_logged(proxy_command(
            routes, aggregate, rates, run_root / "proxy_bytes.csv"), proxy_log)
        for route in routes:
            testbed.wait_tcp_process(
                route.listen_host, route.listen_port, 30, proxy, proxy_log)
        cache_log = run_root / "lmcache-source.log"
        cache = testbed.start_logged(testbed.mp_server_cmd(
            cfg, "source", l2_host="127.0.0.1", l2_port=cfg.lmc_port,
        ), cache_log)
        services.append(cache)
        testbed.wait_tcp_process(
            "127.0.0.1", cfg.src_lmc_port, 300, cache, cache_log)
        source_log = run_root / "source.log"
        source = testbed.start_logged(testbed.vllm_cmd(
            cfg, "source", gpu_index=0, sleep_mode=True), source_log)
        testbed.wait_health_process(
            "127.0.0.1", cfg.src_port, testbed.health_timeout(),
            source, source_log)
        sampler.start()
        remote_roots = {}
        by_id = {node.id: node for node in cluster.destinations}
        for node_id in sorted(by_id):
            node = by_id[node_id]
            remote_root = Path(node.run_root) / run_root.name / node_id
            remote_roots[node_id] = remote_root
            command = ssh_command(node, key, [
                "env", *(f"{name}={os.environ[name]}" for name in RUNTIME_ENV
                         if name in os.environ),
                ".venv/bin/python", "queue-haul/network_campaign.py",
                "node-serve", "--node-id", node_id, "--bind-host", node.host,
                "--source-host", cluster.source.host, "--kv-port",
                str(ports[node_id]["kv"]), "--run-root", str(remote_root),
                "--power-interval-s", str(power_interval_s),
                *(["--kv-blocks", str(round(
                    profile_capacity * fractions[node_id] / KV_BLOCK_SIZE))]
                  if fractions.get(node_id, 1) < 1 else []),
            ])
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            remote[node_id] = process
        reports = _wait_remote_ready(remote, testbed.health_timeout())
        if any(abs(reports[node]["kv_capacity_tokens"] / profile_capacity
                   - fraction) > .01 for node, fraction in fractions.items()):
            raise RuntimeError("destination KV quota differs from hardware")
        local = testbed.Stack(
            lmc, proxy, source, None, run_root, services,
            aggregate or 0,
        )
        return ClusterStack(
            cluster, cfg, local, sampler, remote, remote_roots, ports,
            run_root, key, spot, reports,
        )
    except BaseException:
        for node_id, process in remote.items():
            _stop_remote(by_id[node_id], key, remote_roots[node_id], process)
        if sampler.thread.is_alive():
            sampler.close()
        for process in (source, proxy, *services, lmc):
            if process:
                testbed.stop_proc(process)
        spot.close()
        raise


def _scp(node: Node, key: Path, source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run([
        "scp", "-q", "-r", "-o", "BatchMode=yes", "-o",
        "StrictHostKeyChecking=yes", "-i", str(key), "-P", str(node.ssh_port),
        f"{node.ssh_user}@{node.host}:{source}/.", str(destination),
    ], check=True)


def stop_cluster(stack: ClusterStack, collect: bool = True) -> None:
    nodes = {node.id: node for node in stack.cluster.destinations}
    for node_id, process in stack.remote.items():
        _stop_remote(nodes[node_id], stack.key, stack.remote_roots[node_id],
                     process)
    if stack.sampler.thread.is_alive():
        stack.sampler.close()
    testbed.stop_stack(stack.local)
    stack.spot.close()
    if collect:
        for node_id, root in stack.remote_roots.items():
            _scp(nodes[node_id], stack.key, root,
                 stack.run_root / "nodes" / node_id)


def smoke(cluster: Cluster, key: Path, calibration: dict, bandwidth: str,
          run_root: Path, words: int = 1024) -> dict:
    contract = freeze_contract(calibration)
    stack = start_cluster(cluster, key, contract, bandwidth, run_root)
    report = None
    try:
        prompt = testbed.prompt_text(f"network-smoke-{time.time_ns()}", words)
        source, _ = testbed.warm_source(stack.cfg, run_root, prompt)
        results = {}
        for node in sorted(cluster.destinations, key=lambda value: value.id):
            before = testbed.proxy_counts(run_root / "proxy_bytes.csv")
            result = testbed.post_chat(
                stack.cfg, stack.ports[node.id]["api"], prompt, 4)
            testbed.check_chat(result, f"{node.id} KV continuation")
            time.sleep(1)
            delta = testbed.count_delta(
                before, testbed.proxy_counts(run_root / "proxy_bytes.csv"))
            key_name = f"kv/{node.id}/target_to_client"
            cached = int((result["usage"].get("prompt_tokens_details") or {})
                         .get("cached_tokens", 0))
            if delta.get(key_name, 0) <= 0 or cached <= 0:
                raise RuntimeError(f"{node.id} did not reconstruct remote KV")
            results[node.id] = {
                "request": result, "wire_bytes": delta[key_name],
                "cached_tokens": cached,
            }
        testbed.set_source_sleep(stack.cfg, True)
        time.sleep(5)
        testbed.set_source_sleep(stack.cfg, False)
        report = {
            "schema": "queue-haul-network-smoke-v1", "status": "complete",
            "bandwidth": bandwidth, "source": source,
            "destinations": results,
        }
        (run_root / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n")
        return report
    finally:
        stop_cluster(stack)


def _clear_cluster(stack: ClusterStack) -> None:
    testbed.set_source_sleep(stack.cfg, False)
    with __import__("socket").create_connection((
            stack.cfg.host, stack.cfg.lmc_port)) as sock:
        sock.sendall(b"*1\r\n$8\r\nFLUSHALL\r\n")
        if not sock.recv(64).startswith(b"+OK"):
            raise RuntimeError("Redis FLUSHALL failed")
    testbed.http_text(stack.cfg.host, stack.cfg.src_lmc_http_port,
                       "POST", "/cache/clear")
    testbed.http_text(stack.cfg.host, stack.cfg.src_port,
                       "POST", "/reset_prefix_cache")
    for node in stack.cluster.destinations:
        testbed.http_text(node.host, stack.cfg.sink_lmc_http_port,
                           "POST", "/cache/clear")
        testbed.http_text(node.host, stack.cfg.sink_port,
                           "POST", "/reset_prefix_cache")


def _chat(cfg: testbed.Config, port: int, messages: list[dict], code: str,
          timeout_s: float, bypass_lmcache: bool = False) -> dict:
    messages = messages + [{"role": "user", "content":
                            f"Reply only with session state code {code}."}]
    for attempt in range(2):
        result, text = profiler.stream_chat(
            cfg, port, messages, 128, profiler.messages_hash(messages), timeout_s,
            bypass_lmcache)
        if result.status_code == 200 and code in text:
            return {**asdict(result), "state_code_verified": True,
                    "probe_attempts": attempt + 1}
    raise RuntimeError(
        f"session reconstruction failed after 2 probes: HTTP "
        f"{result.status_code}, response={text[:200]!r}")


def _warm(stack: ClusterStack, messages: list[dict], code: str,
          timeout_s: float) -> dict:
    log = stack.run_root / "lmcache-source.log"
    offset = log.stat().st_size
    result = _chat(stack.cfg, stack.cfg.src_port, messages, code, timeout_s)
    testbed.mp_wait_stored(
        log, offset, result["prompt_tokens"] // 256 * 256)
    return result


def _csv_window(path: Path, start_ns: int, end_ns: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row["start_ns"]) >= start_ns
            and int(row["end_ns"]) <= end_ns]


def summarize_metrics(samples: list[str], target_kv: float,
                      capacity_fraction: float = 1) -> dict:
    if not 0 < capacity_fraction <= 1.01:
        raise ValueError("invalid measured KV capacity fraction")
    names = ("kv_cache_usage_perc", "num_requests_running", "num_requests_waiting")
    values = {name: [] for name in names}
    for sample in samples:
        for name in names:
            match = re.search(rf"^vllm:{name}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$",
                              sample, re.MULTILINE)
            if not match:
                raise ValueError(f"missing vLLM metric {name}")
            values[name].append(float(match.group(1)))
    engine_fraction = statistics.median(values["kv_cache_usage_perc"])
    result = {
        "kv_fraction": max(0, min(
            1, 1 - capacity_fraction * (1 - engine_fraction))),
        "engine_kv_fraction": engine_fraction,
        "kv_capacity_fraction": capacity_fraction,
        "running": statistics.median(values["num_requests_running"]),
        "waiting": max(values["num_requests_waiting"]),
    }
    result["warning"] = abs(result["kv_fraction"] - target_kv) > .05 \
        or result["waiting"] > 0
    return result


def destination_metrics(stack: ClusterStack, node_id: str,
                        target_kv: float,
                        expected_capacity_fraction: float = 1) -> dict:
    samples = []
    for index in range(5):
        samples.append(testbed.http_text(
            stack.cfg.host, stack.ports[node_id]["api"], "GET", "/metrics"))
        if index < 4:
            time.sleep(1)
    capacity = stack.node_reports[node_id]["kv_capacity_tokens"]
    fraction = capacity / ModelProfile.load(MODEL_PATH).kv_capacity_tokens
    result = summarize_metrics(samples, target_kv, fraction)
    result["kv_capacity_tokens"] = capacity
    result["warning"] |= abs(fraction - expected_capacity_fraction) > .01
    return result


def destination_background_work(dtype, rho: float) -> tuple[float, float]:
    service_s = (
        SINK_LOAD_PREFILL_TOKENS
        / dtype.prefill.at(SINK_LOAD_PREFILL_TOKENS)
        + SINK_LOAD_DECODE_TOKENS
        / dtype.decode.at(SINK_LOAD_PREFILL_TOKENS)
    )
    rate = rho / service_s
    return tuple(dtype.work(
        rate * SINK_LOAD_PREFILL_TOKENS,
        rate * SINK_LOAD_DECODE_TOKENS,
        SINK_LOAD_PREFILL_TOKENS,
    ))


def joint_problem(scenario: dict, snapshots: dict[str, dict],
                  profile: ModelProfile,
                  demand: dict[str, tuple[float, float]] | None = None):
    base, _ = policy_campaign._problem(
        profile, scenario["sessions"], 1,
        scenario.get("planning_deadline_s", scenario["deadline_s"]))
    if demand is not None:
        base = replace(base, sessions=tuple(replace(
            session, expected_f=demand[session.session_id][0],
            expected_g=demand[session.session_id][1],
        ) for session in base.sessions))
    requested_shed_w = None
    if "requested_shed_fraction" in scenario:
        initial = source_power(base, profile)
        minimum = source_power(
            base, profile, (session.session_id for session in base.sessions))
        requested_shed_w = scenario["requested_shed_fraction"] * (initial - minimum)
        base = replace(base, power_limit_w=initial - requested_shed_w)
    destinations = tuple(sorted(scenario["bandwidth_mbps"]))
    links = tuple(NetworkLink(
        f"link/{node}", scenario["bandwidth_mbps"][node] * 125_000,
    ) for node in destinations)
    problem = replace(
        base,
        nodes=(base.nodes[0], *(PowerNode(f"{node}-node", 1, False)
                                for node in destinations)),
        instances=(base.instances[0], *(ServingInstance(
            node, (f"{node}-node",)) for node in destinations)),
        links=links,
    )
    template = dedicated_sink_architecture(profile, destinations[0],
                                           (links[0].link_id,))
    dtype = template.types[0]
    pools = tuple(DestinationPool(
        f"pool/{node}", dtype.type_id,
        (DestinationReplica(
            node, destination_background_work(
                dtype, float(scenario["background"][node][0]))
            if scenario.get("load_normalization") == "destination_service"
            else tuple(dtype.work(
                profile.case().F * float(scenario["background"][node][0]),
                0, 512)),
            round(snapshots[node]["kv_fraction"] * dtype.kv_capacity_tokens),
        ),), f"route/{node}", (f"link/{node}",),
        migration_headroom=scenario.get("migration_headroom", {}).get(node),
    ) for node in destinations)
    architecture = DestinationArchitecture(
        template.schema, template.source_compatibility, template.types, pools)
    routes = {("source", node): (f"link/{node}",) for node in destinations}
    return problem, architecture, routes, requested_shed_w


def joint_solver(policy: str, objective: str | None = None) -> str:
    if policy == "queue_haul" and objective == "max_shed":
        return "max_shed"
    return {
        "queue_haul": "lp_work_first", "greedy": "greedy",
        "greedy_lagrangian": "greedy_lagrangian", "random": "random",
        "kv_only": "kv_only", "replay_only": "replay_only",
        "isolated_fastest": "isolated_fastest",
        "queue_haul_power_blind": "lp_power_blind",
        DEADLINE_BLIND_POLICY: "lp_work_first",
    }[policy]


def plan_joint_scenario(scenario: dict, snapshots: dict[str, dict],
                        profile: ModelProfile, seed: int,
                        demand: dict[str, tuple[float, float]] | None = None
                        ) -> list[dict]:
    problem, architecture, routes, requested_shed_w = joint_problem(
        scenario, snapshots, profile, demand)
    partial = scenario.get("design") in {
        "frontier", "constraint", "separation", "hardware_gap"}
    deadline_blind = scenario["policy"] == DEADLINE_BLIND_POLICY
    solver = joint_solver(scenario["policy"], scenario.get("objective"))
    planning_problem = replace(
        problem, deadline_s=DEADLINE_BLIND_HORIZON_S,
        end_s=max(problem.end_s, DEADLINE_BLIND_HORIZON_S),
    ) if deadline_blind else problem
    result = solve(
        planning_problem, profile, routes, solver, seed=seed,
        destination=architecture,
        admission_mode=scenario.get("admission_mode"),
    )
    planned = list(result.moves)
    admitted = {move.session_id for move in planned}
    missing = tuple(row for row in problem.sessions if row.session_id not in admitted)
    if missing and not partial:
        late = replace(problem, sessions=missing, deadline_s=600, end_s=600)
        planned.extend(replace(move, order=move.order + len(planned)) for move in solve(
            late, profile, routes, solver, seed=seed, destination=architecture,
            admission_mode=scenario.get("admission_mode"),
        ).moves)
    moves = [{
        "session_id": move.session_id,
        "destination_instance": move.destination_instance,
        "destination_pool": move.destination_pool,
        "method": move.method, "order": move.order,
        "path": list(move.path),
        "planned_rate_limit_bytes_per_s": move.rate_limit_bytes_per_s,
        "planned_quiesce_s": move.quiesce_s,
        "deadline_admitted": move.session_id in admitted,
    } for move in planned]
    if not partial and {row["session_id"] for row in moves} != {
            row["session_id"] for row in scenario["sessions"]}:
        raise RuntimeError("policy did not plan the complete evacuation")
    if partial and (requested_shed_w is None
            or not set(admitted) <= {row["session_id"] for row in scenario["sessions"]}):
        raise RuntimeError("invalid partial shed decision")
    return moves


def _planned_move(row: dict) -> PlannedMove:
    return PlannedMove(
        row["session_id"], row["destination_instance"], row["method"],
        row["order"], tuple(row["path"]), row.get("rate_limit_bytes_per_s"),
        row.get("quiesce_s"), row.get("destination_pool"),
    )


def _execution_move(row: dict) -> dict:
    return {
        "session_id": row["session_id"],
        "destination_instance": row["destination_instance"],
        "destination_pool": row.get("destination_pool"),
        "method": row["method"], "order": row["order"],
        "path": list(row["path"]),
        "planned_rate_limit_bytes_per_s": row.get(
            "rate_limit_bytes_per_s"),
        "planned_quiesce_s": row.get("quiesce_s"),
        "deadline_admitted": True,
    }


def plan_hardware_gap_scenario(scenario: dict, snapshots: dict[str, dict],
                               profile: ModelProfile, demand: dict
                               ) -> tuple[list[dict], dict]:
    snapshots = _hardware_gap_snapshots(scenario, snapshots, profile)
    if not scenario.get("moves"):
        return plan_joint_scenario(
            scenario, snapshots, profile, scenario["planner_seed"], demand,
        ), {"admission_rejected": False, "capacity_violations": [],
            "admission_mode": scenario["admission_mode"]}
    problem, architecture, _routes, _target = joint_problem(
        scenario, snapshots, profile, demand)
    planned = tuple(_planned_move(row) for row in scenario["moves"])
    policy = scenario["policy"]
    if policy == "queue_haul_stale":
        violations = _stable_plan_violations(
            problem, profile, architecture, planned)
    else:
        horizon = ORACLE_STALE_HORIZON_S \
            if policy == DEADLINE_BLIND_POLICY else None
        violations = _plan_violations(
            problem, profile, architecture, planned, horizon)
    admitted = not violations
    if admitted != scenario["expected_admission"]:
        raise RuntimeError("fixed-plan hardware admission changed")
    fresh = plan_joint_scenario(
        {**scenario, "policy": "queue_haul", "moves": []}, snapshots,
        profile, scenario["planner_seed"], demand,
    ) if policy in {"queue_haul_robust", "queue_haul_stale"} else []
    return ([_execution_move(row) for row in scenario["moves"]]
            if admitted else []), {
        "admission_rejected": not admitted,
        "capacity_violations": list(violations),
        "fresh_moves": fresh, "planning_state": scenario["planning_state"],
        "admission_mode": scenario["admission_mode"],
    }


def agentic_demand(records: dict[str, dict], sessions: list[dict],
                   profile: ModelProfile, total_load: float = .4,
                   ) -> dict[str, tuple[float, float]]:
    demand = {}
    for session in sessions:
        row = records[session["session_id"]]
        rate, turns = float(row["turn_rate_hz"]), row["turns"]
        if rate <= 0 or not turns:
            raise ValueError("agentic trace demand must be positive")
        demand[row["id"]] = (
            rate * statistics.mean(turn["append_tokens"] for turn in turns),
            rate * statistics.mean(turn["output_tokens"] for turn in turns),
        )
    case = profile.case()
    scale = total_load / sum(f / case.F + g / case.G for f, g in demand.values())
    return {session_id: (f * scale, g * scale)
            for session_id, (f, g) in demand.items()}


def deadline_credited_sessions(results: list[dict], started_ns: int,
                               deadline_s: float) -> set[str]:
    deadline_ns = started_ns + int(deadline_s * 1e9)
    return {row["session_id"] for row in results
            if "request" in row and int(row["request"]["end_ns"]) <= deadline_ns}


def diagnostic_attainment(scenario: dict, results: list[dict], demand: dict,
                          profile: ModelProfile, started_ns: int
                          ) -> tuple[float, float]:
    base, initial, requested = diagnostic_power_problem(
        scenario, demand, profile)
    credited = deadline_credited_sessions(
        results, started_ns, scenario["deadline_s"])
    return requested, initial - source_power(base, profile, credited)


def diagnostic_power_problem(scenario: dict, demand: dict,
                             profile: ModelProfile):
    base, _ = policy_campaign._problem(
        profile, scenario["sessions"], 1, scenario["deadline_s"])
    base = replace(base, sessions=tuple(replace(
        session, expected_f=demand[session.session_id][0],
        expected_g=demand[session.session_id][1],
    ) for session in base.sessions))
    initial = source_power(base, profile)
    minimum = source_power(
        base, profile, (session.session_id for session in base.sessions))
    requested = scenario["requested_shed_fraction"] * (initial - minimum)
    return base, initial, requested


def diagnostic_outcomes(scenario: dict, results: list[dict], demand: dict,
                        profile: ModelProfile, started_ns: int) -> dict:
    base, initial, requested = diagnostic_power_problem(
        scenario, demand, profile)
    successful = sorted(
        ((int(row["request"]["end_ns"]), row["session_id"])
         for row in results if "request" in row),
    )
    moved, by_deadline, target_time, curve = [], 0.0, None, [
        {"time_s": 0.0, "shed_w": 0.0}]
    full_horizon = scenario.get("full_horizon_s")
    for ended_ns, session_id in successful:
        elapsed = (ended_ns - started_ns) / 1e9
        if full_horizon is not None and elapsed > full_horizon:
            break
        moved.append(session_id)
        shed = initial - source_power(base, profile, moved)
        curve.append({"time_s": elapsed, "shed_w": shed})
        if elapsed <= scenario["deadline_s"]:
            by_deadline = shed
        if target_time is None and shed >= requested - 1e-8:
            target_time = elapsed
    eventual = initial - source_power(base, profile, moved)
    horizon = full_horizon or max(
        ORACLE_STALE_HORIZON_S, curve[-1]["time_s"])
    if curve[-1]["time_s"] < horizon:
        curve.append({"time_s": horizon, "shed_w": eventual})
    return {
        "requested_shed_w": requested, "realized_shed_w": by_deadline,
        "eventual_shed_w": eventual, "target_met": by_deadline >= requested,
        "eventual_target_met": eventual >= requested,
        "time_to_target_s": target_time, "attainment_curve": curve,
    }


class SinkLoad:
    def __init__(self, cfg: testbed.Config, port: int, prefill_tps: float,
                 rho: float, path: Path, decode_tps: float | None = None):
        if prefill_tps <= 0 or decode_tps is not None and decode_tps <= 0 \
                or not 0 < rho < 1:
            raise ValueError("invalid sink load")
        self.cfg, self.port, self.rho, self.path, self.decode_tps = (
            cfg, port, rho, path, decode_tps)
        self.interval_s = (
            SINK_LOAD_PREFILL_TOKENS / prefill_tps
            + SINK_LOAD_DECODE_TOKENS / decode_tps
        ) / rho if decode_tps is not None else 512 / (rho * prefill_tps)
        self.stop, self.error = threading.Event(), None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _request(self, index: int) -> dict:
        messages = [
            {"role": "system", "content": "You are a tool-using coding agent."},
            {"role": "user", "content":
             "Agentic trace turn 0: analyze tool output. " + "x " * 512},
        ]
        result, _ = profiler.stream_chat(
            self.cfg, self.port, messages, 64,
            profiler.messages_hash(messages), 600, True, f"load-{index}",
            self.decode_tps is not None)
        row = asdict(result)
        if result.status_code != 200 or self.decode_tps is not None and (
            row["prompt_tokens"] != SINK_LOAD_PREFILL_TOKENS
            or row["output_tokens"] != SINK_LOAD_DECODE_TOKENS
        ):
            raise RuntimeError("sink load request violated its service contract")
        return row

    def _run(self) -> None:
        futures, rows, index = [], [], 0
        next_at = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                while not self.stop.is_set():
                    if len(futures) == 8:
                        rows.append(futures.pop(0).result())
                        if self.stop.is_set():
                            break
                    delay = next_at - time.monotonic()
                    if delay > 0 and self.stop.wait(delay):
                        break
                    futures.append(pool.submit(self._request, index))
                    index += 1
                    next_at += self.interval_s
                rows.extend(future.result() for future in futures)
            self.path.write_text("".join(
                json.dumps(row, separators=(",", ":")) + "\n"
                for row in rows))
        except BaseException as exc:
            self.error = exc

    def close(self) -> None:
        self.stop.set()
        self.thread.join(900)
        if self.thread.is_alive():
            raise TimeoutError("sink load did not stop")
        if self.error:
            raise RuntimeError(
                f"sink load failed: {type(self.error).__name__}: {self.error}"
            ) from self.error

    def stop_admissions(self) -> None:
        self.stop.set()


def scenario_records(manifest: dict, scenario: dict) -> dict[str, dict]:
    templates = {row["id"]: row for row in manifest["sessions"]}
    return {
        row["session_id"]: ({
            **templates[row.get("template_id", row["session_id"])],
            "id": row["session_id"], "state_code": f"QH{index:03d}",
        } if "template_id" in row else templates[row["session_id"]])
        for index, row in enumerate(scenario["sessions"])
    }


def run_network_scenario(stack: ClusterStack, manifest: dict, scenario: dict,
                         root: Path, prefill_tps: float) -> dict:
    diagnostic = scenario["design"] in {
        "frontier", "constraint", "separation", "hardware_gap"}
    root.mkdir(parents=True, exist_ok=False)
    (root / "scenario.json").write_text(
        json.dumps(scenario, indent=2, sort_keys=True) + "\n")
    stack.spot.check()
    if any(process.poll() is not None for process in stack.remote.values()):
        raise RuntimeError("remote sink exited")
    _clear_cluster(stack)
    sessions = scenario_records(manifest, scenario)
    messages = {row["session_id"]: profiler.calibration_messages(
        sessions[row["session_id"]], row["initial_tokens"])
        for row in scenario["sessions"]}
    timeout, profile = REQUEST_TIMEOUT_S, ModelProfile.load(MODEL_PATH)
    service_load = scenario.get("load_normalization") == "destination_service"
    case = profile.case()
    destination_rates = (
        case.prefill.rate(SINK_LOAD_PREFILL_TOKENS, 1),
        case.decode.rate(SINK_LOAD_PREFILL_TOKENS, 1),
    )
    loads, snapshots, decision = {}, {}, {}
    for node in stack.cluster.destinations:
        compute, kv = scenario["background"].get(node.id, (0, 0))
        if compute:
            loads[node.id] = SinkLoad(
                stack.cfg, stack.ports[node.id]["api"],
                destination_rates[0] if service_load else prefill_tps, compute,
                root / f"sink_load_{node.id}.jsonl",
                destination_rates[1] if service_load else None)
            loads[node.id].start()
    if scenario.get("source_load"):
        loads["source"] = SinkLoad(
            stack.cfg, stack.cfg.src_port, prefill_tps, scenario["source_load"],
            root / "source_load.jsonl")
        loads["source"].start()
    try:
        if scenario["design"] in {
                "joint", "frontier", "constraint", "separation",
                "hardware_gap"}:
            time.sleep(scenario.get("load_warmup_s", 5))
            nodes = stack.cluster.destinations
            def snapshot(node):
                try:
                    return destination_metrics(
                        stack, node.id, scenario["background"][node.id][1],
                        scenario.get("kv_capacity_fraction", {}).get(
                            node.id, 1))
                except Exception as exc:
                    if not diagnostic or scenario["design"] == "hardware_gap":
                        raise
                    return {"kv_fraction": 0, "warning": True,
                            "error": f"{type(exc).__name__}: {exc}"}
            with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
                snapshots = dict(zip(
                    (node.id for node in nodes),
                    pool.map(snapshot, nodes),
                ))
            demand = agentic_demand(
                sessions, scenario["sessions"], profile,
                scenario.get("source_load", .4))
            if scenario["design"] == "hardware_gap":
                moves, decision = plan_hardware_gap_scenario(
                    scenario, snapshots, profile, demand)
            else:
                moves = plan_joint_scenario(
                    scenario, snapshots, profile, scenario["planner_seed"],
                    demand)
            write_checkpoint(root / "decision.json", {
                "background": snapshots, "moves": moves, **decision,
            })
        else:
            moves = scenario["moves"]
        moves = sorted(moves, key=lambda row: row["order"])
        preparation_errors = {}
        for move in moves:
            if move["method"] == "kv_transfer":
                row = sessions[move["session_id"]]
                try:
                    _warm(stack, messages[move["session_id"]],
                          row["state_code"], timeout)
                except Exception as exc:
                    if not diagnostic:
                        raise
                    preparation_errors[move["session_id"]] = \
                        f"{type(exc).__name__}: {exc}"
    except BaseException:
        for load in loads.values():
            load.close()
        raise
    before = testbed.proxy_counts(stack.run_root / "proxy_bytes.csv")
    start_ns = time.monotonic_ns()
    load_warnings = []
    try:
        def reconstruct(move):
            if move["session_id"] in preparation_errors:
                return {**move, "error": preparation_errors[move["session_id"]]}
            session = sessions[move["session_id"]]
            destination = move["destination_instance"]
            try:
                return {**move, "request": _chat(
                    stack.cfg, stack.ports[destination]["api"],
                    messages[move["session_id"]], session["state_code"],
                    timeout, move["method"] == "replay")}
            except Exception as exc:
                if not diagnostic:
                    raise
                return {**move, "error": f"{type(exc).__name__}: {exc}"}

        if moves:
            with ThreadPoolExecutor(max_workers=len(moves)) as pool:
                results = list(pool.map(reconstruct, moves))
        else:
            results = []
        end_ns = time.monotonic_ns()
    finally:
        for name, load in loads.items():
            try:
                load.close()
            except RuntimeError as exc:
                if not diagnostic:
                    raise
                load_warnings.append(f"{name}: {exc}")
    if not diagnostic and any(row["method"] == "kv_transfer"
           and row["request"]["cached_tokens"] <= 0 for row in results):
        raise RuntimeError("KV reconstruction reported no cached tokens")
    sleep_start_ns = sleep_end_ns = None
    if not diagnostic:
        testbed.set_source_sleep(stack.cfg, True)
        sleep_start_ns = time.monotonic_ns()
        time.sleep(5)
        testbed.set_source_sleep(stack.cfg, False)
        sleep_end_ns = time.monotonic_ns()
        time.sleep(.5)
    stack.spot.check()
    proxy = stack.run_root / "proxy_bytes.csv"
    connections = _csv_window(
        proxy.with_name("proxy_connections.csv"), start_ns, end_ns)
    transfers = _csv_window(
        proxy.with_name("resp_transfers.csv"), start_ns, end_ns)
    elapsed = (end_ns - start_ns) / 1e9
    result = {
        "schema": RESULT_SCHEMA, "status": "complete",
        "scenario_id": scenario["scenario_id"], "started_ns": start_ns,
        "ended_ns": end_ns, "migration_s": elapsed,
        "deadline_met": elapsed <= scenario["deadline_s"],
        "requests": results,
        "background": snapshots,
        "wire_bytes": testbed.count_delta(
            before, testbed.proxy_counts(proxy)),
        "connections": connections, "resp_transfers": transfers,
        "source_sleep_ns": [sleep_start_ns, sleep_end_ns],
    }
    if diagnostic:
        outcomes = diagnostic_outcomes(
            scenario, results, demand, profile, start_ns)
        result.update({
            **outcomes,
            "request_failures": sum("error" in row for row in results),
            "kv_evidence_warnings": sum(
                row["method"] == "kv_transfer" and "request" in row
                and row["request"]["cached_tokens"] <= 0 for row in results),
            "load_warnings": load_warnings,
            **decision,
        })
    write_checkpoint(root / "result.json", result)
    return result


def _serve_window(stack: ClusterStack, messages: dict[str, list[dict]],
                  codes: dict[str, str], routes: dict[str, tuple[str, int]],
                  seconds: float, phase: str) -> list[dict]:
    deadline, rows = time.monotonic() + seconds, []
    with ThreadPoolExecutor(max_workers=len(routes)) as pool:
        while time.monotonic() < deadline:
            started = time.monotonic_ns()
            futures = {session_id: pool.submit(
                _chat, stack.cfg, port, messages[session_id], codes[session_id],
                REQUEST_TIMEOUT_S)
                for session_id, (_, port) in routes.items()}
            for session_id, future in futures.items():
                node, _ = routes[session_id]
                rows.append({"phase": phase, "session_id": session_id,
                             "node": node, "batch_started_ns": started,
                             **future.result()})
    return rows


def observed_demand(rows: list[dict], seconds: float) -> dict[str, tuple[float, float]]:
    if seconds <= 0:
        raise ValueError("observation window must be positive")
    demand = {}
    for row in rows:
        values = demand.setdefault(row["session_id"], [0, 0])
        values[0] += row["prompt_tokens"] - row["cached_tokens"]
        values[1] += row["output_tokens"]
    return {session_id: (values[0] / seconds, values[1] / seconds)
            for session_id, values in demand.items()}


def handoff_scenario(plan: dict, cluster: Cluster, destination_load: float,
                     policy: str = "queue_haul", repeat: int = 0) -> dict:
    matches = [row for row in plan["scenarios"]
               if row["policy"] == policy and row["repeat"] == repeat
               and row["bandwidth"] == "natural"
               and row["deadline_s"] == HANDOFF_DEADLINE_S]
    if len(matches) != 1:
        raise ValueError("handoff policy and repeat must select one scenario")
    scenario = matches[0]
    return {**scenario, "deadline_s": HANDOFF_DEADLINE_S,
            "background": {node.id: [destination_load, 0]
                           for node in cluster.destinations}}


def run_handoff(cluster: Cluster, key: Path, calibration_path: Path,
                plan_path: Path, manifest_path: Path, run_root: Path,
                window_s: float = 300, destination_load: float = .5,
                power_interval_s: float = .1, policy: str = "queue_haul",
                repeat: int = 0) -> dict:
    if run_root.exists() or window_s <= 0 or not 0 < destination_load < 1:
        raise ValueError("handoff requires a new root and positive windows/load")
    plan, manifest = (json.loads(path.read_text())
                      for path in (plan_path, manifest_path))
    validate_plan(plan)
    if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("handoff manifest differs from plan")
    if Cluster.parse(plan["cluster"]) != cluster:
        raise ValueError("handoff cluster differs from plan")
    calibration = json.loads(calibration_path.read_text())
    validate_resume(plan["network_contract"], freeze_contract(calibration))
    host_check(cluster, key)
    scenario = handoff_scenario(
        plan, cluster, destination_load, policy, repeat)
    os.environ.update(HANDOFF_ENV)
    stack = start_cluster(cluster, key, plan["network_contract"], "natural",
                          run_root, power_interval_s)
    loads, metrics, source_load, sleeping, result = {}, {}, None, False, None
    try:
        _clear_cluster(stack)
        records = {row["id"]: row for row in manifest["sessions"]}
        messages = {row["session_id"]: profiler.calibration_messages(
            records[row["session_id"]], row["initial_tokens"])
            for row in scenario["sessions"]}
        codes = {session_id: records[session_id]["state_code"]
                 for session_id in messages}
        prefill_tps = ModelProfile.load(MODEL_PATH).case().F
        for node in cluster.destinations:
            loads[node.id] = SinkLoad(
                stack.cfg, stack.ports[node.id]["api"], prefill_tps,
                destination_load, run_root / f"sink_load_{node.id}.jsonl")
            loads[node.id].start()
        source_load = SinkLoad(
            stack.cfg, stack.cfg.src_port, prefill_tps, .8,
            run_root / "source_load.jsonl")
        source_load.start()
        metrics = {
            "sweden": MetricsSampler(
                stack.cfg.host, stack.cfg.src_port,
                run_root / "metrics_sweden.csv", power_interval_s),
            **{node.id: MetricsSampler(
                stack.cfg.host, stack.ports[node.id]["api"],
                run_root / f"metrics_{node.id}.csv", power_interval_s)
               for node in cluster.destinations},
        }
        for sampler in metrics.values():
            sampler.start()
        phase = {}
        mark = lambda name: phase.update({name: {
            "monotonic_ns": time.monotonic_ns(), "wall_ns": time.time_ns()}})
        mark("pre_start")
        pre = _serve_window(
            stack, messages, codes,
            {session_id: ("sweden", stack.cfg.src_port)
             for session_id in messages}, window_s, "pre")
        mark("pre_end")
        mark("handoff_start")
        nodes = cluster.destinations
        with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            snapshots = dict(zip(
                (node.id for node in nodes), pool.map(
                    lambda node: destination_metrics(stack, node.id, 0), nodes)))
        moves = plan_joint_scenario(
            scenario, snapshots, ModelProfile.load(MODEL_PATH),
            scenario["planner_seed"], observed_demand(
                pre, (phase["pre_end"]["monotonic_ns"]
                      - phase["pre_start"]["monotonic_ns"]) / 1e9))
        write_checkpoint(run_root / "decision.json", {
            "background": snapshots, "moves": moves})
        if not moves or not all(move["deadline_admitted"] for move in moves):
            raise RuntimeError("handoff policy did not admit the full 30-second shed")
        routes = {move["session_id"]: (
            move["destination_instance"],
            stack.ports[move["destination_instance"]]["api"])
            for move in moves}
        handoff = _serve_window(stack, messages, codes, routes, 0.001, "handoff")
        mark("handoff_end")
        migration_s = (phase["handoff_end"]["monotonic_ns"]
                       - phase["handoff_start"]["monotonic_ns"]) / 1e9
        if migration_s > HANDOFF_DEADLINE_S:
            raise RuntimeError(
                f"handoff missed {HANDOFF_DEADLINE_S} s deadline: "
                f"{migration_s:.3f} s")
        mark("switch_start")
        source_load.stop_admissions()
        mark("traffic_switched")

        def power_down():
            nonlocal sleeping
            source_load.close()
            mark("source_drained")
            mark("sleep_start")
            testbed.set_source_sleep(stack.cfg, True)
            sleeping = True
            mark("sleep_ready")

        mark("post_start")
        with ThreadPoolExecutor(max_workers=2) as pool:
            post_future = pool.submit(
                _serve_window, stack, messages, codes, routes, window_s, "post")
            sleep_future = pool.submit(power_down)
            post = post_future.result()
            sleep_future.result()
        mark("post_end")
        result = {"schema": "queue-haul-three-node-handoff-v2",
                  "status": "complete", "window_s": window_s,
                  "destination_load": destination_load,
                  "source_load": .8, "power_interval_s": power_interval_s,
                  "migration_s": migration_s,
                  "deadline_s": HANDOFF_DEADLINE_S, "deadline_met": True,
                  "cache": HANDOFF_ENV, "phases": phase,
                  "scenario": scenario, "decision": {
                      "background": snapshots, "moves": moves},
                  "requests": pre + handoff + post}
    finally:
        try:
            if source_load and source_load.thread.is_alive():
                source_load.close()
            for load in loads.values():
                load.close()
        finally:
            try:
                for sampler in metrics.values():
                    sampler.close()
            finally:
                try:
                    if sleeping:
                        testbed.set_source_sleep(stack.cfg, False)
                finally:
                    stop_cluster(stack)
    write_checkpoint(run_root / "result.json", result)
    return result


def _latest_result(root: Path) -> tuple[int, dict] | None:
    rows = sorted(root.glob("attempt-*/result.json"))
    if not rows:
        return None
    path = rows[-1]
    return int(path.parent.name.rsplit("-", 1)[-1]), json.loads(path.read_text())


def frontier_refinement(plan: dict, run_root: Path) -> dict:
    """Create the next matched boundary-only phase from completed frontier data."""
    validate_plan(plan)
    if plan["design"] != "frontier":
        raise ValueError("refinement requires a frontier plan")
    records = {}
    for scenario in plan["scenarios"]:
        latest = _latest_result(run_root / "scenarios" / scenario["scenario_id"])
        if latest and latest[1].get("status") == "complete":
            result = latest[1]
            records[(scenario["condition_index"], scenario["policy"],
                     scenario["repeat"])] = (
                bool(result.get("target_met")),
                tuple(sorted((row["destination_instance"], row["method"])
                             for row in result.get("requests", [])
                             if "request" in row)),
                float(result.get("realized_shed_w", 0)),
            )
    conditions = {row["condition_index"]: row for row in plan["scenarios"]}
    original_conditions = set(conditions)
    selected = set()
    if plan.get("phase", "pilot") == "pilot":
        next_index = max(conditions) + 1
        symmetric = {}
        for index, row in conditions.items():
            loads = tuple(value[0] for value in row["background"].values())
            if loads[0] == loads[1]:
                symmetric.setdefault(row["pack"], []).append((loads[0], index))
        for cells in symmetric.values():
            cells.sort()
            for (left_load, left), (right_load, right) in zip(cells, cells[1:]):
                if any((records.get((left, policy, 0)) or (None, None))[:2]
                       != (records.get((right, policy, 0)) or (None, None))[:2]
                       for policy in FRONTIER_POLICIES):
                    selected.update((left, right))
                    if (left_load, right_load) in ((.85, .9), (.9, .95)):
                        midpoint = (left_load + right_load) / 2
                        template = conditions[left]
                        conditions[next_index] = {
                            **template, "condition_index": next_index,
                            "background": {node: (midpoint, 0)
                                           for node in template["background"]},
                        }
                        selected.add(next_index)
                        next_index += 1
        # The asymmetric slice competes routes directly, so retain every observed shift.
        canonical = [(index, tuple(value[0] for value in row["background"].values()))
                     for index, row in conditions.items()
                     if index in original_conditions and row["pack"] == "8x16k"]
        for policy in FRONTIER_POLICIES:
            for axis in range(2):
                ordered = sorted((item for item in canonical if item[1][axis] == .5),
                                 key=lambda item: item[1][1 - axis])
                for (left, _), (right, _) in zip(ordered, ordered[1:]):
                    if (records.get((left, policy, 0)) or (None, None))[:2] \
                            != (records.get((right, policy, 0)) or (None, None))[:2]:
                        selected.update((left, right))
        repeats = range(1, 5)
    else:
        for index in conditions:
            for policy in FRONTIER_POLICIES:
                values = [records[key] for key in records
                          if key[:2] == (index, policy)]
                watts = [value[2] for value in values]
                if len({value[1] for value in values}) > 1 or len(watts) > 1 \
                        and 3.92 * statistics.stdev(watts) / len(watts) ** .5 > 10:
                    selected.add(index)
        repeats = range(5, 10)
    scenarios = []
    for index in sorted(selected):
        template = conditions[index]
        cell_repeats = range(5) if index not in original_conditions else repeats
        for repeat in cell_repeats:
            for policy in FRONTIER_POLICIES:
                row = {**template, "repeat": repeat, "policy": policy,
                       "planner_seed": profiler.stable_seed(
                           plan["seed"], index, repeat, policy)}
                row["scenario_id"] = _hash([
                    "frontier-refinement", index, repeat, policy,
                    row["sessions"], row["background"],
                ])[:16]
                scenarios.append(row)
    if not scenarios:
        raise RuntimeError("no frontier boundary needs refinement")
    if plan.get("phase", "pilot") == "pilot":
        scenarios.sort(key=lambda row: (
            row["condition_index"] in original_conditions, row["repeat"],
            row["condition_index"], FRONTIER_POLICIES.index(row["policy"])))
        scenarios = scenarios[:FRONTIER_REFINEMENT_EPISODES]
    output = {**plan, "phase": "refinement", "scenarios": scenarios,
              "repeats": len(repeats)}
    validate_plan(output)
    return output


def deadline_blind_plan(plans: list[dict]) -> dict:
    if not plans:
        raise ValueError("deadline-blind plan requires frontier inputs")
    for plan in plans:
        validate_plan(plan)
        if plan["design"] != "frontier":
            raise ValueError("deadline-blind plan requires frontier inputs")
    templates = {}
    for plan in plans:
        for row in plan["scenarios"]:
            if row["policy"] == "queue_haul":
                templates.setdefault((row["condition_index"], row["repeat"]), row)
    scenarios = []
    for (condition, repeat), template in sorted(templates.items()):
        row = {**template, "policy": DEADLINE_BLIND_POLICY,
               "planner_seed": profiler.stable_seed(
                   plans[0]["seed"], condition, repeat, DEADLINE_BLIND_POLICY)}
        row["scenario_id"] = _hash([
            "deadline-blind", condition, repeat, row["sessions"],
        ])[:16]
        scenarios.append(row)
    output = {**plans[0], "phase": "deadline_blind",
              "policies": [DEADLINE_BLIND_POLICY], "repeats": 1,
              "scenarios": scenarios}
    validate_plan(output)
    return output


def _next_attempt(root: Path) -> int:
    attempts = [int(path.name.rsplit("-", 1)[-1])
                for path in root.glob("attempt-*") if path.is_dir()]
    return max(attempts, default=0) + 1


def checkpoint_progress(plan: dict, run_root: Path) -> dict:
    complete, failed = [], []
    for scenario in plan["scenarios"]:
        latest = _latest_result(run_root / "scenarios" / scenario["scenario_id"])
        if latest:
            (complete if latest[1].get("status") == "complete" else failed) \
                .append(scenario["scenario_id"])
    value = {
        "schema": "queue-haul-network-progress-v1",
        "updated_ns": time.time_ns(), "expected": len(plan["scenarios"]),
        "completed": len(complete), "failed": len(failed),
        "missing": len(plan["scenarios"]) - len(complete) - len(failed),
        "completed_scenario_ids": complete,
    }
    write_checkpoint(run_root / "progress.json", value)
    return value


def plot_frontier(plan: dict, evidence: list[tuple[dict, dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_policies = tuple(dict.fromkeys(
        (*FRONTIER_POLICIES, DEADLINE_BLIND_POLICY)))
    profile = ModelProfile.load(MODEL_PATH)
    colors = dict(zip(plot_policies, TAB10_COLORS))
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for scenario, result in evidence:
        for move in result.get("requests", []):
            node = move["destination_instance"]
            load = scenario["background"][node][0]
            axis.scatter(
                scenario["bandwidth_mbps"][node] / 1000,
                profile.case().F * (1 - load),
                marker="^" if move["method"] == "kv_transfer" else "o",
                facecolors=colors[scenario["policy"]] if result.get("target_met")
                else "none", edgecolors=colors[scenario["policy"]], alpha=.75)
    case, tokens = profile.case(), 16_384
    eta = (tokens - case.kv_transfer.tail_tokens(tokens)) / (
        case.kv_transfer.sealed_bytes(tokens) - 2 * tokens)
    xs = sorted(path["natural_mbps"] / 1000
                for path in plan["network_contract"]["paths"].values())
    axis.plot(xs, [eta * x * 125_000_000 for x in xs], "k--",
              label=r"analytical 16K boundary $e=\eta\rho$")
    axis.set(xlabel="Measured natural network goodput (Gb/s)",
             ylabel="Available destination prefill (tokens/s)")
    axis.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"prefill_network_mechanism.{suffix}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for policy in plan["policies"]:
        rows = [(scenario, result) for scenario, result in evidence
                if scenario["policy"] == policy]
        sizes = sorted({sum(item["initial_tokens"] for item in scenario["sessions"])
                        for scenario, _ in rows})
        loads = sorted({statistics.mean(value[0] for value in scenario["background"].values())
                        for scenario, _ in rows})
        axes[0].plot([size / 1000 for size in sizes], [statistics.mean(
            result.get("target_met", False) for scenario, result in rows
            if sum(item["initial_tokens"] for item in scenario["sessions"]) == size)
            for size in sizes], marker="o", color=colors[policy], label=policy)
        axes[1].plot(loads, [statistics.mean(
            result.get("target_met", False) for scenario, result in rows
            if statistics.mean(value[0] for value in scenario["background"].values()) == load)
            for load in loads], marker="o", color=colors[policy])
    axes[0].set(xlabel="Movement size (thousand tokens)", ylabel="30 s target attainment")
    axes[1].set(xlabel="Mean destination load", ylim=(-.05, 1.05))
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"frontier_target_attainment.{suffix}", dpi=200)
    plt.close(fig)


def _constraint_action_counts(moves) -> dict[str, int]:
    counts = dict.fromkeys(CONSTRAINT_ACTIONS, 0)
    for move in moves:
        destination = (move["destination_instance"] if isinstance(move, dict)
                       else move.destination_instance)
        method = move["method"] if isinstance(move, dict) else move.method
        counts[f"{destination}_{method}"] += 1
    return counts


def plot_constraint(rows: list[dict], duals: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policy_colors = dict(zip(CONSTRAINT_POLICIES, TAB10_COLORS))
    policy_labels = {
        "queue_haul": "Queue-Haul exact max", "greedy": "Queue-Haul greedy",
        "kv_only": "KV only", "replay_only": "Replay only",
        "isolated_fastest": "Per-session fastest",
        "queue_haul_power_blind": "Power blind",
    }
    action_labels = {
        "germany_kv_transfer": "KV → Germany",
        "east_replay": "Replay → East",
        "east_kv_transfer": "KV → East",
        "germany_replay": "Replay → Germany",
    }
    cells = [cell[0] for cell in CONSTRAINT_CELLS]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    windows = cells[:3]
    for policy in CONSTRAINT_POLICIES:
        selected = {row["condition_id"]: row for row in rows
                    if row["policy"] == policy}
        axes[0].plot(
            [selected[cell]["deadline_s"] for cell in windows],
            [selected[cell]["attained_shed_w"] for cell in windows],
            marker="o", color=policy_colors[policy], label=policy_labels[policy],
        )
    request = next(row["requested_shed_w"] for row in rows
                   if row["condition_id"] == windows[0])
    axes[0].axhline(request, color="black", linestyle="--", linewidth=1,
                    label="Full-pack request")
    selected = {row["policy"]: row for row in rows
                if row["condition_id"] == cells[3]}
    axes[1].bar(
        [policy_labels[policy] for policy in CONSTRAINT_POLICIES],
        [selected[policy]["attained_shed_w"] for policy in CONSTRAINT_POLICIES],
        color=[policy_colors[policy] for policy in CONSTRAINT_POLICIES],
    )
    axes[1].axhline(request, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Deadline (s)", ylabel="Source power shed (W)",
                title="Full-pack request by deadline")
    axes[1].set_title("30 s with Germany replay quota")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"constraint_attainment.{suffix}", dpi=200)
    plt.close(fig)

    action_colors = dict(zip(CONSTRAINT_ACTIONS, TAB10_COLORS))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for axis, condition in zip(axes.flat, cells):
        selected = {row["policy"]: row for row in rows
                    if row["condition_id"] == condition}
        policies = [policy for policy in CONSTRAINT_POLICIES if policy in selected]
        bottom = [0.0] * len(policies)
        for action in CONSTRAINT_ACTIONS:
            values = [selected[policy][action] /
                      max(1, selected[policy]["selected_sessions"])
                      for policy in policies]
            axis.bar([policy_labels[policy] for policy in policies], values,
                     bottom=bottom, color=action_colors[action],
                     label=action_labels[action])
            bottom = [left + value for left, value in zip(bottom, values)]
        axis.set_title(condition)
        axis.tick_params(axis="x", rotation=35, labelsize=8)
    axes[0, 0].set_ylabel("Fraction of selected actions")
    axes[1, 0].set_ylabel("Fraction of selected actions")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4,
               loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"constraint_actions.{suffix}", dpi=200)
    plt.close(fig)

    migration = [row for row in duals if row["resource"].startswith("migration:")]
    if migration:
        fig, axis = plt.subplots(figsize=(8, 4.5))
        resources = [f"migration:pool/{action.split('_', 1)[0]}:"
                     f"{action.split('_', 1)[1]}"
                     for action in CONSTRAINT_ACTIONS]
        offsets = (-.18, -.06, .06, .18)
        for offset, color, resource in zip(offsets, TAB10_COLORS, resources):
            values = {row["condition_id"]: row for row in migration
                      if row["resource"] == resource}
            axis.scatter([index + offset for index in range(len(cells))],
                         [values[cell]["shadow_w_per_unit"] for cell in cells],
                         color=color,
                         label=resource.removeprefix("migration:pool/")
                         .replace(":", " ").replace("kv_transfer", "KV"))
        axis.set(xlabel="Frozen condition",
                 ylabel="Phase-I shadow price (W / replica-second)")
        axis.set_xticks(range(len(cells)), cells)
        axis.legend(frameon=False, ncol=2, fontsize=8)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"constraint_duals.{suffix}", dpi=200)
        plt.close(fig)


def _validate_constraint_simulation(rows: list[dict], duals: list[dict]) -> None:
    by_cell = {(row["condition_id"], row["policy"]): row for row in rows}
    if any(row["target_met"] for row in rows):
        raise RuntimeError("the full-pack request must exceed constrained capacity")
    if any(by_cell[cell, "queue_haul"]["attained_shed_w"] < max(
            by_cell[cell, policy]["attained_shed_w"]
            for policy in CONSTRAINT_POLICIES[1:]) - 1e-8
           for cell, *_ in CONSTRAINT_CELLS):
        raise RuntimeError("Queue-Haul is below a baseline capacity point")
    migration = [row for row in duals if row["resource"].startswith("migration:")]
    if any(len([row for row in migration if row["condition_id"] == cell
                and row["shadow_w_per_full_capacity"] > 1e-8]) != 4
           for cell, *_ in CONSTRAINT_CELLS):
        raise RuntimeError("a frozen condition does not exhaust four migration windows")
    base, quota = by_cell["window-30", "queue_haul"], \
        by_cell["quota-30", "queue_haul"]
    favored = quota["germany_kv_transfer"] + quota["east_replay"]
    if quota["germany_replay"] >= base["germany_replay"] \
            or quota["east_replay"] <= base["east_replay"] \
            or favored / quota["selected_sessions"] < .70:
        raise RuntimeError("replay quota did not cause the required destination shift")


def simulate_constraint(plan_path: Path, out: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if plan["design"] != "constraint":
        raise ValueError("constraint simulation requires a constraint plan")
    if profiler.file_hash(MODEL_PATH) != plan["model_profile"]["sha256"]:
        raise RuntimeError("constraint model profile changed")
    manifest_path = Path(plan["manifest"]["path"])
    if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("constraint manifest changed")
    manifest = json.loads(manifest_path.read_text())
    profile, rows, duals, seen = ModelProfile.load(MODEL_PATH), [], [], set()
    for scenario in plan["scenarios"]:
        snapshots = {node: {"kv_fraction": values[1]}
                     for node, values in scenario["background"].items()}
        demand = agentic_demand(
            scenario_records(manifest, scenario), scenario["sessions"], profile,
            scenario["source_load"],
        )
        problem, architecture, routes, requested = joint_problem(
            scenario, snapshots, profile, demand)
        result = solve(
            problem, profile, routes, joint_solver(
                scenario["policy"], scenario.get("objective")),
            seed=scenario["planner_seed"], destination=architecture,
        )
        counts = _constraint_action_counts(result.moves)
        attained = result.initial_source_power_w - result.planned_source_power_w
        rows.append({
            "condition_index": scenario["condition_index"],
            "condition_id": scenario["condition_id"],
            "policy": scenario["policy"], "objective": scenario["objective"],
            "solver": result.solver, "deadline_s": scenario["deadline_s"],
            "session_count": len(scenario["sessions"]),
            "movement_tokens": sum(row["initial_tokens"]
                                   for row in scenario["sessions"]),
            "requested_shed_w": requested, "attained_shed_w": attained,
            "attainment_fraction": attained / requested,
            "target_met": bool(result.feasible),
            "selected_sessions": len(result.moves),
            "predicted_makespan_s": result.predicted_migration_makespan_s,
            "bottleneck": result.bottleneck or "",
            "binding_resources": result.binding_resources,
            **counts,
        })
        if scenario["condition_id"] not in seen:
            seen.add(scenario["condition_id"])
            selection = replace(
                problem, final_state="awake", assumed_shutdown_s=None)
            table = candidate_table(
                problem, profile, architecture, "normal",
                ExpectedPower(selection, profile))
            ceiling, prices = phase_one_capacity_duals(table)
            for name, capacity, unit, price in zip(
                table.resource_names, table.resource_capacities,
                table.resource_units, prices,
            ):
                duals.append({
                    "condition_index": scenario["condition_index"],
                    "condition_id": scenario["condition_id"],
                    "phase_one_marginal_ceiling_w": ceiling,
                    "resource": name, "capacity": capacity, "unit": unit,
                    "shadow_w_per_full_capacity": float(price),
                    "shadow_w_per_unit": float(price / capacity),
                })
    order = {policy: index for index, policy in enumerate(CONSTRAINT_POLICIES)}
    rows.sort(key=lambda row: (row["condition_index"], order[row["policy"]]))
    duals.sort(key=lambda row: (row["condition_index"], row["resource"]))
    _validate_constraint_simulation(rows, duals)
    out.mkdir(parents=True, exist_ok=False)
    prediction_path, dual_path = (out / "constraint_predictions.csv",
                                  out / "constraint_duals.csv")
    profiler.write_csv(prediction_path, rows)
    profiler.write_csv(dual_path, duals)
    plot_constraint(rows, duals, out)
    artifacts = {path.name: profiler.file_hash(path)
                 for path in sorted(out.iterdir()) if path.is_file()}
    metadata = {
        "schema": "queue-haul-constraint-simulation-v2",
        "plan": {"path": str(plan_path), "sha256": profiler.file_hash(plan_path)},
        "manifest_sha256": plan["manifest"]["sha256"],
        "model_profile_sha256": plan["model_profile"]["sha256"],
        "context_selection": {
            "window-19": {"support": "recorded exact", "seed": 15},
            "window-30/quota-30": {"support": "recorded exact", "seed": 8},
            "window-60": {"support": "eight copies of each trace at 14042"},
            "disclosure": "19 s and 30 s seeds were selected offline to expose stress",
        },
        "constructed_quota": {
            "condition": "quota-30", "pool": "germany", "method": "replay",
            "headroom_fraction": .25, "capacity_replica_s": 6.25,
            "claim": "operator counterfactual, not a measured load coefficient",
        },
        "dual_semantics": (
            "Phase-I maximum additive initial marginal-power surrogate shadow "
            "price because every exact request exceeds its marginal ceiling; "
            "exact bundle shed is recomputed after integral packing"
        ),
        "objective": (
            "maximize total removable single-source load with an exact binary "
            "joint action/destination solve, then minimize migration work"
        ),
        "big_shed_request": "100% of modeled removable pack power",
        "policy_colors": dict(zip(CONSTRAINT_POLICIES, TAB10_COLORS)),
        "artifacts": artifacts,
    }
    write_checkpoint(out / "metadata.json", metadata)
    return {"scenarios": len(rows), "conditions": len(seen),
            "out": str(out), "valid": True}


def plot_separation(rows: list[dict], resources: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = {
        "queue_haul": "Queue-Haul", "greedy": "Greedy",
        "kv_only": "KV only", "replay_only": "Replay only",
        "isolated_fastest": "Isolated fastest",
        "queue_haul_power_blind": "Power blind",
        DEADLINE_BLIND_POLICY: "Deadline blind",
    }
    colors = dict(zip(SEPARATION_POLICIES, TAB10_COLORS))
    cells = [row[0] for row in SEPARATION_CELLS]
    fig, axes = plt.subplots(2, len(cells), figsize=(15, 8))
    for column, cell in enumerate(cells):
        selected = {(row["repeat"], row["policy"]): row for row in rows
                    if row["condition_id"] == cell}
        values = [statistics.median(selected[repeat, policy]["attained_shed_w"]
                                    for repeat in range(SEPARATION_REPEATS))
                  for policy in SEPARATION_POLICIES]
        target = next(iter(selected.values()))["requested_shed_w"]
        axes[0, column].bar(
            [labels[policy] for policy in SEPARATION_POLICIES], values,
            color=[colors[policy] for policy in SEPARATION_POLICIES],
        )
        axes[0, column].axhline(target, color="black", linestyle="--")
        axes[0, column].set_title(cell)
        axes[0, column].tick_params(axis="x", rotation=35, labelsize=7)
        if column == 0:
            axes[0, column].set_ylabel("Shed by 45 s (W)")
        bottom = [0.0, 0.0]
        for action, color in zip(CONSTRAINT_ACTIONS, TAB10_COLORS):
            values = [statistics.median(
                selected[repeat, policy][action]
                / selected[repeat, policy]["selected_sessions"]
                for repeat in range(SEPARATION_REPEATS)
            ) for policy in SEPARATION_POLICIES[:2]]
            axes[1, column].bar(
                [labels[policy] for policy in SEPARATION_POLICIES[:2]], values,
                bottom=bottom, color=color,
                label=action.replace("_kv_transfer", " KV")
                .replace("_replay", " replay"),
            )
            bottom = [a + b for a, b in zip(bottom, values)]
        axes[1, column].set_ylim(0, 1)
        if column == 0:
            axes[1, column].set_ylabel("Completed-action fraction")
    handles, legend = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, legend, frameon=False, ncol=4, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, .95))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"separation_campaign.{suffix}", dpi=200)
    plt.close(fig)

    if not resources:
        return
    names = sorted({row["resource"] for row in resources})
    values = np.array([[next(
        row["utilization"] for row in resources
        if row["condition_id"] == cell and row["resource"] == name
    ) for cell in cells] for name in names])
    fig, axis = plt.subplots(figsize=(7, max(4, len(names) * .35)))
    image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(cells)), cells, rotation=15, ha="right")
    axis.set_yticks(range(len(names)), names, fontsize=7)
    for row, column in np.ndindex(values.shape):
        if values[row, column] >= .8:
            axis.text(column, row, f"{values[row, column]:.2f}",
                      ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=axis, label="Queue-Haul utilization")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"separation_resources.{suffix}", dpi=200)
    plt.close(fig)


def _validate_separation_simulation(rows: list[dict], resources: list[dict],
                                    duals: list[dict]) -> None:
    for row in rows:
        winner = row["policy"] in SEPARATION_POLICIES[:2]
        ratio = row["attained_shed_w"] / row["requested_shed_w"]
        if winner and (ratio < 1 + SEPARATION_MARGIN
                       or not row["deadline_met"]):
            raise RuntimeError("a joint planner lacks the required robust margin")
        if not winner and ratio > 1 - SEPARATION_MARGIN:
            raise RuntimeError("a baseline is too close to the separating target")
        if winner and (row["east_replay"] + row["germany_replay"] < 2
                       or row["east_kv_transfer"]
                       + row["germany_kv_transfer"] < 2
                       or row["east_replay"] + row["east_kv_transfer"] == 0
                       or row["germany_replay"]
                       + row["germany_kv_transfer"] == 0):
            raise RuntimeError("a joint planner does not require both actions")
        if row["policy"] == DEADLINE_BLIND_POLICY and (
                not row["planner_feasible"] or row["deadline_met"]):
            raise RuntimeError("deadline-blind planning is not a real deadline trap")
    utilization = {(row["condition_id"], row["resource"]): row["utilization"]
                   for row in resources}
    prices = {(row["condition_id"], row["resource"]):
              row["shadow_w_per_full_capacity"] for row in duals}
    for cell, expected in SEPARATION_BINDINGS.items():
        if any(utilization.get((cell, resource), 0) < minimum
               for resource, minimum in expected.items()) or any(
                   prices.get((cell, resource), 0) <= 0
                   for resource in expected
                   if resource != "service:pool/east:0"):
            raise RuntimeError("a separating cell does not severely bind resources")


def simulate_separation(plan_path: Path, out: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if plan["design"] != "separation":
        raise ValueError("separation simulation requires a separation plan")
    if profiler.file_hash(MODEL_PATH) != plan["model_profile"]["sha256"]:
        raise RuntimeError("separation model profile changed")
    manifest_path = Path(plan["manifest"]["path"])
    if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("separation manifest changed")
    manifest, profile = (json.loads(manifest_path.read_text()),
                         ModelProfile.load(MODEL_PATH))
    rows, resources, duals, seen = [], [], [], set()
    for scenario in plan["scenarios"]:
        snapshots = {node: {"kv_fraction": values[1]}
                     for node, values in scenario["background"].items()}
        demand = agentic_demand(
            scenario_records(manifest, scenario), scenario["sessions"], profile,
            scenario["source_load"],
        )
        problem, architecture, routes, requested = joint_problem(
            scenario, snapshots, profile, demand)
        planning = replace(
            problem, deadline_s=DEADLINE_BLIND_HORIZON_S,
            end_s=DEADLINE_BLIND_HORIZON_S,
        ) if scenario["policy"] == DEADLINE_BLIND_POLICY else problem
        result = solve(
            planning, profile, routes,
            joint_solver(scenario["policy"], scenario.get("objective")),
            seed=scenario["planner_seed"], destination=architecture,
        )
        actual = replace(
            problem, deadline_s=scenario["deadline_s"],
            end_s=scenario["deadline_s"],
        )
        execution = predict(
            _expected_scenario(actual, result.moves), profile, result.moves,
            destination=architecture,
        )
        completed = {row.session_id for row in execution.sessions
                     if row.committed_s is not None
                     and row.committed_s <= scenario["deadline_s"]}
        moves = [move for move in result.moves if move.session_id in completed]
        attained = result.initial_source_power_w \
            - execution.modeled_source_power_at_deadline_w
        rows.append({
            "condition_index": scenario["condition_index"],
            "condition_id": scenario["condition_id"],
            "repeat": scenario["repeat"], "policy": scenario["policy"],
            "solver": result.solver, "deadline_s": scenario["deadline_s"],
            "planning_deadline_s": scenario["planning_deadline_s"],
            "requested_shed_w": requested, "attained_shed_w": attained,
            "attainment_fraction": attained / requested,
            "target_met": attained >= requested - 1e-8,
            "deadline_met": execution.deadline_met,
            "planner_feasible": result.feasible,
            "planned_sessions": len(result.moves),
            "selected_sessions": len(moves),
            "predicted_makespan_s": execution.migration_makespan_s,
            **_constraint_action_counts(moves),
        })
        key = scenario["condition_id"]
        if scenario["policy"] == "queue_haul" and key not in seen:
            seen.add(key)
            resources.extend({
                "condition_id": key, "resource": row.name,
                "unit": row.unit, "used": row.used,
                "capacity": row.capacity, "utilization": row.utilization,
            } for row in result.resource_uses)
            table = candidate_table(
                problem, profile, architecture, "normal",
                ExpectedPower(replace(
                    problem, final_state="awake", assumed_shutdown_s=None),
                    profile),
            )
            ceiling, prices = phase_one_capacity_duals(table)
            duals.extend({
                "condition_id": key,
                "phase_one_marginal_ceiling_w": ceiling,
                "resource": name, "capacity": capacity, "unit": unit,
                "shadow_w_per_full_capacity": float(price),
                "shadow_w_per_unit": float(price / capacity),
            } for name, capacity, unit, price in zip(
                table.resource_names, table.resource_capacities,
                table.resource_units, prices,
            ))
    order = {policy: index for index, policy in enumerate(SEPARATION_POLICIES)}
    rows.sort(key=lambda row: (
        row["condition_index"], row["repeat"], order[row["policy"]]))
    _validate_separation_simulation(rows, resources, duals)
    out.mkdir(parents=True, exist_ok=False)
    profiler.write_csv(out / "separation_predictions.csv", rows)
    profiler.write_csv(out / "separation_resources.csv", resources)
    profiler.write_csv(out / "separation_duals.csv", duals)
    plot_separation(rows, resources, out)
    artifacts = {path.name: profiler.file_hash(path)
                 for path in sorted(out.iterdir()) if path.is_file()}
    write_checkpoint(out / "metadata.json", {
        "schema": "queue-haul-separation-simulation-v1",
        "plan": {"path": str(plan_path),
                 "sha256": profiler.file_hash(plan_path)},
        "load_support": {"path": str(LOAD_SUPPORT_PATH.relative_to(ROOT)),
                         "sha256": profiler.file_hash(LOAD_SUPPORT_PATH)},
        "margin_fraction": SEPARATION_MARGIN,
        "minimum_binding_utilization": SEPARATION_BINDINGS,
        "deadline_s": SEPARATION_DEADLINE_S,
        "planning_deadline_s": SEPARATION_PLANNING_DEADLINE_S,
        "objective": "exact maximum shed for Queue-Haul; target-aware greedy",
        "policy_colors": dict(zip(SEPARATION_POLICIES, TAB10_COLORS)),
        "artifacts": artifacts,
    })
    return {"scenarios": len(rows), "conditions": len(seen),
            "out": str(out), "valid": True}


def _restricted_architecture(architecture: DestinationArchitecture,
                             restriction: str) -> DestinationArchitecture:
    if restriction not in ORACLE_RESTRICTIONS:
        raise ValueError(f"unknown oracle restriction: {restriction}")
    pools = architecture.pools
    methods = {"kv_only": ("kv_transfer",),
               "replay_only": ("replay",)}.get(restriction)
    if methods:
        pools = tuple(replace(
            pool, methods=methods,
            migration_headroom=None if pool.migration_headroom is None else {
                name: value for name, value in pool.migration_headroom.items()
                if name in methods
            },
        ) for pool in pools)
    if restriction in {"east_only", "germany_only"}:
        destination = restriction.removesuffix("_only")
        pools = tuple(pool for pool in pools
                      if pool.pool_id == f"pool/{destination}")
    return replace(architecture, pools=pools)


def _oracle_row(family: str, condition: str, restriction: str, problem,
                architecture, routes, profile, target: float, **fields):
    result = solve(
        problem, profile, routes, "max_shed",
        destination=_restricted_architecture(architecture, restriction),
        admission_mode="normal",
    )
    shed = result.initial_source_power_w - result.planned_source_power_w
    return result, {
        "family": family, "condition_id": condition,
        "restriction": restriction, "requested_shed_w": target,
        "shed_w": shed, "attainment_fraction": shed / target,
        "target_met": shed >= target - 1e-8,
        "admission_mode": result.admission_mode,
        "selected_sessions": len(result.moves),
        "predicted_makespan_s": result.predicted_migration_makespan_s,
        **_constraint_action_counts(result.moves), **fields,
    }


def _plan_violations(problem, profile, architecture, moves,
                     horizon_s: float | None = None,
                     prefixes: tuple[str, ...] = ()) -> tuple[str, ...]:
    checking = (replace(problem, deadline_s=horizon_s,
                        end_s=max(problem.end_s, horizon_s))
                if horizon_s is not None else problem)
    table = candidate_table(
        checking, profile, architecture, "normal",
        ExpectedPower(replace(
            checking, final_state="awake", assumed_shutdown_s=None), profile),
    )
    candidates = {
        (table.sessions[row.session].session_id, row.method,
         architecture.pools[row.pool].pool_id): index
        for index, row in enumerate(table.candidates)
    }
    signatures = [(move.session_id, move.method, move.destination_pool)
                  for move in moves]
    missing = [f"candidate:{'/'.join(map(str, signature))}"
               for signature in signatures if signature not in candidates]
    if missing:
        return tuple(missing)
    selected = [candidates[signature] for signature in signatures]
    usage = table.resources[:, selected].sum(axis=1).A1
    return tuple(name for name, value in zip(table.resource_names, usage)
                 if (not prefixes or name.startswith(prefixes))
                 and value > 1 + 1e-8)


def _stable_plan_violations(problem, profile, architecture,
                            moves) -> tuple[str, ...]:
    return _plan_violations(
        problem, profile, architecture, moves, ORACLE_STALE_HORIZON_S,
        ("service:", "kv:"))


def _evaluate_fixed_plan(state: str, label: str, problem, architecture,
                         profile, moves, target: float,
                         curves: list[dict]) -> dict:
    counts = _constraint_action_counts(moves)
    initial = source_power(problem, profile)
    planned = initial - source_power(
        problem, profile, (move.session_id for move in moves))
    row = {
        "state": state, "plan": label, "requested_shed_w": target,
        "planned_shed_w": planned, "selected_sessions": len(moves), **counts,
        "admission_mode": "normal",
    }
    violations = _stable_plan_violations(
        problem, profile, architecture, moves)
    if violations:
        return {**row, "status": "capacity_infeasible",
                "capacity_violations": violations, "plan_valid": False,
                "shed_by_deadline_w": None, "eventual_shed_w": None,
                "target_by_deadline": False, "eventual_target_met": False,
                "time_to_target_s": None, "migration_makespan_s": None,
                "simulator_deadline_met": False}
    actual = replace(
        problem, deadline_s=SEPARATION_DEADLINE_S,
        end_s=ORACLE_STALE_HORIZON_S,
    )
    execution = predict(
        _expected_scenario(actual, moves), profile, moves,
        destination=architecture,
    )
    moved, shed_by_deadline, target_time = [], 0.0, None
    curves.append({"state": state, "plan": label, "time_s": 0.0,
                   "shed_w": 0.0})
    for committed_s, session_id in sorted(
        (item.committed_s, item.session_id) for item in execution.sessions
        if item.committed_s is not None
    ):
        if committed_s > ORACLE_STALE_HORIZON_S:
            break
        moved.append(session_id)
        shed = initial - source_power(problem, profile, moved)
        curves.append({"state": state, "plan": label,
                       "time_s": committed_s, "shed_w": shed})
        if committed_s <= SEPARATION_DEADLINE_S:
            shed_by_deadline = shed
        if target_time is None and shed >= target - 1e-8:
            target_time = committed_s
    eventual = initial - source_power(problem, profile, moved)
    curves.append({"state": state, "plan": label,
                   "time_s": ORACLE_STALE_HORIZON_S, "shed_w": eventual})
    by_deadline = shed_by_deadline >= target - 1e-8
    eventual_met = eventual >= target - 1e-8
    status = "target_met" if by_deadline else "late" if eventual_met else "insufficient"
    return {
        **row, "status": status, "capacity_violations": (),
        "plan_valid": True, "shed_by_deadline_w": shed_by_deadline,
        "eventual_shed_w": eventual, "target_by_deadline": by_deadline,
        "eventual_target_met": eventual_met,
        "time_to_target_s": target_time,
        "migration_makespan_s": execution.migration_makespan_s,
        "simulator_deadline_met": execution.deadline_met,
    }


def _validate_oracle_stale(oracles: list[dict], predictions: list[dict],
                           resources: list[dict], duals: list[dict]) -> None:
    if {row["admission_mode"] for row in (*oracles, *predictions)} \
            != {"normal"}:
        raise RuntimeError("restricted oracles do not share normal admission")
    oracle = {(row["family"], row["condition_id"], row["restriction"]): row
              for row in oracles}
    for condition, *_ in SEPARATION_CELLS:
        joint = oracle["original", condition, "joint"]["shed_w"]
        restricted = max(oracle["original", condition, name]["shed_w"]
                         for name in ORACLE_RESTRICTIONS[1:])
        if joint - restricted < (.4 if H100_CAMPAIGN else 12):
            raise RuntimeError("an original exact restricted-oracle gap is too small")
    target = oracle["toggle", "all-bind", "joint"]["requested_shed_w"]
    joint = oracle["toggle", "all-bind", "joint"]
    restricted = [oracle["toggle", "all-bind", name]
                  for name in ORACLE_RESTRICTIONS[1:]]
    if joint["shed_w"] < (1.015 if H100_CAMPAIGN else 1.2) * target \
            or max(row["shed_w"] for row in restricted) \
            > (.985 if H100_CAMPAIGN else .8) * target \
            or joint["shed_w"] - max(row["shed_w"] for row in restricted) \
            < (3 if H100_CAMPAIGN else 18) \
            or any(joint[action] == 0 for action in CONSTRAINT_ACTIONS):
        raise RuntimeError("the all-bind exact-oracle separation is not severe")
    releases = (
        ("free-kv", "east_only", 1.5 if H100_CAMPAIGN else 4),
        ("free-service", "germany_only", 12),
        ("free-bandwidth", "kv_only", 1.5 if H100_CAMPAIGN else 4),
    )
    if any(oracle["toggle", state, restriction]["shed_w"]
           - oracle["toggle", "all-bind", restriction]["shed_w"] < margin
           for state, restriction, margin in releases) \
            or not any(oracle["toggle", "all-release", name]["target_met"]
                       for name in ORACLE_RESTRICTIONS[1:]):
        raise RuntimeError("switching off a constraint did not release its oracle")
    predicted = {(row["state"], row["plan"]): row for row in predictions}
    if any(not predicted[state, "robust"]["target_by_deadline"]
           or not predicted[state, "adaptive"]["target_by_deadline"]
           or predicted[state, "adaptive"]["shed_by_deadline_w"] + 1e-8
           < predicted[state, "robust"]["shed_by_deadline_w"]
           for state, *_ in ORACLE_STALE_STATES) \
            or predicted["all-bind", "stale-optimistic"]["status"] \
            != "capacity_infeasible":
        raise RuntimeError("the robust or stale-plan certificate failed")
    aware, blind = (predicted["all-bind", name] for name in (
        "deadline-aware", "deadline-blind"))
    blind_valid = (blind["target_by_deadline"] if H100_CAMPAIGN else
                   blind["status"] == "late"
                   and blind["shed_by_deadline_w"] <= .85 * target
                   and blind["eventual_shed_w"] >= 1.2 * target
                   and blind["time_to_target_s"] is not None
                   and blind["time_to_target_s"] >= 55)
    if aware["shed_by_deadline_w"] < (1.015 if H100_CAMPAIGN else 1.2) * target \
            or aware["time_to_target_s"] >= SEPARATION_DEADLINE_S \
            or not blind_valid:
        raise RuntimeError("deadline blindness is not a severe late-power trap")
    utilization = {(row["state"], row["resource"]): row["utilization"]
                   for row in resources}
    prices = {(row["state"], row["resource"]):
              row["shadow_w_per_full_capacity"] for row in duals}
    bindings = ({"service:pool/germany:0": .90} if H100_CAMPAIGN else
                {"kv:pool/east": .94, "service:pool/germany:0": .97})
    if any(utilization.get(("all-bind", name), 0) < minimum
           or prices.get(("all-bind", name), 0) <= 0
           for name, minimum in bindings.items()):
        raise RuntimeError("the all-bind KV/service constraints are not severe")


def plot_oracle_stale(oracles: list[dict], predictions: list[dict],
                      curves: list[dict], resources: list[dict],
                      duals: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    restrictions = dict(zip(ORACLE_RESTRICTIONS, TAB10_COLORS))
    labels = {"joint": "Queue-Haul", "kv_only": "KV only",
              "replay_only": "Replay only", "east_only": "East only",
              "germany_only": "Germany only"}
    cells = [row[0] for row in SEPARATION_CELLS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    x, width = np.arange(len(cells)), .16
    for index, restriction in enumerate(ORACLE_RESTRICTIONS):
        selected = {(row["condition_id"], row["restriction"]): row
                    for row in oracles if row["family"] == "original"}
        axes[0].bar(x + (index - 2) * width,
                    [selected[cell, restriction]["shed_w"] for cell in cells],
                    width, color=restrictions[restriction],
                    label=labels[restriction])
    axes[0].plot(x, [selected[cell, "joint"]["requested_shed_w"]
                     for cell in cells], "k--", marker="_", label="Target")
    axes[0].set_xticks(x, cells, rotation=20, ha="right")
    axes[0].set(ylabel="Maximum shed by 30 s (W)",
                title="Exact restricted-oracle envelopes")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    states = [row[0] for row in ORACLE_STALE_STATES]
    predicted = {(row["state"], row["plan"]): row for row in predictions}
    colors = {"adaptive": TAB10_COLORS[0], "robust": TAB10_COLORS[2],
              "stale-optimistic": TAB10_COLORS[3]}
    for label, color in colors.items():
        values = [predicted[state, label]["shed_by_deadline_w"]
                  if predicted[state, label]["plan_valid"] else np.nan
                  for state in states]
        axes[1].plot(states, values, marker="o", color=color,
                     label=label.replace("-", " "))
        invalid = [index for index, state in enumerate(states)
                   if not predicted[state, label]["plan_valid"]]
        axes[1].scatter(invalid, [0] * len(invalid), marker="x", s=45,
                        color=color, label="capacity infeasible"
                        if label == "stale-optimistic" else None)
    toggles = {(row["condition_id"], row["restriction"]): row
               for row in oracles if row["family"] == "toggle"}
    axes[1].plot(states, [max(toggles[state, name]["shed_w"]
                              for name in ORACLE_RESTRICTIONS[1:])
                          for state in states], marker="s", linestyle="--",
                 color=TAB10_COLORS[1], label="best restricted oracle")
    target = predicted["all-bind", "adaptive"]["requested_shed_w"]
    axes[1].axhline(target, color="black", linestyle="--", label="Target")
    axes[1].tick_params(axis="x", rotation=35, labelsize=7)
    axes[1].set(ylabel="Shed by 45 s (W)",
                title="Fresh, robust, and stale plans")
    axes[1].legend(frameon=False, fontsize=7)

    for label, color in (("deadline-aware", TAB10_COLORS[0]),
                         ("deadline-blind", TAB10_COLORS[3])):
        rows = [row for row in curves
                if row["state"] == "all-bind" and row["plan"] == label]
        axes[2].step([row["time_s"] for row in rows],
                     [row["shed_w"] for row in rows], where="post",
                     color=color, label=label.replace("-", " "))
    axes[2].axvline(SEPARATION_DEADLINE_S, color="black", linestyle=":",
                    label="45 s deadline")
    axes[2].axhline(target, color="black", linestyle="--", label="Target")
    axes[2].set(xlim=(0, ORACLE_STALE_HORIZON_S), xlabel="Time (s)",
                ylabel="Nonlinear source power shed (W)",
                title="Enough power, but after the deadline")
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"oracle_stale_summary.{suffix}", dpi=200)
    plt.close(fig)

    names = sorted({row["resource"] for row in resources})
    values = np.array([[next(row["utilization"] for row in resources
                             if row["state"] == state
                             and row["resource"] == name)
                        for state in states] for name in names])
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(names) * .34)))
    cmap = LinearSegmentedColormap.from_list(
        "tab10-blue", ("white", TAB10_COLORS[0]))
    image = axes[0].imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    axes[0].set_xticks(range(len(states)), states, rotation=35, ha="right",
                       fontsize=7)
    axes[0].set_yticks(range(len(names)), names, fontsize=7)
    for row, column in np.ndindex(values.shape):
        if values[row, column] >= .8:
            axes[0].text(column, row, f"{values[row, column]:.2f}",
                         ha="center", va="center", color="white", fontsize=6)
    fig.colorbar(image, ax=axes[0], label="Adaptive Queue-Haul utilization")
    axes[0].set_title("Constraint toggles")
    positive = [row for row in duals if row["state"] == "all-bind"
                and row["shadow_w_per_full_capacity"] > 1e-8]
    axes[1].barh(
        [row["resource"] for row in positive],
        [row["shadow_w_per_full_capacity"] for row in positive],
        color=[TAB10_COLORS[index % len(TAB10_COLORS)]
               for index in range(len(positive))],
    )
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].set(xlabel="Phase-I shadow W per full capacity",
                title="All-bind positive duals")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"oracle_stale_resources.{suffix}", dpi=200)
    plt.close(fig)


def simulate_oracle_stale(plan_path: Path, out: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if plan["design"] != "separation":
        raise ValueError("oracle/stale simulation requires a separation plan")
    if profiler.file_hash(MODEL_PATH) != plan["model_profile"]["sha256"]:
        raise RuntimeError("oracle/stale model profile changed")
    manifest_path = Path(plan["manifest"]["path"])
    if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("oracle/stale manifest changed")
    manifest = json.loads(manifest_path.read_text())
    profile = ModelProfile.load(MODEL_PATH)
    templates = [next(row for row in plan["scenarios"]
                      if row["condition_id"] == condition
                      and row["repeat"] == 0 and row["policy"] == "queue_haul")
                 for condition, *_ in SEPARATION_CELLS]

    def make_problem(scenario):
        snapshots = {node: {"kv_fraction": values[1]}
                     for node, values in scenario["background"].items()}
        demand = agentic_demand(
            scenario_records(manifest, scenario), scenario["sessions"], profile,
            scenario["source_load"],
        )
        return joint_problem(scenario, snapshots, profile, demand)

    oracles, resources, duals, predictions, curves = [], [], [], [], []
    for scenario in templates:
        problem, architecture, routes, target = make_problem(scenario)
        for restriction in ORACLE_RESTRICTIONS:
            _, row = _oracle_row(
                "original", scenario["condition_id"], restriction,
                problem, architecture, routes, profile, target,
                bandwidth=scenario["bandwidth"],
                germany_load=scenario["background"]["germany"][0],
                east_kv_fraction=scenario["background"]["east"][1],
            )
            oracles.append(row)

    problems, solutions = {}, {}
    template = templates[-1]
    background_kv = (HARDWARE_GAP_BACKGROUND_KV_TOKENS
                     / profile.kv_capacity_tokens if H100_CAMPAIGN else 0)
    for state, germany_load, east_kv, bandwidth in ORACLE_STALE_STATES:
        scenario = {**template, "condition_id": state,
                    "background": {"east": (.25, east_kv + background_kv),
                                   "germany": (germany_load, background_kv)},
                    "bandwidth": bandwidth,
                    "bandwidth_mbps": _bandwidths(
                        plan["network_contract"], bandwidth),
                    "requested_shed_fraction": ORACLE_STALE_TARGET_FRACTION,
                    "migration_headroom": ({
                        "east": {"replay": .35},
                        "germany": {"replay": .35},
                    } if H100_CAMPAIGN else {})}
        problem, architecture, routes, target = make_problem(scenario)
        problems[state] = problem, architecture, routes, target
        solutions[state] = {}
        for restriction in ORACLE_RESTRICTIONS:
            result, row = _oracle_row(
                "toggle", state, restriction, problem, architecture, routes,
                profile, target, bandwidth=bandwidth,
                germany_load=germany_load, east_kv_fraction=east_kv,
            )
            solutions[state][restriction] = result
            oracles.append(row)
        joint = solutions[state]["joint"]
        resources.extend({"state": state, "resource": row.name,
                          "unit": row.unit, "used": row.used,
                          "capacity": row.capacity,
                          "utilization": row.utilization}
                         for row in joint.resource_uses)
        table = candidate_table(
            problem, profile, architecture, "normal",
            ExpectedPower(replace(
                problem, final_state="awake", assumed_shutdown_s=None), profile),
        )
        ceiling, prices = phase_one_capacity_duals(table)
        duals.extend({
            "state": state, "phase_one_marginal_ceiling_w": ceiling,
            "resource": name, "capacity": capacity, "unit": unit,
            "shadow_w_per_full_capacity": float(price),
            "shadow_w_per_unit": float(price / capacity),
        } for name, capacity, unit, price in zip(
            table.resource_names, table.resource_capacities,
            table.resource_units, prices,
        ))

    robust = solutions["all-bind"]["joint"].moves
    stale = solutions["all-release"]["joint"].moves
    for state, *_ in ORACLE_STALE_STATES:
        problem, architecture, _routes, target = problems[state]
        for label, moves in (("adaptive", solutions[state]["joint"].moves),
                             ("robust", robust),
                             ("stale-optimistic", stale)):
            predictions.append(_evaluate_fixed_plan(
                state, label, problem, architecture, profile, moves,
                target, curves))
    problem, architecture, routes, target = problems["all-bind"]
    predictions.append(_evaluate_fixed_plan(
        "all-bind", "deadline-aware", problem, architecture, profile,
        robust, target, curves))
    blind = solve(
        replace(problem, deadline_s=ORACLE_STALE_HORIZON_S,
                end_s=ORACLE_STALE_HORIZON_S),
        profile, routes, "max_shed", destination=architecture,
        admission_mode="normal",
    )
    predictions.append(_evaluate_fixed_plan(
        "all-bind", "deadline-blind", problem, architecture, profile,
        blind.moves, target, curves))
    _validate_oracle_stale(oracles, predictions, resources, duals)

    out.mkdir(parents=True, exist_ok=False)
    profiler.write_csv(out / "restricted_oracles.csv", oracles)
    profiler.write_csv(out / "toggle_predictions.csv", predictions)
    profiler.write_csv(out / "toggle_resources.csv", resources)
    profiler.write_csv(out / "toggle_duals.csv", duals)
    profiler.write_csv(out / "attainment_curves.csv", curves)
    write_checkpoint(out / "plans.json", {
        "schema": "queue-haul-oracle-stale-plans-v1",
        "robust_all_bind": [asdict(move) for move in robust],
        "stale_all_release": [asdict(move) for move in stale],
        "deadline_blind_90s": [asdict(move) for move in blind.moves],
    })
    plot_oracle_stale(oracles, predictions, curves, resources, duals, out)
    artifacts = {path.name: profiler.file_hash(path)
                 for path in sorted(out.iterdir()) if path.is_file()}
    write_checkpoint(out / "metadata.json", {
        "schema": "queue-haul-oracle-stale-simulation-v1",
        "plan": {"path": str(plan_path),
                 "sha256": profiler.file_hash(plan_path)},
        "manifest_sha256": plan["manifest"]["sha256"],
        "model_profile_sha256": plan["model_profile"]["sha256"],
        "load_support": {"path": str(LOAD_SUPPORT_PATH.relative_to(ROOT)),
                         "sha256": profiler.file_hash(LOAD_SUPPORT_PATH)},
        "uncertainty_set": {
            "states": [{"state": state, "germany_service_load": load,
                        "east_kv_fraction": kv, "bandwidth": bandwidth}
                       for state, load, kv, bandwidth in ORACLE_STALE_STATES],
            "robust_plan": (
                "exact max-shed plan at the componentwise worst corner; "
                "service/KV capacity only increases and bandwidth only rises "
                "in every other rectangular corner"
            ),
        },
        "constructed_constraint": (
            "East KV occupancy is a controlled 90% reserve against the profiled "
            "A100 KV capacity; controlled route caps derive from measured natural "
            "paths, while service-load support, workload pack, and destination "
            "throughput are measured inputs"
        ),
        "requested_shed_fraction": ORACLE_STALE_TARGET_FRACTION,
        "deadline_s": SEPARATION_DEADLINE_S,
        "planning_deadline_s": SEPARATION_PLANNING_DEADLINE_S,
        "full_horizon_s": ORACLE_STALE_HORIZON_S,
        "oracle_semantics": (
            "exact binary max-shed solve under forced normal admission with only "
            "the named methods or pools removed; sessions, source power, routes, "
            "and deadline are matched"
        ),
        "policy_colors": {
            "adaptive": TAB10_COLORS[0], "best_restricted": TAB10_COLORS[1],
            "robust": TAB10_COLORS[2], "stale_optimistic": TAB10_COLORS[3],
        },
        "artifacts": artifacts,
    })
    return {"oracle_conditions": len(SEPARATION_CELLS),
            "toggle_states": len(ORACLE_STALE_STATES),
            "out": str(out), "valid": True}


def _scenario_problem(scenario: dict, manifest: dict, profile: ModelProfile):
    snapshots = {node: {"kv_fraction": values[1]}
                 for node, values in scenario["background"].items()}
    snapshots = _hardware_gap_snapshots(scenario, snapshots, profile)
    demand = agentic_demand(
        scenario_records(manifest, scenario), scenario["sessions"], profile,
        scenario["source_load"],
    )
    return (*joint_problem(scenario, snapshots, profile, demand), demand)


def _hardware_gap_snapshots(scenario: dict, snapshots: dict[str, dict],
                            profile: ModelProfile) -> dict[str, dict]:
    if scenario.get("design") != "hardware_gap":
        return snapshots
    return {node: {
        **snapshot,
        "kv_fraction": min(1, snapshot["kv_fraction"] + scenario[
            "background_kv_headroom_tokens"][node]
            / profile.kv_capacity_tokens),
    } for node, snapshot in snapshots.items()}


def _move_rows(moves) -> list[dict]:
    return json.loads(json.dumps([asdict(move) for move in moves]))


def hardware_gap_plan(parent_path: Path, oracle_path: Path) -> dict:
    parent = json.loads(parent_path.read_text())
    validate_plan(parent)
    if parent["design"] != "separation":
        raise ValueError("hardware gap requires a separation plan")
    frozen = json.loads(oracle_path.read_text())
    if frozen.get("schema") != "queue-haul-oracle-stale-plans-v1":
        raise ValueError("invalid oracle/stale plans")
    manifest = json.loads(Path(parent["manifest"]["path"]).read_text())
    profile = ModelProfile.load(MODEL_PATH)
    template = next(row for row in parent["scenarios"]
                    if row["condition_id"] == "joint-shaped"
                    and row["repeat"] == 0 and row["policy"] == "queue_haul")
    states, exact = {}, {}
    for state, germany, kv, bandwidth, _policies in HARDWARE_GAP_MATRIX:
        scenario = {
            **template, "design": "hardware_gap", "condition_id": state,
            "background": {"east": (.25, kv), "germany": (germany, 0)},
            "bandwidth": bandwidth,
            "bandwidth_mbps": _bandwidths(
                parent["network_contract"], bandwidth),
            "requested_shed_fraction": HARDWARE_GAP_TARGET_FRACTION,
            "admission_mode": "normal",
            "migration_headroom": ({
                "east": {"replay": .35}, "germany": {"replay": .35},
            } if H100_CAMPAIGN else {}),
            "full_horizon_s": ORACLE_STALE_HORIZON_S,
            "background_kv_headroom_tokens": {
                "east": HARDWARE_GAP_BACKGROUND_KV_TOKENS,
                "germany": HARDWARE_GAP_BACKGROUND_KV_TOKENS,
            },
        }
        problem, architecture, routes, target, _demand = _scenario_problem(
            scenario, manifest, profile)
        states[state] = scenario, problem, architecture, routes, target
        for restriction in ORACLE_RESTRICTIONS:
            result, _ = _oracle_row(
                "hardware_gap", state, restriction, problem, architecture,
                routes, profile, target)
            exact[state, restriction] = result
    all_bind = states["all-bind"]
    robust = exact["all-bind", "joint"].moves
    stale = exact["all-release", "joint"].moves
    blind = solve(
        replace(all_bind[1], deadline_s=ORACLE_STALE_HORIZON_S,
                end_s=ORACLE_STALE_HORIZON_S),
        profile, all_bind[3], "max_shed", destination=all_bind[2],
        admission_mode="normal",
    ).moves
    generated = {
        "robust_all_bind": _move_rows(robust),
        "stale_all_release": _move_rows(stale),
        "deadline_blind_90s": _move_rows(blind),
    }
    if any(generated[key] != frozen[key] for key in generated):
        raise RuntimeError("frozen oracle/stale decisions changed")
    scenarios = []
    for index, (state, germany, kv, bandwidth, policies) \
            in enumerate(HARDWARE_GAP_MATRIX):
        base = states[state][0]
        for repeat in range(HARDWARE_GAP_REPEATS):
            for policy in policies:
                if policy.startswith("oracle_"):
                    moves = _move_rows(exact[state, policy[7:]].moves)
                elif policy == "queue_haul_robust":
                    moves = generated["robust_all_bind"]
                elif policy == "queue_haul_stale":
                    moves = generated["stale_all_release"]
                elif policy == DEADLINE_BLIND_POLICY:
                    moves = generated["deadline_blind_90s"]
                else:
                    moves = []
                row = {
                    **base, "condition_index": index, "repeat": repeat,
                    "policy": policy,
                    "planning_state": (
                        "all-bind" if policy == "queue_haul_robust" else
                        "all-release" if policy == "queue_haul_stale" else
                        "all-bind-90s" if policy == DEADLINE_BLIND_POLICY else
                        state),
                    "planner_seed": profiler.stable_seed(
                        parent["seed"], "hardware-gap", index, repeat, policy),
                    "kv_capacity_fraction": {
                        "east": round(1 - kv, 10), "germany": 1},
                    "expected_admission": not (
                        state == "all-bind" and policy == "queue_haul_stale"),
                }
                if moves:
                    row["moves"] = moves
                row["scenario_id"] = _hash([
                    "hardware_gap", state, repeat, policy, moves, bandwidth,
                    germany, kv, HARDWARE_GAP_TARGET_FRACTION,
                ])[:16]
                scenarios.append(row)
    blocks = [[row for row in scenarios
               if row["condition_index"] == index and row["repeat"] == repeat]
              for index in range(len(HARDWARE_GAP_MATRIX))
              for repeat in range(HARDWARE_GAP_REPEATS)]
    rng = random.Random(parent["seed"])
    for block in blocks:
        rng.shuffle(block)
    scenarios = [row for block in blocks for row in block]
    output = {
        **{key: parent[key] for key in (
            "schema", "seed", "manifest", "model_profile",
            "network_contract", "cluster", "calibration", "load_support",
        )},
        "design": "hardware_gap", "policies": list(HARDWARE_GAP_POLICIES),
        "conditions": [{
            "condition_id": state, "germany_service_load": germany,
            "east_kv_reserved_fraction": kv, "bandwidth": bandwidth,
            "policies": list(policies),
        } for state, germany, kv, bandwidth, policies in HARDWARE_GAP_MATRIX],
        "repeats": HARDWARE_GAP_REPEATS, "sessions_per_scenario": 28,
        "parent_plan": {"path": str(parent_path),
                        "sha256": profiler.file_hash(parent_path)},
        "oracle_plans": {"path": str(oracle_path),
                         "sha256": profiler.file_hash(oracle_path)},
        "scenarios": scenarios,
    }
    validate_plan(output)
    return output


def write_hardware_gap_plan(parent: Path, oracle: Path, out: Path) -> dict:
    plan = hardware_gap_plan(parent, oracle)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _validate_hardware_gap_simulation(rows: list[dict]) -> None:
    if {row["admission_mode"] for row in rows} != {"normal"}:
        raise RuntimeError("hardware gap admission modes are unmatched")
    selected = {(row["state"], row["policy"]): row for row in rows}
    target = selected["all-bind", "queue_haul_robust"]["requested_shed_w"]
    winner = selected["all-bind", "queue_haul_robust"]
    losers = [selected["all-bind", policy] for policy in (
        "greedy", "oracle_kv_only", "oracle_replay_only",
        "oracle_east_only", "oracle_germany_only", "isolated_fastest",
        "queue_haul_power_blind",
    )]
    winner_ratio = 1.015 if H100_CAMPAIGN else 1.1
    loser_ratio = .985 if H100_CAMPAIGN else .9
    if not winner["target_by_deadline"] \
            or winner["shed_by_deadline_w"] < winner_ratio * target \
            or max(row["shed_by_deadline_w"] for row in losers) \
            > loser_ratio * target:
        raise RuntimeError("all-bind hardware separation is not severe")
    stale = selected["all-bind", "queue_haul_stale"]
    blind = selected["all-bind", DEADLINE_BLIND_POLICY]
    blind_valid = (blind["target_by_deadline"] if H100_CAMPAIGN else
                   blind["status"] == "late"
                   and blind["shed_by_deadline_w"] <= .85 * target
                   and blind["eventual_shed_w"] >= 1.2 * target
                   and blind["time_to_target_s"] is not None
                   and 55 <= blind["time_to_target_s"]
                   <= ORACLE_STALE_HORIZON_S)
    stale_violations = set(stale["capacity_violations"])
    stale_valid = stale["status"] == "capacity_infeasible" and (
        bool(stale_violations) if H100_CAMPAIGN else
        {"kv:pool/east", "service:pool/germany:0"} <= stale_violations)
    if not stale_valid or not blind_valid:
        raise RuntimeError("stale or deadline hardware trap is not severe")
    if any(selected[state, "queue_haul"]["shed_by_deadline_w"]
           < winner_ratio * target
           for state in ("free-kv", "free-service", "free-bandwidth",
                         "all-release")) \
            or any(selected[state, "queue_haul_robust"][
                "shed_by_deadline_w"] < winner_ratio * target for state in (
                    "free-kv", "free-service", "free-bandwidth",
                    "all-release")) \
            or not H100_CAMPAIGN and selected[
                "free-kv", "greedy"]["shed_by_deadline_w"] < 1.1 * target \
            or selected["all-release", "queue_haul_stale"][
                "shed_by_deadline_w"] < 1.2 * target \
            or selected["all-release", "oracle_germany_only"][
                "shed_by_deadline_w"] > .95 * target:
        raise RuntimeError("hardware release controls are not separated")
    releases = (
        ("free-kv", "oracle_east_only", 1.5 if H100_CAMPAIGN else 4),
        ("free-service", "oracle_germany_only", 12),
        ("free-bandwidth", "oracle_kv_only", 1.5 if H100_CAMPAIGN else 7),
    )
    if any(selected[state, policy]["planned_shed_w"]
           - selected["all-bind", policy]["planned_shed_w"] < margin
           for state, policy, margin in releases):
        raise RuntimeError("a hardware constraint release is too small")


def plot_hardware_gap(rows: list[dict], curves: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    selected = {(row["state"], row["policy"]): row for row in rows}
    target = selected["all-bind", "queue_haul_robust"]["requested_shed_w"]
    policies = [policy for policy in HARDWARE_GAP_MATRIX[0][-1]
                if policy != "queue_haul_stale"]
    labels = [policy.replace("queue_haul_", "").replace("oracle_", "")
              .replace("_", " ") for policy in policies]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].bar(
        np.arange(len(policies)),
        [selected["all-bind", policy]["shed_by_deadline_w"]
         for policy in policies],
        color=[HARDWARE_GAP_COLORS[policy] for policy in policies],
    )
    axes[0].axhline(target, color="black", linestyle="--", label="Target")
    axes[0].set_xticks(np.arange(len(policies)), labels, rotation=35,
                       ha="right", fontsize=7)
    axes[0].set(ylabel="Shed by 45 s (W)", title="All constraints bind")
    axes[0].legend(frameon=False)

    releases = (
        ("all-bind", "queue_haul_robust", "oracle_replay_only"),
        ("free-kv", "queue_haul", "oracle_east_only"),
        ("free-service", "queue_haul", "oracle_germany_only"),
        ("free-bandwidth", "queue_haul", "oracle_kv_only"),
        ("all-release", "queue_haul", "oracle_germany_only"),
    )
    x = np.arange(len(releases))
    axes[1].bar(x - .18, [selected[state, joint]["shed_by_deadline_w"]
                          for state, joint, _ in releases], .36,
                color=HARDWARE_GAP_COLORS["queue_haul"], label="Queue-Haul")
    seen = set()
    for index, (state, _, policy) in enumerate(releases):
        label = policy.removeprefix("oracle_").replace("_", " ")
        axes[1].bar(
            index + .18, selected[state, policy]["shed_by_deadline_w"], .36,
            color=HARDWARE_GAP_COLORS[policy],
            label=label if policy not in seen else None,
        )
        seen.add(policy)
    axes[1].plot(
        x, [selected[state, "queue_haul_robust"]["shed_by_deadline_w"]
            for state, *_ in releases], marker="D", linestyle=":",
        color=HARDWARE_GAP_COLORS["queue_haul_robust"],
        label="Worst-corner robust plan",
    )
    axes[1].axhline(target, color="black", linestyle="--", label="Target")
    axes[1].set_xticks(x, [state for state, *_ in releases], rotation=25,
                       ha="right", fontsize=8)
    axes[1].set(ylabel="Shed by 45 s (W)", title="One-at-a-time releases")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)

    for policy in ("queue_haul_robust", DEADLINE_BLIND_POLICY):
        values = [row for row in curves if row["state"] == "all-bind"
                  and row["plan"] == policy]
        for index, repeat in enumerate(sorted(
                {row.get("repeat", 0) for row in values})):
            trace = sorted((row for row in values
                            if row.get("repeat", 0) == repeat),
                           key=lambda row: row["time_s"])
            axes[2].step(
                [row["time_s"] for row in trace],
                [row["shed_w"] for row in trace], where="post",
                color=HARDWARE_GAP_COLORS[policy],
                alpha=1 if len(values) == len(trace) else .45,
                label=(policy.replace("queue_haul_", "").replace("_", " ")
                       if index == 0 else None))
    axes[2].axvline(SEPARATION_DEADLINE_S, color="black", linestyle=":",
                    label="45 s deadline")
    axes[2].axhline(target, color="black", linestyle="--", label="Target")
    axes[2].set(xlim=(0, ORACLE_STALE_HORIZON_S), xlabel="Time (s)",
                ylabel="Source power shed (W)",
                title="Enough power, but too late")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"hardware_gap.{suffix}", dpi=200)
    plt.close(fig)


def simulate_hardware_gap(plan_path: Path, out: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if plan["design"] != "hardware_gap":
        raise ValueError("hardware gap simulation requires a gap plan")
    manifest = json.loads(Path(plan["manifest"]["path"]).read_text())
    profile = ModelProfile.load(MODEL_PATH)
    rows, curves, resources, duals = [], [], [], []
    for state, *_ in HARDWARE_GAP_MATRIX:
        scenarios = [row for row in plan["scenarios"]
                     if row["condition_id"] == state and row["repeat"] == 0]
        for scenario in scenarios:
            problem, architecture, routes, target, _demand = _scenario_problem(
                scenario, manifest, profile)
            policy = scenario["policy"]
            if scenario.get("moves"):
                moves = tuple(_planned_move(row) for row in scenario["moves"])
                admission_mode = scenario["admission_mode"]
            else:
                result = solve(
                    problem, profile, routes,
                    joint_solver(policy, scenario["objective"]),
                    seed=scenario["planner_seed"], destination=architecture,
                    admission_mode=scenario["admission_mode"],
                )
                moves, admission_mode = result.moves, result.admission_mode
            row = _evaluate_fixed_plan(
                state, policy, problem, architecture, profile, moves, target,
                curves)
            rows.append({**row, "policy": policy,
                         "admission_mode": admission_mode,
                         "bandwidth": scenario["bandwidth"],
                         "germany_service_load":
                         scenario["background"]["germany"][0],
                         "east_kv_reserved_fraction":
                         scenario["background"]["east"][1],
                         **_constraint_action_counts(moves)})
        problem, architecture, routes, _target, _demand = _scenario_problem(
            scenarios[0], manifest, profile)
        result = solve(
            problem, profile, routes, "max_shed", destination=architecture,
            admission_mode="normal")
        resources.extend({"state": state, "resource": row.name,
                          "unit": row.unit, "used": row.used,
                          "capacity": row.capacity,
                          "utilization": row.utilization}
                         for row in result.resource_uses)
        table = candidate_table(
            problem, profile, architecture, "normal",
            ExpectedPower(replace(
                problem, final_state="awake", assumed_shutdown_s=None),
                profile))
        ceiling, prices = phase_one_capacity_duals(table)
        duals.extend({
            "state": state, "phase_one_marginal_ceiling_w": ceiling,
            "resource": name, "capacity": capacity, "unit": unit,
            "shadow_w_per_full_capacity": float(price),
            "shadow_w_per_unit": float(price / capacity),
        } for name, capacity, unit, price in zip(
            table.resource_names, table.resource_capacities,
            table.resource_units, prices))
    _validate_hardware_gap_simulation(rows)
    out.mkdir(parents=True, exist_ok=False)
    profiler.write_csv(out / "hardware_gap_predictions.csv", rows)
    profiler.write_csv(out / "hardware_gap_curves.csv", curves)
    profiler.write_csv(out / "hardware_gap_resources.csv", resources)
    profiler.write_csv(out / "hardware_gap_duals.csv", duals)
    plot_hardware_gap(rows, curves, out)
    artifacts = {path.name: profiler.file_hash(path)
                 for path in sorted(out.iterdir()) if path.is_file()}
    write_checkpoint(out / "metadata.json", {
        "schema": "queue-haul-hardware-gap-simulation-v1",
        "plan": {"path": str(plan_path),
                 "sha256": profiler.file_hash(plan_path)},
        "target_fraction": HARDWARE_GAP_TARGET_FRACTION,
        "east_all_bind_kv_capacity_fraction": .1,
        "background_kv_headroom_tokens": HARDWARE_GAP_BACKGROUND_KV_TOKENS,
        "deadline_s": SEPARATION_DEADLINE_S,
        "full_horizon_s": ORACLE_STALE_HORIZON_S,
        "policy_colors": HARDWARE_GAP_COLORS,
        "artifacts": artifacts,
    })
    return {"scenarios": len(plan["scenarios"]),
            "conditions": len(HARDWARE_GAP_MATRIX),
            "out": str(out), "valid": True}


def _valid_constraint_evidence(scenario: dict, result: dict) -> bool:
    background = result.get("background")
    return all(key in result for key in (
        "deadline_met", "target_met", "request_failures",
        "kv_evidence_warnings", "load_warnings", "background",
    )) and result["deadline_met"] is True \
        and result["target_met"] is False \
        and result["request_failures"] == 0 \
        and result["kv_evidence_warnings"] == 0 \
        and result["load_warnings"] == [] \
        and set(background) == {"east", "germany"} \
        and not any(row.get("warning") for row in background.values())


def _valid_separation_evidence(scenario: dict, result: dict) -> bool:
    background = result.get("background")
    if not all(key in result for key in (
        "deadline_met", "target_met", "requested_shed_w", "realized_shed_w",
        "request_failures", "kv_evidence_warnings", "load_warnings",
        "background",
    )) or result["request_failures"] or result["kv_evidence_warnings"] \
            or result["load_warnings"] or set(background) != {"east", "germany"} \
            or any(row.get("warning") for row in background.values()):
        return False
    ratio = result["realized_shed_w"] / result["requested_shed_w"]
    winner = scenario["policy"] in SEPARATION_POLICIES[:2]
    moves = result.get("requests", ())
    methods = {row.get("method") for row in moves if "request" in row}
    destinations = {row.get("destination_instance") for row in moves
                    if "request" in row}
    return (
        ratio >= 1 + SEPARATION_MARGIN and result["deadline_met"] is True
        and methods == {"replay", "kv_transfer"}
        and destinations == {"east", "germany"}
        if winner else ratio <= 1 - SEPARATION_MARGIN
    )


def _valid_hardware_gap_evidence(scenario: dict, result: dict) -> bool:
    background = result.get("background", {})
    required = {
        "request_failures", "kv_evidence_warnings", "load_warnings",
        "requested_shed_w", "realized_shed_w", "eventual_shed_w",
        "time_to_target_s", "deadline_met", "target_met",
    }
    if not required <= result.keys() \
            or result.get("request_failures") or result.get("kv_evidence_warnings") \
            or result.get("load_warnings") or set(background) \
            != {"east", "germany"} or any(
                row.get("warning") for row in background.values()) \
            or result.get("admission_mode") != scenario["admission_mode"]:
        return False
    expected = scenario["kv_capacity_fraction"]
    if any(abs(background[node].get("kv_capacity_fraction", -1)
               - expected[node]) > .01 for node in expected):
        return False
    if not scenario["expected_admission"]:
        violations = set(result.get("capacity_violations", ()))
        return result.get("admission_rejected") is True \
            and not result.get("requests") \
            and (bool(violations) if H100_CAMPAIGN else
                 {"kv:pool/east", "service:pool/germany:0"} <= violations)
    if result.get("admission_rejected") or result.get("request_failures"):
        return False
    ratio = result["realized_shed_w"] / result["requested_shed_w"]
    state, policy = scenario["condition_id"], scenario["policy"]
    if policy == DEADLINE_BLIND_POLICY:
        target_time = result["time_to_target_s"]
        return result["deadline_met"] is False \
            and ratio <= (.985 if H100_CAMPAIGN else .85) \
            and result["eventual_shed_w"] >= (
                1.015 if H100_CAMPAIGN else 1.2) * result["requested_shed_w"] \
            and target_time is not None \
            and (scenario.get("deadline_s", SEPARATION_DEADLINE_S)
                 if H100_CAMPAIGN else 55) \
            < target_time <= scenario["full_horizon_s"]
    if state == "all-bind":
        if policy == "queue_haul_robust":
            moves = result.get("requests", ())
            return ratio >= (1.015 if H100_CAMPAIGN else 1.1) \
                and result["deadline_met"] is True \
                and {row["method"] for row in moves} \
                == {"replay", "kv_transfer"} \
                and {row["destination_instance"] for row in moves} \
                == {"east", "germany"}
        return ratio <= (.985 if H100_CAMPAIGN else .9) \
            and result["deadline_met"] is True
    if policy in {
            "queue_haul", "queue_haul_robust", "greedy",
            "queue_haul_stale"}:
        if H100_CAMPAIGN and policy == "greedy":
            return result["deadline_met"] is True
        return ratio >= (1.015 if H100_CAMPAIGN else 1.1) \
            and result["deadline_met"] is True
    return result["deadline_met"] is True and result["target_met"] is False


def _valid_hardware_gap_block(evidence: list[tuple[dict, dict]]) -> bool:
    groups = {}
    for scenario, result in evidence:
        groups.setdefault((scenario["condition_id"], scenario["policy"]),
                          []).append(result)
    if any(len(rows) != HARDWARE_GAP_REPEATS for rows in groups.values()):
        return False
    med = {key: {
        field: statistics.median(row[field] for row in rows)
        for field in ("requested_shed_w", "realized_shed_w",
                      "eventual_shed_w")
    } for key, rows in groups.items() if all(
        field in row for row in rows for field in (
            "requested_shed_w", "realized_shed_w", "eventual_shed_w"))}
    target = med["all-bind", "queue_haul_robust"]["requested_shed_w"]
    releases = (
        ("free-kv", "oracle_east_only", 1.5 if H100_CAMPAIGN else 4),
        ("free-service", "oracle_germany_only", 12),
        ("free-bandwidth", "oracle_kv_only", 1.5 if H100_CAMPAIGN else 7),
    )
    return all(
        med[state, policy]["realized_shed_w"]
        - med["all-bind", policy]["realized_shed_w"] >= margin
        for state, policy, margin in releases
    ) and med["all-release", "oracle_germany_only"]["realized_shed_w"] \
        <= .95 * target


def reduce_run(plan: dict, run_root: Path) -> dict:
    rows, evidence, completed, failed, missing, invalid_evidence = [], [], 0, 0, 0, 0
    constraint = plan.get("design") == "constraint"
    separation = plan.get("design") == "separation"
    hardware_gap = plan.get("design") == "hardware_gap"
    manifest = json.loads(Path(plan["manifest"]["path"]).read_text()) \
        if separation or hardware_gap else None
    profile = ModelProfile.load(MODEL_PATH) \
        if separation or hardware_gap else None
    for scenario in plan["scenarios"]:
        latest = _latest_result(run_root / "scenarios" / scenario["scenario_id"])
        if latest is None:
            missing += 1
            attempt, result = 0, {"status": "missing"}
        else:
            attempt, result = latest
            completed += result["status"] == "complete"
            failed += result["status"] == "failed"
            if (separation or hardware_gap) \
                    and result["status"] == "complete":
                demand = agentic_demand(
                    scenario_records(manifest, scenario), scenario["sessions"],
                    profile, scenario["source_load"])
                outcomes = diagnostic_outcomes(
                    scenario, result["requests"], demand, profile,
                    result["started_ns"])
                result = {**result, **outcomes}
            if result["status"] == "complete":
                evidence.append((scenario, result))
        if constraint and result.get("status") == "complete" \
                and not _valid_constraint_evidence(scenario, result):
            invalid_evidence += 1
        if separation and result.get("status") == "complete" \
                and not _valid_separation_evidence(scenario, result):
            invalid_evidence += 1
        if hardware_gap and result.get("status") == "complete" \
                and not _valid_hardware_gap_evidence(scenario, result):
            invalid_evidence += 1
        connections = result.get("connections", [])
        rtts = [float(row["target_rtt_us"]) / 1000 for row in connections
                if row.get("target_rtt_us")]
        wire = result.get("wire_bytes", {})
        destinations = sorted({request.get("destination_instance", "")
                               for request in result.get("requests", [])} - {""})
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "condition_index": scenario["condition_index"],
            "repeat": scenario["repeat"], "policy": scenario["policy"],
            "destination": ",".join(destinations) or scenario.get("destination", ""),
            "workload": scenario["workload"],
            "bandwidth": scenario["bandwidth"],
            "background": json.dumps(scenario.get("background", {}), sort_keys=True),
            "pack": scenario.get("pack", ""),
            "session_count": len(scenario.get("sessions", ())),
            "movement_tokens": sum(item.get("initial_tokens", 0)
                                   for item in scenario.get("sessions", ())),
            "deadline_s": scenario["deadline_s"], "attempt": attempt,
            "status": result["status"],
            "migration_s": result.get("migration_s", ""),
            "deadline_met": result.get("deadline_met", ""),
            "requested_shed_w": result.get("requested_shed_w", ""),
            "realized_shed_w": result.get("realized_shed_w", ""),
            "target_met": result.get("target_met", ""),
            "eventual_shed_w": result.get("eventual_shed_w", ""),
            "eventual_target_met": result.get("eventual_target_met", ""),
            "time_to_target_s": result.get("time_to_target_s", ""),
            "admission_rejected": result.get("admission_rejected", ""),
            "capacity_violations": json.dumps(
                result.get("capacity_violations", ())),
            "request_failures": result.get("request_failures", ""),
            "kv_evidence_warnings": result.get("kv_evidence_warnings", ""),
            "load_warnings": len(result.get("load_warnings", ())),
            "background_warnings": sum(
                bool(value.get("warning"))
                for value in result.get("background", {}).values()),
            "api_request_bytes": sum(value for key, value in wire.items()
                                     if key.endswith("/client_to_target")
                                     and key.startswith("api/")),
            "kv_response_bytes": sum(value for key, value in wire.items()
                                    if key.endswith("/target_to_client")
                                    and key.startswith("kv/")),
            "median_tcp_rtt_ms": statistics.median(rtts) if rtts else "",
            "retransmissions": sum(int(row.get("target_total_retrans") or 0)
                                   for row in connections),
            "error": result.get("error", ""),
        })
    profiler.write_csv(run_root / "results.csv", rows)
    frontier = plan.get("design") == "frontier"
    if frontier and evidence:
        plot_frontier(plan, evidence, run_root)
    if constraint and len(evidence) == len(plan["scenarios"]):
        plotted = []
        for scenario, result in evidence:
            moves = result.get("requests", [])
            plotted.append({
                "condition_id": scenario["condition_id"],
                "deadline_s": scenario["deadline_s"],
                "policy": scenario["policy"],
                "deadline_s": scenario["deadline_s"],
                "requested_shed_w": result["requested_shed_w"],
                "attained_shed_w": result["realized_shed_w"],
                "selected_sessions": len(moves),
                **_constraint_action_counts(moves),
            })
        plot_constraint(plotted, [], run_root)
    if separation and len(evidence) == len(plan["scenarios"]):
        plotted = []
        for scenario, result in evidence:
            moves = [row for row in result.get("requests", ()) if "request" in row]
            plotted.append({
                "condition_id": scenario["condition_id"],
                "repeat": scenario["repeat"], "policy": scenario["policy"],
                "requested_shed_w": result["requested_shed_w"],
                "attained_shed_w": result["realized_shed_w"],
                "selected_sessions": len(moves),
                **_constraint_action_counts(moves),
            })
        plot_separation(plotted, [], run_root)
    if hardware_gap and len(evidence) == len(plan["scenarios"]):
        if not _valid_hardware_gap_block(evidence):
            invalid_evidence += 1
        plotted, curves = [], []
        for scenario, result in evidence:
            plotted.append({
                "state": scenario["condition_id"],
                "policy": scenario["policy"],
                "requested_shed_w": result["requested_shed_w"],
                "shed_by_deadline_w": result["realized_shed_w"],
            })
            curves.extend({
                **row, "state": scenario["condition_id"],
                "plan": scenario["policy"], "repeat": scenario["repeat"],
            } for row in result["attainment_curve"])
        medians = []
        for key in {(row["state"], row["policy"]) for row in plotted}:
            group = [row for row in plotted
                     if (row["state"], row["policy"]) == key]
            medians.append({
                "state": key[0], "policy": key[1],
                "requested_shed_w": statistics.median(
                    row["requested_shed_w"] for row in group),
                "shed_by_deadline_w": statistics.median(
                    row["shed_by_deadline_w"] for row in group),
            })
        plot_hardware_gap(medians, curves, run_root)
    summary = {
        "schema": "queue-haul-network-summary-v1",
        "expected": len(plan["scenarios"]), "completed": completed,
        "failed": failed, "missing": missing,
        "valid": not missing and not (
            (constraint or separation or hardware_gap)
            and invalid_evidence) and (
            completed == len(plan["scenarios"]) if not frontier else
            failed / len(plan["scenarios"]) <= FRONTIER_FAILURE_GATE),
    }
    if constraint or separation or hardware_gap:
        summary["invalid_evidence"] = invalid_evidence
    (run_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    artifacts = [path for path in sorted(run_root.rglob("*"))
                 if path.is_file() and path.name != "artifacts.sha256"]
    (run_root / "artifacts.sha256").write_text("".join(
        f"{profiler.file_hash(path)}  {path.relative_to(run_root)}\n"
        for path in artifacts))
    return summary


def merge_metadata(current: dict, previous: dict | None) -> dict:
    if previous is None:
        return current
    def core(row):
        return {key: ({node: {field: value for field, value in report.items()
                              if field != "git_sha"}
                       for node, report in value.items()}
                      if key == "hosts" else value)
                for key, value in row.items()
                if key not in {"checks", "git_sha"}}
    if core(current) != core(previous):
        raise RuntimeError("run metadata changed; use a new run root")
    current["checks"] = previous["checks"] + current["checks"]
    return current


def run_campaign(cluster: Cluster, key: Path, current_calibration: Path,
                 plan_path: Path, run_root: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    if Cluster.parse(plan["cluster"]) != cluster:
        raise ValueError("run cluster differs from the prepared plan")
    if profiler.file_hash(Path(plan["manifest"]["path"])) \
            != plan["manifest"]["sha256"] \
            or profiler.file_hash(MODEL_PATH) != plan["model_profile"]["sha256"]:
        raise RuntimeError("pinned plan input changed")
    current = freeze_contract(json.loads(current_calibration.read_text()))
    validate_resume(plan["network_contract"], current)
    reports = host_check(cluster, key)
    sha, dirty = profiler.git_state(False)
    run_root.mkdir(parents=True, exist_ok=True)
    identity_fields = (
        "git_sha", "dirty", "gpu", "gpu_memory_mib", "vllm", "lmcache",
        "vm_size", "priority", "ptp", "datadrive", "private_ip", "region",
    )
    metadata = {
        "schema": "queue-haul-network-run-v1",
        "plan_sha256": profiler.file_hash(plan_path), "git_sha": sha,
        "dirty": dirty,
        "hosts": {node: {field: report.get(field) for field in identity_fields}
                  for node, report in reports.items()},
        "checks": [{"wall_ns": time.time_ns(), "hosts": reports,
            "calibration": {"path": str(current_calibration),
                "sha256": profiler.file_hash(current_calibration),
                "contract": current}}],
    }
    metadata_path = run_root / "run_metadata.json"
    previous = json.loads(metadata_path.read_text()) \
        if metadata_path.exists() else None
    metadata = merge_metadata(metadata, previous)
    write_checkpoint(metadata_path, metadata)
    write_checkpoint(run_root / "plan.json", plan)
    checkpoint_progress(plan, run_root)
    manifest = json.loads(Path(plan["manifest"]["path"]).read_text())
    prefill_tps = ModelProfile.load(MODEL_PATH).case().F
    stack, stack_key = None, None
    try:
        for scenario in plan["scenarios"]:
            scenario_root = run_root / "scenarios" / scenario["scenario_id"]
            latest = _latest_result(scenario_root)
            if latest and latest[1].get("status") == "complete":
                continue
            wanted = (scenario["bandwidth"], tuple(sorted(
                scenario.get("kv_capacity_fraction", {}).items())))
            if stack and stack_key != wanted:
                stop_cluster(stack)
                stack = None
            if stack is None:
                stack_key = wanted
                stack = start_cluster(
                    cluster, key, plan["network_contract"],
                    scenario["bandwidth"],
                    run_root / "stacks" /
                    f"{scenario['bandwidth']}-{time.time_ns()}",
                    kv_capacity_fraction=scenario.get(
                        "kv_capacity_fraction"),
                )
            attempt = _next_attempt(scenario_root)
            attempt_root = scenario_root / f"attempt-{attempt:04d}"
            try:
                run_network_scenario(
                    stack, manifest, scenario, attempt_root, prefill_tps)
            except Exception as exc:
                attempt_root.mkdir(parents=True, exist_ok=True)
                write_checkpoint(attempt_root / "result.json", {
                    "schema": RESULT_SCHEMA, "status": "failed",
                    "scenario_id": scenario["scenario_id"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
                stop_cluster(stack)
                stack = None
                checkpoint_progress(plan, run_root)
                if plan["design"] not in {
                        "frontier", "constraint", "separation",
                        "hardware_gap"}:
                    raise
            checkpoint_progress(plan, run_root)
    finally:
        if stack:
            stop_cluster(stack)
    summary = reduce_run(plan, run_root)
    if not summary["valid"]:
        raise RuntimeError(
            f"network campaign incomplete: {summary['failed']} failed, "
            f"{summary['missing']} missing, "
            f"{summary.get('invalid_evidence', 0)} invalid evidence")
    return summary


def prepare(cluster_path: Path, calibration_path: Path, manifest_path: Path,
            out: Path, seed: int = 1, sessions: int = 8,
            design: str = "joint") -> dict:
    cluster = Cluster.load(cluster_path)
    calibration = json.loads(calibration_path.read_text())
    plan = make_plan(manifest_path, freeze_contract(calibration), seed, sessions,
                     design)
    plan["cluster"] = cluster.as_dict()
    plan["calibration"] = {
        "path": str(calibration_path),
        "sha256": profiler.file_hash(calibration_path),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def write_frontier_refinement(plan_path: Path, run_root: Path, out: Path) -> dict:
    refined = frontier_refinement(json.loads(plan_path.read_text()), run_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(refined, indent=2, sort_keys=True) + "\n")
    return refined


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--seed", type=int, default=1)
    command.add_argument("--sessions", type=int, default=8)
    command.add_argument("--design",
                         choices=("joint", "isolated", "frontier", "constraint",
                                  "separation"),
                         default="joint")
    command = sub.add_parser("check")
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path,
                         default=Path("~/.ssh/azrs").expanduser())
    command = sub.add_parser("calibrate")
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path,
                         default=Path("~/.ssh/azrs").expanduser())
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--seconds", type=int, default=60)
    command.add_argument("--repeats", type=int, default=3)
    command.add_argument("--ping-count", type=int, default=200)
    sub.add_parser("node-check")
    command = sub.add_parser("node-serve")
    command.add_argument("--node-id", required=True)
    command.add_argument("--bind-host", required=True)
    command.add_argument("--source-host", required=True)
    command.add_argument("--kv-port", type=int, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--power-interval-s", type=float, default=.25)
    command.add_argument("--kv-blocks", type=int)
    command = sub.add_parser("smoke")
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path,
                         default=Path("~/.ssh/azrs").expanduser())
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--bandwidth", default="natural",
                         choices=("natural", "controlled_40", "controlled_80"))
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--words", type=int, default=1024)
    command = sub.add_parser("run")
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path,
                         default=Path("~/.ssh/azrs").expanduser())
    command.add_argument("--current-calibration", type=Path, required=True)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command = sub.add_parser("reduce")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command = sub.add_parser("simulate-constraint")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("simulate-separation")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("simulate-oracle-stale")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("hardware-gap")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--oracle-plans", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("simulate-hardware-gap")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("refine")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("deadline-blind")
    command.add_argument("--plan", type=Path, action="append", required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("handoff")
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path,
                         default=Path("~/.ssh/azrs").expanduser())
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--window-s", type=float, default=300)
    command.add_argument("--destination-load", type=float, default=.5)
    command.add_argument("--power-interval-s", type=float, default=.1)
    command.add_argument("--policy", choices=HANDOFF_POLICIES,
                         default="queue_haul")
    command.add_argument("--repeat", type=int, choices=range(REPEATS), default=0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args.cluster, args.calibration, args.manifest, args.out,
                args.seed, args.sessions, args.design)
    elif args.command == "node-check":
        print(json.dumps(node_report(), sort_keys=True))
    elif args.command == "check":
        print(json.dumps(host_check(
            Cluster.load(args.cluster), args.ssh_key.expanduser()),
            indent=2, sort_keys=True))
    elif args.command == "calibrate":
        print(json.dumps(calibrate(
            Cluster.load(args.cluster), args.ssh_key.expanduser(), args.out,
            args.seconds, args.repeats, args.ping_count,
        ), indent=2, sort_keys=True))
    elif args.command == "node-serve":
        node_serve(args.node_id, args.bind_host, args.source_host,
                   args.kv_port, args.run_root, args.power_interval_s,
                   args.kv_blocks)
    elif args.command == "smoke":
        print(json.dumps(smoke(
            Cluster.load(args.cluster), args.ssh_key.expanduser(),
            json.loads(args.calibration.read_text()), args.bandwidth,
            args.run_root, args.words,
        ), indent=2, sort_keys=True))
    elif args.command == "run":
        print(json.dumps(run_campaign(
            Cluster.load(args.cluster), args.ssh_key.expanduser(),
            args.current_calibration, args.plan, args.run_root,
        ), indent=2, sort_keys=True))
    elif args.command == "reduce":
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        print(json.dumps(reduce_run(plan, args.run_root),
                         indent=2, sort_keys=True))
    elif args.command == "simulate-constraint":
        print(json.dumps(simulate_constraint(args.plan, args.out),
                         indent=2, sort_keys=True))
    elif args.command == "simulate-separation":
        print(json.dumps(simulate_separation(args.plan, args.out),
                         indent=2, sort_keys=True))
    elif args.command == "simulate-oracle-stale":
        print(json.dumps(simulate_oracle_stale(args.plan, args.out),
                         indent=2, sort_keys=True))
    elif args.command == "hardware-gap":
        plan = write_hardware_gap_plan(
            args.plan, args.oracle_plans, args.out)
        print(json.dumps({"scenarios": len(plan["scenarios"]),
                          "out": str(args.out)}, sort_keys=True))
    elif args.command == "simulate-hardware-gap":
        print(json.dumps(simulate_hardware_gap(args.plan, args.out),
                         indent=2, sort_keys=True))
    elif args.command == "refine":
        refined = write_frontier_refinement(
            args.plan, args.run_root, args.out)
        print(json.dumps({"scenarios": len(refined["scenarios"]),
                          "out": str(args.out)}, sort_keys=True))
    elif args.command == "deadline-blind":
        plan = deadline_blind_plan([
            json.loads(path.read_text()) for path in args.plan])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"scenarios": len(plan["scenarios"]),
                          "out": str(args.out)}, sort_keys=True))
    elif args.command == "handoff":
        print(json.dumps(run_handoff(
            Cluster.load(args.cluster), args.ssh_key.expanduser(),
            args.calibration, args.plan, args.manifest, args.run_root,
            args.window_s, args.destination_load, args.power_interval_s,
            args.policy, args.repeat,
        ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
