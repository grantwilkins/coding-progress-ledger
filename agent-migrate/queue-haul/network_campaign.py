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
from planner import plan as solve, source_power
from pool_planner import candidate_table, phase_one_capacity_duals
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile
from simulate import NetworkLink, PowerNode, ServingInstance


CLUSTER_SCHEMA = "queue-haul-azure-cluster-v1"
CALIBRATION_SCHEMA = "queue-haul-network-calibration-v1"
PLAN_SCHEMA = "queue-haul-network-plan-v2"
RESULT_SCHEMA = "queue-haul-network-result-v2"
CLOCK_LIMIT_MS = 2.0
RESUME_DRIFT = .10
REQUEST_TIMEOUT_S = 600.0
REPEATS = 3
ISOLATED_PROMPT_HEADROOM_TOKENS = 512
POLICIES = (
    "queue_haul", "greedy", "greedy_lagrangian", "kv_only", "replay_only",
    "random",
)
FRONTIER_POLICIES = (
    "queue_haul", "greedy", "replay_only", "kv_only",
    "queue_haul_power_blind",
)
CONSTRAINT_POLICIES = (
    "queue_haul", "greedy", "kv_only", "replay_only", "isolated_fastest",
    "queue_haul_power_blind",
)
CONSTRAINT_CELLS = (
    ("window-19", 19, 22, 15, 1.0),
    ("window-30", 30, 28, 8, 1.0),
    ("window-60", 60, 64, None, 1.0),
    ("quota-30", 30, 28, 8, 1.0),
)
CONSTRAINT_ACTIONS = (
    "germany_kv_transfer", "east_replay", "east_kv_transfer",
    "germany_replay",
)
TAB10_COLORS = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
FRONTIER_PACKS = (
    ("4x16k", 4, 16_384), ("8x16k", 8, 16_384),
    ("16x16k", 16, 16_384), ("8x8k", 8, 8_192),
    ("8x24k", 8, 24_576), ("8x31k", 8, 31_488),
)
FRONTIER_LOADS = (0, .5, .85, .9, .95)
FRONTIER_FAILURE_GATE = .5
FRONTIER_REFINEMENT_EPISODES = 65
DEADLINE_BLIND_POLICY = "queue_haul_deadline_blind"
DEADLINE_BLIND_HORIZON_S = 600
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json"
WORKLOAD_PATHS = {name: ROOT / f"profiles/{name}.json" for name in (
    "coding", "interactive_coding", "agentic_tool_loop",
)}
EXPECTED_RUNTIME = {"vllm": "0.22.0", "lmcache": "0.5.1"}
HANDOFF_DEADLINE_S = 30
HANDOFF_POLICIES = ("queue_haul", "kv_only", "replay_only")
HANDOFF_ENV = {
    "QH_KV_ROLE_SOURCE": "kv_both", "QH_KV_ROLE_SINK": "kv_both",
    "QH_LMCACHE_L1_GB": "33", "QH_PREFIX_CACHING": "off",
    "QH_REDIS_MAXMEMORY_GB": "32",
}


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
        if value.source.region != "swedencentral" \
                or not {node.region for node in value.destinations} \
                <= {"eastus2", "westeurope", "germanywestcentral"}:
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
        "uv", "run", "python", "queue-haul/network_campaign.py", "node-check",
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
        if report.get("dirty") or report.get("vm_size") \
                != "Standard_NC24ads_A100_v4" \
                or "A100" not in report.get("gpu", "") \
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
    if design not in {"joint", "isolated", "frontier", "constraint"} \
            or len(destinations) != (
        2 if design in {"joint", "frontier", "constraint"} else 1
    ):
        raise ValueError(f"{design} design requires "
                         f"{'two destinations' if design in {'joint', 'frontier', 'constraint'} else 'one destination'}")
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
            (pack, loads) for pack in FRONTIER_PACKS for loads in load_pairs
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
    if design == "constraint":
        blocks = [[row for row in scenarios if row["condition_index"] == index]
                  for index in range(len(CONSTRAINT_CELLS))]
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
                         CONSTRAINT_POLICIES if design == "constraint" else POLICIES),
        "conditions": target_conditions(destinations) if design == "joint" else
            [{"condition_id": row[0], "deadline_s": row[1],
              "sessions": row[2], "context_seed": row[3],
              "requested_shed_fraction": row[4]}
             for row in CONSTRAINT_CELLS] if design == "constraint" else [],
        "repeats": 1 if design in {"frontier", "constraint"} else REPEATS,
        "sessions_per_scenario": None if design == "constraint" else sessions,
        "scenarios": scenarios,
    }
    validate_plan(output)
    return output


def validate_plan(plan: dict) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid network plan schema")
    scenarios = plan.get("scenarios", [])
    design = plan.get("design")
    expected = 126 if design == "joint" else 54 if design == "isolated" \
        else (185 if plan.get("phase", "pilot") == "pilot" else
              len(scenarios)) if design == "frontier" \
        else 24 if design == "constraint" else 0
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


def node_serve(node_id: str, bind_host: str, source_host: str, kv_port: int,
               run_root: Path, power_interval_s: float = .25) -> None:
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
        sink = testbed.start_logged(testbed.vllm_cmd(
            cfg, "sink", gpu_index=0, bind_host=bind_host), sink_log)
        testbed.wait_health_process(
            bind_host, cfg.sink_port, testbed.health_timeout(), sink, sink_log)
        sampler.start()
        print(json.dumps({
            "status": "ready", "node_id": node_id, "host": bind_host,
            "vllm_port": cfg.sink_port, "kv_port": kv_port,
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
                       timeout_s: float) -> None:
    with ThreadPoolExecutor(max_workers=len(remote)) as pool:
        futures = {node_id: pool.submit(_remote_ready, process, timeout_s)
                   for node_id, process in remote.items()}
        for node_id in sorted(futures):
            futures[node_id].result()


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
                  power_interval_s: float = .25) -> ClusterStack:
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
                "env", *(f"{name}={os.environ[name]}" for name in HANDOFF_ENV
                         if name in os.environ),
                "uv", "run", "python", "queue-haul/network_campaign.py",
                "node-serve", "--node-id", node_id, "--bind-host", node.host,
                "--source-host", cluster.source.host, "--kv-port",
                str(ports[node_id]["kv"]), "--run-root", str(remote_root),
                "--power-interval-s", str(power_interval_s),
            ])
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            remote[node_id] = process
        _wait_remote_ready(remote, testbed.health_timeout())
        local = testbed.Stack(
            lmc, proxy, source, None, run_root, services,
            aggregate or 0,
        )
        return ClusterStack(
            cluster, cfg, local, sampler, remote, remote_roots, ports,
            run_root, key, spot,
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


def summarize_metrics(samples: list[str], target_kv: float) -> dict:
    names = ("kv_cache_usage_perc", "num_requests_running", "num_requests_waiting")
    values = {name: [] for name in names}
    for sample in samples:
        for name in names:
            match = re.search(rf"^vllm:{name}(?:\{{[^\n]*\}})?\s+([0-9.eE+-]+)$",
                              sample, re.MULTILINE)
            if not match:
                raise ValueError(f"missing vLLM metric {name}")
            values[name].append(float(match.group(1)))
    result = {
        "kv_fraction": statistics.median(values["kv_cache_usage_perc"]),
        "running": statistics.median(values["num_requests_running"]),
        "waiting": max(values["num_requests_waiting"]),
    }
    result["warning"] = abs(result["kv_fraction"] - target_kv) > .05 \
        or result["waiting"] > 0
    return result


def destination_metrics(stack: ClusterStack, node_id: str,
                        target_kv: float) -> dict:
    samples = []
    for index in range(5):
        samples.append(testbed.http_text(
            stack.cfg.host, stack.ports[node_id]["api"], "GET", "/metrics"))
        if index < 4:
            time.sleep(1)
    return summarize_metrics(samples, target_kv)


def joint_problem(scenario: dict, snapshots: dict[str, dict],
                  profile: ModelProfile,
                  demand: dict[str, tuple[float, float]] | None = None):
    base, _ = policy_campaign._problem(
        profile, scenario["sessions"], 1, scenario["deadline_s"])
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
            node, tuple(dtype.work(
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
    partial = scenario.get("design") in {"frontier", "constraint"}
    deadline_blind = scenario["policy"] == DEADLINE_BLIND_POLICY
    solver = joint_solver(scenario["policy"], scenario.get("objective"))
    planning_problem = replace(
        problem, deadline_s=DEADLINE_BLIND_HORIZON_S,
        end_s=max(problem.end_s, DEADLINE_BLIND_HORIZON_S),
    ) if deadline_blind else problem
    result = solve(planning_problem, profile, routes, solver, seed=seed,
                   destination=architecture)
    planned = list(result.moves)
    admitted = {move.session_id for move in planned}
    missing = tuple(row for row in problem.sessions if row.session_id not in admitted)
    if missing and not partial:
        late = replace(problem, sessions=missing, deadline_s=600, end_s=600)
        planned.extend(replace(move, order=move.order + len(planned)) for move in solve(
            late, profile, routes, solver, seed=seed, destination=architecture,
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


class SinkLoad:
    def __init__(self, cfg: testbed.Config, port: int, prefill_tps: float,
                 rho: float, path: Path):
        if prefill_tps <= 0 or not 0 < rho < 1:
            raise ValueError("invalid sink load")
        self.cfg, self.port, self.prefill_tps, self.rho, self.path = (
            cfg, port, prefill_tps, rho, path)
        self.stop, self.error = threading.Event(), None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _request(self, index: int) -> dict:
        messages = [
            {"role": "system", "content": "You are a tool-using coding agent."},
            {"role": "user", "content":
             f"Agentic trace turn {index}: analyze tool output. " + "x " * 512},
        ]
        result, _ = profiler.stream_chat(
            self.cfg, self.port, messages, 64,
            profiler.messages_hash(messages), 600, True, f"load-{index}")
        if result.status_code != 200:
            raise RuntimeError(f"sink load request failed: {result.status_code}")
        return asdict(result)

    def _run(self) -> None:
        futures, index = [], 0
        interval = 512 / (self.rho * self.prefill_tps)
        next_at = time.monotonic()
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                while not self.stop.is_set():
                    delay = next_at - time.monotonic()
                    if delay > 0 and self.stop.wait(delay):
                        break
                    futures.append(pool.submit(self._request, index))
                    index += 1
                    next_at += interval
                rows = [future.result() for future in futures]
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
            raise RuntimeError("sink load failed") from self.error

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
    diagnostic = scenario["design"] in {"frontier", "constraint"}
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
    timeout = REQUEST_TIMEOUT_S
    loads, snapshots = {}, {}
    for node in stack.cluster.destinations:
        compute, kv = scenario["background"].get(node.id, (0, 0))
        if compute:
            loads[node.id] = SinkLoad(
                stack.cfg, stack.ports[node.id]["api"], prefill_tps, compute,
                root / f"sink_load_{node.id}.jsonl")
            loads[node.id].start()
    if scenario.get("source_load"):
        loads["source"] = SinkLoad(
            stack.cfg, stack.cfg.src_port, prefill_tps, scenario["source_load"],
            root / "source_load.jsonl")
        loads["source"].start()
    try:
        if scenario["design"] in {"joint", "frontier", "constraint"}:
            time.sleep(5)
            nodes = stack.cluster.destinations
            def snapshot(node):
                try:
                    return destination_metrics(
                        stack, node.id, scenario["background"][node.id][1])
                except Exception as exc:
                    if not diagnostic:
                        raise
                    return {"kv_fraction": 0, "warning": True,
                            "error": f"{type(exc).__name__}: {exc}"}
            with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
                snapshots = dict(zip(
                    (node.id for node in nodes),
                    pool.map(snapshot, nodes),
                ))
            profile = ModelProfile.load(MODEL_PATH)
            demand = agentic_demand(
                sessions, scenario["sessions"], profile,
                scenario.get("source_load", .4))
            moves = plan_joint_scenario(
                scenario, snapshots, profile, scenario["planner_seed"], demand)
            write_checkpoint(root / "decision.json", {
                "background": snapshots, "moves": moves,
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
    finally:
        for name, load in loads.items():
            try:
                load.close()
            except RuntimeError as exc:
                if not diagnostic:
                    raise
                load_warnings.append(f"{name}: {exc}")
    end_ns = time.monotonic_ns()
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
        profile = ModelProfile.load(MODEL_PATH)
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
        credited = {row["session_id"] for row in results
                    if "request" in row
                    and (row["request"]["end_ns"] - row["request"]["start_ns"])
                    / 1e9 <= scenario["deadline_s"]}
        realized = initial - source_power(base, profile, credited)
        result.update({
            "requested_shed_w": requested, "realized_shed_w": realized,
            "target_met": realized >= requested,
            "request_failures": sum("error" in row for row in results),
            "kv_evidence_warnings": sum(
                row["method"] == "kv_transfer" and "request" in row
                and row["request"]["cached_tokens"] <= 0 for row in results),
            "load_warnings": load_warnings,
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

    plot_policies = (*FRONTIER_POLICIES, DEADLINE_BLIND_POLICY)
    profile, colors = ModelProfile.load(MODEL_PATH), {
        policy: color for policy, color in zip(plot_policies, (
            "#B1040E", "#008566", "#E98300", "#006CB8", "#6F42C1",
            "#17BECF"))}
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


def reduce_run(plan: dict, run_root: Path) -> dict:
    rows, evidence, completed, failed, missing, invalid_evidence = [], [], 0, 0, 0, 0
    constraint = plan.get("design") == "constraint"
    for scenario in plan["scenarios"]:
        latest = _latest_result(run_root / "scenarios" / scenario["scenario_id"])
        if latest is None:
            missing += 1
            attempt, result = 0, {"status": "missing"}
        else:
            attempt, result = latest
            completed += result["status"] == "complete"
            failed += result["status"] == "failed"
            if result["status"] == "complete":
                evidence.append((scenario, result))
        if constraint and result.get("status") == "complete" \
                and not _valid_constraint_evidence(scenario, result):
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
    if constraint and evidence:
        plotted = []
        for scenario, result in evidence:
            moves = result.get("requests", [])
            plotted.append({
                "condition_id": scenario["condition_id"],
                "policy": scenario["policy"],
                "requested_shed_w": result["requested_shed_w"],
                "attained_shed_w": result["realized_shed_w"],
                "selected_sessions": len(moves),
                **_constraint_action_counts(moves),
            })
        plot_constraint(plotted, [], run_root)
    summary = {
        "schema": "queue-haul-network-summary-v1",
        "expected": len(plan["scenarios"]), "completed": completed,
        "failed": failed, "missing": missing,
        "valid": not missing and not (constraint and invalid_evidence) and (
            completed == len(plan["scenarios"]) if not frontier else
            failed / len(plan["scenarios"]) <= FRONTIER_FAILURE_GATE),
    }
    if constraint:
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
    stack, bandwidth = None, None
    try:
        for scenario in plan["scenarios"]:
            scenario_root = run_root / "scenarios" / scenario["scenario_id"]
            latest = _latest_result(scenario_root)
            if latest and latest[1].get("status") == "complete":
                continue
            if stack and bandwidth != scenario["bandwidth"]:
                stop_cluster(stack)
                stack = None
            if stack is None:
                bandwidth = scenario["bandwidth"]
                stack = start_cluster(
                    cluster, key, plan["network_contract"], bandwidth,
                    run_root / "stacks" /
                    f"{bandwidth}-{time.time_ns()}",
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
                if plan["design"] not in {"frontier", "constraint"}:
                    raise
            checkpoint_progress(plan, run_root)
    finally:
        if stack:
            stop_cluster(stack)
    summary = reduce_run(plan, run_root)
    if not summary["valid"]:
        raise RuntimeError(
            f"network campaign incomplete: {summary['failed']} failed, "
            f"{summary['missing']} missing")
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
                         choices=("joint", "isolated", "frontier", "constraint"),
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
                   args.kv_port, args.run_root, args.power_interval_s)
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
