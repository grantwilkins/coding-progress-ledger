"""Run the private three-region Azure Queue-Haul campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path


CLUSTER_SCHEMA = "queue-haul-azure-cluster-v1"
CALIBRATION_SCHEMA = "queue-haul-network-calibration-v1"
CLOCK_LIMIT_MS = 2.0
RESUME_DRIFT = .10
REPEATS = 3
POLICIES = (
    "queue_haul", "greedy", "greedy_lagrangian", "kv_only", "replay_only",
    "random",
)
ROOT = Path(__file__).parent
EXPECTED_RUNTIME = {"vllm": "0.22.0", "lmcache": "0.5.1"}


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
        if len(value.destinations) != 2 or len({n.id for n in nodes}) != 3 \
                or len({n.host for n in nodes}) != 3:
            raise ValueError("cluster node ids and hosts must be unique")
        if value.source.region != "swedencentral" \
                or {node.region for node in value.destinations} \
                != {"eastus2", "westeurope"}:
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
    if len(raw.get("paths", {})) != 2 \
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
        if node and (report["region"] != node.region
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
                time.sleep(.5)
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
        time.sleep(.5)
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


def target_conditions() -> list[dict]:
    anchor = {
        "workload": "interactive_coding", "bandwidth": "controlled_80",
        "sink_load": "idle", "deadline_s": 30,
    }
    changes = (
        {}, {"workload": "coding"}, {"workload": "agentic_tool_loop"},
        {"bandwidth": "controlled_40"}, {"bandwidth": "natural"},
        {"sink_load": "rho_0.8"}, {"deadline_s": 19},
    )
    return [{**anchor, **change} for change in changes]


def prepare(cluster_path: Path, calibration_path: Path, out: Path) -> dict:
    cluster = Cluster.load(cluster_path)
    calibration = json.loads(calibration_path.read_text())
    plan = {
        "schema": "queue-haul-network-plan-v1",
        "cluster": cluster.as_dict(),
        "network": freeze_contract(calibration),
        "policies": list(POLICIES), "repeats": REPEATS,
        "conditions": target_conditions(),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
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
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args.cluster, args.calibration, args.out)
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


if __name__ == "__main__":
    main()
