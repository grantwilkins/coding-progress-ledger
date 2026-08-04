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
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import migration_testbed as testbed
import migration_profiler as profiler
import policy_hardware_campaign as policy_campaign
from profiles import ModelProfile, WorkloadProfile


CLUSTER_SCHEMA = "queue-haul-azure-cluster-v1"
CALIBRATION_SCHEMA = "queue-haul-network-calibration-v1"
PLAN_SCHEMA = "queue-haul-network-plan-v1"
RESULT_SCHEMA = "queue-haul-network-result-v1"
CLOCK_LIMIT_MS = 2.0
RESUME_DRIFT = .10
REQUEST_TIMEOUT_S = 600.0
REPEATS = 3
POLICIES = (
    "queue_haul", "greedy", "greedy_lagrangian", "kv_only", "replay_only",
    "random",
)
ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
WORKLOAD_PATHS = {name: ROOT / f"profiles/{name}.json" for name in (
    "coding", "interactive_coding", "agentic_tool_loop",
)}
EXPECTED_RUNTIME = {"vllm": "0.22.0", "lmcache": "0.5.1"}


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
                <= {"eastus2", "westeurope"}:
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


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_plan(manifest_path: Path, contract: dict, seed: int = 1,
              sessions: int = 8) -> dict:
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_manifest(manifest)
    available = sorted(manifest["sessions"], key=lambda row: row["id"])
    if not 0 < sessions <= len(available):
        raise ValueError("invalid session count")
    model, scenarios = ModelProfile.load(MODEL_PATH), []
    destinations = tuple(sorted(contract["paths"]))
    if not 1 <= len(destinations) <= 2:
        raise ValueError("network contract requires one or two destinations")
    for condition_index, condition in enumerate(target_conditions()):
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
            for policy_index, policy in enumerate(POLICIES):
                destination = destinations[
                    (condition_index + repeat + policy_index)
                    % len(destinations)]
                path = contract["paths"][destination]
                bandwidth = path["natural_mbps"] if condition["bandwidth"] \
                    == "natural" else path["controlled_mbps"][
                        condition["bandwidth"].rsplit("_", 1)[-1]]
                problem, routes = policy_campaign._problem(
                    model, session_rows, bandwidth, condition["deadline_s"])
                moves = policy_campaign._moves(
                    policy, problem, routes, model,
                    profiler.stable_seed(
                        seed, condition_index, repeat, policy),
                )
                scenario_id = _hash([
                    condition_index, repeat, policy, destination, session_rows,
                    moves,
                ])[:16]
                scenarios.append({
                    "scenario_id": scenario_id,
                    "condition_index": condition_index, "repeat": repeat,
                    **condition, "policy": policy,
                    "destination": destination,
                    "bandwidth_mbps": bandwidth,
                    "sessions": session_rows, "moves": moves,
                })
    rng = random.Random(seed)
    for bandwidth in ("controlled_80", "controlled_40", "natural"):
        rows = [row for row in scenarios if row["bandwidth"] == bandwidth]
        rng.shuffle(rows)
        scenarios = [row for row in scenarios if row["bandwidth"] != bandwidth]
        scenarios.extend(rows)
    output = {
        "schema": PLAN_SCHEMA, "seed": seed,
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "model_profile": {"path": str(MODEL_PATH),
                          "sha256": profiler.file_hash(MODEL_PATH)},
        "network_contract": contract, "policies": list(POLICIES),
        "conditions": target_conditions(), "repeats": REPEATS,
        "sessions_per_scenario": sessions, "scenarios": scenarios,
    }
    validate_plan(output)
    return output


def validate_plan(plan: dict) -> None:
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("invalid network plan schema")
    scenarios = plan.get("scenarios", [])
    if len(scenarios) != len(target_conditions()) * REPEATS * len(POLICIES) \
            or len({row["scenario_id"] for row in scenarios}) != len(scenarios):
        raise ValueError("network plan must contain exactly 126 unique scenarios")
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
                path = contract["paths"][row["destination"]]
                expected = path["natural_mbps"] if row["bandwidth"] \
                    == "natural" else path["controlled_mbps"][
                        row["bandwidth"].rsplit("_", 1)[-1]]
                if row["bandwidth_mbps"] != expected \
                        or {move["session_id"] for move in row["moves"]} \
                        != {item["session_id"] for item in row["sessions"]}:
                    raise ValueError("scenario route or move contract changed")
    if any({row["destination"] for row in scenarios
            if row["policy"] == policy} != set(contract["paths"])
           for policy in POLICIES):
        raise ValueError("every policy must cover every destination")


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
               run_root: Path) -> None:
    import migration_profiler

    cfg = testbed.Config(host="127.0.0.1")
    testbed.preflight(cfg, 1)
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "node-serve.pid").write_text(str(os.getpid()))
    cache = sink = None
    sampler = migration_profiler.PowerSampler(run_root / "power.csv")
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
                  bandwidth: str, run_root: Path) -> ClusterStack:
    import migration_profiler

    cfg = testbed.Config(host="127.0.0.1")
    testbed.preflight(cfg, 1)
    run_root.mkdir(parents=True, exist_ok=False)
    routes, ports = cluster_routes(cluster)
    aggregate, rates = bandwidth_limits(contract, bandwidth)
    lmc = proxy = source = None
    services, remote = [], {}
    sampler = migration_profiler.PowerSampler(run_root / "power.csv")
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
                "uv", "run", "python", "queue-haul/network_campaign.py",
                "node-serve", "--node-id", node_id, "--bind-host", node.host,
                "--source-host", cluster.source.host, "--kv-port",
                str(ports[node_id]["kv"]), "--run-root", str(remote_root),
            ])
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, start_new_session=True,
            )
            remote[node_id] = process
            _remote_ready(process, testbed.health_timeout())
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
          timeout_s: float) -> dict:
    messages = messages + [{"role": "user", "content":
                            f"Reply only with session state code {code}."}]
    result, text = profiler.stream_chat(
        cfg, port, messages, 128, profiler.messages_hash(messages), timeout_s)
    if result.status_code != 200 or code not in text:
        raise RuntimeError(
            f"session reconstruction failed: HTTP {result.status_code}, "
            f"state code present={code in text}")
    return {**asdict(result), "state_code_verified": True}


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


class SinkLoad:
    def __init__(self, cfg: testbed.Config, port: int, prefill_tps: float,
                 path: Path):
        if prefill_tps <= 0:
            raise ValueError("sink prefill throughput must be positive")
        self.cfg, self.port, self.prefill_tps, self.path = (
            cfg, port, prefill_tps, path)
        self.stop, self.error = threading.Event(), None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _request(self, index: int) -> dict:
        messages = [{"role": "user", "content":
                     f"load-{index} " + "x " * 512}]
        result, _ = profiler.stream_chat(
            self.cfg, self.port, messages, 1,
            profiler.messages_hash(messages), 600)
        if result.status_code != 200:
            raise RuntimeError(f"sink load request failed: {result.status_code}")
        return asdict(result)

    def _run(self) -> None:
        futures, index = [], 0
        interval = 512 / (.8 * self.prefill_tps)
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


def run_network_scenario(stack: ClusterStack, manifest: dict, scenario: dict,
                         root: Path, prefill_tps: float) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    (root / "scenario.json").write_text(
        json.dumps(scenario, indent=2, sort_keys=True) + "\n")
    stack.spot.check()
    if any(process.poll() is not None for process in stack.remote.values()):
        raise RuntimeError("remote sink exited")
    _clear_cluster(stack)
    sessions = {row["id"]: row for row in manifest["sessions"]}
    messages = {row["session_id"]: profiler.calibration_messages(
        sessions[row["session_id"]], row["initial_tokens"])
        for row in scenario["sessions"]}
    moves = sorted(scenario["moves"], key=lambda row: row["order"])
    timeout = REQUEST_TIMEOUT_S
    for move in moves:
        if move["method"] == "replay":
            row = sessions[move["session_id"]]
            _warm(stack, messages[move["session_id"]],
                  row["state_code"], timeout)
    _clear_cluster(stack)
    for move in moves:
        if move["method"] == "kv_transfer":
            row = sessions[move["session_id"]]
            _warm(stack, messages[move["session_id"]],
                  row["state_code"], timeout)
    node = next(node for node in stack.cluster.destinations
                if node.id == scenario["destination"])
    load = SinkLoad(stack.cfg, stack.ports[node.id]["api"], prefill_tps,
                    root / "sink_load.jsonl") \
        if scenario["sink_load"] == "rho_0.8" else None
    before = testbed.proxy_counts(stack.run_root / "proxy_bytes.csv")
    start_ns = time.monotonic_ns()
    if load:
        load.start()
    try:
        def reconstruct(move):
            session = sessions[move["session_id"]]
            return {
                **move, "request": _chat(
                    stack.cfg, stack.ports[node.id]["api"],
                    messages[move["session_id"]], session["state_code"],
                    timeout,
                ),
            }

        with ThreadPoolExecutor(max_workers=len(moves)) as pool:
            results = list(pool.map(reconstruct, moves))
    finally:
        if load:
            load.close()
    end_ns = time.monotonic_ns()
    if any(row["method"] == "kv_transfer"
           and row["request"]["cached_tokens"] <= 0 for row in results):
        raise RuntimeError("KV reconstruction reported no cached tokens")
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
        "wire_bytes": testbed.count_delta(
            before, testbed.proxy_counts(proxy)),
        "connections": connections, "resp_transfers": transfers,
        "source_sleep_ns": [sleep_start_ns, sleep_end_ns],
    }
    write_checkpoint(root / "result.json", result)
    return result


def _latest_result(root: Path) -> tuple[int, dict] | None:
    rows = sorted(root.glob("attempt-*/result.json"))
    if not rows:
        return None
    path = rows[-1]
    return int(path.parent.name.rsplit("-", 1)[-1]), json.loads(path.read_text())


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


def reduce_run(plan: dict, run_root: Path) -> dict:
    rows, completed, failed, missing = [], 0, 0, 0
    for scenario in plan["scenarios"]:
        latest = _latest_result(run_root / "scenarios" / scenario["scenario_id"])
        if latest is None:
            missing += 1
            attempt, result = 0, {"status": "missing"}
        else:
            attempt, result = latest
            completed += result["status"] == "complete"
            failed += result["status"] == "failed"
        connections = result.get("connections", [])
        rtts = [float(row["target_rtt_us"]) / 1000 for row in connections
                if row.get("target_rtt_us")]
        wire = result.get("wire_bytes", {})
        rows.append({
            "scenario_id": scenario["scenario_id"],
            "condition_index": scenario["condition_index"],
            "repeat": scenario["repeat"], "policy": scenario["policy"],
            "destination": scenario["destination"],
            "workload": scenario["workload"],
            "bandwidth": scenario["bandwidth"],
            "sink_load": scenario["sink_load"],
            "deadline_s": scenario["deadline_s"], "attempt": attempt,
            "status": result["status"],
            "migration_s": result.get("migration_s", ""),
            "deadline_met": result.get("deadline_met", ""),
            "api_request_bytes": wire.get(
                f"api/{scenario['destination']}/client_to_target", 0),
            "kv_response_bytes": wire.get(
                f"kv/{scenario['destination']}/target_to_client", 0),
            "median_tcp_rtt_ms": statistics.median(rtts) if rtts else "",
            "retransmissions": sum(int(row.get("target_total_retrans") or 0)
                                   for row in connections),
            "error": result.get("error", ""),
        })
    profiler.write_csv(run_root / "results.csv", rows)
    summary = {
        "schema": "queue-haul-network-summary-v1",
        "expected": len(plan["scenarios"]), "completed": completed,
        "failed": failed, "missing": missing,
        "valid": completed == len(plan["scenarios"])
        and not failed and not missing,
    }
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
    core = lambda row: {key: value for key, value in row.items()
                        if key != "checks"}
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
            out: Path, seed: int = 1, sessions: int = 8) -> dict:
    cluster = Cluster.load(cluster_path)
    calibration = json.loads(calibration_path.read_text())
    plan = make_plan(manifest_path, freeze_contract(calibration), seed, sessions)
    plan["cluster"] = cluster.as_dict()
    plan["calibration"] = {
        "path": str(calibration_path),
        "sha256": profiler.file_hash(calibration_path),
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
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--seed", type=int, default=1)
    command.add_argument("--sessions", type=int, default=8)
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
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args.cluster, args.calibration, args.manifest, args.out,
                args.seed, args.sessions)
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
                   args.kv_port, args.run_root)
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


if __name__ == "__main__":
    main()
