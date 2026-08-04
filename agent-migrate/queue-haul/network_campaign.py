"""Run the private three-region Azure Queue-Haul campaign."""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import shlex
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import migration_testbed as testbed


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
    cache = sink = None
    sampler = migration_profiler.PowerSampler(run_root / "power.csv")
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, lambda *_args: stopped.set())
    try:
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
    try:
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
            run_root, key,
        )
    except BaseException:
        for process in remote.values():
            testbed.stop_proc(process)
        if sampler.thread.is_alive():
            sampler.close()
        for process in (source, proxy, *services, lmc):
            if process:
                testbed.stop_proc(process)
        raise


def _scp(node: Node, key: Path, source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    subprocess.run([
        "scp", "-q", "-r", "-o", "BatchMode=yes", "-o",
        "StrictHostKeyChecking=yes", "-i", str(key), "-P", str(node.ssh_port),
        f"{node.ssh_user}@{node.host}:{source}/.", str(destination),
    ], check=True)


def stop_cluster(stack: ClusterStack, collect: bool = True) -> None:
    for process in stack.remote.values():
        testbed.stop_proc(process)
    if stack.sampler.thread.is_alive():
        stack.sampler.close()
    testbed.stop_stack(stack.local)
    if collect:
        nodes = {node.id: node for node in stack.cluster.destinations}
        for node_id, root in stack.remote_roots.items():
            _scp(nodes[node_id], stack.key, root,
                 stack.run_root / "nodes" / node_id)


def smoke(cluster: Cluster, key: Path, calibration: dict, bandwidth: str,
          run_root: Path, words: int = 4096) -> dict:
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
    command.add_argument("--words", type=int, default=4096)
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
    elif args.command == "node-serve":
        node_serve(args.node_id, args.bind_host, args.source_host,
                   args.kv_port, args.run_root)
    elif args.command == "smoke":
        print(json.dumps(smoke(
            Cluster.load(args.cluster), args.ssh_key.expanduser(),
            json.loads(args.calibration.read_text()), args.bandwidth,
            args.run_root, args.words,
        ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
