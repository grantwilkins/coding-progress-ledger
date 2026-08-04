"""Run the private three-region Azure Queue-Haul campaign."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
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
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args.cluster, args.calibration, args.out)


if __name__ == "__main__":
    main()
