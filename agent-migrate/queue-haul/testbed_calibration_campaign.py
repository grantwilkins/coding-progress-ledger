"""Minimal destination, migration, and operational calibration design."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from phase_power_calibration import MIXTURES


STATES = ("slack", "bandwidth-only", "east-kv-only",
          "germany-service-only", "jointly-binding")
LOADS = (.25, .6, .9)
CONCURRENCIES = (1, 2, 4, 8)
ACTION_MIXES = ("replay_only", "kv_only", "mixed")
BANDWIDTHS = ("natural", "controlled_40")


def make_plan(parent_path: Path, seed: int = 1) -> dict:
    parent = json.loads(parent_path.read_text())
    template = next(row for row in parent["scenarios"]
                    if row["condition_id"] == "joint-shaped" and row["repeat"] == 0)
    contexts = sorted(int(row["initial_tokens"]) for row in template["sessions"])
    anchors = [contexts[0], contexts[len(contexts) // 2], contexts[-1]]
    service = [{"destination": destination, "boundary": boundary,
                "mixture": mixture, "repeat": repeat}
               for destination in ("east", "germany")
               for boundary in ("slack", "normal", "emergency")
               for mixture in MIXTURES for repeat in range(3)]
    migration = [{"destination": destination, "destination_prefill_load": load,
                  "concurrency": concurrency, "action_mix": action_mix,
                  "context_tokens": context, "bandwidth": bandwidth,
                  "repeat": repeat}
                 for destination in ("east", "germany")
                 for load in LOADS for concurrency in CONCURRENCIES
                 for action_mix in ACTION_MIXES for context in anchors
                 for bandwidth in BANDWIDTHS for repeat in range(3)]
    operational = [{"state": state, "repeat": repeat}
                   for state in STATES for repeat in range(3)]
    rng = random.Random(seed)
    for cells in (service, migration, operational):
        rng.shuffle(cells)
    screening = [{"destination": destination, "destination_prefill_load": load,
                  "concurrency": concurrency, "action_mix": action_mix,
                  "context_tokens": anchors[(i + j + k) % len(anchors)],
                  "bandwidth": BANDWIDTHS[(i + 2 * j + k) % len(BANDWIDTHS)],
                  "repeat": 0}
                 for i, destination in enumerate(("east", "germany"))
                 for j, load in enumerate(LOADS)
                 for k, concurrency in enumerate(CONCURRENCIES)
                 for action_mix in ACTION_MIXES]
    rng.shuffle(screening)
    return {
        "schema": "queue-haul-targeted-calibration-plan-v1",
        "parent": str(parent_path), "pack": "recorded-28-seed-8", "seed": seed,
        "service_cells": service, "migration_cells": migration,
        "migration_screening_cells": screening,
        "migration_campaign_mode": "run screening first; add two confirmation repeats only "
                                   "for retained cells after grouped residual analysis",
        "operational_cells": operational,
        "telemetry_required": [
            "migration_start_ns", "destination_ready_ns", "commit_ns",
            "replay_tokens", "kv_bytes", "destination", "method",
            "concurrent_replay", "concurrent_kv", "destination_prefill_load",
            "route_goodput_bytes_per_s", "effective_replay_tokens_per_s",
            "effective_kv_ingest_bytes_per_s", "setup_s", "completion_s",
            "switch_s",
        ],
        "validation_holdouts": ["action_mix", "destination_prefill_load+concurrency"],
        "gates": {"median_timing_relative_error": .10,
                  "p90_timing_relative_error": .15,
                  "p90_timing_absolute_error_s": 1,
                  "false_feasible": 0, "correctness_failures": 0},
    }


def network_plan(parent_path: Path, campaign_path: Path) -> dict:
    parent, campaign = json.loads(parent_path.read_text()), json.loads(campaign_path.read_text())
    template = next(row for row in parent["scenarios"]
                    if row["condition_id"] == "joint-shaped" and row["repeat"] == 0)
    available = template["sessions"]
    scenarios = []
    for index, cell in enumerate(campaign["migration_screening_cells"]):
        selected = sorted(available, key=lambda row: abs(
            int(row["initial_tokens"]) - int(cell["context_tokens"])))[:cell["concurrency"]]
        sessions = [{**row, "order": order,
                     "session_id": f"{row['template_id']}-calibration-{index}-{order}"}
                    for order, row in enumerate(selected)]
        methods = (["replay"] * len(sessions) if cell["action_mix"] == "replay_only" else
                   ["kv_transfer"] * len(sessions) if cell["action_mix"] == "kv_only" else
                   ["replay" if order % 2 == 0 else "kv_transfer"
                    for order in range(len(sessions))])
        moves = [{"session_id": session["session_id"],
                  "destination_instance": cell["destination"],
                  "destination_pool": f"pool/{cell['destination']}",
                  "method": method, "order": order,
                  "path": [f"link/{cell['destination']}"]}
                 for order, (session, method) in enumerate(zip(sessions, methods))]
        identity = json.dumps(cell, sort_keys=True, separators=(",", ":"))
        bandwidths = {node: (parent["network_contract"]["paths"][node]["natural_mbps"]
                             if cell["bandwidth"] == "natural" else
                             parent["network_contract"]["paths"][node]
                             ["controlled_mbps"]["40"])
                      for node in ("east", "germany")}
        scenarios.append({
            "scenario_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
            "design": "calibration", "condition_id": "migration-screening",
            "workload": "migration_calibration",
            "condition_index": index, "policy": cell["action_mix"],
            "repeat": cell["repeat"], "pack": campaign["pack"],
            "background": {node: [cell["destination_prefill_load"]
                                   if node == cell["destination"] else 0, 0]
                           for node in ("east", "germany")},
            "load_normalization": "destination_service", "load_warmup_s": 15,
            "bandwidth": cell["bandwidth"],
            "bandwidth_mbps": bandwidths,
            "deadline_s": 300, "sessions": sessions, "moves": moves,
            "calibration_cell": cell,
        })
    return {**parent, "design": "calibration", "policies": list(ACTION_MIXES),
            "conditions": [], "repeats": 1, "sessions_per_scenario": None,
            "scenarios": scenarios}


def _first_response(request: dict) -> int:
    chunks = request.get("stream_chunks", [])
    return int(chunks[0]["monotonic_ns"] if chunks else
               request.get("first_byte_ns", request["end_ns"]))


def _overlap(rows: list[dict], item: dict, method: str) -> int:
    destination = item["move"]["destination_instance"]
    instants = [item["start_ns"], *(row["start_ns"] for row in rows
                if item["start_ns"] < row["start_ns"] < item["ready_ns"])]
    return max((sum(row["method"] == method
                    and row["move"]["destination_instance"] == destination
                    and row["start_ns"] <= instant < row["end_ns"] for row in rows)
                for instant in instants), default=0)


def inventory(roots: list[Path]) -> tuple[list[dict], dict]:
    rows = []
    for root in roots:
        route_by_connection = {}
        for connection_path in root.glob("stacks/*/proxy_connections.csv"):
            with connection_path.open(newline="") as handle:
                route_by_connection.update({row["connection_id"]: row["route"]
                                            for row in csv.DictReader(handle)})
        for path in sorted(root.glob("scenarios/*/attempt-*/result.json")):
            scenario_path = path.with_name("scenario.json")
            if not scenario_path.exists():
                continue
            scenario, result = json.loads(scenario_path.read_text()), json.loads(path.read_text())
            sessions = {row["session_id"]: row for row in scenario.get("sessions", [])}
            requests = []
            for move in result.get("requests", []):
                if "request" not in move:
                    continue
                request = move["request"]
                requests.append({"move": move, "method": move["method"],
                                 "start_ns": int(request["start_ns"]),
                                 "ready_ns": _first_response(request),
                                 "end_ns": int(request["end_ns"])})
            route_transfers = {}
            for transfer in result.get("resp_transfers", []):
                route = route_by_connection.get(transfer["connection_id"])
                if route and transfer["command"] == "GET":
                    route_transfers.setdefault(route, []).append(transfer)
            for item in requests:
                move, request = item["move"], item["move"]["request"]
                destination, route = move["destination_instance"], f"kv/{move['destination_instance']}"
                transfers = route_transfers.get(route, [])
                route_bytes = int(result.get("wire_bytes", {}).get(
                    f"{route}/target_to_client", 0))
                route_span = (max((int(row["end_ns"]) for row in transfers), default=0)
                              - min((int(row["start_ns"]) for row in transfers), default=0))
                payload = sum(int(row["payload_bytes"]) for row in transfers)
                same_kv = [row for row in requests if row["method"] == "kv_transfer"
                           and row["move"]["destination_instance"] == destination]
                direct_kv = payload if item["method"] == "kv_transfer" and len(same_kv) == 1 else None
                replay_tokens = int(request["prompt_tokens"]) - int(request.get("cached_tokens", 0))
                ready_s = max((item["ready_ns"] - item["start_ns"]) / 1e9, 1e-9)
                background = scenario.get("background", {}).get(destination, [None])
                switch = result.get("source_sleep_ns") or [None]
                rows.append({
                    "scenario_id": scenario.get("scenario_id"), "session_id": move["session_id"],
                    "destination": destination, "method": item["method"],
                    "migration_start_ns": item["start_ns"],
                    "destination_ready_ns": item["ready_ns"], "commit_ns": item["end_ns"],
                    "replay_tokens": replay_tokens if item["method"] == "replay" else 0,
                    "kv_bytes": direct_kv, "episode_route_kv_bytes": payload,
                    "kv_bytes_attribution": "direct_single_migration" if direct_kv is not None
                    else "episode_route_only",
                    "context_tokens": sessions.get(move["session_id"], {}).get("initial_tokens"),
                    "concurrent_replay": _overlap(requests, item, "replay"),
                    "concurrent_kv": _overlap(requests, item, "kv_transfer"),
                    "destination_prefill_load": background[0] if isinstance(background, list) else None,
                    "route_goodput_bytes_per_s": route_bytes / (route_span / 1e9) if route_span > 0 else None,
                    "effective_replay_tokens_per_s": replay_tokens / ready_s
                    if item["method"] == "replay" else None,
                    "effective_kv_ingest_bytes_per_s": direct_kv / ready_s
                    if direct_kv is not None else None,
                    "setup_s": (item["start_ns"] - int(result["started_ns"])) / 1e9,
                    "completion_s": (item["end_ns"] - item["ready_ns"]) / 1e9,
                    "switch_s": (int(switch[0]) - int(result["ended_ns"])) / 1e9
                    if switch[0] is not None else None,
                    "bandwidth": scenario.get("bandwidth"), "deadline_s": scenario.get("deadline_s"),
                    "deadline_admitted": move.get("deadline_admitted"),
                    "episode_deadline_met": result.get("deadline_met"), "source": str(path),
                })
    kv = [row for row in rows if row["method"] == "kv_transfer"]
    summary = {
        "schema": "queue-haul-migration-telemetry-inventory-v1", "migrations": len(rows),
        "scenarios": len({row["scenario_id"] for row in rows}),
        "destinations": sorted({row["destination"] for row in rows}),
        "methods": sorted({row["method"] for row in rows}),
        "bandwidths": sorted({str(row["bandwidth"]) for row in rows}),
        "direct_per_migration_kv_bytes": sum(row["kv_bytes"] is not None for row in kv),
        "kv_migrations": len(kv),
        "requires_new_per_migration_transfer_ids": any(row["kv_bytes"] is None for row in kv),
        "timestamp_semantics": {"migration_start_ns": "destination request start",
                                "destination_ready_ns": "first response stream chunk",
                                "commit_ns": "verified reconstruction response end"},
    }
    return rows, summary


def _write(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no migration telemetry found")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _timing_gate(rows: list[dict]) -> tuple[bool, dict]:
    if not rows:
        return False, {"median_relative_error": None, "p90_relative_error": None}
    absolute = [abs(float(row["observed_s"]) - float(row["predicted_s"])) for row in rows]
    relative = [error / max(float(row["observed_s"]), 1e-9)
                for error, row in zip(absolute, rows)]
    p90_index = max(0, -(-9 * len(rows) // 10) - 1)
    p90_relative, p90_absolute = sorted(relative)[p90_index], sorted(absolute)[p90_index]
    median = statistics.median(relative)
    return median <= .10 and (p90_relative <= .15 or p90_absolute <= 1), {
        "median_relative_error": median, "p90_relative_error": p90_relative,
        "p90_absolute_error_s": p90_absolute,
    }


def reduce(service_path: Path, migration_path: Path,
           operational_path: Path) -> dict:
    service, migration, operational = map(_read, (
        service_path, migration_path, operational_path))
    service_gate, service_error = _timing_gate(service)
    migration_gate, migration_error = _timing_gate(migration)
    false_feasible = sum(row.get("predicted_feasible") == "True"
                         and row.get("observed_feasible") != "True"
                         for row in operational)
    failures = sum(int(row.get("correctness_failures", 0))
                   for row in migration + operational)
    expected_states = {(state, str(repeat)) for state in STATES for repeat in range(3)}
    complete = {(row.get("state"), row.get("repeat")) for row in operational} == expected_states
    return {
        "schema": "queue-haul-targeted-calibration-summary-v1",
        "service_gate_passed": service_gate,
        "migration_gate_passed": migration_gate,
        "correctness_failures": failures, "false_feasible": false_feasible,
        "operational_gate_passed": complete and false_feasible == 0,
        "service_error": service_error, "migration_error": migration_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parent", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("reduce")
    r.add_argument("--service", type=Path, required=True)
    r.add_argument("--migration", type=Path, required=True)
    r.add_argument("--operational", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("inventory")
    i.add_argument("--root", type=Path, action="append", required=True)
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("--summary", type=Path, required=True)
    n = sub.add_parser("network-plan")
    n.add_argument("--parent", type=Path, required=True)
    n.add_argument("--campaign", type=Path, required=True)
    n.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "network-plan":
        args.out.write_text(json.dumps(network_plan(args.parent, args.campaign),
                                       indent=2, sort_keys=True) + "\n")
    elif args.command == "inventory":
        rows, value = inventory(args.root); _write(args.out, rows)
        args.summary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    else:
        value = make_plan(args.parent, args.seed) if args.command == "prepare" else reduce(
            args.service, args.migration, args.operational)
        args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
