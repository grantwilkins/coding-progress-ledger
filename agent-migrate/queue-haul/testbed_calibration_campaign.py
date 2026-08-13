"""Minimal destination, migration, and operational calibration design."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

from phase_power_calibration import MIXTURES


STATES = ("slack", "bandwidth-only", "east-kv-only",
          "germany-service-only", "jointly-binding")


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
    migration = [{"destination": destination, "method": method,
                  "context_tokens": context, "repeat": repeat}
                 for destination in ("east", "germany")
                 for method in ("replay", "kv_transfer")
                 for context in anchors for repeat in range(3)]
    operational = [{"state": state, "repeat": repeat}
                   for state in STATES for repeat in range(3)]
    rng = random.Random(seed)
    for cells in (service, migration, operational):
        rng.shuffle(cells)
    return {
        "schema": "queue-haul-targeted-calibration-plan-v1",
        "parent": str(parent_path), "pack": "recorded-28-seed-8", "seed": seed,
        "service_cells": service, "migration_cells": migration,
        "operational_cells": operational,
        "gates": {"median_timing_relative_error": .10,
                  "p90_timing_relative_error": .15,
                  "p90_timing_absolute_error_s": 1,
                  "false_feasible": 0, "correctness_failures": 0},
    }


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
    args = parser.parse_args()
    value = make_plan(args.parent, args.seed) if args.command == "prepare" else reduce(
        args.service, args.migration, args.operational)
    args.out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
