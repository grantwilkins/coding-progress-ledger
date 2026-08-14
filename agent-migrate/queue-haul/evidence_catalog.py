"""Checksum-bound provenance and trailing-window power reduction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path


KINDS = {"direct_measurement", "derived_from_measurement", "model_credited",
         "estimated", "assumed"}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> tuple[str, list[str]]:
    name, text = path.name, str(path).lower()
    if name == "power.csv" or name.startswith("power-") and name.endswith(".csv") \
            or name.startswith("metrics_") or name in {
                "proxy_bytes.csv", "proxy_connections.csv", "resp_transfers.csv",
                "source_load.jsonl", "sink_load_east.jsonl", "sink_load_germany.jsonl",
            }:
        return "direct_measurement", ["raw telemetry"]
    if name in {"results.csv", "result.json"} and (
            "hardware-gap" in text or "separation" in text):
        return "model_credited", ["completion-derived source-power shed"]
    if name in {"reduced.json", "summary.json", "power_curve.csv", "trailing_power.csv",
                "power_summary.csv", "queue_summary.csv",
                "migration-telemetry-existing.csv"} \
            or name.endswith(".summary.json") or "calibration" in name:
        return "derived_from_measurement", ["reduced measurement"]
    if name in {"plan.json", "scenario.json"} or name.startswith("plan-"):
        return "assumed", ["predeclared experiment input"]
    return "estimated", ["unclassified supporting artifact"]


def catalog(root: Path) -> dict:
    names = {"power.csv", "results.csv", "result.json", "summary.json", "reduced.json",
             "power_curve.csv", "levels.json", "run_metadata.json", "plan.json",
             "scenario.json", "source_load.jsonl", "power_summary.csv", "queue_summary.csv",
             "trailing_power.csv", "migration-telemetry-existing.csv",
             "proxy_bytes.csv", "proxy_connections.csv", "resp_transfers.csv",
             "metrics_sweden.csv", "metrics_east.csv", "metrics_germany.csv"}
    paths = sorted(path for path in root.rglob("*") if path.is_file()
                   and (path.name in names or path.name.endswith(".summary.json")
                        or path.name.startswith(("calibration-", "plan-", "power-"))))
    entries = []
    for path in paths:
        kind, quantities = classify(path)
        entries.append({"path": str(path.relative_to(root)), "sha256": file_hash(path),
                        "bytes": path.stat().st_size, "evidence_kind": kind,
                        "quantities": quantities})
    return {"schema": "queue-haul-evidence-catalog-v1", "root": str(root),
            "entries": entries}


def verify(catalog_value: dict, root: Path, path: Path) -> dict:
    relative = str(path.relative_to(root))
    matches = [row for row in catalog_value["entries"] if row["path"] == relative]
    if len(matches) != 1 or matches[0]["evidence_kind"] not in KINDS:
        raise ValueError(f"missing provenance for {relative}")
    if matches[0]["sha256"] != file_hash(path):
        raise ValueError(f"provenance checksum changed for {relative}")
    return matches[0]


def _power_files(run_root: Path) -> list[tuple[Path, int, int]]:
    files = []
    for path in run_root.glob("stacks/*/power.csv"):
        with path.open(newline="") as handle:
            times = [int(row["monotonic_ns"]) for row in csv.DictReader(handle)
                     if row.get("valid", "1") == "1"]
        if times:
            files.append((path, min(times), max(times)))
    return files


def _window(samples: list[tuple[int, float]], start: int, end: int) -> float:
    selected = [(time, watts) for time, watts in samples if start <= time < end]
    if len(selected) < 15 or selected[0][0] - start > 500_000_000 \
            or end - selected[-1][0] > 500_000_000 \
            or any(right[0] - left[0] > 500_000_000
                   for left, right in zip(selected, selected[1:])):
        raise ValueError("power samples do not cover the five-second window")
    return statistics.fmean(watts for _, watts in selected)


def trailing_power_rows(run_root: Path, window_s: float = 5) -> list[dict]:
    if window_s <= 0:
        raise ValueError("power window must be positive")
    available, rows = _power_files(run_root), []
    for scenario_path in sorted(run_root.glob("scenarios/*/attempt-*/scenario.json")):
        attempt = scenario_path.parent
        result_path = attempt / "result.json"
        source_path = attempt / "source_load.jsonl"
        if not result_path.exists() or not source_path.exists():
            continue
        scenario, result = json.loads(scenario_path.read_text()), json.loads(result_path.read_text())
        if not result.get("started_ns"):
            continue
        start = int(result["started_ns"])
        deadline = start + int(float(scenario["deadline_s"]) * 1e9)
        width = int(window_s * 1e9)
        matches = [item for item in available
                   if item[1] <= start - width and item[2] >= deadline]
        if len(matches) != 1:
            continue
        last_load_start = max(
            int(json.loads(line)["start_ns"]) for line in source_path.read_text().splitlines()
        )
        if last_load_start < deadline - width:
            continue
        with matches[0][0].open(newline="") as handle:
            samples = [(int(row["monotonic_ns"]), float(row["power_w"]))
                       for row in csv.DictReader(handle)
                       if row.get("valid", "1") == "1" and int(row.get("gpu", 0)) == 0]
        try:
            baseline = _window(samples, start - width, start)
            trailing = _window(samples, deadline - width, deadline)
        except ValueError:
            continue
        rows.append({
            "scenario_id": scenario["scenario_id"], "condition": scenario["condition_id"],
            "policy": scenario["policy"], "repeat": scenario["repeat"],
            "deadline_s": scenario["deadline_s"], "power_window_s": window_s,
            "baseline_source_power_w": baseline,
            "measured_trailing_source_power_w": trailing,
            "measured_trailing_shed_w": baseline - trailing,
            "modeled_deadline_shed_w": result.get("realized_shed_w"),
            "power_evidence_kind": "direct_measurement",
            "power_path": str(matches[0][0]),
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("no valid trailing power windows")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)


def trailing_summary(run_root: Path, rows: list[dict]) -> dict:
    expected = {path.parent.parent.name
                for path in run_root.glob("scenarios/*/attempt-*/result.json")
                if json.loads(path.read_text()).get("started_ns")}
    valid = {row["scenario_id"] for row in rows}
    differences = [abs(float(row["measured_trailing_shed_w"])
                       - float(row["modeled_deadline_shed_w"])) for row in rows]
    return {
        "schema": "queue-haul-trailing-power-summary-v1",
        "expected_scenarios": len(expected), "valid_scenarios": len(valid),
        "rejected_scenario_ids": sorted(expected - valid),
        "modeled_vs_measured_mae_w": statistics.fmean(differences),
        "within_5w": sum(value <= 5 for value in differences),
        "promotion_gate_passed": len(valid) >= 15
        and statistics.fmean(differences) <= 5,
        "realized_shed_w_legacy_semantics": "model_credited",
        "measured_trailing_shed_w_semantics": "direct five-second total Sweden GPU power",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    catalog_command = sub.add_parser("catalog")
    catalog_command.add_argument("--root", type=Path, required=True)
    catalog_command.add_argument("--out", type=Path, required=True)
    trailing = sub.add_parser("trailing-power")
    trailing.add_argument("--run-root", type=Path, required=True)
    trailing.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "catalog":
        args.out.write_text(json.dumps(catalog(args.root), indent=2, sort_keys=True) + "\n")
    else:
        rows = trailing_power_rows(args.run_root)
        write_csv(args.out, rows)
        args.out.with_suffix(".summary.json").write_text(
            json.dumps(trailing_summary(args.run_root, rows), indent=2,
                       sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
