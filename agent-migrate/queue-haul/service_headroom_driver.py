"""Durably execute one hardware arm of the service-headroom campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import service_headroom_campaign as campaign


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def invoke(*args: str | Path) -> None:
    subprocess.run(
        [sys.executable, str(Path(campaign.__file__).resolve()),
         *map(str, args)],
        check=True,
    )


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "complete"
    except (json.JSONDecodeError, OSError):
        return False


def run_cells(plan: dict, plan_path: Path, cell_ids: list[str], runs: Path,
              normalization: Path | None, status_path: Path,
              stage: str, retry_delay_s: float, max_attempts: int) -> None:
    for index, cell_id in enumerate(cell_ids):
        result = runs / cell_id / "result.json"
        attempts = 0
        while True:
            attempts += 1
            write_json(status_path, {
                "state": "running", "stage": stage, "cell_id": cell_id,
                "cell_index": index + 1, "cell_count": len(cell_ids),
                "attempt": attempts, "updated_wall_ns": time.time_ns(),
                "plan_sha256": campaign.digest(plan),
            })
            command = ["run-cell", "--plan", plan_path,
                       "--cell-id", cell_id, "--out", runs]
            if normalization is not None:
                command.extend(("--normalization", normalization))
            try:
                invoke(*command)
                if not complete(result):
                    raise RuntimeError(f"cell did not produce a complete result: {cell_id}")
                break
            except subprocess.CalledProcessError:
                if max_attempts and attempts >= max_attempts:
                    raise
                time.sleep(retry_delay_s)
        write_json(status_path, {
            "state": "running", "stage": stage, "cell_id": cell_id,
            "cell_index": index + 1, "cell_count": len(cell_ids),
            "completed_cells": index + 1, "updated_wall_ns": time.time_ns(),
            "plan_sha256": campaign.digest(plan),
        })


def execute(root: Path, hardware: str, ttft_target_s: float,
            tpot_target_s: float, retry_delay_s: float,
            max_attempts: int) -> None:
    root.mkdir(parents=True, exist_ok=True)
    status = root / "status.json"
    plan_path = root / "plan.json"
    core = campaign.make_plan()
    if plan_path.exists():
        campaign.validate_plan(json.loads(plan_path.read_text()))
    else:
        write_json(plan_path, core)
    plan = campaign.read_plan(plan_path)
    discovery = root / "discovery"
    normalization = root / f"{hardware}-normalization.json"
    calibration_ids = plan["run_order"][hardware]["calibration"]
    run_cells(plan, plan_path, calibration_ids, discovery, None, status,
              "calibration", retry_delay_s, max_attempts)
    invoke("reduce-calibration", "--plan", plan_path, "--hardware", hardware,
           "--runs", discovery, "--out", normalization)

    measurement_ids = plan["run_order"][hardware]["measurement"]
    run_cells(plan, plan_path, measurement_ids, discovery, normalization,
              status, "discovery", retry_delay_s, max_attempts)
    scout_path = root / f"{hardware}-scout.json"
    invoke("reduce", "--plan", plan_path, "--hardware", hardware,
           "--runs", discovery, "--ttft-target-s", ttft_target_s,
           "--tpot-target-s", tpot_target_s, "--out", scout_path)
    scout = json.loads(scout_path.read_text())
    if not scout["selection_ready"]:
        write_json(status, {
            "state": "complete", "stage": "discovery",
            "selection_ready": False,
            "reason": "discovery did not contain a valid pass/fail bracket",
            "updated_wall_ns": time.time_ns(),
            "plan_sha256": campaign.digest(plan),
        })
        return

    confirmation_plan_path = root / f"{hardware}-confirmation-plan.json"
    invoke("prepare-confirmation", "--plan", plan_path, "--scout", scout_path,
           "--hardware", hardware, "--out", confirmation_plan_path)
    confirmation_plan = campaign.read_plan(confirmation_plan_path)
    confirmation_runs = root / "confirmation"
    run_cells(confirmation_plan, confirmation_plan_path,
              confirmation_plan["run_order"], confirmation_runs,
              normalization, status, "confirmation", retry_delay_s,
              max_attempts)
    confirmed_path = root / f"{hardware}-confirmed.json"
    invoke("reduce-confirmation", "--plan", confirmation_plan_path,
           "--core-plan", plan_path, "--scout", scout_path,
           "--runs", confirmation_runs, "--out", confirmed_path)
    confirmed = json.loads(confirmed_path.read_text())
    write_json(status, {
        "state": "complete", "stage": "confirmation",
        "selection_ready": True,
        "planner_usable": confirmed["planner_usable"],
        "supported_bound": confirmed["supported_bound"],
        "updated_wall_ns": time.time_ns(),
        "plan_sha256": campaign.digest(plan),
        "confirmation_plan_sha256": campaign.digest(confirmation_plan),
    })


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--hardware", choices=campaign.HARDWARE, required=True)
    parser.add_argument("--ttft-target-s", type=float, required=True)
    parser.add_argument("--tpot-target-s", type=float, required=True)
    parser.add_argument("--retry-delay-s", type=float, default=30)
    parser.add_argument("--max-attempts", type=int, default=3,
                        help="Invalid-cell attempts before the service restarts; 0 retries forever")
    args = parser.parse_args(argv)
    if min(args.ttft_target_s, args.tpot_target_s) <= 0 \
            or args.retry_delay_s < 0 or args.max_attempts < 0:
        parser.error("targets must be positive and retry controls nonnegative")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    execute(args.run_root, args.hardware, args.ttft_target_s,
            args.tpot_target_s, args.retry_delay_s, args.max_attempts)


if __name__ == "__main__":
    main()
