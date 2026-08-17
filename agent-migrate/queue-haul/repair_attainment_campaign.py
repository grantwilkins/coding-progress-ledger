"""Large, paired simulation population for mid-flight plan repair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import migration_profiler as profiler
import network_campaign as network
import repair_hardware_campaign as hardware
import repair_stress_campaign as stress
from profiles import ModelProfile


SCHEMA = "queue-haul-repair-attainment-population-v1"
DEFAULT_OUT = Path("outputs/repair-attainment-simulation-20260817/population.json")
DEFAULT_SEED_START = 10_000
DEFAULT_SAMPLES = 4_096
TARGET_FRACTION = .50


def _summary(row: dict) -> dict:
    omitted = {
        "initial_moves", "repair_moves", "stable_schedule",
        "repair_schedule", "control_schedule", "repair_curve",
        "control_curve", "repair_result",
    }
    return {key: value for key, value in row.items() if key not in omitted}


def run(base_plan_path: Path, timing_path: Path, out: Path,
        seed_start: int, samples: int) -> dict:
    if samples < 1 or seed_start < 0:
        raise ValueError("samples must be positive and seeds nonnegative")
    plan = json.loads(base_plan_path.read_text())
    hardware.validate_plan(plan)
    timing = json.loads(timing_path.read_text())
    if timing.get("schema") != "queue-haul-repair-10x-timing-fit-v3" \
            or not timing.get("passed"):
        raise RuntimeError("population requires the passing live 0.1x fit")
    parent_path = stress._resolve(plan["parent"]["path"])
    manifest_path = stress._resolve(plan["manifest"]["path"])
    parent = json.loads(parent_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    profile = ModelProfile.load(network.MODEL_PATH)
    seeds = tuple(range(seed_start, seed_start + samples))
    cells = []
    for index, seed in enumerate(seeds, 1):
        for axis in stress.FAULT_AXES:
            cells.append(_summary(stress._simulate_target(
                parent, plan, timing, manifest, profile, TARGET_FRACTION,
                stress.HEALTHY_EAST_LOAD, stress.MOVE_CONCURRENCY, seed, axis,
                "attainment_population")))
        if index % 100 == 0 or index == samples:
            print(f"completed {index}/{samples} workload packs", flush=True)
    bundle = {
        "schema": SCHEMA,
        "semantics": (
            "fresh simulator-generated workload packs crossed with all three "
            "fault axes; the stale original schedule and repaired residual "
            "schedule are paired under identical 0.1x degraded timing"),
        "base_plan": {"path": str(base_plan_path),
                      "sha256": profiler.file_hash(base_plan_path)},
        "timing": {"path": str(timing_path),
                   "sha256": profiler.file_hash(timing_path),
                   "schema": timing["schema"]},
        "parent": {"path": str(parent_path),
                   "sha256": profiler.file_hash(parent_path)},
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "model_profile": {"path": str(network.MODEL_PATH),
                          "sha256": profiler.file_hash(network.MODEL_PATH)},
        "target_fraction": TARGET_FRACTION,
        "fault_axes": {key: list(value)
                       for key, value in stress.FAULT_AXES.items()},
        "healthy_east_load": stress.HEALTHY_EAST_LOAD,
        "move_concurrency": stress.MOVE_CONCURRENCY,
        "context_support": list(stress.CONTEXT_SUPPORT),
        "seed_start": seed_start,
        "samples": samples,
        "context_seeds": list(seeds),
        "cells": cells,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out, bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-plan", type=Path,
                        default=stress.DEFAULT_BASE_PLAN)
    parser.add_argument("--timing", type=Path, default=stress.DEFAULT_TIMING)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    args = parser.parse_args()
    run(args.base_plan, args.timing, args.out, args.seed_start, args.samples)


if __name__ == "__main__":
    main()
