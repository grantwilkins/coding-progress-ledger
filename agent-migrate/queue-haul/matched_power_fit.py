"""Freeze model-specific H100 power on the matched campaign load path."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull


ROOT = Path(__file__).parent
SPECS = (
    ("qwen3.8-27b", "Qwen/Qwen3.8-27B", 7366.927533417684, 440.2443056839966,
     Path("/datadrive/h100-phase-power-20260820-r5/qwen3_8_27b/measurements.csv"),
     Path("/datadrive/h100-phase-power-20260820-r5/qwen3_8_27b/idle.jsonl"), "phase"),
    ("gpt-oss-20b", "openai/gpt-oss-20b", 50519.60708244544, 451.32,
     Path("/datadrive/queue-haul-power/h100-realized-20260814-005/cells.jsonl"),
     None, "cells"),
    ("gemma-4-26b", "google/gemma-4-26B-A4B-it", 40069.547946926155,
     1796.4137476159165,
     Path("/datadrive/h100-serving-gemma-vllm024-20260819-r3/power/cells.jsonl"),
     None, "cells"),
)


def _digest(*paths: Path) -> str:
    value = hashlib.sha256()
    for path in paths:
        value.update(path.read_bytes())
    return value.hexdigest()


def _hull(rows: list[dict]) -> list[list[float]]:
    points = np.unique([[0., 0.], *[[row["f"], row["g"]] for row in rows]], axis=0)
    return points[ConvexHull(points).vertices].tolist()


def _validation(groups: dict[float, list[dict]]) -> dict:
    errors = [statistics.median(other["power"] for j, other in enumerate(rows) if j != i)
              - row["power"] for rows in groups.values() if len(rows) > 1
              for i, row in enumerate(rows)]
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    within = float(np.mean(np.abs(errors) <= 5))
    result = {"repeat_holdout_rmse_w": rmse, "within_5w_fraction": within,
              "replicated_points": sum(len(rows) > 1 for rows in groups.values()),
              "heldout_rows": len(errors), "gate_passed": rmse <= 5 and within >= .8}
    if result["replicated_points"] < 3 or not result["gate_passed"]:
        raise RuntimeError(f"campaign power repeat gate failed: {result}")
    return result


def fit_spec(spec: tuple, bootstrap_samples: int = 200, seed: int = 1) -> dict:
    name, model, F, G, path, idle_path, kind = spec
    if kind == "phase":
        raw = list(csv.DictReader(path.open()))
        all_rows = [{"f": float(row["f_tps"]), "g": float(row["g_tps"]),
                     "power": float(row["power_mean_w"])} for row in raw]
        chosen = [{**row, "group": float(source["target_service_load"])}
                  for row, source in zip(all_rows, raw) if source["mixture"] == "mixed"]
        idle = [json.loads(line)["power_mean_w"] for line in idle_path.read_text().splitlines()]
        evidence = (path, idle_path)
    else:
        raw = [json.loads(line) for line in path.read_text().splitlines()]
        all_rows = [{"f": float(row["realized_prefill_tps"]),
                     "g": float(row["realized_decode_tps"]),
                     "power": float(row["power_mean_w"])}
                    for row in raw if row["family"] != "idle"]
        chosen = [{"f": float(row["realized_prefill_tps"]),
                   "g": float(row["realized_decode_tps"]),
                   "power": float(row["power_mean_w"]),
                   "group": int(row["concurrency"])}
                  for row in raw if row["family"] == "campaign"]
        idle = [float(row["power_mean_w"]) for row in raw if row["family"] == "idle"]
        evidence = (path,)
    groups = {group: [row for row in chosen if row["group"] == group]
              for group in sorted({row["group"] for row in chosen})}
    xs = ({group: group for group in groups} if kind == "phase" else
          {group: statistics.median(row["f"] / F + row["g"] / G
                                    for row in rows) for group, rows in groups.items()})
    p0 = statistics.median(idle)
    curve = [[0., p0], *[[xs[group], statistics.median(row["power"] for row in rows)]
                          for group, rows in groups.items()]]
    rng = np.random.default_rng(seed)
    bootstrap = [[[0., float(rng.choice(idle))],
                  *[[xs[group], float(rng.choice([row["power"] for row in rows]))]
                    for group, rows in groups.items()]] for _ in range(bootstrap_samples)]
    validation = _validation(groups)
    digest = _digest(*evidence)
    phase = {
        "p0_w": p0, "delta_w": max(row[1] for row in curve) - p0,
        "a_s_per_prefill_token": 1 / F, "b_s_per_decode_token": 1 / G,
        "valid_hull": _hull(all_rows),
        "grouped_cv_rmse_w": validation["repeat_holdout_rmse_w"],
        "within_5w_fraction": validation["within_5w_fraction"], "bootstrap": [],
        "provenance_sha256": digest, "measured_power_curve": curve,
        "measured_power_bootstrap": bootstrap,
    }
    return {
        "schema": "queue-haul-campaign-power-fit-v1", "model": model,
        "hardware": "H100", "F_prefill_tps": F, "G_decode_tps": G,
        "scope": "matched East/Germany coding-session load path",
        "source": {"path": str(path), "sha256": digest, "kind": kind,
                   "observed_rows": len(all_rows), "selected_rows": len(chosen)},
        "validation": validation, "max_power_load": curve[-1][0],
        "phase_power": phase,
    }


def run(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for spec in SPECS:
        result = fit_spec(spec)
        (out / f"{spec[0]}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=ROOT / "profiles/matched_action_h100_power")
    run(parser.parse_args().out)
