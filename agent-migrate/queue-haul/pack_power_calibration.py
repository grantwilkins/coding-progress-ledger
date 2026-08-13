"""Fit fixed-pack phase load to direct trailing-window Sweden power."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression

import network_campaign as network
from profiles import ModelProfile


def fit_curve(points: list[dict], idle_w: float, far_load: float,
              far_power_w: float) -> tuple[list[list[float]], dict]:
    grouped = {}
    for row in points:
        grouped.setdefault(round(float(row["load"]), 12), []).append(float(row["power_w"]))
    loads = np.asarray([0., *sorted(grouped), far_load])
    watts = np.asarray([idle_w, *(statistics.median(grouped[x]) for x in sorted(grouped)),
                        far_power_w])
    fitted = isotonic_regression(watts, increasing=True).x
    curve = [[float(x), float(y)] for x, y in zip(loads, fitted)]
    errors = [float(np.interp(row["load"], loads, fitted) - row["power_w"])
              for row in points]
    return curve, {"mae_w": statistics.fmean(map(abs, errors)),
                   "rmse_w": float(np.sqrt(np.mean(np.square(errors)))),
                   "within_5w_fraction": statistics.fmean(abs(x) <= 5 for x in errors)}


def bootstrap_curves(points: list[dict], idle_w: float, far_load: float,
                     far_power_w: float, samples: int = 200, seed: int = 1) -> list[list[list[float]]]:
    grouped = {repeat: [row for row in points if row["repeat"] == repeat]
               for repeat in sorted({row["repeat"] for row in points})}
    rng, curves = np.random.default_rng(seed), []
    for _ in range(samples):
        selected = [row for repeat in rng.choice(list(grouped), len(grouped))
                    for row in grouped[repeat]]
        curves.append(fit_curve(selected, idle_w, far_load, far_power_w)[0])
    return curves


def grouped_repeat_cv(points: list[dict], idle_w: float, far_load: float,
                      far_power_w: float) -> dict:
    errors = []
    for repeat in sorted({row["repeat"] for row in points}):
        curve, _ = fit_curve([row for row in points if row["repeat"] != repeat],
                             idle_w, far_load, far_power_w)
        x, y = np.asarray(curve).T
        errors.extend(float(np.interp(row["load"], x, y) - row["power_w"])
                      for row in points if row["repeat"] == repeat)
    return {"grouped_repeat_cv_mae_w": statistics.fmean(map(abs, errors)),
            "grouped_repeat_cv_rmse_w": float(np.sqrt(np.mean(np.square(errors)))),
            "grouped_repeat_cv_bias_w": statistics.fmean(errors),
            "grouped_repeat_cv_within_5w_fraction": statistics.fmean(
                abs(error) <= 5 for error in errors)}


def baseline_gate(rows: list[dict]) -> tuple[set[str], dict]:
    values = [float(row["baseline_source_power_w"]) for row in rows]
    median = statistics.median(values)
    mad = statistics.median(abs(value - median) for value in values)
    tolerance = max(5., 3 * 1.4826 * mad)
    rejected = {row["scenario_id"] for row in rows
                if abs(float(row["baseline_source_power_w"]) - median) > tolerance}
    return rejected, {"baseline_median_w": median, "baseline_mad_w": mad,
                      "baseline_tolerance_w": tolerance,
                      "baseline_rejected_scenarios": sorted(rejected)}


def collect(root: Path, profile: ModelProfile) -> tuple[list[dict], dict]:
    plan = json.loads((root / "plan.json").read_text())
    manifest_path = Path(plan["manifest"]["path"])
    if manifest_path.parts[0] == "queue-haul":
        manifest_path = Path(__file__).parent.joinpath(*manifest_path.parts[1:])
    manifest = json.loads(manifest_path.read_text())
    scenarios = {row["scenario_id"]: row for row in plan["scenarios"]}
    with (root / "results.csv").open(newline="") as handle:
        results = {row["scenario_id"]: row for row in csv.DictReader(handle)}
    with (root / "trailing_power.csv").open(newline="") as handle:
        trailing = list(csv.DictReader(handle))
    rejected, gate = baseline_gate(trailing)
    phase, points = profile.case().phase_power, []
    if phase is None:
        raise ValueError("pack fit requires phase power")
    for measured in trailing:
        if measured["scenario_id"] in rejected:
            continue
        scenario, result_row = scenarios[measured["scenario_id"]], results[measured["scenario_id"]]
        problem, *_ = network._scenario_problem(scenario, manifest, profile)
        result = json.loads((root / "scenarios" / measured["scenario_id"]
                             / f"attempt-{int(result_row['attempt']):04d}"
                             / "result.json").read_text())
        cutoff = int(result["started_ns"]) + int(float(scenario["deadline_s"]) * 1e9)
        moved = {row["session_id"] for row in result["requests"] if "request" in row
                 and int(row["request"]["end_ns"]) <= cutoff}
        for window, removed, watts in (
            ("pre", set(), measured["baseline_source_power_w"]),
            ("post", moved, measured["measured_trailing_source_power_w"]),
        ):
            if window == "post" and result_row["deadline_met"] != "True":
                continue
            kept = [row for row in problem.sessions if row.session_id not in removed]
            points.append({"scenario_id": measured["scenario_id"], "window": window,
                           "repeat": int(scenario["repeat"]), "moved": len(removed),
                           "load": phase.load(sum(row.expected_f for row in kept),
                                              sum(row.expected_g for row in kept)),
                           "power_w": float(watts)})
    return points, gate


def fit(profile_path: Path, root: Path, out_profile: Path, summary_path: Path,
        points_path: Path, far_power_w: float) -> dict:
    profile, raw = ModelProfile.load(profile_path), json.loads(profile_path.read_text())
    phase = profile.case().phase_power
    if phase is None:
        raise ValueError("pack fit requires phase power")
    points, baseline = collect(root, profile)
    curve, metrics = fit_curve(points, phase.p0_w, profile.max_power_load, far_power_w)
    metrics.update(grouped_repeat_cv(points, phase.p0_w, profile.max_power_load,
                                     far_power_w))
    bootstraps = bootstrap_curves(points, phase.p0_w, profile.max_power_load, far_power_w)
    phase_raw = raw["cases"]["central"]["phase_power"]
    phase_raw["measured_power_curve"] = curve
    phase_raw["measured_power_bootstrap"] = bootstraps
    phase_raw["delta_w"] = far_power_w - phase.p0_w
    phase_raw["bootstrap"] = []
    digest = hashlib.sha256((root / "trailing_power.csv").read_bytes()).hexdigest()
    phase_raw["provenance_sha256"] = digest
    raw["profile_id"] += "-pack-power-v1"
    raw["sources"]["power"] = {"kind": "measured", "reference": str(root / "trailing_power.csv"),
        "valid_range": [0, profile.max_power_load],
        "relative_error": metrics["mae_w"] / (far_power_w - phase.p0_w)}
    out_profile.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    ModelProfile.load(out_profile)
    with points_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=points[0]); writer.writeheader(); writer.writerows(points)
    summary = {"schema": "queue-haul-pack-power-fit-v1", **metrics,
               "gate_passed": metrics["grouped_repeat_cv_mae_w"] <= 5
               and metrics["grouped_repeat_cv_within_5w_fraction"] >= .8,
               "points": len(points), "curve": curve, "scope": "recorded-28-seed-8",
               **baseline}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, required=True); parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-profile", type=Path, required=True); parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--points", type=Path, required=True); parser.add_argument("--far-power-w", type=float, required=True)
    args = parser.parse_args(); fit(args.profile, args.root, args.out_profile,
                                    args.summary, args.points, args.far_power_w)


if __name__ == "__main__":
    main()
