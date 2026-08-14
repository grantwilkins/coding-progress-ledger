"""Recalibrate H100 power load and audit a completed frontier campaign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

import network_campaign as campaign
from planner import _expected_scenario, plan as solve, source_power
from profiles import ModelProfile, PowerCurve
from simulate import predict


def read_levels(path: Path, max_rate: float) -> list[dict]:
    return [row for row in json.loads(path.read_text())
            if row["rate_rps"] <= max_rate]


def concave_envelope(points) -> list[list[float]]:
    curve = []
    for point in sorted(map(list, points)):
        point[1] = max(point[1], curve[-1][1] if curve else point[1])
        curve.append(point)
        while len(curve) > 2:
            left = (curve[-2][1] - curve[-3][1]) / (curve[-2][0] - curve[-3][0])
            right = (curve[-1][1] - curve[-2][1]) / (curve[-1][0] - curve[-2][0])
            if left >= right:
                break
            curve.pop(-2)
    return curve


def calibrate_linear_load(prefill: list[dict], mixed: list[dict],
                          F: float, idle_w: float, max_ell: float) -> dict:
    alpha = 1 / F
    curve = concave_envelope([(0, idle_w), *((
        alpha * row["prompt_tokens"] / row["window_s"], row["power_mean_w"]
    ) for row in prefill)])
    if curve[-1][0] < max_ell:
        curve.append([max_ell, curve[-1][1]])
    x, y = map(np.asarray, zip(*curve))
    f = np.asarray([row["prompt_tokens"] / row["window_s"] for row in mixed])
    g = np.asarray([row["output_tokens"] / row["window_s"] for row in mixed])
    watts = np.asarray([row["power_mean_w"] for row in mixed])
    fitted = least_squares(
        lambda value: np.interp(alpha * f + value[0] * g, x, y) - watts,
        (1 / F,), bounds=(0, np.inf),
    )
    beta = float(fitted.x[0])
    errors = np.interp(alpha * f + beta * g, x, y) - watts
    return {
        "alpha": alpha, "beta": beta, "effective_F": F,
        "effective_G": 1 / beta, "beta_over_alpha": beta / alpha,
        "mixed_rmse_w": float(np.sqrt(np.mean(errors ** 2))),
        "mixed_mae_w": float(np.mean(abs(errors))), "power_curve": curve,
        "prefill_points": len(prefill), "mixed_points": len(mixed),
    }


def service_load(path: Path, start_ns: int, window_s: float,
                 prefill_tps: float, decode_tps: float) -> float:
    lo = start_ns - round(window_s * 1e9)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [row for row in rows if lo <= row["start_ns"] < start_ns]
    return sum(row["prompt_tokens"] for row in rows) / window_s / prefill_tps \
        + sum(row["output_tokens"] for row in rows) / window_s / decode_tps


def quantile(values, q):
    return float(np.quantile(values, q)) if values else None


def grouped(rows, key, value) -> dict:
    result = {}
    for label in sorted({row[key] for row in rows}):
        values = [row[value] for row in rows if row[key] == label
                  and row[value] not in (None, "")]
        result[str(label)] = {"n": len(values), "median": quantile(values, .5),
                              "p90": quantile(values, .9)}
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit(run_root: Path, manifest_path: Path, profile_path: Path,
          out: Path, calibration: dict, load_window_s: float = 4) -> dict:
    plan = json.loads((run_root / "plan.json").read_text())
    regions = {row["id"]: row["region"]
               for row in plan["cluster"]["destinations"]}
    manifest = json.loads(manifest_path.read_text())
    profile = ModelProfile.load(profile_path)
    central = replace(
        profile.case(), power_alpha=calibration["alpha"],
        power_beta=calibration["beta"],
        power_curve=PowerCurve.parse(calibration["power_curve"]),
    )
    profile = replace(profile, cases={**profile.cases, "central": central})
    scenarios, sessions, loads = [], [], []
    by_condition = {}
    for scenario in plan["scenarios"]:
        latest = campaign._latest_result(
            run_root / "scenarios" / scenario["scenario_id"])
        if latest is None or latest[1].get("status") != "complete":
            continue
        attempt, result = latest
        demand = campaign.agentic_demand(
            campaign.scenario_records(manifest, scenario), scenario["sessions"],
            profile, scenario["source_load"],
        )
        problem, architecture, routes, requested = campaign.joint_problem(
            scenario, result["background"], profile, demand)
        moves = tuple(campaign._planned_move(row) for row in result["requests"])
        modeled = predict(
            _expected_scenario(problem, moves), profile, moves,
            destination=architecture,
        )
        recalibrated = campaign.diagnostic_outcomes(
            scenario, result["requests"], demand, profile, result["started_ns"])
        selected = [move.session_id for move in moves]
        initial = source_power(problem, profile)
        planned_shed = initial - source_power(problem, profile, selected)
        removable = recalibrated["requested_shed_w"] \
            / scenario["requested_shed_fraction"]
        predicted_s = modeled.migration_makespan_s or 0
        observed_s = result["migration_s"]
        load_pair = "/".join(f"{value[0]:g}" for _, value in
                             sorted(scenario["background"].items()))
        scenarios.append({
            "scenario_id": scenario["scenario_id"], "policy": scenario["policy"],
            "pack": scenario["pack"], "destination_load": load_pair,
            "selected_sessions": len(moves), "request_failures": result["request_failures"],
            "predicted_makespan_s": predicted_s, "observed_makespan_s": observed_s,
            "timing_ratio": observed_s / predicted_s if predicted_s else None,
            "recorded_requested_shed_w": result["requested_shed_w"],
            "recalibrated_requested_shed_w": recalibrated["requested_shed_w"],
            "recalibrated_planned_shed_w": planned_shed,
            "recalibrated_realized_shed_w": recalibrated["realized_shed_w"],
            "recalibrated_eventual_shed_w": recalibrated["eventual_shed_w"],
            "recalibrated_planned_fraction": planned_shed / removable,
            "recalibrated_realized_fraction": recalibrated["realized_shed_w"] / removable,
            "recalibrated_eventual_fraction": recalibrated["eventual_shed_w"] / removable,
            "recorded_target_met": result["target_met"],
            "recalibrated_target_met": recalibrated["target_met"],
            "recalibrated_eventual_target_met": recalibrated["eventual_target_met"],
        })
        predicted = {row.session_id: row for row in modeled.sessions}
        for row in result["requests"]:
            if "request" not in row or predicted[row["session_id"]].committed_s is None:
                continue
            predicted_s = predicted[row["session_id"]].committed_s
            observed_s = (row["request"]["end_ns"] - result["started_ns"]) / 1e9
            sessions.append({
                "scenario_id": scenario["scenario_id"], "pack": scenario["pack"],
                "destination_load": load_pair, "width": len(moves),
                "destination_id": row["destination_instance"],
                "destination": regions[row["destination_instance"]],
                "method": row["method"],
                "path": f'{regions[row["destination_instance"]]}:{row["method"]}',
                "context_tokens": next(item["initial_tokens"] for item in
                                       scenario["sessions"]
                                       if item["session_id"] == row["session_id"]),
                "predicted_s": predicted_s, "observed_s": observed_s,
                "timing_ratio": observed_s / predicted_s,
            })
        attempt_root = run_root / "scenarios" / scenario["scenario_id"] \
            / f"attempt-{attempt:04d}"
        dtype = architecture.types[0]
        for node, nominal in scenario["background"].items():
            path = attempt_root / f"sink_load_{node}.jsonl"
            if path.exists():
                loads.append({
                    "scenario_id": scenario["scenario_id"], "destination_id": node,
                    "destination": regions[node],
                    "load_cell": f"{regions[node]}:{nominal[0]:g}",
                    "nominal_load": nominal[0],
                    "achieved_load": service_load(
                        path, result["started_ns"], load_window_s,
                        dtype.prefill.at(campaign.SINK_LOAD_PREFILL_TOKENS),
                        dtype.decode.at(campaign.SINK_LOAD_PREFILL_TOKENS),
                    ),
                })
        if scenario["policy"] == "queue_haul":
            by_condition[scenario["condition_index"]] = (
                scenario, problem, architecture, routes, requested)
    ceilings = []
    for condition, (scenario, problem, architecture, routes, requested) \
            in sorted(by_condition.items()):
        result = solve(problem, profile, routes, "max_shed",
                       seed=scenario["planner_seed"], destination=architecture)
        bottleneck = result.bottleneck or ""
        for node, region in regions.items():
            bottleneck = bottleneck.replace(node, region)
        ceilings.append({
            "condition_index": condition, "pack": scenario["pack"],
            "destination_load": "/".join(f"{value[0]:g}" for _, value in
                                          sorted(scenario["background"].items())),
            "requested_shed_w": requested,
            "maximum_planned_shed_w": result.initial_source_power_w
            - result.planned_source_power_w,
            "target_feasible": bool(result.feasible),
            "selected_sessions": len(result.moves),
            "predicted_makespan_s": result.predicted_migration_makespan_s,
            "bottleneck": bottleneck,
        })
    valid_timing = [row["timing_ratio"] for row in scenarios
                    if row["timing_ratio"] and not row["request_failures"]]
    fractions = (.4, .5, .6, .7, .8, .9)
    watts = (30, 35, 40, 45, 50, 55)
    summary = {
        "schema": "queue-haul-frontier-model-audit-v1", "scenarios": len(scenarios),
        "calibration": calibration,
        "power": {
            "recorded_target_met": sum(row["recorded_target_met"] for row in scenarios),
            "recalibrated_target_met": sum(row["recalibrated_target_met"] for row in scenarios),
            "recalibrated_eventual_target_met": sum(
                row["recalibrated_eventual_target_met"] for row in scenarios),
            "median_recorded_requested_shed_w": statistics.median(
                row["recorded_requested_shed_w"] for row in scenarios),
            "median_recalibrated_requested_shed_w": statistics.median(
                row["recalibrated_requested_shed_w"] for row in scenarios),
            "exact_feasible_conditions": sum(
                bool(row["target_feasible"]) for row in ceilings),
            "conditions": len(ceilings),
            "fixed_plan_by_fraction": {
                str(value): {
                    "deadline": sum(row["recalibrated_realized_fraction"] >= value
                                    for row in scenarios),
                    "eventual": sum(row["recalibrated_eventual_fraction"] >= value
                                    for row in scenarios),
                } for value in fractions
            },
            "fixed_plan_by_watts": {
                str(value): {
                    "deadline": sum(row["recalibrated_realized_shed_w"] >= value
                                    for row in scenarios),
                    "eventual": sum(row["recalibrated_eventual_shed_w"] >= value
                                    for row in scenarios),
                } for value in watts
            },
        },
        "timing": {
            "median_scenario_actual_over_predicted": quantile(valid_timing, .5),
            "p90_scenario_actual_over_predicted": quantile(valid_timing, .9),
            "by_method": grouped(sessions, "method", "timing_ratio"),
            "by_destination": grouped(sessions, "destination", "timing_ratio"),
            "by_path": grouped(sessions, "path", "timing_ratio"),
            "by_context": grouped(sessions, "context_tokens", "timing_ratio"),
            "by_pack": grouped(scenarios, "pack", "timing_ratio"),
        },
        "destination_load": grouped(loads, "load_cell", "achieved_load"),
        "limitations": [
            "Power attainment is model-recomputed; migrated sessions did not drive source power.",
            "Destination achieved load is inferred from request starts, not server work counters.",
            "The counterfactual ceiling is model-based and does not repair timing error.",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (("scenarios.csv", scenarios), ("session_timing.csv", sessions),
                       ("destination_load.csv", loads), ("condition_ceiling.csv", ceilings)):
        write_csv(out / name, rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--prefill-levels", type=Path, required=True)
    parser.add_argument("--mixed-levels", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefill-max-rate", type=float, default=8)
    parser.add_argument("--mixed-max-rate", type=float, default=4)
    args = parser.parse_args()
    profile = ModelProfile.load(args.profile)
    prefill = read_levels(args.prefill_levels, args.prefill_max_rate)
    mixed = read_levels(args.mixed_levels, args.mixed_max_rate)
    calibration = calibrate_linear_load(
        prefill, mixed, profile.case().F,
        profile.case().power_curve.power(0), profile.max_ell,
    )
    calibration["previous_alpha"] = profile.case().power_alpha or 1 / profile.case().F
    calibration["previous_beta"] = profile.case().power_beta or 1 / profile.case().G
    previous_errors = np.asarray([
        profile.case().power_curve.power(
            calibration["previous_alpha"] * row["prompt_tokens"] / row["window_s"]
            + calibration["previous_beta"] * row["output_tokens"] / row["window_s"]
        ) - row["power_mean_w"] for row in mixed
    ])
    calibration["previous_mixed_bias_w"] = float(np.mean(previous_errors))
    calibration["previous_mixed_rmse_w"] = float(
        np.sqrt(np.mean(previous_errors ** 2)))
    calibration["beta_sensitivity"] = {
        str(rate): calibrate_linear_load(
            prefill, [row for row in mixed if row["rate_rps"] <= rate],
            profile.case().F, profile.case().power_curve.power(0), profile.max_ell,
        )["beta"]
        for rate in sorted({row["rate_rps"] for row in mixed})[1:]
    }
    print(json.dumps(audit(args.run_root, args.manifest, args.profile,
                           args.out, calibration), indent=2))


if __name__ == "__main__":
    main()
