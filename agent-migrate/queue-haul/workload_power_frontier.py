"""Pooled power-shed frontiers over paired workload/calibration draws."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from planner import plan, source_power
import plot_style
from plot_hardware_shed_frontier import (
    evaluated_source_power, planning_problem, plateau_attainment,
)
from plot_pooled_shed_frontier import write_csv
from pool_planner import MAX_SHED_CAPACITY_GAP
from profiles import ModelProfile
import workload_adaptation_campaign as adaptation


OUT = adaptation.ROOT / "outputs/workload-power-frontier-20260814/pooled_shed_frontier"
DEFAULT_SAMPLES = 100
DEFAULT_POINTS = 9
MAX_REQUEST = 1.0
SOLVERS = {"queue_haul_lp": "lp_highs"}
CAPACITY_SOLVER = "max_shed_capacity"
CAPACITY_ORDER_TOLERANCE = 4 * MAX_SHED_CAPACITY_GAP
DISPLAY_STATES = adaptation.DISPLAY_CASES
LINE_ZORDERS = {"bandwidth": 3, "none": 2}
FIGSIZE = (3.35, 2.5)
plot_style.apply()


def request_grid(points, maximum=MAX_REQUEST):
    if points < 2 or not 0 < maximum <= 1:
        raise ValueError("invalid workload frontier grid")
    return tuple(sorted(set(np.linspace(0, maximum, points)) | {2 / 3}))


def planning_request_w(fraction, maximum):
    return fraction * maximum + (
        adaptation.POWER_TOLERANCE_W if fraction == MAX_REQUEST else 0
    )


def capacity_release_audit(rows, close=False):
    by_constraints = {
        constraints: case_id for case_id, _, constraints in adaptation.factorial_cases()
    }
    capacities = {}
    for row in rows:
        key = row["replicate"], row["factor_case_id"]
        value = row["capacity_mip_shed_w"]
        if key in capacities and not np.isclose(capacities[key], value):
            raise RuntimeError("capacity reference changed within a paired case")
        capacities[key] = value
    comparisons, inversions, maximum_uplift = 0, 0, 0.0
    cases = sorted(adaptation.factorial_cases(), key=lambda item: -len(item[2]))
    for replicate in {row["replicate"] for row in rows}:
        maximum = next(row["maximum_removable_w"] for row in rows
                       if row["replicate"] == replicate)
        for case_id, _, constraints in cases:
            for factor in constraints:
                comparisons += 1
                released = by_constraints[constraints - {factor}]
                difference = capacities[replicate, case_id] \
                    - capacities[replicate, released]
                if difference > CAPACITY_ORDER_TOLERANCE * maximum + 1e-8:
                    raise RuntimeError("capacity solver exceeded its release tolerance")
                if difference > 0 and close:
                    capacities[replicate, released] = \
                        capacities[replicate, case_id]
                if difference > 1e-8:
                    inversions += 1
                    maximum_uplift = max(maximum_uplift, difference / maximum)
    if close:
        for row in rows:
            value = capacities[row["replicate"], row["factor_case_id"]]
            row["maximum_attainable_shed_w"] = value
            row["maximum_attainable_fraction"] = \
                value / row["maximum_removable_w"]
    return {
        "comparisons": comparisons, "raw_solver_inversions": inversions,
        "maximum_monotone_uplift_fraction": maximum_uplift, "violations": 0,
    }


def sweep(samples=DEFAULT_SAMPLES, points=DEFAULT_POINTS,
          seed=adaptation.DEFAULT_SEED, sessions=28):
    if samples < 1 or sessions < 1:
        raise ValueError("invalid workload frontier controls")
    profile = ModelProfile.load(adaptation.PROFILE)
    templates, workload = adaptation.load_templates(adaptation.MANIFEST, profile)
    timing_rows = adaptation.read_csv(adaptation.TIMING)
    parent = json.loads(adaptation.TIMING_PARENT.read_text())
    adaptation.central_timing_fits()
    fractions, rng, rows = request_grid(points), np.random.default_rng(seed), []
    for replicate in range(samples):
        sampled, pack, fits, power_index, timing_hash, projected = \
            adaptation.sample_draw(
                profile, templates, timing_rows, parent, rng, replicate, seed,
                sessions,
            )
        for factor_id, label, constraints in adaptation.factorial_cases():
            problem, architecture, routes, _ = adaptation.build_problem(
                sampled, pack, constraints, 2 / 3, fits,
            )
            initial = source_power(problem, sampled)
            minimum = source_power(
                problem, sampled,
                (session.session_id for session in problem.sessions),
            )
            maximum = initial - minimum
            if maximum <= 0:
                raise RuntimeError("sampled workload has no removable power")
            capacity_result = plan(
                problem, sampled, routes, CAPACITY_SOLVER, seed=replicate,
                destination=architecture, admission_mode="normal",
            )
            capacity_safe = capacity_result.failure_reason in {None, "target_unmet"} \
                and all(use.utilization <= 1 + 1e-8
                        for use in capacity_result.resource_uses)
            capacity = max(0.0, initial - evaluated_source_power(
                problem, problem, capacity_result, sampled, architecture,
            ))
            if not capacity_safe or capacity > maximum + 1e-8:
                raise RuntimeError("maximum-shed capacity reference is invalid")
            phase = sampled.case().phase_power
            if phase is None:
                raise RuntimeError("capacity reference requires phase-aware power")
            for policy, solver in SOLVERS.items():
                raw, admissible, pending = [], [], []
                for fraction in fractions:
                    target = fraction * maximum
                    actual = replace(
                        problem,
                        power_limit_w=initial - planning_request_w(fraction, maximum),
                    )
                    planned = planning_problem(actual, policy)
                    if fraction == 0:
                        shed, safe, failure = 0.0, True, ""
                    else:
                        result = plan(
                            planned, sampled, routes, solver, seed=replicate,
                            destination=architecture, admission_mode="normal",
                        )
                        shed = max(0.0, initial - evaluated_source_power(
                            actual, planned, result, sampled, architecture,
                        ))
                        safe = result.failure_reason in {None, "target_unmet"} \
                            and all(use.utilization <= 1 + 1e-8
                                    for use in result.resource_uses)
                        failure = result.failure_reason or ""
                    raw.append(shed)
                    admissible.append(safe)
                    pending.append({
                        "case_id": f"{replicate}/{factor_id}",
                        "replicate": replicate, "factor_case_id": factor_id,
                        "bound_constraint": label,
                        "constraints": "+".join(sorted(constraints)) or "none",
                        "policy": policy, "requested_fraction": fraction,
                        "maximum_removable_w": maximum,
                        "requested_shed_w": target, "raw_safe_shed_w": shed,
                        "capacity_mip_shed_w": capacity,
                        "capacity_solver": CAPACITY_SOLVER,
                        "plan_safe": safe,
                        "target_met_by_30s": safe and shed >= target - 1e-8,
                        "failure_reason": failure,
                        "power_bootstrap_index": power_index,
                        "scoring_deadline_s": adaptation.SCORING_DEADLINE_S,
                        "power_window_s": sampled.power_window_s,
                        "controller_delay_s": problem.controller_delay_s,
                        "migration_budget_s": adaptation.migration_budget_s(sampled),
                        "phase_p0_w": phase.p0_w, "phase_delta_w": phase.delta_w,
                        "phase_a_s_per_prefill_token": phase.a_s_per_prefill_token,
                        "phase_b_s_per_decode_token": phase.b_s_per_decode_token,
                        "phase_power_provenance_sha256": phase.provenance_sha256,
                        "timing_fit_sha256": timing_hash,
                        "bandwidth_projection_regions": projected,
                    })
                attained = plateau_attainment(
                    [fraction * maximum for fraction in fractions], raw, admissible,
                )
                rows.extend({
                    **row, "safely_attained_shed_w": value,
                    "safely_attained_fraction": value / maximum,
                } for row, value in zip(pending, attained))
    expected = samples * len(adaptation.ORDER) * len(SOLVERS) * len(fractions)
    if len(rows) != expected:
        raise RuntimeError("workload frontier is incomplete")
    capacity_release_audit(rows, close=True)
    return rows, workload


def power_summary(rows):
    cases = {row["case_id"] for row in rows}
    summary = []
    for policy in SOLVERS:
        for fraction in sorted({row["requested_fraction"] for row in rows
                                if row["policy"] == policy}):
            selected = [row for row in rows if row["policy"] == policy
                        and row["requested_fraction"] == fraction]
            if len(selected) != len(cases) or \
                    {row["case_id"] for row in selected} != cases:
                raise RuntimeError("power summary does not weight each case once")
            maximum = np.quantile(
                [row["maximum_removable_w"] for row in selected], (.05, .5, .95),
            )
            attained = np.quantile(
                [row["safely_attained_shed_w"] for row in selected], (.05, .5, .95),
            )
            summary.append({
                "policy": policy, "requested_fraction": fraction,
                "cases": len(cases),
                **{f"maximum_removable_w_{name}": value
                   for name, value in zip(("p05", "median", "p95"), maximum)},
                **{f"safely_attained_w_{name}": value
                   for name, value in zip(("p05", "median", "p95"), attained)},
            })
    return summary


def capacity_summary(rows, grid=np.linspace(0, 1, 101)):
    if not len(grid) or grid[0] != 0 or grid[-1] != 1 \
            or any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("capacity grid must increase from zero to one")
    selected = [row for row in rows if row["policy"] == "queue_haul_lp"]
    output = []
    for state in DISPLAY_STATES:
        candidates = [row for row in selected
                      if row["factor_case_id"] == state]
        capacities = {}
        for row in candidates:
            replicate, value = row["replicate"], row["maximum_attainable_fraction"]
            if replicate in capacities and not np.isclose(capacities[replicate], value):
                raise RuntimeError("capacity summary has conflicting paired maxima")
            capacities[replicate] = value
        if not capacities:
            raise RuntimeError("capacity summary requires paired maxima")
        values = np.asarray(tuple(capacities.values()))
        output.extend({
            "constraint_state": state,
            "bound_constraint": plot_style.RESOURCE_STATE_NAMES[state],
            "requested_fraction": float(request),
            "attainment_rate": float(np.mean(values >= request - 1e-9)),
            "cases": len(values),
        } for request in grid)
    return output


def write_capacity_plot(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    fig, axis = plt.subplots(figsize=FIGSIZE)
    for state in DISPLAY_STATES:
        selected = [row for row in summary if row["constraint_state"] == state]
        axis.step(
            [row["requested_fraction"] for row in selected],
            [row["attainment_rate"] for row in selected],
            where="post",
            color=plot_style.RESOURCE_STATE_COLORS[state],
            linestyle=plot_style.RESOURCE_STATE_LINESTYLES[state],
            zorder=LINE_ZORDERS.get(state, 1),
            label=plot_style.RESOURCE_STATE_NAMES[state],
        )
    axis.set(xlim=(0, 1), ylim=(0, 1),
             xlabel="Requested Source-Power Fraction",
             ylabel="Cases Meeting Target")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.tick_params(labelsize=10)
    axis.xaxis.label.set_size(10)
    axis.yaxis.label.set_size(10)
    axis.grid(alpha=.2)
    axis.legend(frameon=False, fontsize=7.5, loc="lower left")
    fig.subplots_adjust(left=.25, right=.97, bottom=.2, top=.97)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--points", type=int, default=DEFAULT_POINTS)
    parser.add_argument("--seed", type=int, default=adaptation.DEFAULT_SEED)
    parser.add_argument("--sessions", type=int, default=28)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    validation = adaptation.validate_surface()
    rows, workload = sweep(args.samples, args.points, args.seed, args.sessions)
    profile = ModelProfile.load(adaptation.PROFILE)
    summary = capacity_summary(rows)
    write_csv(rows, args.out.with_name(f"{args.out.name}_cases.csv"))
    write_csv(summary, args.out.with_suffix(".csv"))
    write_csv(power_summary(rows), args.out.with_name(
        f"{args.out.name}_power.csv"
    ))
    write_capacity_plot(summary, args.out)
    metadata = {
        "schema": "queue-haul-workload-power-frontier-v9",
        "claim": "modeled source-power capacity distribution across regional constraint states using a certified integer maximum-phase-load reference",
        "samples": args.samples, "factor_states": len(adaptation.ORDER),
        "pooled_cases": args.samples * len(adaptation.ORDER),
        "sessions_per_pack": args.sessions, "seed": args.seed,
        "scoring_deadline_s": adaptation.SCORING_DEADLINE_S,
        "power_window_s": profile.power_window_s,
        "controller_delay_s": 0,
        "migration_budget_s": adaptation.migration_budget_s(profile),
        "requested_fractions": list(request_grid(args.points)),
        "policies": list(SOLVERS), "solvers": SOLVERS, "workload": workload,
        "factor_levels": adaptation.LEVELS, "regions": list(adaptation.REGIONS),
        "bandwidth_bottleneck": {
            "physical_route_mbps": adaptation.BANDWIDTH_BOTTLENECK_MBPS,
            "natural_physical_route_mbps": adaptation.physical_route_mbps(),
            "pipeline_timing_condition": "controlled_40",
        },
        "normalization": "safely attained watts / draw-specific removable watts",
        "figure_metric": "fraction of paired draws whose certified maximum attainable source-power fraction meets each request",
        "capacity_reference": {
            "solver": CAPACITY_SOLVER,
            "objective": "maximize removed z=sum(a*f+b*g) with integer session actions",
            "mip_relative_gap": MAX_SHED_CAPACITY_GAP,
            "release_audit": capacity_release_audit(rows),
        },
        "plotted_constraint_states": list(DISPLAY_STATES),
        "constraint_style": "canonical color and line styles distinguish the five displayed constraint states",
        "power_target": "invert sampled monotone phase power once and constrain additive removed phase load; verify exact nonlinear watts after packing",
        "power_model": {
            "profile": str(adaptation.PROFILE.relative_to(adaptation.ROOT)),
            "a_s_per_prefill_token": profile.case().phase_power.a_s_per_prefill_token,
            "b_s_per_decode_token": profile.case().phase_power.b_s_per_decode_token,
            "bootstrap_draw": "one joint (p0, delta, a, b) tuple per replicate",
        },
        "power_scope": "steady awake source-region power; destination power excluded",
        "source_load_definition": "sum(f/F + g/G)=0.4; distinct from sampled phase load z=af+bg",
        "surface_validation": adaptation.validation_summary(validation),
        "inputs": {
            str(path.relative_to(adaptation.ROOT)): adaptation.file_hash(path)
            for path in (
                adaptation.PROFILE, adaptation.MANIFEST, adaptation.TIMING,
                adaptation.TIMING_SUMMARY, adaptation.TIMING_PARENT,
                adaptation.LOADED_SERVICE, adaptation.NETWORK_CALIBRATION,
                adaptation.LOCAL_TIMING / "scenarios.csv",
                adaptation.LOCAL_TIMING / "migrations.csv",
                adaptation.WIDTH8_TIMING / "scenarios.csv",
                adaptation.WIDTH8_TIMING / "migration_stages.csv",
            )
        },
        "limitations": [
            "the first deterministic paired draws are a sensitivity ensemble, not independent observations or a confidence interval",
            "each workload draw and global constraint state receives equal weight",
            "the main figure shows the three single bottlenecks plus all-bound and none-bound states; the raw case table retains all eight factorial states",
            "the three intermediate two-factor states are retained in raw tables but omitted from the main figure",
            "the frontier remains modeled rather than hardware-measured",
            "the integer capacity reference is solved to a certified 0.25% relative MIP gap",
            "the bandwidth state caps both physical destination routes at the 1-Gbit/s lower boundary of existing A100 loaded-migration validation",
            "reported shed is awake source-region power rather than net fleet power or energy",
            "Replay endpoint work uses the measured prefill-heavy relative load factor; KV is load-neutral centrally",
            "short-context Replay inside regional support uses a constant minimum-base-rate sensitivity extension",
            "the relative load factor is transported from a fixed width-eight pack to regional concurrency-one timing as a sensitivity",
            "the loaded-service slope is fixed at its central fit rather than sampled from its bootstrap",
            "bandwidth-bound East routes fall below the load factor's 1-Gbit/s validation floor",
            "route and endpoint stages overlap under the calibrated effective pipeline rate; Replay and KV endpoint work remain fully shared",
            adaptation.surface_scope_limitation(validation),
        ],
    }
    args.out.with_name(f"{args.out.name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
