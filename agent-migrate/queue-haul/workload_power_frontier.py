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
from profiles import ModelProfile
import workload_adaptation_campaign as adaptation


OUT = adaptation.ROOT / "outputs/workload-power-frontier-20260814/pooled_shed_frontier"
DEFAULT_SAMPLES = 100
DEFAULT_POINTS = 9
MAX_REQUEST = 1.0
SOLVERS = {"queue_haul_lp": "lp_highs"}
DISPLAY_STATES = {
    "none": "none", "hbm": "hbm", "dest_compute": "dest_compute",
    "all": "bandwidth-dest_compute-hbm",
}
BANDWIDTH_NULL_PAIRS = (
    ("none", "bandwidth"), ("hbm", "bandwidth-hbm"),
    ("dest_compute", "bandwidth-dest_compute"),
    ("dest_compute-hbm", "bandwidth-dest_compute-hbm"),
)
plot_style.apply()


def request_grid(points, maximum=MAX_REQUEST):
    if points < 2 or not 0 < maximum <= 1:
        raise ValueError("invalid workload frontier grid")
    return tuple(sorted(set(np.linspace(0, maximum, points)) | {2 / 3}))


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
            for policy, solver in SOLVERS.items():
                raw, admissible, pending = [], [], []
                for fraction in fractions:
                    target = fraction * maximum
                    actual = replace(problem, power_limit_w=initial - target)
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
                        "plan_safe": safe,
                        "target_met_by_30s": safe and shed >= target - 1e-8,
                        "failure_reason": failure,
                        "power_bootstrap_index": power_index,
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


def assert_bandwidth_null(rows):
    selected = [row for row in rows if row["policy"] == "queue_haul_lp"]
    for baseline, constrained in BANDWIDTH_NULL_PAIRS:
        paired = []
        for state in (baseline, constrained):
            values = {(row["replicate"], row["requested_fraction"]):
                      row["safely_attained_fraction"] for row in selected
                      if row["factor_case_id"] == state}
            if len(values) != sum(row["factor_case_id"] == state
                                  for row in selected):
                raise RuntimeError("duplicate paired frontier point")
            paired.append(values)
        if paired[0].keys() != paired[1].keys() or any(
                not np.isclose(paired[0][key], paired[1][key], rtol=0, atol=1e-12)
                for key in paired[0]):
            raise RuntimeError("bandwidth state is not null and cannot be collapsed")


def capacity_summary(rows, grid=np.linspace(0, 1, 101)):
    if not len(grid) or grid[0] != 0 or grid[-1] != 1 \
            or any(right <= left for left, right in zip(grid, grid[1:])):
        raise ValueError("capacity grid must increase from zero to one")
    selected = [row for row in rows if row["policy"] == "queue_haul_lp"]
    maximum_request = max(row["requested_fraction"] for row in selected)
    output = []
    for state, factor_case in DISPLAY_STATES.items():
        candidates = [row for row in selected
                      if row["factor_case_id"] == factor_case
                      and row["requested_fraction"] == maximum_request]
        capacities = {row["replicate"]: row["safely_attained_fraction"]
                      for row in candidates}
        if len(candidates) != len(capacities) or not capacities:
            raise RuntimeError("capacity summary requires one maximum per paired draw")
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

    fig, axis = plt.subplots(figsize=(4, 3))
    for state in DISPLAY_STATES:
        selected = [row for row in summary if row["constraint_state"] == state]
        axis.step(
            [row["requested_fraction"] for row in selected],
            [row["attainment_rate"] for row in selected],
            where="post",
            color=plot_style.RESOURCE_STATE_COLORS[state],
            linestyle=plot_style.RESOURCE_STATE_LINESTYLES[state],
            label=plot_style.RESOURCE_STATE_NAMES[state],
        )
    axis.set(xlim=(0, 1), ylim=(0, 1),
             xlabel="Requested Source-Power Fraction",
             ylabel="Cases Meeting Target")
    axis.xaxis.set_major_formatter(PercentFormatter(1))
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.tick_params(labelsize=11)
    axis.xaxis.label.set_size(12)
    axis.yaxis.label.set_size(12)
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=10, ncol=1,
               loc="center left", bbox_to_anchor=(.94, .5))
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight")
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
    assert_bandwidth_null(rows)
    summary = capacity_summary(rows)
    write_csv(rows, args.out.with_name(f"{args.out.name}_cases.csv"))
    write_csv(summary, args.out.with_suffix(".csv"))
    write_csv(power_summary(rows), args.out.with_name(
        f"{args.out.name}_power.csv"
    ))
    write_capacity_plot(summary, args.out)
    metadata = {
        "schema": "queue-haul-workload-power-frontier-v4",
        "claim": "modeled Queue-Haul source-power capacity distribution across regional constraint states with exact nonlinear one-source power targets",
        "samples": args.samples, "factor_states": len(adaptation.ORDER),
        "pooled_cases": args.samples * len(adaptation.ORDER),
        "sessions_per_pack": args.sessions, "seed": args.seed,
        "requested_fractions": list(request_grid(args.points)),
        "policies": list(SOLVERS), "solvers": SOLVERS, "workload": workload,
        "factor_levels": adaptation.LEVELS, "regions": list(adaptation.REGIONS),
        "normalization": "safely attained watts / draw-specific removable watts",
        "figure_metric": "fraction of paired draws whose maximum safely attained Queue-Haul source-power fraction meets each request",
        "plotted_constraint_states": DISPLAY_STATES,
        "bandwidth_null": "each measured-bandwidth state is exactly paired-equal to its released counterpart before being collapsed from the main figure",
        "power_target": "invert sampled monotone phase power once and constrain additive removed phase load; verify exact nonlinear watts after packing",
        "power_scope": "steady awake source-region power; destination power excluded",
        "source_load_definition": "sum(f/F + g/G)=0.4; distinct from sampled phase load z=af+bg",
        "surface_validation": adaptation.validation_summary(validation),
        "inputs": {
            str(path.relative_to(adaptation.ROOT)): adaptation.file_hash(path)
            for path in (
                adaptation.PROFILE, adaptation.MANIFEST, adaptation.TIMING,
                adaptation.TIMING_SUMMARY, adaptation.TIMING_PARENT,
                adaptation.LOADED_SERVICE,
                adaptation.LOCAL_TIMING / "scenarios.csv",
                adaptation.LOCAL_TIMING / "migrations.csv",
                adaptation.WIDTH8_TIMING / "scenarios.csv",
                adaptation.WIDTH8_TIMING / "migration_stages.csv",
            )
        },
        "limitations": [
            "the first deterministic paired draws are a sensitivity ensemble, not independent observations or a confidence interval",
            "each workload draw and global constraint state receives equal weight",
            "the main figure and raw tables show Queue-Haul capacity across all eight constraint states",
            "measured bandwidth is a paired null effect in this workload ensemble and is omitted from the main legend after an exact equality gate",
            "the frontier remains modeled rather than hardware-measured",
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
