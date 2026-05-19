from __future__ import annotations

import csv
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from baselines import (
    solve_crossover_greedy,
    solve_mixed_greedy,
    solve_replay_only,
    solve_state_only,
)
from catalog import catalog_models, get_model
from coefficients import compute_coefficients
from cvxpy_solver import solve_cvxpy, solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from metrics import (
    allocation_diagnostics,
    action_mix,
    source_load_action_mix,
    source_load_destination_mix,
    source_load_moved_s,
    utilization,
)
from mirror_descent import solve_mirror_descent
from problem import make_problem, saturation_diagnostics
from queueing import queue_metrics
from workload import assert_workload_quality

REGIMES = ("bandwidth-spread", "prefill-spread", "background-load-spread")
TRANSITION_REGIME = "transition-coupled"
DEST_LABELS = {
    "bandwidth-spread": ("1 Gbps", "10 Gbps", "100 Gbps"),
    "prefill-spread": ("20% prefill load", "50% prefill load", "80% prefill load"),
    "background-load-spread": ("low network load", "balanced load", "cached context"),
    "transition-coupled": ("4 Gbps", "6 Gbps", "9 Gbps"),
}


def _selected_mirror_descent(problem):
    eta_x0 = 500.0 if problem.regime == TRANSITION_REGIME else 10.0
    return solve_mirror_descent(problem, eta_x0=eta_x0)


def _policy_row(model, regime, policy, result, problem, coeffs, diagnostics, cvx_obj):
    net_util, prefill_util = utilization(
        problem,
        coeffs,
        result.allocation if hasattr(result, "allocation") else result.y,
    )
    y = result.allocation if hasattr(result, "allocation") else result.y
    mix = action_mix(problem, y)
    source_load_mix = source_load_action_mix(problem, y)
    alloc = allocation_diagnostics(problem, coeffs, y)
    source_load_moved = source_load_moved_s(problem, y)
    source_load_shortfall = max(0.0, problem.source_load_target_s - source_load_moved)
    excess_source_load = max(0.0, source_load_moved - problem.source_load_target_s)
    capacity_feasible = bool(np.max(net_util) < 1.0 and np.max(prefill_util) < 1.0)
    objective = getattr(result, "objective", None)
    feasible = bool(
        getattr(result, "feasible", True)
        and objective is not None
        and np.isfinite(objective)
        and source_load_shortfall <= 1e-5
        and capacity_feasible
    )
    obj_gap = objective - cvx_obj if feasible else None
    rel_gap = obj_gap / max(1.0, abs(cvx_obj)) if feasible else None
    return {
        "model": model,
        "regime": regime,
        "policy": policy,
        "objective": f"{objective:.10g}" if feasible else "INFEASIBLE",
        "objective_gap_to_cvx": f"{max(0.0, obj_gap):.10g}" if feasible else "INFEASIBLE",
        "relative_gap_to_cvx": f"{max(0.0, rel_gap):.10g}" if feasible else "INFEASIBLE",
        "source_load_moved_s": f"{source_load_moved:.10g}",
        "source_load_target_s": f"{problem.source_load_target_s:.10g}",
        "source_load_shortfall_s": f"{source_load_shortfall:.10g}",
        "excess_source_load_s": f"{excess_source_load:.10g}",
        "max_net_util": f"{float(np.max(net_util)):.10g}",
        "max_prefill_util": f"{float(np.max(prefill_util)):.10g}",
        "capacity_feasible": str(capacity_feasible),
        "alpha": f"{getattr(result, 'alpha', np.nan):.10g}",
        "eta_x0": f"{getattr(result, 'eta_x0', np.nan):.10g}",
        "bisection_iterations": f"{getattr(result, 'bisection_iterations', np.nan):.10g}",
        "active_classes_moved": f"{alloc['active_classes_moved']:.10g}",
        "active_destinations_used": f"{alloc['active_destinations_used']:.10g}",
        "destination_entropy": f"{alloc['destination_entropy']:.10g}",
        "action_entropy": f"{alloc['action_entropy']:.10g}",
        **{key: f"{value:.10g}" for key, value in mix.items()},
        **{key: f"{value:.10g}" for key, value in source_load_mix.items()},
        "replay_demand_over_capacity": f"{diagnostics[0]:.10g}",
        "state_bytes_over_capacity": f"{diagnostics[1]:.10g}",
        "deadline_overrun_mean": f"{(getattr(result, 'diagnostics', None) or {}).get('deadline_overrun_mean', np.nan):.10g}",
        "deadline_overrun_p95": f"{(getattr(result, 'diagnostics', None) or {}).get('deadline_overrun_p95', np.nan):.10g}",
        "deadline_overrun_max": f"{(getattr(result, 'diagnostics', None) or {}).get('deadline_overrun_max', np.nan):.10g}",
        "deadline_load_max": f"{(getattr(result, 'diagnostics', None) or {}).get('deadline_load_max', np.nan):.10g}",
        "feasible": str(bool(feasible)),
    }


def _log_run(model, regime, problem, coeffs, diagnostics, cvx, md):
    net_util, prefill_util = utilization(problem, coeffs, md.y)
    gap = max(0.0, (md.objective - cvx.objective) / max(1.0, abs(cvx.objective)))
    source_load_moved = source_load_moved_s(problem, md.y)
    md_mix = source_load_action_mix(problem, md.y)
    cvx_mix = source_load_action_mix(problem, cvx.y)
    print(
        f"{model.name} / {regime}: diagnostics replay_demand/cap={diagnostics[0]:.3f}, "
        f"state_bytes/cap={diagnostics[1]:.3f}"
    )
    print(
        "  CVXPY oracle objective="
        f"{cvx.objective:.6g}; mirror best feasible objective={md.objective:.6g}; "
        f"relative_gap={gap:.3g}; source_load={source_load_moved:.6g}/{problem.source_load_target_s:.6g}; "
        f"max_net_util={float(np.max(net_util)):.3f}; "
        f"max_prefill_util={float(np.max(prefill_util)):.3f}; "
        f"alpha={md.alpha:.3g}; "
        f"replay_load MD/CVXPY={md_mix['replay_load_frac']:.3f}/{cvx_mix['replay_load_frac']:.3f}"
    )


def run_sweep(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    rows = []

    for model in catalog_models():
        for regime in REGIMES:
            problem = make_problem(model, regime, **workload_config.problem_kwargs())
            coeffs = compute_coefficients(problem)
            diagnostics = saturation_diagnostics(problem)
            cvx = solve_cvxpy(problem)
            md = _selected_mirror_descent(problem)
            _log_run(model, regime, problem, coeffs, diagnostics, cvx, md)
            replay = solve_replay_only(problem)
            state = solve_state_only(problem)
            results = {
                "CVXPY": cvx,
                "mirror-descent-best": md,
                "replay-only": replay,
                "state-only": state,
            }
            for policy, result in results.items():
                rows.append(
                    _policy_row(
                        model.name,
                        regime,
                        policy,
                        result,
                        problem,
                        coeffs,
                        diagnostics,
                        cvx.objective,
                    )
                )

    summary = out / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    run_transition_coupled(out, workload_config)


def run_transition_coupled(out, workload_config: WorkloadConfig = WorkloadConfig()):
    model = get_model("GLM-5")
    problem = make_problem(model, TRANSITION_REGIME, **workload_config.problem_kwargs())
    coeffs = compute_coefficients(problem)
    diagnostics = saturation_diagnostics(problem)
    cvx = solve_cvxpy(problem)
    soft = solve_soft_deadline_cvxpy(problem)
    md = _selected_mirror_descent(problem)
    crossover = solve_crossover_greedy(problem)
    results = {
        "CVXPY": cvx,
        "deadline-penalty": soft,
        "mirror-descent-best": md,
        "crossover-greedy": crossover,
        "mixed-greedy": solve_mixed_greedy(problem),
        "replay-only": solve_replay_only(problem),
        "state-only": solve_state_only(problem),
    }
    _require_transition_quality(problem, coeffs, soft, crossover)
    rows = [
        _policy_row(
            model.name,
            TRANSITION_REGIME,
            policy,
            result,
            problem,
            coeffs,
            diagnostics,
            cvx.objective,
        )
        for policy, result in results.items()
    ]
    _write_rows(out / "transition_coupled_policy_table.csv", rows)
    summary = _allocation_summary_rows(problem, results)
    _write_rows(out / "transition_coupled_allocation_summary.csv", summary)
    queue_rows = _transition_queue_rows(problem, results)
    _write_rows(out / "transition_coupled_queue_table.csv", queue_rows)
    _print_transition_outputs(rows, summary, queue_rows)


def _require_transition_quality(problem, coeffs, cvx, crossover):
    assert_workload_quality(
        problem,
        cvx.y,
        crossover.allocation,
        cvx.objective,
        crossover.objective,
        crossover.feasible,
    )


def _allocation_summary_rows(problem, results):
    rows = []
    labels = DEST_LABELS[problem.regime]
    for policy, result in results.items():
        y = result.allocation if hasattr(result, "allocation") else result.y
        action = source_load_action_mix(problem, y)
        dest = source_load_destination_mix(problem, y)
        rows.append(
            {
                "policy": policy,
                "replay_load_fraction": f"{action['replay_load_frac']:.6g}",
                "state_transfer_load_fraction": f"{action['state_load_frac']:.6g}",
                **{f"{label}_load_fraction": f"{share:.6g}" for label, share in zip(labels, dest)},
            }
        )
    return rows


def _write_rows(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _transition_queue_rows(problem, results):
    policies = (
        ("CVXPY", "CVXPY-rounded"),
        ("deadline-penalty", "deadline-penalty-rounded"),
        ("mirror-descent-best", "mirror-descent-rounded"),
        ("crossover-greedy", "crossover-greedy"),
        ("mixed-greedy", "mixed-greedy"),
        ("replay-only", "replay-only"),
        ("state-only", "state-only"),
    )
    rows = []
    for source, policy in policies:
        if source not in results:
            continue
        result = results[source]
        y = result.allocation if hasattr(result, "allocation") else result.y
        row = _empty_transition_queue_row(policy)
        if not getattr(result, "feasible", True) or source_load_moved_s(problem, y) < problem.source_load_target_s - 1e-5:
            rows.append(row)
            continue
        try:
            metrics = queue_metrics(problem, y)
        except ValueError:
            row["status"] = "ROUNDING_FAILED"
            rows.append(row)
            continue
        row.update(
            {
                "status": "OK",
                "source_load_moved_s": f"{metrics['source_load_moved_s']:.10g}",
                "source_load_target_s": f"{metrics['source_load_target_s']:.10g}",
                "source_load_ratio": f"{metrics['source_load_ratio']:.10g}",
                "mean_reconstruction_delay": f"{metrics['mean_reconstruction_delay']:.10g}",
                "p50_reconstruction_delay": f"{metrics['p50_reconstruction_delay']:.10g}",
                "p95_reconstruction_delay": f"{metrics['p95_reconstruction_delay']:.10g}",
                "deadline_miss_rate": f"{metrics['deadline_miss_rate']:.10g}",
                "max_network_busy_window": f"{metrics['max_network_busy_window']:.10g}",
                "max_prefill_busy_window": f"{metrics['max_prefill_busy_window']:.10g}",
                "replay_load_fraction": f"{metrics['replay_load_frac']:.10g}",
                "state_transfer_load_fraction": f"{metrics['state_load_frac']:.10g}",
            }
        )
        rows.append(row)
    return rows


def _empty_transition_queue_row(policy):
    return {
        "policy": policy,
        "status": "INFEASIBLE",
        "source_load_moved_s": "INFEASIBLE",
        "source_load_target_s": "INFEASIBLE",
        "source_load_ratio": "INFEASIBLE",
        "mean_reconstruction_delay": "INFEASIBLE",
        "p50_reconstruction_delay": "INFEASIBLE",
        "p95_reconstruction_delay": "INFEASIBLE",
        "deadline_miss_rate": "INFEASIBLE",
        "max_network_busy_window": "INFEASIBLE",
        "max_prefill_busy_window": "INFEASIBLE",
        "replay_load_fraction": "INFEASIBLE",
        "state_transfer_load_fraction": "INFEASIBLE",
    }


def _print_transition_outputs(rows, summary, queue_rows):
    policy_cols = [
        "policy",
        "objective",
        "relative_gap_to_cvx",
        "replay_load_frac",
        "state_load_frac",
        "active_classes_moved",
        "active_destinations_used",
        "destination_entropy",
        "action_entropy",
        "max_net_util",
        "max_prefill_util",
        "feasible",
    ]
    print("\ntransition-coupled policy table")
    _print_table(rows, policy_cols)
    print("\ntransition-coupled allocation summary")
    _print_table(summary, list(summary[0].keys()))
    print("\ntransition-coupled queue table")
    _print_table(queue_rows, list(queue_rows[0].keys()))
    _print_queue_latex(queue_rows)
    _print_queue_finding(queue_rows)


def _print_table(rows, cols):
    widths = {col: max(len(col), *(len(str(row[col])) for row in rows)) for col in cols}
    print(" | ".join(col.ljust(widths[col]) for col in cols))
    print("-+-".join("-" * widths[col] for col in cols))
    for row in rows:
        print(" | ".join(str(row[col]).ljust(widths[col]) for col in cols))


def _print_queue_latex(rows):
    print("\ntransition-coupled queue table (LaTeX)")
    print("\\begin{tabular}{lrrrrrrrrr}")
    print(
        "policy & source load/target & mean & p50 & p95 & miss & net/H & prefill/H & replay & state \\\\"
    )
    print("\\hline")
    for row in rows:
        if row.get("status", "OK") != "OK":
            print(f"{row['policy']} & \\multicolumn{{9}}{{c}}{{{row['status']}}} \\\\")
            continue
        print(
            f"{row['policy']} & {float(row['source_load_ratio']):.3f} & "
            f"{float(row['mean_reconstruction_delay']):.3f} & "
            f"{float(row['p50_reconstruction_delay']):.3f} & "
            f"{float(row['p95_reconstruction_delay']):.3f} & "
            f"{float(row['deadline_miss_rate']):.3f} & "
            f"{float(row['max_network_busy_window']):.3f} & "
            f"{float(row['max_prefill_busy_window']):.3f} & "
            f"{float(row['replay_load_fraction']):.3f} & "
            f"{float(row['state_transfer_load_fraction']):.3f} \\\\"
        )
    print("\\end{tabular}")


def _print_queue_finding(rows):
    by_policy = {row["policy"]: row for row in rows}
    cvx = by_policy["CVXPY-rounded"]
    comparators = ("crossover-greedy", "replay-only", "state-only")
    better_p95 = all(
        _queue_metric(cvx, "p95_reconstruction_delay")
        < _queue_metric(by_policy[policy], "p95_reconstruction_delay")
        for policy in comparators
    )
    better_miss = all(
        _queue_metric(cvx, "deadline_miss_rate")
        < _queue_metric(by_policy[policy], "deadline_miss_rate")
        for policy in comparators
    )
    if better_p95 or better_miss:
        print("\nQueue finding: CVXPY-rounded improves p95 delay or deadline misses.")
    else:
        print(
            "\nQueue finding: CVXPY-rounded does not improve p95 delay or "
            "deadline misses over crossover-greedy and both single-action policies."
        )


def _queue_metric(row, key):
    return float(row[key]) if row.get("status", "OK") == "OK" else np.inf


if __name__ == "__main__":
    run_sweep(parse_workload_config("Run catalog allocation sweep."))
