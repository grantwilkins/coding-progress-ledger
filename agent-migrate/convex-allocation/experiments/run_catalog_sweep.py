from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context="talk", style="whitegrid")

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
from coefficients import ACTIONS, compute_coefficients
from cvxpy_solver import solve_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from metrics import (
    allocation_diagnostics,
    action_mix,
    shed_achieved,
    shed_action_mix,
    shed_destination_mix,
    utilization,
)
from mirror_descent import solve_mirror_descent
from problem import make_problem, saturation_diagnostics
from queueing import queue_metrics
from workload import assert_workload_quality

REGIMES = ("bandwidth-spread", "prefill-spread", "background-load-spread")
TRANSITION_REGIME = "transition-coupled"
MODEL_LABELS = {
    "DeepSeek-V4-Pro": "DeepSeek",
    "GLM-5": "GLM-5",
    "Qwen3-Next-80B-A3B": "Qwen3-Next",
}
MODEL_COLORS = {
    "DeepSeek-V4-Pro": "#4c78a8",
    "GLM-5": "#f58518",
    "Qwen3-Next-80B-A3B": "#54a24b",
}
REGIME_LABELS = {
    "bandwidth-spread": "Link bandwidth varies",
    "prefill-spread": "Prefill load varies",
    "background-load-spread": "Background load + cache",
}
DEST_LABELS = {
    "bandwidth-spread": ("1 Gbps", "10 Gbps", "100 Gbps"),
    "prefill-spread": ("20% prefill load", "50% prefill load", "80% prefill load"),
    "background-load-spread": ("low network load", "balanced load", "cached context"),
    "transition-coupled": ("4 Gbps", "6 Gbps", "9 Gbps"),
}
ACTION_COLORS = {
    "replay_frac": "#4c78a8",
    "state_frac": "#f58518",
    "stay_frac": "#c9c2bd",
}
REGIME_XLABELS = ("bandwidth", "prefill load", "background load")


def _relative_gap(values, cvx_obj):
    return np.maximum((values - cvx_obj) / max(1.0, abs(cvx_obj)), 0.0)


def _best_objective_gap(hist, cvx_obj):
    return _relative_gap(np.asarray(hist["best_feasible_objective"], dtype=float), cvx_obj)


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
    shed_mix = shed_action_mix(problem, y)
    alloc = allocation_diagnostics(problem, coeffs, y)
    shed = shed_achieved(problem, y)
    shed_violation = max(0.0, problem.B_shed - shed)
    excess_shed = max(0.0, shed - problem.B_shed)
    capacity_feasible = bool(np.max(net_util) < 1.0 and np.max(prefill_util) < 1.0)
    objective = getattr(result, "objective", None)
    feasible = bool(
        getattr(result, "feasible", True)
        and objective is not None
        and np.isfinite(objective)
        and shed_violation <= 1e-5
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
        "shed_achieved": f"{shed:.10g}",
        "shed_target": f"{problem.B_shed:.10g}",
        "shed_violation": f"{shed_violation:.10g}",
        "excess_shed": f"{excess_shed:.10g}",
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
        **{key: f"{value:.10g}" for key, value in shed_mix.items()},
        "replay_demand_over_capacity": f"{diagnostics[0]:.10g}",
        "state_bytes_over_capacity": f"{diagnostics[1]:.10g}",
        "feasible": str(bool(feasible)),
    }


def _log_run(model, regime, problem, coeffs, diagnostics, cvx, md):
    net_util, prefill_util = utilization(problem, coeffs, md.y)
    gap = max(0.0, (md.objective - cvx.objective) / max(1.0, abs(cvx.objective)))
    shed = shed_achieved(problem, md.y)
    md_mix = shed_action_mix(problem, md.y)
    cvx_mix = shed_action_mix(problem, cvx.y)
    print(
        f"{model.name} / {regime}: diagnostics replay_demand/cap={diagnostics[0]:.3f}, "
        f"state_bytes/cap={diagnostics[1]:.3f}"
    )
    print(
        "  CVXPY oracle objective="
        f"{cvx.objective:.6g}; mirror best feasible objective={md.objective:.6g}; "
        f"relative_gap={gap:.3g}; shed={shed:.6g}/{problem.B_shed:.6g}; "
        f"max_net_util={float(np.max(net_util)):.3f}; "
        f"max_prefill_util={float(np.max(prefill_util)):.3f}; "
        f"alpha={md.alpha:.3g}; "
        f"replay_shed MD/CVXPY={md_mix['replay_shed_frac']:.3f}/{cvx_mix['replay_shed_frac']:.3f}"
    )


def run_sweep(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = {}

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
            cells[(model.name, regime)] = (problem, coeffs, results)

    summary = out / "summary.csv"
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    plot_headline(cells, out)
    plot_heatmaps(cells, out)
    plot_utilization(cells, out)
    plot_policy_objectives(cells, out)
    plot_convergence(cells, out)
    plot_crossover(out)
    run_transition_coupled(out, workload_config)


def run_transition_coupled(out, workload_config: WorkloadConfig = WorkloadConfig()):
    model = get_model("GLM-5")
    problem = make_problem(model, TRANSITION_REGIME, **workload_config.problem_kwargs())
    coeffs = compute_coefficients(problem)
    diagnostics = saturation_diagnostics(problem)
    cvx = solve_cvxpy(problem)
    md = _selected_mirror_descent(problem)
    crossover = solve_crossover_greedy(problem)
    results = {
        "CVXPY": cvx,
        "mirror-descent-best": md,
        "crossover-greedy": crossover,
        "mixed-greedy": solve_mixed_greedy(problem),
        "replay-only": solve_replay_only(problem),
        "state-only": solve_state_only(problem),
    }
    _require_transition_quality(problem, coeffs, cvx, crossover)
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
        action = shed_action_mix(problem, y)
        dest = shed_destination_mix(problem, y)
        rows.append(
            {
                "policy": policy,
                "replay_shed_frac": f"{action['replay_shed_frac']:.6g}",
                "state_shed_frac": f"{action['state_shed_frac']:.6g}",
                **{f"{label}_shed_frac": f"{share:.6g}" for label, share in zip(labels, dest)},
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
        ("mirror-descent-best", "mirror-descent-rounded"),
        ("crossover-greedy", "crossover-greedy"),
        ("mixed-greedy", "mixed-greedy"),
        ("replay-only", "replay-only"),
        ("state-only", "state-only"),
    )
    rows = []
    for source, policy in policies:
        result = results[source]
        y = result.allocation if hasattr(result, "allocation") else result.y
        row = _empty_transition_queue_row(policy)
        if not getattr(result, "feasible", True) or shed_achieved(problem, y) < problem.B_shed - 1e-5:
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
                "rounded_shed_achieved": f"{metrics['rounded_shed_achieved']:.10g}",
                "rounded_shed_target": f"{metrics['rounded_shed_target']:.10g}",
                "rounded_shed_ratio": f"{metrics['rounded_shed_ratio']:.10g}",
                "mean_reconstruction_delay": f"{metrics['mean_reconstruction_delay']:.10g}",
                "p50_reconstruction_delay": f"{metrics['p50_reconstruction_delay']:.10g}",
                "p95_reconstruction_delay": f"{metrics['p95_reconstruction_delay']:.10g}",
                "deadline_miss_rate": f"{metrics['deadline_miss_rate']:.10g}",
                "max_network_busy_window": f"{metrics['max_network_busy_window']:.10g}",
                "max_prefill_busy_window": f"{metrics['max_prefill_busy_window']:.10g}",
                "replay_shed_frac": f"{metrics['replay_shed_frac']:.10g}",
                "state_shed_frac": f"{metrics['state_shed_frac']:.10g}",
            }
        )
        rows.append(row)
    return rows


def _empty_transition_queue_row(policy):
    return {
        "policy": policy,
        "status": "INFEASIBLE",
        "rounded_shed_achieved": "INFEASIBLE",
        "rounded_shed_target": "INFEASIBLE",
        "rounded_shed_ratio": "INFEASIBLE",
        "mean_reconstruction_delay": "INFEASIBLE",
        "p50_reconstruction_delay": "INFEASIBLE",
        "p95_reconstruction_delay": "INFEASIBLE",
        "deadline_miss_rate": "INFEASIBLE",
        "max_network_busy_window": "INFEASIBLE",
        "max_prefill_busy_window": "INFEASIBLE",
        "replay_shed_frac": "INFEASIBLE",
        "state_shed_frac": "INFEASIBLE",
    }


def _print_transition_outputs(rows, summary, queue_rows):
    policy_cols = [
        "policy",
        "objective",
        "relative_gap_to_cvx",
        "replay_shed_frac",
        "state_shed_frac",
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
        "policy & shed/target & mean & p50 & p95 & miss & net/H & prefill/H & replay & state \\\\"
    )
    print("\\hline")
    for row in rows:
        if row.get("status", "OK") != "OK":
            print(f"{row['policy']} & \\multicolumn{{9}}{{c}}{{{row['status']}}} \\\\")
            continue
        print(
            f"{row['policy']} & {float(row['rounded_shed_ratio']):.3f} & "
            f"{float(row['mean_reconstruction_delay']):.3f} & "
            f"{float(row['p50_reconstruction_delay']):.3f} & "
            f"{float(row['p95_reconstruction_delay']):.3f} & "
            f"{float(row['deadline_miss_rate']):.3f} & "
            f"{float(row['max_network_busy_window']):.3f} & "
            f"{float(row['max_prefill_busy_window']):.3f} & "
            f"{float(row['replay_shed_frac']):.3f} & "
            f"{float(row['state_shed_frac']):.3f} \\\\"
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


def plot_headline(cells, out):
    models = list(catalog_models())
    regimes = list(REGIMES)

    data = np.zeros((len(models), len(regimes)))

    for i, model in enumerate(models):
        for j, regime in enumerate(regimes):
            problem, _, results = cells[(model.name, regime)]
            mix = shed_action_mix(problem, results["CVXPY"].y)
            data[i, j] = mix["replay_shed_frac"]

    row_labels = [
        f"{MODEL_LABELS[m.name]}\n{m.published_crossover_gbps:.1f} Gbps" for m in models
    ]
    col_labels = [
        "Bandwidth\nvaries",
        "Prefill load\nvaries",
        "Background load\n+ cache",
    ]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)

    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlBu")

    ax.set_xticks(np.arange(len(regimes)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(row_labels)

    ax.set_title("Optimal replay share of shed work", pad=12)

    for i in range(len(models)):
        for j in range(len(regimes)):
            value = data[i, j]
            text_color = "white" if value < 0.25 or value > 0.75 else "black"
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=13,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Replay fraction of shed work")
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["0%\nstate", "50%\nmix", "100%\nreplay"])

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig(out / "headline_action_mix.png", dpi=300)
    plt.close(fig)


def plot_heatmaps(cells, out):
    chosen = max(
        cells,
        key=lambda key: min(
            shed_action_mix(cells[key][0], cells[key][2]["CVXPY"].y)[
                "replay_shed_frac"
            ],
            shed_action_mix(cells[key][0], cells[key][2]["CVXPY"].y)["state_shed_frac"],
        ),
    )
    model, regime = chosen
    problem, _, results = cells[chosen]
    data = results["CVXPY"].y / problem.d[:, None]
    cols = [
        f"{dest}\n{action}" for dest in DEST_LABELS[regime] for action in ACTIONS
    ] + ["Stay"]
    rows = [
        f"{int(t):,} tokens\n{int(d)} requests" for t, d in zip(problem.T, problem.d)
    ]
    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)
    sns.heatmap(
        data,
        vmin=0,
        vmax=1,
        cmap="magma",
        ax=ax,
        xticklabels=cols,
        yticklabels=rows,
        cbar_kws={"label": "fraction of class"},
    )
    ax.set_title(
        f"Class allocation for the most mixed cell: {MODEL_LABELS[model]}, {REGIME_LABELS[regime]}"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
    ax.set_ylabel("workload class")
    ax.tick_params(length=0)
    fig.savefig(out / "allocation_heatmap_per_scenario.png", dpi=200)
    plt.close(fig)


def plot_utilization(cells, out):
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = np.arange(len(REGIMES))
    for model in catalog_models():
        net = []
        prefill = []
        for regime in REGIMES:
            problem, coeffs, results = cells[(model.name, regime)]
            u_net, u_prefill = utilization(problem, coeffs, results["CVXPY"].y)
            net.append(float(np.max(u_net)))
            prefill.append(float(np.max(u_prefill)))
        color = MODEL_COLORS[model.name]
        ax.plot(
            x,
            net,
            marker="o",
            color=color,
            linewidth=2.4,
            label=f"{MODEL_LABELS[model.name]} network",
        )
        ax.plot(
            x,
            prefill,
            marker="s",
            color=color,
            linewidth=2.4,
            linestyle="--",
            label=f"{MODEL_LABELS[model.name]} prefill",
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x, REGIME_XLABELS)
    ax.set_ylim(0.45, 1.03)
    ax.set_ylabel("max destination utilization")
    ax.set_title("Resource ceilings used by the convex optimum")
    ax.legend(
        frameon=False,
        fontsize=10,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.savefig(out / "utilization_vs_policy.png", dpi=200)
    plt.close(fig)


def plot_policy_objectives(cells, out):
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    x = np.arange(len(REGIMES))
    for model in catalog_models():
        replay_ratio = []
        state_ratio = []
        for regime in REGIMES:
            _, _, results = cells[(model.name, regime)]
            cvx_obj = results["CVXPY"].objective
            replay_ratio.append(
                results["replay-only"].objective / cvx_obj
                if results["replay-only"].feasible
                else np.nan
            )
            state_ratio.append(
                results["state-only"].objective / cvx_obj
                if results["state-only"].feasible
                else np.nan
            )
        color = MODEL_COLORS[model.name]
        ax.plot(
            x,
            replay_ratio,
            marker="o",
            color=color,
            linewidth=2.4,
            label=f"{MODEL_LABELS[model.name]} replay greedy",
        )
        ax.plot(
            x,
            state_ratio,
            marker="s",
            color=color,
            linewidth=2.4,
            linestyle="--",
            label=f"{MODEL_LABELS[model.name]} state greedy",
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x, REGIME_XLABELS)
    ax.set_ylabel("objective / CVXPY optimum")
    ax.set_title("Greedy objective relative to CVXPY")
    ax.set_ylim(0.9, 5.4)
    ax.legend(
        frameon=False,
        fontsize=10,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.savefig(out / "objective_vs_policy.png", dpi=200)
    plt.close(fig)


def plot_convergence(cells, out):
    key = ("GLM-5", "bandwidth-spread")
    _, _, results = cells[key]
    hist = results["mirror-descent-best"].history
    cvx_obj = results["CVXPY"].objective
    t = np.arange(1, hist["best_feasible_objective"].size + 1)
    best_gap = _best_objective_gap(hist, cvx_obj)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    ax.plot(
        t,
        best_gap,
        color=MODEL_COLORS[key[0]],
        linewidth=2.4,
    )
    ax.set_xlabel("gradient evaluations")
    ax.set_ylabel("best feasible objective gap to CVXPY")
    ax.set_title("Mirror descent with scalar search on the GLM-5 transition case")
    ax.set_ylim(bottom=0.0)
    fig.savefig(out / "convergence_one_scenario.png", dpi=200)
    plt.close(fig)


def plot_crossover(out):
    fig, ax = plt.subplots(figsize=(8, 3.8), constrained_layout=True)
    y = np.arange(len(catalog_models()))
    for row, model in zip(y, catalog_models()):
        computed = model.crossover_gbps
        published = model.published_crossover_gbps
        ax.hlines(
            row,
            0.1,
            computed,
            color="#4c78a8",
            linewidth=6,
            alpha=0.75,
            label="replay faster" if row == 0 else None,
        )
        ax.hlines(
            row,
            computed,
            220,
            color="#f58518",
            linewidth=6,
            alpha=0.75,
            label="state faster" if row == 0 else None,
        )
        ax.plot(
            computed,
            row,
            marker="o",
            color="black",
            markersize=5,
            label="computed crossover" if row == 0 else None,
        )
        ax.plot(
            published,
            row,
            marker="|",
            color="black",
            markersize=14,
            label="FINDINGS.md crossover" if row == 0 else None,
        )
        ax.text(computed * 1.08, row + 0.13, f"{computed:.1f} Gbps", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 220)
    ax.set_yticks(y, [MODEL_LABELS[m.name] for m in catalog_models()])
    ax.set_xlabel("bandwidth (Gbps)")
    ax.set_title("Single-request replay/state crossover")
    ax.grid(axis="x", color="#eeeeee", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.savefig(out / "crossover_recovery.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    run_sweep(parse_workload_config("Run catalog allocation sweep."))
