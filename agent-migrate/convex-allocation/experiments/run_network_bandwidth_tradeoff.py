from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from catalog import get_model
from cvxpy_solver import solve_soft_deadline_cvxpy
from evaluation import WorkloadConfig, parse_workload_config
from experiments.plot_queue_centered import _max_waiting_depth_points
from problem import ProblemData, make_problem
from queueing import evaluate_rounded_queue_trace, round_allocation

NETWORK_SCALES = (0.60, 0.80, 1.00, 1.25, 1.50, 2.00)
SOURCE_WORKING_SET_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
DRAIN_WINDOW_S = 1800.0
COLUMNS = (
    "network_bandwidth_scale",
    "drain_window_s",
    "max_safe_source_working_set_fraction",
    "evacuated_state_tb",
    "evacuated_nvl72_hbm_fraction",
    "retained_prefill_removal_rate_s_per_s",
    "request_migration_fraction",
    "p95_delay_over_deadline",
    "deadline_miss_rate",
    "max_network_queue_depth",
    "max_prefill_queue_depth",
    "network_capacity_pressure",
    "prefill_capacity_pressure",
    "drain_completion_s",
    "deadline_overrun_max",
    "replay_retained_prefill_fraction",
    "state_transfer_retained_prefill_fraction",
)


def run_network_bandwidth_tradeoff(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_scale_row(scale, workload_config) for scale in NETWORK_SCALES]
    _write_rows(out / "network_bandwidth_tradeoff.csv", rows)
    _plot(rows, out / "network_bandwidth_tradeoff.pdf")
    return rows


def _scale_row(scale: float, workload_config: WorkloadConfig) -> dict[str, float | str]:
    candidates = []
    for source_working_set_fraction in SOURCE_WORKING_SET_FRACTIONS:
        problem = _scale_network(
            make_problem(
                get_model("GLM-5"),
                "transition-coupled",
                source_working_set_fraction=source_working_set_fraction,
                deadline_scale=1.0,
                **workload_config.problem_kwargs(),
            ),
            scale,
        )
        try:
            result = solve_soft_deadline_cvxpy(problem)
            rounded = round_allocation(problem, result.y)
            metrics, trace = evaluate_rounded_queue_trace(problem, rounded.y, drain_window_s=DRAIN_WINDOW_S)
        except (RuntimeError, ValueError):
            continue
        if _safe(metrics):
            candidates.append((source_working_set_fraction, result, rounded, metrics, trace))
    if not candidates:
        return {"network_bandwidth_scale": scale, "drain_window_s": DRAIN_WINDOW_S, **{key: math.nan for key in COLUMNS[2:]}}
    source_working_set_fraction, result, rounded, metrics, trace = max(candidates, key=lambda item: item[0])
    diagnostics = result.diagnostics or {}
    return {
        "network_bandwidth_scale": scale,
        "drain_window_s": DRAIN_WINDOW_S,
        "max_safe_source_working_set_fraction": source_working_set_fraction,
        "evacuated_state_tb": metrics["evacuated_state_tb"],
        "evacuated_nvl72_hbm_fraction": metrics["evacuated_nvl72_hbm_fraction"],
        "retained_prefill_removal_rate_s_per_s": metrics["retained_prefill_removal_rate_s_per_s"],
        "request_migration_fraction": float(np.sum(rounded.y[:, :-1]) / np.sum(problem.d)),
        "p95_delay_over_deadline": metrics["p95_reconstruction_delay_ratio"],
        "deadline_miss_rate": metrics["deadline_miss_rate"],
        "max_network_queue_depth": _queue_depth(trace, "network"),
        "max_prefill_queue_depth": _queue_depth(trace, "prefill"),
        "network_capacity_pressure": metrics["network_capacity_pressure"],
        "prefill_capacity_pressure": metrics["prefill_capacity_pressure"],
        "drain_completion_s": metrics["drain_completion_s"],
        "deadline_overrun_max": diagnostics.get("deadline_overrun_max", math.nan),
        "replay_retained_prefill_fraction": metrics["replay_retained_prefill_fraction"],
        "state_transfer_retained_prefill_fraction": metrics["state_transfer_retained_prefill_fraction"],
    }


def _scale_network(problem: ProblemData, scale: float) -> ProblemData:
    return ProblemData(
        model=problem.model,
        regime=problem.regime,
        T=problem.T,
        d=problem.d,
        deadline_s=problem.deadline_s,
        lambda_Bps=problem.lambda_Bps * scale,
        rho_prefill=problem.rho_prefill,
        C_net=problem.C_net * scale,
        C_prefill=problem.C_prefill,
        ell_net=problem.ell_net * scale,
        ell_prefill=problem.ell_prefill,
        h_ctx=problem.h_ctx,
        h_kv=problem.h_kv,
        retained_prefill_target_s=problem.retained_prefill_target_s,
        w=problem.w,
    )


def _safe(metrics: dict[str, float]) -> bool:
    return (
        metrics["retained_prefill_moved_s"] >= metrics["retained_prefill_target_s"] - 1e-9
        and metrics["deadline_miss_rate"] <= 0.01
        and metrics["p95_reconstruction_delay_ratio"] <= 1.0
    )


def _queue_depth(trace, resource: str) -> float:
    return max(depth for _, depth in _max_waiting_depth_points(trace, resource))


def _write_rows(path: Path, rows: list[dict[str, float | str]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows, path: Path) -> None:
    x = np.array([row["network_bandwidth_scale"] for row in rows], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.8), sharex=True, constrained_layout=True)
    _line(axes[0], x, rows, "max_safe_source_working_set_fraction", "Source working set")
    _line(axes[0], x, rows, "request_migration_fraction", "Requests moved")
    axes[0].set_ylabel("Fraction of workload")
    axes[0].set_ylim(bottom=0.0)
    _line(axes[1], x, rows, "max_network_queue_depth", "Network queue")
    _line(axes[1], x, rows, "max_prefill_queue_depth", "GPU prefill queue")
    axes[1].set_xlabel("Network bandwidth (x baseline)")
    axes[1].set_ylabel("Max queued requests")
    for ax in axes:
        ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _line(ax, x, rows, key, label):
    y = np.array([row[key] for row in rows], dtype=float)
    finite = np.isfinite(y)
    if np.any(finite):
        ax.plot(x[finite], y[finite], marker="o", linewidth=1.5, label=label)


if __name__ == "__main__":
    run_network_bandwidth_tradeoff(parse_workload_config("Plot network bandwidth tradeoff."))
