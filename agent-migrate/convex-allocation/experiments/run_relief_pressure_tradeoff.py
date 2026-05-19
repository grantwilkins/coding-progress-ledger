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
RELIEF_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
COLUMNS = (
    "network_bandwidth_scale",
    "max_safe_relief_fraction",
    "moved_request_frac",
    "p95_delay_ratio",
    "deadline_miss_rate",
    "network_queue_depth",
    "prefill_queue_depth",
    "max_network_busy_window",
    "max_prefill_busy_window",
    "deadline_debt_max",
    "replay_relief_frac",
    "state_relief_frac",
)


def run_relief_pressure_tradeoff(workload_config: WorkloadConfig = WorkloadConfig()):
    out = workload_config.output_dir(ROOT)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_scale_row(scale, workload_config) for scale in NETWORK_SCALES]
    _write_rows(out / "relief_pressure_tradeoff.csv", rows)
    _plot(rows, out / "relief_pressure_tradeoff.pdf")
    return rows


def _scale_row(scale: float, workload_config: WorkloadConfig) -> dict[str, float | str]:
    candidates = []
    for relief_fraction in RELIEF_FRACTIONS:
        problem = _scale_network(
            make_problem(
                get_model("GLM-5"),
                "transition-coupled",
                relief_fraction=relief_fraction,
                deadline_scale=1.0,
                **workload_config.problem_kwargs(),
            ),
            scale,
        )
        try:
            result = solve_soft_deadline_cvxpy(problem)
            rounded = round_allocation(problem, result.y)
            metrics, trace = evaluate_rounded_queue_trace(problem, rounded.y)
        except (RuntimeError, ValueError):
            continue
        if _safe(metrics):
            candidates.append((relief_fraction, result, rounded, metrics, trace))
    if not candidates:
        return {"network_bandwidth_scale": scale, **{key: math.nan for key in COLUMNS[1:]}}
    relief_fraction, result, rounded, metrics, trace = max(candidates, key=lambda item: item[0])
    diagnostics = result.diagnostics or {}
    return {
        "network_bandwidth_scale": scale,
        "max_safe_relief_fraction": relief_fraction,
        "moved_request_frac": float(np.sum(rounded.y[:, :-1]) / np.sum(problem.d)),
        "p95_delay_ratio": metrics["p95_reconstruction_delay_ratio"],
        "deadline_miss_rate": metrics["deadline_miss_rate"],
        "network_queue_depth": _queue_depth(trace, "network"),
        "prefill_queue_depth": _queue_depth(trace, "prefill"),
        "max_network_busy_window": metrics["max_network_busy_window"],
        "max_prefill_busy_window": metrics["max_prefill_busy_window"],
        "deadline_debt_max": diagnostics.get("deadline_debt_max", math.nan),
        "replay_relief_frac": metrics["replay_relief_frac"],
        "state_relief_frac": metrics["state_relief_frac"],
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
        relief_target_s=problem.relief_target_s,
        w=problem.w,
    )


def _safe(metrics: dict[str, float]) -> bool:
    return (
        metrics["rounded_relief_achieved_s"] >= metrics["relief_target_s"] - 1e-9
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
    fig, axes = plt.subplots(2, 1, figsize=(5.7, 5.2), sharex=True, constrained_layout=True)
    _line(axes[0], x, rows, "max_safe_relief_fraction", "source relief")
    _line(axes[0], x, rows, "moved_request_frac", "evacuated requests")
    axes[0].set_ylabel("fraction")
    axes[0].set_ylim(bottom=0.0)
    _line(axes[1], x, rows, "network_queue_depth", "network")
    _line(axes[1], x, rows, "prefill_queue_depth", "prefill")
    axes[1].set_xlabel("network bandwidth scale")
    axes[1].set_ylabel("max waiting queue depth")
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
    run_relief_pressure_tradeoff(parse_workload_config("Plot relief pressure tradeoff."))
