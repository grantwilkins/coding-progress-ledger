from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from power_profile_reduce import binned_curve

CONFIG = "gpt-oss-20b-h100_tp1"
ROOT = Path(__file__).resolve().parents[3]
PTRACE = ROOT / "powertrace-sim"
RAW_DIR = PTRACE / "data" / "sharegpt-benchmark-gpt-oss-20b-a100"
OUT_STEM = Path(__file__).resolve().parent / "outputs" / "stage1_gpt_oss_20b_h100_tp1_window_sensitivity"
WINDOWS = (1.0, 2.0, 5.0, 10.0, 30.0)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_windows(windows) -> tuple[float, ...]:
    out = tuple(float(w) for w in windows)
    if any(w <= 0 for w in out) or len(set(out)) != len(out):
        raise ValueError("windows must be unique positive seconds")
    return tuple(sorted(out))


def concat_runs(runs: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not runs:
        raise ValueError("no matched raw runs")
    fields = {"P": "power_w", "f": "f_tps", "g": "g_tps", "itl": "itl_ms", "ttft_pt": "ttft_per_tok_ms"}
    return {k: np.concatenate([r[v] for r in runs]) for k, v in fields.items()}


def summary_row(window: float, summary: dict, arrays: dict[str, np.ndarray]) -> dict[str, float | int]:
    ell = np.asarray(arrays["ell"], float)
    return {
        "window_s": window,
        "n_windows": int(summary["n_windows"]),
        "F_prefill_tps": float(summary["F_prefill_tps"]),
        "G_decode_tps": float(summary["G_decode_tps"]),
        "P0_w": float(summary["P0_w"]),
        "P_max_w": float(summary["P_max_w"]),
        "r2_linear": float(summary["r2_linear"]),
        "r2_saturating": float(summary["r2_saturating"]),
        "ell_power_knee": float(summary["ell_power_knee"]),
        "rho_star": float(summary["rho_star"]),
        "ell_p95": float(np.nanquantile(ell, 0.95)),
        "ell_max": float(np.nanmax(ell)),
    }


def curve_rows(window: float, arrays: dict[str, np.ndarray], bins: int) -> list[dict[str, float | int]]:
    return [{"window_s": window, **r} for r in binned_curve(arrays["ell"], arrays["P"], bins)]


def fit_rows(window: float, arrays: dict, points: int) -> list[dict[str, float]]:
    ell = np.linspace(0.0, float(np.nanmax(arrays["ell"])), points)
    power = arrays["P_of_ell"](ell)
    return [{"window_s": window, "ell": float(x), "power_w": float(y)} for x, y in zip(ell, power)]


def analyze_windows(raw_dir: Path, powertrace_root: Path, windows: tuple[float, ...], tp: int, bins: int, points: int):
    two = load_module(powertrace_root / "scripts" / "eval" / "two_price_fit.py", "powertrace_two_price_fit")
    sat = load_module(powertrace_root / "scripts" / "eval" / "saturating_fit.py", "powertrace_saturating_fit")
    summaries, curves, fits = [], [], []
    for window in validate_windows(windows):
        runs = two.collect_runs(str(raw_dir), window).get(tp, [])
        summary, arrays = sat.analyze(CONFIG, concat_runs(runs))
        summaries.append(summary_row(window, summary, arrays))
        curves.extend(curve_rows(window, arrays, bins))
        fits.extend(fit_rows(window, arrays, points))
    return summaries, curves, fits


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def make_plot(summaries: list[dict], curves: list[dict], fits: list[dict], out_stem: Path) -> None:
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(summaries)))
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.0), sharey=True)
    by_w = {r["window_s"]: r for r in summaries}
    for ax, (window, row), color in zip(axes.flat, by_w.items(), colors):
        c = [r for r in curves if r["window_s"] == window]
        f = [r for r in fits if r["window_s"] == window]
        ax.scatter([r["ell_mean"] for r in c], [r["power_mean_w"] for r in c], s=18, color=color)
        ax.plot([r["ell"] for r in f], [r["power_w"] for r in f], color="black", lw=1.2)
        ax.axvline(row["rho_star"], color="red", ls="--", lw=1.0)
        ax.axvline(row["ell_power_knee"], color="purple", ls=":", lw=1.0)
        ax.set_title(
            f"W={window:g}s  R2 {row['r2_linear']:.2f}->{row['r2_saturating']:.2f}\n"
            f"rho*={row['rho_star']:.2f}, p95 ell={row['ell_p95']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("offered load ell = f/F + g/G")
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("average node power [W]")
    fig.suptitle("gpt-oss-20b A100 TP=1: power-load curve sensitivity to window size", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=160)
    fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Queue-Haul power-window sensitivity")
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--powertrace-root", type=Path, default=PTRACE)
    p.add_argument("--windows", nargs="+", type=float, default=list(WINDOWS))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--bins", type=int, default=24)
    p.add_argument("--fit-points", type=int, default=200)
    p.add_argument("--out-stem", type=Path, default=OUT_STEM)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summaries, curves, fits = analyze_windows(
        args.raw_dir, args.powertrace_root, tuple(args.windows), args.tp, args.bins, args.fit_points
    )
    write_csv(args.out_stem.with_name(args.out_stem.name + "_summary.csv"), summaries)
    write_csv(args.out_stem.with_name(args.out_stem.name + "_binned.csv"), curves)
    make_plot(summaries, curves, fits, args.out_stem)
    print(args.out_stem)


if __name__ == "__main__":
    main()
