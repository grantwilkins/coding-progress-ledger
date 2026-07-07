from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONFIG = "gpt-oss-20b-a100_tp1"
PTRACE = (
    Path(__file__).resolve().parents[3] / "powertrace-sim" / "results" / "two_price_fit"
)
OUT_STEM = Path(__file__).resolve().parent / "outputs" / "stage1_gpt_oss_20b_a100_tp1"
POWER_KNEE_FRAC = 0.8
CONSTANT_FIELDS = (
    "F_prefill_tps",
    "G_decode_tps",
    "P0_w",
    "P_max_w",
    "c1_j_per_prefill_tok",
    "c2_j_per_decode_tok",
    "p_pre_w_per_busy_s",
    "p_dec_w_per_busy_s",
    "ell_power_knee",
    "ell_latency_knee",
    "rho_star",
    "s_plat_w_per_ell",
    "p_amort_w_per_ell",
    "amort_over_plat",
    "r2_linear",
    "r2_saturating",
)


def read_constants(path: Path, config: str) -> dict[str, float | str]:
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["config"] == config:
                return {"config": config, **{k: float(row[k]) for k in CONSTANT_FIELDS}}
    raise ValueError(f"missing constants for {config} in {path}")


def load_windows(
    path: Path, constants: dict[str, float | str]
) -> dict[str, np.ndarray]:
    windows = dict(np.load(path, allow_pickle=False))
    missing = sorted({"P", "f", "g"} - windows.keys())
    if missing:
        raise ValueError(f"missing window fields: {', '.join(missing)}")
    F, G = float(constants["F_prefill_tps"]), float(constants["G_decode_tps"])
    if F <= 0 or G <= 0:
        raise ValueError("F_prefill_tps and G_decode_tps must be positive")
    windows["ell"] = (
        np.asarray(windows["f"], float) / F + np.asarray(windows["g"], float) / G
    )
    return windows


def binned_curve(ell, power, bins: int) -> list[dict[str, float | int]]:
    ell = np.asarray(ell, float)
    power = np.asarray(power, float)
    ok = np.isfinite(ell) & np.isfinite(power)
    if not ok.any():
        raise ValueError("no finite ell/power samples")
    ell, power = ell[ok], power[ok]
    edges = np.linspace(0.0, float(ell.max()), bins + 1)
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (ell >= lo) & (ell <= hi if i == bins - 1 else ell < hi)
        if sel.any():
            p = power[sel]
            rows.append(
                {
                    "ell_lo": lo,
                    "ell_hi": hi,
                    "ell_mean": float(ell[sel].mean()),
                    "power_mean_w": float(p.mean()),
                    "power_p10_w": float(np.quantile(p, 0.10)),
                    "power_p90_w": float(np.quantile(p, 0.90)),
                    "n": int(sel.sum()),
                }
            )
    return rows


def concave_power_curve(
    constants: dict[str, float | str], max_ell: float, points: int = 200
) -> list[dict[str, float]]:
    p0, pmax = float(constants["P0_w"]), float(constants["P_max_w"])
    knee = float(constants["ell_power_knee"])
    if pmax <= p0 or knee <= 0:
        raise ValueError("invalid concave power constants")
    ell = np.linspace(0.0, max_ell, points)
    w = (POWER_KNEE_FRAC / (1 - POWER_KNEE_FRAC)) * ell / knee
    power = p0 + (pmax - p0) * w / (1 + w)
    return [{"ell": float(x), "power_w": float(y)} for x, y in zip(ell, power)]


def write_csv(path: Path, rows: list[dict], fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_plot(windows, curve, power_curve, constants, out_stem: Path) -> None:
    ell, power = windows["ell"], windows["P"]
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    sc = ax.scatter(ell, power, s=7, alpha=0.25, vmin=0, vmax=1)
    ax.plot(
        [r["ell_mean"] for r in curve],
        [r["power_mean_w"] for r in curve],
        color="black",
        marker="o",
        ms=3,
        lw=1.5,
        label="binned average power",
    )
    ax.plot(
        [r["ell"] for r in power_curve],
        [r["power_w"] for r in power_curve],
        color="green",
        ls="--",
        lw=1.2,
        label="concave power curve",
    )
    ax.set_xlabel("ell load = f/F + g/G")
    ax.set_ylabel("average node power [W]")
    ax.set_title("gpt-oss-20b on A100, TP=1")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out_stem}.png", dpi=160)
    fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser(description="Queue-Haul Stage 1a ell-vs-power reducer")
    p.add_argument("--windows", type=Path, default=PTRACE / f"windows_{CONFIG}.npz")
    p.add_argument("--summary", type=Path, default=PTRACE / "saturating_summary.csv")
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--bins", type=int, default=24)
    p.add_argument("--out-stem", type=Path, default=OUT_STEM)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    constants = read_constants(args.summary, args.config)
    windows = load_windows(args.windows, constants)
    curve = binned_curve(windows["ell"], windows["P"], args.bins)
    write_csv(
        args.out_stem.with_name(args.out_stem.name + "_curve.csv"),
        curve,
        curve[0].keys(),
    )
    power_curve = concave_power_curve(constants, float(np.nanmax(windows["ell"])))
    write_csv(
        args.out_stem.with_name(args.out_stem.name + "_power_curve.csv"),
        power_curve,
        power_curve[0].keys(),
    )
    write_csv(
        args.out_stem.with_name(args.out_stem.name + "_constants.csv"),
        [constants],
        constants.keys(),
    )
    make_plot(windows, curve, power_curve, constants, args.out_stem)
    print(args.out_stem)


if __name__ == "__main__":
    main()
