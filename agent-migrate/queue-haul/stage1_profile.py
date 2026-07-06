from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONFIG = "gpt-oss-20b-a100_tp1"
PTRACE = Path(__file__).resolve().parents[3] / "powertrace-sim" / "results" / "two_price_fit"
OUT_STEM = Path(__file__).resolve().parent / "outputs" / "stage1_gpt_oss_20b_a100_tp1"
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


def write_csv(path: Path, rows: list[dict], fieldnames) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_plot(windows, curve, constants, out_stem: Path) -> None:
    ell, power = windows["ell"], windows["P"]
    f_share = windows.get("f_share", np.zeros_like(ell))
    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    sc = ax.scatter(ell, power, c=f_share, s=7, alpha=0.25, cmap="coolwarm", vmin=0, vmax=1)
    ax.plot(
        [r["ell_mean"] for r in curve],
        [r["power_mean_w"] for r in curve],
        color="black",
        marker="o",
        ms=3,
        lw=1.5,
        label="binned average power",
    )
    ax.axvline(constants["ell_power_knee"], color="purple", ls=":", lw=1.2, label="power knee")
    ax.axvline(constants["rho_star"], color="red", ls="--", lw=1.2, label="rho*")
    note = (
        f"F={constants['F_prefill_tps']:.0f} tok/s, G={constants['G_decode_tps']:.0f} tok/s\n"
        f"P0={constants['P0_w']:.0f} W, Pmax={constants['P_max_w']:.0f} W\n"
        f"s_plat={constants['s_plat_w_per_ell']:.0f} W/ell, "
        f"p_amort={constants['p_amort_w_per_ell']:.0f} W/ell"
    )
    ax.text(0.98, 0.05, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    ax.set_xlabel("ell load = f/F + g/G")
    ax.set_ylabel("average node power [W]")
    ax.set_title("gpt-oss-20b on A100, TP=1")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, label="prefill token share")
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
    windows = dict(np.load(args.windows, allow_pickle=False))
    constants = read_constants(args.summary, args.config)
    curve = binned_curve(windows["ell"], windows["P"], args.bins)
    write_csv(args.out_stem.with_name(args.out_stem.name + "_curve.csv"), curve, curve[0].keys())
    write_csv(
        args.out_stem.with_name(args.out_stem.name + "_constants.csv"),
        [constants],
        constants.keys(),
    )
    make_plot(windows, curve, constants, args.out_stem)
    print(args.out_stem)


if __name__ == "__main__":
    main()
