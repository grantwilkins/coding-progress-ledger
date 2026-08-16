"""Plot measured service-headroom points in the prefill/decode load plane."""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import plot_style
import service_headroom_campaign as campaign


SCHEMA = "queue-haul-service-headroom-phase-plane-v1"


def load(run_root: Path) -> list[dict]:
    plan = campaign.read_plan(run_root / "plan.json")
    artifacts = (("discovery", "h100-scout.json", campaign.SCHEMA),
                 ("confirmation", "h100-confirmed.json", campaign.CONFIRM_SCHEMA))
    samples, identities = [], set()
    for stage, filename, schema in artifacts:
        artifact = json.loads((run_root / filename).read_text())
        if artifact.get("schema") != schema or artifact.get("stage") != (
                "scout" if stage == "discovery" else "confirmation"):
            raise ValueError(f"invalid {stage} service-headroom artifact")
        identities.add((artifact["hardware"], artifact["runtime_identity_sha256"],
                        artifact["normalization_sha256"]))
        for row in artifact["rows"]:
            trace = json.loads((run_root / stage / row["cell_id"]
                                / "offered.json").read_text())
            rho_p, rho_d = campaign.offered_phase_rho(plan, trace)
            if not math.isclose(row["offered_rho"], rho_p + rho_d, abs_tol=1e-12):
                raise ValueError(f"offered load changed: {row['cell_id']}")
            samples.append({"stage": stage, "direction": row["direction"],
                            "target_rho": row["target_rho"], "block": row["block"],
                            "rho_p": rho_p, "rho_d": rho_d,
                            "p90_ttft_s": row["p90_ttft_s"],
                            "p90_mean_tpot_s": row["p90_mean_tpot_s"],
                            "stable": row["stable"]})
    if len(identities) != 1:
        raise ValueError("phase plane mixes hardware, runtime, or normalization")
    return aggregate(samples)


def aggregate(samples: list[dict]) -> list[dict]:
    grouped = {}
    for row in samples:
        grouped.setdefault((row["direction"], row["target_rho"]), []).append(row)
    points = []
    for (direction, target), rows in grouped.items():
        rho_p = statistics.fmean(row["rho_p"] for row in rows)
        rho_d = statistics.fmean(row["rho_d"] for row in rows)
        if max(abs(row["rho_p"] - rho_p) + abs(row["rho_d"] - rho_d)
               for row in rows) > 1e-12:
            raise ValueError(f"replicate load changed: {direction}/{target}")
        points.append({"direction": direction, "target_rho": target,
                       "rho_p": rho_p, "rho_d": rho_d,
                       "rho_total": rho_p + rho_d, "replicates": len(rows),
                       "stages": "+".join(sorted({row["stage"] for row in rows})),
                       "all_stable": all(row["stable"] for row in rows),
                       "worst_p90_ttft_s": max(row["p90_ttft_s"] for row in rows),
                       "worst_p90_mean_tpot_s": max(
                           row["p90_mean_tpot_s"] for row in rows)})
    return sorted(points, key=lambda row: (row["direction"], row["target_rho"]))


def parse_slo(value: str) -> tuple[str, float, float]:
    try:
        name, ttft, tpot = value.split(":")
        result = name, float(ttft), float(tpot)
    except ValueError as error:
        raise argparse.ArgumentTypeError("SLO must be NAME:TTFT_SECONDS:TPOT_SECONDS") \
            from error
    if not name or min(result[1:]) <= 0:
        raise argparse.ArgumentTypeError("SLO name and thresholds must be positive")
    return result


def feasible(row: dict, slo: tuple[str, float, float]) -> bool:
    return bool(row["all_stable"] and row["worst_p90_ttft_s"] <= slo[1]
                and row["worst_p90_mean_tpot_s"] <= slo[2])


def write(rows: list[dict], slos: list[tuple[str, float, float]], out: Path) -> None:
    if not rows or not slos or len({slo[0] for slo in slos}) != len(slos):
        raise ValueError("phase plane needs points and uniquely named SLOs")
    plot_style.apply()
    fig, axes = plt.subplots(1, len(slos), squeeze=False,
                             figsize=(4.6 * len(slos), 4.4), sharex=True, sharey=True)
    for axis, slo in zip(axes[0], slos):
        for direction in plot_style.SERVICE_DIRECTIONS:
            selected = sorted((row for row in rows if row["direction"] == direction),
                              key=lambda row: row["rho_total"])
            if not selected:
                continue
            color = plot_style.SERVICE_DIRECTION_COLORS[direction]
            axis.plot([row["rho_p"] for row in selected],
                      [row["rho_d"] for row in selected], color=color,
                      linestyle=plot_style.SERVICE_DIRECTION_LINESTYLES[direction],
                      linewidth=1.5)
            for row in selected:
                physical = row["all_stable"]
                fill = ({"facecolors": color if feasible(row, slo) else "none"}
                        if physical else {})
                axis.scatter(row["rho_p"], row["rho_d"], s=48, color=color,
                             marker=(plot_style.SERVICE_DIRECTION_MARKERS[direction]
                                     if physical else "x"),
                             linewidths=1.6, zorder=3, **fill)
                axis.annotate(f"{row['target_rho']:.2g}",
                              (row["rho_p"], row["rho_d"]), xytext=(3, 3),
                              textcoords="offset points", fontsize=8)
        axis.set(title=f"{slo[0]}\nTTFT ≤ {slo[1]:g}s, TPOT ≤ {slo[2]:g}s",
                 xlabel=r"Measured prefill load $\rho_p$", aspect="equal")
        axis.grid(alpha=.2)
    axes[0, 0].set_ylabel(r"Measured decode load $\rho_d$")
    direction_handles = [Line2D([], [],
        color=plot_style.SERVICE_DIRECTION_COLORS[key],
        linestyle=plot_style.SERVICE_DIRECTION_LINESTYLES[key],
        marker=plot_style.SERVICE_DIRECTION_MARKERS[key],
        label=plot_style.SERVICE_DIRECTION_NAMES[key])
        for key in plot_style.SERVICE_DIRECTIONS]
    status_handles = [
        Line2D([], [], color="black", marker="o", linestyle="", label="SLO pass"),
        Line2D([], [], color="black", marker="o", markerfacecolor="none",
               linestyle="", label="SLO miss"),
        Line2D([], [], color="black", marker="x", linestyle="", label="Unstable"),
    ]
    fig.legend(handles=direction_handles + status_handles, loc="lower center",
               ncol=len(direction_handles) + len(status_handles), frameon=False,
               fontsize=plot_style.LEGEND_FONT_SIZE)
    fig.text(.5, .01, "Measured directional rays only; no 2D contour interpolation.",
             ha="center", fontsize=plot_style.ANNOTATION_FONT_SIZE)
    fig.tight_layout(rect=(0, .12, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)
    long = [{**row, "slo": slo[0], "ttft_target_s": slo[1],
             "tpot_target_s": slo[2], "feasible": feasible(row, slo)}
            for slo in slos for row in rows]
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=long[0], lineterminator="\n")
        writer.writeheader(); writer.writerows(long)
    out.with_suffix(".json").write_text(json.dumps(
        {"schema": SCHEMA, "slos": [dict(zip(("name", "ttft_target_s",
         "tpot_target_s"), slo)) for slo in slos], "points": rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--slo", action="append", type=parse_slo, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write(load(args.run_root), args.slo, args.out)


if __name__ == "__main__":
    main()
