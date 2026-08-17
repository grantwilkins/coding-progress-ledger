"""Plot fixed-shape P90 latency degradation and boundary variability."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import fixed_shape_slo_campaign as campaign
import plot_style


SCHEMA = "queue-haul-fixed-shape-slo-plot-v1"


def load(roots: list[Path]) -> tuple[list[dict], dict]:
    summaries = [json.loads((root / "summary.json").read_text()) for root in roots]
    if {row.get("schema") for row in summaries} != {campaign.SCHEMA} \
            or {row["model"] for row in summaries} != set(plot_style.MODELS):
        raise ValueError("plot needs one complete summary per canonical model")
    contract = {(row["hardware"], row["input_tokens"], row["output_tokens"],
                 row["requests_per_point"], tuple(row["rates_rps"]),
                 row["ttft_slo_s"], row["tpot_slo_s"]) for row in summaries}
    if len(contract) != 1:
        raise ValueError("fixed-shape summaries use different contracts")
    rows = [{**point, "boundary": point["offered_rps"] in
             summary.get("whisker_rates_rps", ())}
            for summary in summaries for point in summary["curve"]]
    hardware, input_tokens, output_tokens, requests, rates, ttft, tpot = contract.pop()
    return rows, {"hardware": hardware, "input_tokens": input_tokens,
                  "output_tokens": output_tokens, "requests_per_point": requests,
                  "rates_rps": list(rates), "ttft_slo_s": ttft,
                  "tpot_slo_s": tpot}


def write(rows: list[dict], contract: dict, out: Path) -> None:
    plot_style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), sharex=True)
    panels = (("p90_ttft_s", "P90 TTFT (s)", contract["ttft_slo_s"]),
              ("p90_mean_tpot_s", "P90 mean TPOT (s/token)",
               contract["tpot_slo_s"]))
    for axis, (metric, ylabel, slo) in zip(axes, panels):
        for model in plot_style.MODELS:
            selected = sorted((row for row in rows if row["model"] == model),
                              key=lambda row: row["offered_rps"])
            x = [row["offered_rps"] for row in selected]
            y = [row[metric] for row in selected]
            axis.plot(x, y, color=plot_style.MODEL_COLORS[model],
                      linestyle=plot_style.MODEL_LINESTYLES[model],
                      marker=plot_style.MODEL_MARKERS[model],
                      label=plot_style.MODEL_NAMES[model])
            boundary = [row for row in selected if row["boundary"]
                        and row[metric] is not None]
            if boundary:
                axis.errorbar(
                    [row["offered_rps"] for row in boundary],
                    [row[metric] for row in boundary],
                    yerr=[[row[metric] - row[f"{metric}_min"] for row in boundary],
                          [row[f"{metric}_max"] - row[metric] for row in boundary]],
                    fmt="none", capsize=4, linewidth=1.5,
                    color=plot_style.MODEL_COLORS[model],
                )
        if slo is not None:
            axis.axhline(slo, color="black", linestyle=(0, (3, 1)), linewidth=1.5,
                         label=f"SLO = {slo:g}")
        axis.set_xscale("log", base=2)
        ticks = sorted({row["offered_rps"] for row in rows})
        axis.set_xticks(ticks, [f"{rate:g}" for rate in ticks])
        axis.set_xlabel("Offered request rate (RPS)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle(f"{contract['input_tokens']:,} input + "
                 f"{contract['output_tokens']:,} forced output tokens, "
                 f"{contract['requests_per_point']} requests/point")
    fig.tight_layout(rect=(0, .13, 1, .96))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)
    with out.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(
        {"schema": SCHEMA, "contract": contract, "points": rows}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write(*load(args.run_root), args.out)


if __name__ == "__main__":
    main()
