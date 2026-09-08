"""Plot action mix versus full power-target attainment for recorded episodes."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

import plot_style
from plot_wan_prefill_results import POLICIES, STYLE_IDS, plt, save


def episode_point(row, pack, target, window):
    sessions = {s["session_id"] for s in pack["sessions"]}
    gains = pack["power_gains"]
    if len(sessions) != 8 or not np.isclose(gains[-1], target) or max(gains[:-1]) >= target:
        raise ValueError("target must require all eight sessions")
    decisions = json.loads(row["decisions"])
    ids = [d["session_id"] for d in decisions]
    if not decisions or len(ids) != len(set(ids)) or not set(ids) <= sessions:
        raise ValueError("invalid selected sessions")
    if any(d["action"] not in ("replay", "kv_transfer") for d in decisions):
        raise ValueError("unknown action")
    times = [d["completion_s"] for d in decisions if d["completion_s"] is not None and not d["error"]]
    if any(not np.isfinite(t) or t < 0 for t in times):
        raise ValueError("invalid completion time")
    time = max(times) + window if len(times) == len(sessions) else None
    recorded = row["target_time_s"]
    if recorded and (time is None or not np.isclose(time, float(recorded))):
        raise ValueError("attainment time disagrees with archived result")
    if (time is not None and time <= float(row["deadline_s"])) != (row["target_attained"] == "True"):
        raise ValueError("deadline attainment disagrees with archived result")
    return 100 * sum(d["action"] == "kv_transfer" for d in decisions) / len(decisions), time


def plot(root, out):
    plot_style.apply()
    with (root / "episodes.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if len({(r["campaign"], r["episode_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate episodes")
    points = []
    for campaign in ("wan", "prefill"):
        frozen = json.loads((root / "0" / campaign / "prepared/plan.json").read_text())["frozen"]
        packs = {p["pack_id"]: p for p in frozen["inputs"]["packs"]}
        for row in rows:
            if row["campaign"] == campaign:
                share, time = episode_point(row, packs[row["pack_id"]], frozen["inputs"]["target"],
                                            frozen["constants"]["power_window_s"])
                points.append({**{k: row[k] for k in ("campaign", "episode_id", "state_id", "pack_id",
                                                     "policy", "repeat", "phase")},
                               "wan_mbps": float(row["wan_mbps"]),
                               "prefill_rps": json.loads(row["capacity_inputs"])["background_rps"],
                               "kv_share_percent": share, "attainment_time_s": "" if time is None else time})
    counts = Counter((p["campaign"], p["state_id"], p["pack_id"], p["policy"]) for p in points)
    if any(n != 13 for n in counts.values()):
        raise ValueError("expected 13 repeats per case and policy")
    horizon = max(r["attainment_time_s"] for r in points if r["attainment_time_s"] != "")
    not_met = horizon + 7
    out.mkdir(parents=True, exist_ok=True)
    for campaign in ("wan", "prefill"):
        cases = sorted({(r["wan_mbps"], r["prefill_rps"], r["state_id"]) for r in points
                        if r["campaign"] == campaign})
        nrows = (len(cases) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(3.5, 1.45 * nrows + .8), sharex=True, sharey=True)
        for ax, (wan, prefill, state) in zip(axes.flat, cases):
            case_points = [r for r in points if r["campaign"] == campaign and r["state_id"] == state]
            for policy in POLICIES:
                selected = sorted((r for r in case_points if r["policy"] == policy), key=lambda r: int(r["repeat"]))
                identity = STYLE_IDS[policy]
                offsets = np.linspace(-4, 4, len(selected))
                if policy == "greedy":
                    offsets = offsets[::-1]
                ax.scatter(np.array([r["kv_share_percent"] for r in selected]) + offsets,
                           [r["attainment_time_s"] if r["attainment_time_s"] != "" else not_met for r in selected],
                           marker=plot_style.POLICY_MARKERS[identity], s=9, alpha=.7,
                           facecolors="none" if policy == "greedy" else plot_style.POLICY_COLORS[identity],
                           edgecolors=plot_style.POLICY_COLORS[identity], linewidths=.5,
                           label=plot_style.PAPER_POLICY_NAMES[identity], zorder=3)
            ax.axhspan(horizon + 3, not_met + 3, color=".94", zorder=0)
            ax.axhline(30, color="black", linestyle=":", linewidth=.8)
            ax.text(50, 31, "30 s deadline", ha="center", fontsize=6, fontstyle="italic")
            title = (f"{wan / 1000:g} Gb/s" + (" · control" if wan == 10000 else " · constrained")
                     if campaign == "wan" else f"{prefill:g} prefill req/s" + (" · control" if prefill == 0 else ""))
            ax.set_title(title, fontsize=7.5, pad=5)
            ax.set(xlim=(-7, 107), ylim=(0, not_met + 3), xticks=(0, 50, 100),
                   yticks=(0, 15, 30, not_met), yticklabels=("0", "15", "30", "Not met"))
            plot_style.half_column(ax)
            ax.tick_params(axis="y", labelsize=6)
            ax.grid(alpha=.2, linewidth=.5)
            ax.set_axisbelow(True)
            ax.text(.5, .04, "13 episodes per policy", transform=ax.transAxes,
                    ha="center", fontsize=5.5, color=".35")
        for ax in list(axes.flat)[len(cases):]:
            ax.set_visible(False)
        fig.supylabel("Time to requested power reduction (s)", x=.01, y=.59, fontsize=8)
        fig.supxlabel("KV-transfer share (%)", x=.58, y=.155, fontsize=8)
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.55, .045),
                   ncol=3, frameon=False, fontsize=6, handlelength=1.2,
                   handletextpad=.3, columnspacing=.7, labelspacing=.4)
        fig.text(.5, .012, "Horizontal offsets ±4 pp separate repeats; time values are exact.",
                 ha="center", fontsize=5.5)
        fig.subplots_adjust(left=.20, right=.97, bottom=.22, top=.95, hspace=.38, wspace=.18)
        save(fig, out / f"{campaign}_action_attainment")
    with (out / "action_attainment.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=points[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/robustness-a100-20260907"))
    parser.add_argument("--out", type=Path, default=Path("outputs/robustness-a100-20260907/pooled"))
    args = parser.parse_args()
    plot(args.root, args.out)
