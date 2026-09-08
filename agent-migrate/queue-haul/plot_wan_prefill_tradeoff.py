"""Plot action mix versus full power-target attainment for recorded episodes."""

import argparse
import csv
import json
from collections import defaultdict
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
    summaries = []
    for campaign in ("wan", "prefill"):
        frozen = json.loads((root / "0" / campaign / "prepared/plan.json").read_text())["frozen"]
        packs = {p["pack_id"]: p for p in frozen["inputs"]["packs"]}
        grouped = defaultdict(list)
        for row in rows:
            if row["campaign"] == campaign:
                grouped[row["state_id"], row["pack_id"], row["policy"]].append(
                    episode_point(row, packs[row["pack_id"]], frozen["inputs"]["target"],
                                  frozen["constants"]["power_window_s"]))
        for (state, pack, policy), points in grouped.items():
            times = [t for _, t in points if t is not None]
            if len(points) != 13:
                raise ValueError("expected 13 repeats per case and policy")
            summaries.append(dict(campaign=campaign, state_id=state, pack_id=pack, policy=policy,
                                  episodes=len(points), attained=len(times),
                                  kv_share_percent=np.mean([x for x, _ in points]),
                                  attainment_time_s=np.mean(times) if len(times) == len(points) else ""))
    horizon = max(r["attainment_time_s"] for r in summaries if r["attainment_time_s"] != "")
    not_met = horizon + 7
    out.mkdir(parents=True, exist_ok=True)
    for campaign in ("wan", "prefill"):
        fig, ax = plt.subplots(figsize=(2.1, 1.75))
        for policy in POLICIES:
            points = [r for r in summaries if r["campaign"] == campaign and r["policy"] == policy]
            identity = STYLE_IDS[policy]
            ax.scatter([r["kv_share_percent"] for r in points],
                       [r["attainment_time_s"] if r["attainment_time_s"] != "" else not_met for r in points],
                       marker=plot_style.POLICY_MARKERS[identity], s=20,
                       facecolors="none" if policy == "greedy" else plot_style.POLICY_COLORS[identity],
                       edgecolors=plot_style.POLICY_COLORS[identity], linewidths=.7,
                       label=plot_style.PAPER_POLICY_NAMES[identity], zorder=3)
        ax.axhspan(horizon + 3, not_met + 3, color=".94", zorder=0)
        ax.axhline(30, color="black", linestyle=":", linewidth=.8)
        ax.text(3, 31, "30 s deadline", fontsize=6, fontstyle="italic")
        ax.set(xlabel="KV-transfer share (%)", ylabel="Time to target (s)",
               xlim=(-5, 105), ylim=(0, not_met + 3), xticks=(0, 50, 100),
               yticks=(0, 15, 30, not_met), yticklabels=("0", "15", "30", "Not met"))
        plot_style.half_column(ax)
        ax.tick_params(axis="y", labelsize=6)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 2
        ax.grid(alpha=.2, linewidth=.5)
        ax.set_axisbelow(True)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.61, .01),
                   ncol=2, frameon=False, fontsize=5.5, handlelength=1.2,
                   handletextpad=.3, columnspacing=.7, labelspacing=.3)
        fig.subplots_adjust(left=.27, right=.96, bottom=.43, top=.97)
        save(fig, out / f"{campaign}_action_attainment")
    with (out / "action_attainment.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/robustness-a100-20260907"))
    parser.add_argument("--out", type=Path, default=Path("outputs/robustness-a100-20260907/pooled"))
    args = parser.parse_args()
    plot(args.root, args.out)
