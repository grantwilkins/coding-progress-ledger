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


def kv_estimates(root, rows):
    frozen = json.loads((root / "0/wan/prepared/plan.json").read_text())["frozen"]
    controls = [r for r in rows if r["campaign"] == "wan" and float(r["wan_mbps"]) == 10000
                and r["policy"] == "per_session_greedy"]
    if len(controls) != 13 or len({r["pack_id"] for r in controls}) != 1:
        raise ValueError("expected thirteen matched all-KV control episodes")
    demands = [d for d in frozen["inputs"]["action_demands"]
               if d["pack_id"] == controls[0]["pack_id"] and d["action"] == "kv_transfer"]
    if len(demands) != 8:
        raise ValueError("expected eight KV byte demands")
    total_bytes = sum(d["d_wan"] for d in demands)
    control_times = []
    for row in controls:
        decisions = json.loads(row["decisions"])
        if len(decisions) != 8 or any(d["action"] != "kv_transfer" or d["error"] for d in decisions):
            raise ValueError("control must complete all eight sessions with KV")
        control_times.append(max(d["completion_s"] for d in decisions))
    overhead = float(np.mean(control_times)) - 8 * total_bytes / 1e10
    if overhead < 0:
        raise ValueError("control transfer faster than its byte budget")
    rates = sorted({float(r["wan_mbps"]) for r in rows if r["campaign"] == "wan" and float(r["wan_mbps"]) != 10000})
    return [dict(wan_mbps=rate, kv_share_percent=100, total_kv_bytes=total_bytes,
                 control_mean_completion_s=float(np.mean(control_times)), overhead_s=overhead,
                 completion_time_s=8 * total_bytes / (rate * 1e6) + overhead,
                 power_window_s=frozen["constants"]["power_window_s"],
                 attainment_time_s=8 * total_bytes / (rate * 1e6) + overhead + frozen["constants"]["power_window_s"])
            for rate in rates]


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
    points = [p for p in points if (p["wan_mbps"] != 10000 if p["campaign"] == "wan" else p["prefill_rps"] > 0)]
    points = [p for p in points if p["policy"] in POLICIES[:3]]
    if any(p["attainment_time_s"] == "" for p in points):
        raise ValueError("full-plan episode has no finite attainment time")
    estimates = kv_estimates(root, rows)
    horizon = max(r["attainment_time_s"] for r in points + estimates)
    out.mkdir(parents=True, exist_ok=True)
    for campaign in ("wan", "prefill"):
        selected_campaign = [r for r in points if r["campaign"] == campaign]
        fig, ax = plt.subplots(figsize=(2.1, 1.6))
        for policy in POLICIES:
            shared = policy in ("kv_only", "replay_only")
            source_policy = "per_session_greedy" if shared else policy
            selected = sorted((r for r in selected_campaign if r["policy"] == source_policy),
                              key=lambda r: (r["state_id"], int(r["repeat"])))
            identity = STYLE_IDS[policy]
            offsets = np.random.default_rng(0).permutation(np.linspace(-1, 1, len(selected)))
            if shared:
                keep = [i for i, r in enumerate(selected) if r["kv_share_percent"] == (100 if policy == "kv_only" else 0)]
                selected, offsets = [selected[i] for i in keep], offsets[keep]
            estimated = campaign == "wan" and policy == "kv_only"
            if estimated:
                selected, offsets = estimates, np.zeros(len(estimates))
            if not selected:
                continue
            ax.scatter(np.array([r["kv_share_percent"] for r in selected]) + offsets,
                       [r["attainment_time_s"] for r in selected],
                       marker=plot_style.POLICY_MARKERS[identity], s=20 if shared else 7, alpha=.45 if shared else .6,
                       facecolors="none" if shared or policy == "greedy" else plot_style.POLICY_COLORS[identity],
                       edgecolors=plot_style.POLICY_COLORS[identity], linewidths=1 if shared else .8,
                       label=plot_style.STRESS_POLICY_NAMES[identity] if policy == "per_session_greedy" else plot_style.PAPER_POLICY_NAMES[identity],
                       zorder=4 if policy == "per_session_greedy" else 3)
        ax.axhline(30, color="black", linestyle=":", linewidth=.8)
        ax.text(50, 31, "30 s deadline", ha="center", fontsize=6, fontstyle="italic")
        ax.set(xlim=(-5, 105), ylim=(0, horizon + 2), xticks=(0, 50, 100),
               yticks=(0, 15, 30, 45),
               xlabel="KV-transfer share (%)", ylabel="Time to target (s)")
        plot_style.half_column(ax)
        ax.tick_params(axis="y", labelsize=6)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 2
        ax.grid(alpha=.2, linewidth=.5)
        ax.set_axisbelow(True)
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .01),
                   ncol=3, frameon=False, fontsize=5.5, handlelength=.8,
                   handletextpad=.3, columnspacing=.4, labelspacing=.4)
        fig.subplots_adjust(left=.27, right=.96, bottom=.38, top=.97)
        save(fig, out / f"{campaign}_action_attainment")
    with (out / "wan_kv_estimates.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=estimates[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(estimates)
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
