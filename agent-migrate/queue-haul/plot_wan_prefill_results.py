"""Pool recorded WAN/prefill episode completions and policy action counts."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import plot_style

POLICIES = ("queue_haul", "greedy", "per_session_greedy", "kv_only", "replay_only")
STYLE_IDS = {p: "isolated_fastest" if p == "per_session_greedy" else p for p in POLICIES}


def pooled(rows, campaign, policy):
    episodes = [r for r in rows if r["campaign"] == campaign and r["policy"] == policy]
    if not episodes:
        raise ValueError(f"missing episodes: {campaign}/{policy}")
    events, counts, total = [], Counter(), 0
    for row in episodes:
        decisions = json.loads(row["decisions"])
        counts.update(d["action"] for d in decisions)
        counts["not_selected"] += int(row["not_moved_count"])
        size = len(decisions) + int(row["not_moved_count"])
        if size != 8 or len({d["session_id"] for d in decisions}) != len(decisions):
            raise ValueError("expected eight distinct source sessions per episode")
        total += size
        completed = []
        for decision in decisions:
            at = decision["completion_s"]
            if at is not None and not decision["error"]:
                if not np.isfinite(at) or at < 0:
                    raise ValueError("invalid completion time")
                completed.append(at)
        if len(completed) == size:
            events.append(max(completed))
    if counts.keys() - {"replay", "kv_transfer", "not_selected"}:
        raise ValueError("unknown action")
    return np.r_[0, sorted(events)], np.arange(len(events) + 1) / len(episodes), counts, total


def save(fig, out):
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)


def plot(source, out):
    plot_style.apply()
    with source.open() as stream:
        rows = list(csv.DictReader(stream))
    ids = [(r["campaign"], r["episode_id"]) for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate episodes")
    groups = {(c, p): pooled(rows, c, p) for c in ("wan", "prefill") for p in POLICIES}
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    horizon = max(x[-1] for x, _, _, _ in groups.values()) * 1.04
    for campaign, title in (("wan", "WAN constrained"), ("prefill", "Prefill / compute headroom constrained")):
        coverage = [Counter((r["state_id"], r["pack_id"], r["repeat"]) for r in rows
                            if r["campaign"] == campaign and r["policy"] == p) for p in POLICIES]
        if any(c != coverage[0] for c in coverage):
            raise ValueError("policies must cover the same cases and repeats")
        fig, ax = plt.subplots()
        for policy in POLICIES:
            x, y, counts, total = groups[campaign, policy]
            ax.step(np.r_[x, horizon], np.r_[y, y[-1]], where="post",
                    **plot_style.policy_style(STYLE_IDS[policy]))
            summaries.append(dict(campaign=campaign, policy=policy, episodes=total // 8,
                                  sessions=total, completed_episodes=len(x) - 1,
                                  episodes_completed_by_30s=int(np.count_nonzero(x[1:] <= 30)),
                                  replay=counts["replay"], kv_transfer=counts["kv_transfer"],
                                  not_selected=counts["not_selected"]))
        ax.axvline(30, color="black", linestyle=":", linewidth=1, label="30 s deadline")
        ax.set(xlabel="Time since migration start (s)", ylabel="Fraction of episodes finished",
               title=title, xlim=(0, horizon), ylim=(0, 1.02))
        ax.grid(alpha=.2)
        ax.legend(loc="upper left")
        save(fig, out / f"{campaign}_completion_ecdf")
    fig, ax = plt.subplots()
    policies = POLICIES[:3]
    bottom = np.zeros(len(policies))
    for action in ("replay", "kv_transfer", "not_selected"):
        shares = [sum(groups[c, p][2][action] for c in ("wan", "prefill")) /
                  sum(groups[c, p][3] for c in ("wan", "prefill")) for p in policies]
        if not any(shares):
            continue
        ax.bar(range(3), shares, bottom=bottom, label=plot_style.ACTION_NAMES[action],
               color=plot_style.ACTION_COLORS[action])
        for i, share in enumerate(shares):
            if share:
                ax.text(i, bottom[i] + share / 2, f"{share:.1%}", ha="center", va="center", color="white")
        bottom += shares
    ax.set(xticks=range(3), xticklabels=[plot_style.POLICY_NAMES[STYLE_IDS[p]] for p in policies],
           ylabel="Share of source sessions", ylim=(0, 1), title="Action mix · WAN + prefill cases pooled")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, frameon=False)
    save(fig, out / "pooled_action_mix")
    with (out / "pooled_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=summaries[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/robustness-a100-20260907/episodes.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/robustness-a100-20260907/pooled"))
    args = parser.parse_args()
    plot(args.source, args.out)
