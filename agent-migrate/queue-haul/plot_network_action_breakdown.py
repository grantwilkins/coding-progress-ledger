"""Plot joint planner action choices by policy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_network_ttft_cdf import attempts


ROOT = Path(__file__).parent
NETWORK_ROOT = ROOT / "outputs/network-campaign-20260805"
ACTIONS = (("east", "replay"), ("east", "kv_transfer"),
           ("west", "replay"), ("west", "kv_transfer"))
POLICIES = ("queue_haul", "greedy", "kv_only", "replay_only")
LABELS = {
    "queue_haul": "Queue-Haul LP", "greedy": "Queue-Haul Greedy",
    "kv_only": "KV only", "replay_only": "Replay only",
}


def _summary(label: str, counts: Counter) -> dict:
    total = sum(counts.values())
    if not total or set(counts) - set(ACTIONS):
        raise ValueError(f"invalid action counts for {label}")
    return {"label": label, "total": total,
            "counts": {action: counts[action] for action in ACTIONS}}


def joint(root: Path) -> list[dict]:
    counts = defaultdict(Counter)
    for path in attempts(root):
        scenario = json.loads((path.parent / "scenario.json").read_text())
        moves = json.loads((path.parent / "decision.json").read_text())["moves"]
        sessions = {row["session_id"] for row in scenario["sessions"]}
        if len(moves) != len(sessions) or {row["session_id"] for row in moves} \
                != sessions:
            raise ValueError(f"incomplete joint decision: {path}")
        policy = scenario["policy"]
        for move in moves:
            action = move["destination_instance"], move["method"]
            if action not in ACTIONS:
                raise ValueError(f"invalid joint action: {action}")
            counts[policy][action] += 1
    return [_summary(LABELS[policy], counts[policy])
            for policy in POLICIES if counts[policy]]


def rows(root: Path = NETWORK_ROOT) -> list[dict]:
    return joint(root / "joint-queue-002-partial-086")


def write(root: Path = NETWORK_ROOT, out: Path | None = None) -> list[dict]:
    summaries = rows(root)
    out = out or root / "network_action_breakdown"
    positions = range(len(summaries))

    styles = {
        ("east", "replay"): ("Replay to East", "#E98300", ""),
        ("east", "kv_transfer"): ("KV to East", "#006CB8", ""),
        ("west", "replay"): ("Replay to West", "#E98300", "///"),
        ("west", "kv_transfer"): ("KV to West", "#006CB8", "///"),
    }
    figure, axis = plt.subplots(figsize=(6, 3))
    for y, row in zip(positions, summaries):
        left = 0.
        for action in ACTIONS:
            count = row["counts"][action]
            share = 100 * count / row["total"]
            label, color, hatch = styles[action]
            axis.barh(y, share, left=left, height=.62, color=color, hatch=hatch,
                      edgecolor="white", linewidth=.5,
                      label=label if y == positions[0] else None)
            left += share
        axis.text(101.5, y, f"n={row['total']}", va="center", fontsize=8)

    axis.set(xlim=(0, 112), xticks=range(0, 101, 20),
             yticks=positions,
             yticklabels=[row["label"] for row in summaries],
             xlabel="Share within each bar (%)")
    axis.invert_yaxis()
    axis.spines[["left", "right", "top"]].set_visible(False)
    axis.tick_params(axis="x", length=0)
    axis.grid(axis="x", alpha=.2, zorder=0)
    axis.legend(frameon=False, ncol=2, loc="lower center",
                bbox_to_anchor=(.5, 1.01))
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(out.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)

    with out.with_suffix(".csv").open("w", newline="") as stream:
        fields = ("label", "total", "site", "method", "count",
                  "share_percent")
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            for site, method in ACTIONS:
                count = row["counts"][site, method]
                writer.writerow({"label": row["label"], "total": row["total"],
                                 "site": site, "method": method, "count": count,
                                 "share_percent": 100 * count / row["total"]})
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=NETWORK_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    write(args.root, args.out)


if __name__ == "__main__":
    main()
