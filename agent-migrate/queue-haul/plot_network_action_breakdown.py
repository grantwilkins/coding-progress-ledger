"""Plot isolated method winners and joint planner actions on one axis."""

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
POLICIES = ("queue_haul", "greedy", "greedy_lagrangian", "kv_only",
            "replay_only", "random")
LABELS = {
    "queue_haul": "Queue-Haul LP", "greedy": "Queue-Haul Greedy",
    "greedy_lagrangian": "Lagrangian greedy", "kv_only": "KV only",
    "replay_only": "Replay only", "random": "Random",
}


def _summary(group: str, label: str, unit: str, counts: Counter) -> dict:
    total = sum(counts.values())
    if not total or set(counts) - set(ACTIONS):
        raise ValueError(f"invalid action counts for {group}")
    return {"group": group, "label": label, "unit": unit, "total": total,
            "counts": {action: counts[action] for action in ACTIONS}}


def isolated(root: Path, site: str) -> dict:
    cells = defaultdict(dict)
    with (root / "results.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "complete":
                continue
            if row["destination"] != site or row["policy"] not in {
                    "replay", "kv_transfer"}:
                raise ValueError(f"invalid isolated action in {root}")
            key = row["bandwidth"], row["condition_index"], row["repeat"]
            if row["policy"] in cells[key]:
                raise ValueError(f"duplicate isolated action in {root}: {key}")
            cells[key][row["policy"]] = float(row["migration_s"])
    counts = Counter()
    for key, methods in cells.items():
        if set(methods) != {"replay", "kv_transfer"} \
                or methods["replay"] == methods["kv_transfer"]:
            raise ValueError(f"unpaired or tied isolated cell in {root}: {key}")
        counts[site, min(methods, key=methods.get)] += 1
    return _summary(f"{site.title()} validation", "Faster measured action",
                    "matched cells", counts)


def joint(root: Path) -> list[dict]:
    counts, scenarios = defaultdict(Counter), Counter()
    for path in attempts(root):
        scenario = json.loads((path.parent / "scenario.json").read_text())
        moves = json.loads((path.parent / "decision.json").read_text())["moves"]
        sessions = {row["session_id"] for row in scenario["sessions"]}
        if len(moves) != len(sessions) or {row["session_id"] for row in moves} \
                != sessions:
            raise ValueError(f"incomplete joint decision: {path}")
        policy = scenario["policy"]
        scenarios[policy] += 1
        for move in moves:
            action = move["destination_instance"], move["method"]
            if action not in ACTIONS:
                raise ValueError(f"invalid joint action: {action}")
            counts[policy][action] += 1
    return [_summary("Joint selection", LABELS[policy], "sessions",
                     counts[policy]) | {"scenarios": scenarios[policy]}
            for policy in POLICIES if counts[policy]]


def rows(root: Path = NETWORK_ROOT) -> list[dict]:
    return [isolated(root / "single-east", "east"),
            isolated(root / "single-west", "west"),
            *joint(root / "joint-queue-002-partial-086")]


def write(root: Path = NETWORK_ROOT, out: Path | None = None) -> list[dict]:
    summaries = rows(root)
    out = out or root / "network_action_breakdown"
    positions, cursor, previous = [], 0., None
    for row in summaries:
        if previous is not None and row["group"] != previous:
            cursor += .8
        positions.append(cursor)
        cursor += 1
        previous = row["group"]

    styles = {
        ("east", "replay"): ("Replay to East", "#E98300", ""),
        ("east", "kv_transfer"): ("KV to East", "#006CB8", ""),
        ("west", "replay"): ("Replay to West", "#E98300", "///"),
        ("west", "kv_transfer"): ("KV to West", "#006CB8", "///"),
    }
    figure, axis = plt.subplots(figsize=(6, 4.5))
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

    for group in dict.fromkeys(row["group"] for row in summaries):
        y = next(y for y, row in zip(positions, summaries)
                 if row["group"] == group)
        axis.text(0, y - .55, group, ha="left", va="bottom", fontsize=9,
                  weight="bold")
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
        fields = ("group", "label", "unit", "total", "site", "method",
                  "count", "share_percent")
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        for row in summaries:
            for site, method in ACTIONS:
                count = row["counts"][site, method]
                writer.writerow({"group": row["group"], "label": row["label"],
                                 "unit": row["unit"], "total": row["total"],
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
