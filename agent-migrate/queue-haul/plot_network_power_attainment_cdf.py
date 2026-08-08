"""Plot matched East/Germany modeled power-target attainment ECDFs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from network_campaign import agentic_demand
from planner import source_power
from policy_hardware_campaign import _problem
from profiles import ModelProfile


ROOT = Path(__file__).parent
RUN_ROOT = ROOT / "outputs/east-germany-frontier-20260808"
PHASES = ("pilot", "refinement", "deadline-blind")
POLICIES = (
    "queue_haul", "greedy", "kv_only", "replay_only",
    "queue_haul_power_blind", "queue_haul_deadline_blind",
)
LABELS = {
    "queue_haul": "Queue-Haul LP", "greedy": "Queue-Haul Greedy",
    "kv_only": "KV Migrate Only", "replay_only": "Replay Context Only",
    "queue_haul_power_blind": "Queue-Haul Power Blind",
    "queue_haul_deadline_blind": "Queue-Haul Deadline Blind",
}
COLORS = dict(zip(POLICIES, (
    "#B1040E", "#008566", "#006CB8", "#E98300", "#6F42C1", "#17BECF",
)))
LINESTYLES = dict(zip(POLICIES, (
    "-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)),
)))
FIELDS = (
    "phase", "scenario_id", "condition_index", "repeat", "policy", "pack",
    "movement_tokens", "mean_destination_load", "deadline_s",
    "requested_shed_w", "attainment_s", "attained_by_deadline",
)


def _pinned(record: dict) -> Path:
    saved = Path(record["path"])
    candidates = {
        saved, ROOT.parent / saved, ROOT / "profiles" / saved.name,
        ROOT / "outputs" / saved.name,
    }
    for path in candidates:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() \
                == record["sha256"]:
            return path
    raise RuntimeError("campaign input is unavailable or changed")


def completion_times(result: dict) -> list[tuple[float, str]]:
    start, output, seen = int(result["started_ns"]), [], set()
    for move in result.get("requests", []):
        if "request" not in move:
            continue
        session = move["session_id"]
        time_s = (int(move["request"]["end_ns"]) - start) / 1e9
        if session in seen or time_s < 0:
            raise ValueError("request completions must be unique and nonnegative")
        seen.add(session)
        output.append((time_s, session))
    return sorted(output)


def attainment_time(completions, target_w, shed) -> float | None:
    if target_w <= 0:
        raise ValueError("power target must be positive")
    credited = set()
    for time_s, session in sorted(completions):
        if session in credited or time_s < 0:
            raise ValueError("completions must uniquely identify sessions")
        credited.add(session)
        if shed(credited) >= target_w - 1e-9:
            return time_s
    return None


def attainment_curve(rows: list[dict], policy: str):
    selected = [row for row in rows if row["policy"] == policy]
    events = sorted(row["attainment_s"] for row in selected
                    if row["attainment_s"] is not None)
    return np.r_[0, events], np.r_[0, np.arange(1, len(events) + 1)
                                      / len(selected)]


def _episode(scenario, result, profile, templates):
    records = {row["session_id"]: {
        **templates[row.get("template_id", row["session_id"])],
        "id": row["session_id"],
    } for row in scenario["sessions"]}
    demand = agentic_demand(
        records, scenario["sessions"], profile, scenario["source_load"])
    problem, _ = _problem(
        profile, scenario["sessions"], 1, scenario["deadline_s"])
    problem = replace(problem, sessions=tuple(replace(
        session, expected_f=demand[session.session_id][0],
        expected_g=demand[session.session_id][1],
    ) for session in problem.sessions))
    initial = source_power(problem, profile)
    minimum = source_power(
        problem, profile, (session.session_id for session in problem.sessions))
    target = scenario["requested_shed_fraction"] * (initial - minimum)
    if not math.isclose(target, float(result["requested_shed_w"]), abs_tol=1e-9):
        raise ValueError("recomputed and recorded power targets differ")
    event = attainment_time(
        completion_times(result), target,
        lambda credited: initial - source_power(problem, profile, credited),
    )
    if bool(result["target_met"]) != (
            event is not None and event <= scenario["deadline_s"]):
        raise ValueError("recorded and recomputed target attainment differ")
    return event, target


def extract(run_root: Path = RUN_ROOT) -> list[dict]:
    output, pins = [], None
    for phase in PHASES:
        root = run_root / phase
        plan = json.loads((root / "plan.json").read_text())
        current = (plan["model_profile"]["sha256"], plan["manifest"]["sha256"])
        if pins is None:
            pins = current
            profile = ModelProfile.load(_pinned(plan["model_profile"]))
            manifest = json.loads(_pinned(plan["manifest"]).read_text())
            templates = {row["id"]: row for row in manifest["sessions"]}
        elif current != pins:
            raise ValueError("frontier phases use different pinned inputs")
        with (root / "results.csv").open() as stream:
            selected = {row["scenario_id"]: row for row in csv.DictReader(stream)}
        scenarios = {row["scenario_id"]: row for row in plan["scenarios"]}
        if len(selected) != len(plan["scenarios"]) or set(selected) != set(scenarios):
            raise ValueError("plan and selected frontier results do not match")
        for identifier, scenario in scenarios.items():
            row = selected[identifier]
            event = target = None
            if row["status"] == "complete":
                result = json.loads((
                    root / "scenarios" / identifier
                    / f"attempt-{int(row['attempt']):04d}" / "result.json"
                ).read_text())
                if result["status"] != "complete":
                    raise ValueError("selected result is not complete")
                event, target = _episode(scenario, result, profile, templates)
            output.append({
                "phase": phase, "scenario_id": identifier,
                "condition_index": scenario["condition_index"],
                "repeat": scenario["repeat"], "policy": scenario["policy"],
                "pack": scenario["pack"],
                "movement_tokens": sum(item["initial_tokens"]
                                       for item in scenario["sessions"]),
                "mean_destination_load": np.mean(
                    [value[0] for value in scenario["background"].values()]),
                "deadline_s": scenario["deadline_s"],
                "requested_shed_w": target, "attainment_s": event,
                "attained_by_deadline": event is not None
                and event <= scenario["deadline_s"],
            })
    blocks = defaultdict(set)
    for row in output:
        blocks[row["policy"]].add((row["condition_index"], row["repeat"]))
    if set(blocks) != set(POLICIES) or len({frozenset(value)
                                           for value in blocks.values()}) != 1:
        raise ValueError("policies do not share matched operating points")
    return output


def write(run_root: Path = RUN_ROOT) -> list[dict]:
    rows = extract(run_root)
    stem = run_root / "frontier_power_attainment_cdf"
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    deadlines = {row["deadline_s"] for row in rows}
    if len(deadlines) != 1:
        raise ValueError("power-attainment CDF mixes deadlines")
    deadline = deadlines.pop()
    events = [row["attainment_s"] for row in rows
              if row["attainment_s"] is not None]
    right = max(deadline * 1.08, max(events) * 1.05)
    figure, axis = plt.subplots(figsize=(6.4, 5))
    for policy in POLICIES:
        x, y = attainment_curve(rows, policy)
        axis.step(np.r_[x, right], np.r_[y, y[-1]], where="post",
                  color=COLORS[policy], linestyle=LINESTYLES[policy],
                  linewidth=2.5, label=LABELS[policy])
    axis.axvline(deadline, color="black", linestyle="--", linewidth=1.5)
    axis.text(deadline, 1.01, f"{deadline:g} s Deadline",
              transform=axis.get_xaxis_transform(), ha="center", va="bottom")
    axis.set(xlabel="Time to 80% Modeled Power-Shed Attainment (s)",
             ylabel="Cumulative Fraction of Episodes", xlim=(0, right),
             ylim=(0, 1.02))
    axis.grid(alpha=.25)
    axis.legend(frameon=False, ncol=2, loc="upper center",
                bbox_to_anchor=(.5, -.2), fontsize=9)
    figure.text(.5, .01,
                "50 matched episodes per policy; unreached targets remain missing mass",
                ha="center", fontsize=8, color="#555555")
    figure.subplots_adjust(bottom=.3)
    for suffix in ("png", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=220,
                       bbox_inches="tight")
    plt.close(figure)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    write(parser.parse_args().run_root)


if __name__ == "__main__":
    main()
