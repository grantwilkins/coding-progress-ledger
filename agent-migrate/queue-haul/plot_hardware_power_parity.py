"""Plot planned versus directly measured source-power shed."""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style
from profiles import ModelProfile


ROOT = Path(__file__).parent
SCRATCH = Path("/scratch/users") / getpass.getuser()
CONTEMPORANEOUS = tuple(
    SCRATCH / f"qh-policy-width8-packing-contemporaneous-shard{i}-20260811"
    for i in range(2)
)
FIXED = SCRATCH / "qh-policy-width8-packing-20260730"
OUTPUT = ROOT / "outputs/policy-hardware-width8-packing-contemporaneous-20260811/policy_hardware_power_parity"
METHODS = (
    "queue_haul", "greedy", "isolated_fastest", "kv_only", "replay_only",
    "queue_haul_power_blind", "queue_haul_deadline_blind",
)
PLOT_METHODS = ("queue_haul", "greedy")
CONTEMPORANEOUS_METHODS = set(METHODS) - {"kv_only", "replay_only"}
MARKERS = dict(zip(METHODS, "os^vPDX"))
NS = 10**9
WINDOW_NS = SETTLE_NS = NS
plot_style.apply()


def _mean(rows: list[dict], start: int, end: int) -> float:
    if end <= start or not any(row["monotonic_ns"] <= start for row in rows) \
            or not any(row["monotonic_ns"] >= end for row in rows):
        raise RuntimeError("power samples do not cover settled window")
    value = next(row["power_w"] for row in reversed(rows)
                 if row["monotonic_ns"] <= start)
    area, cursor = 0, start
    for row in rows:
        sample = row["monotonic_ns"]
        if sample <= start:
            continue
        stop = min(sample, end)
        area += (stop - cursor) * value
        if sample >= end:
            return area / (end - start)
        cursor, value = sample, row["power_w"]
    raise RuntimeError("power samples do not cover settled window")


def source_power_shed(path: Path, result: dict,
                      pre_guard_ns=SETTLE_NS) -> tuple[float, float, float]:
    with path.open() as handle:
        rows = [{"monotonic_ns": int(row["monotonic_ns"]),
                 "gpu": int(row["gpu"]), "power_w": float(row["power_w"])}
                for row in csv.DictReader(handle) if row["valid"] == "1"]
    gpus = sorted({row["gpu"] for row in rows})
    if gpus != [0, 1]:
        raise RuntimeError(f"expected two measured GPUs in {path}, found {gpus}")
    source = sorted((row for row in rows if row["gpu"] == 0),
                    key=lambda row: row["monotonic_ns"])
    migrations = result["migrations"]
    start = min(row["initial_start_ns"] for row in migrations)
    switched = max(row["switch_end_ns"] for row in migrations)
    before = _mean(source, start - pre_guard_ns - WINDOW_NS,
                   start - pre_guard_ns)
    after = _mean(source, switched + SETTLE_NS,
                  switched + SETTLE_NS + WINDOW_NS)
    return before, after, before - after


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predicted_shed(scenario: dict, curve) -> float:
    admitted = sum(row["deadline_admitted"] for row in scenario["moves"])
    remaining = .4 * (1 - admitted / len(scenario["sessions"]))
    return curve.power(.4) - curve.power(remaining)


def _last_source_request(path: Path, port: int, before: int) -> int:
    events = (json.loads(line) for line in path.open())
    return max(row["monotonic_ns"] for row in events
               if row.get("event") == "request_end"
               and row.get("route_port") == port
               and row["monotonic_ns"] <= before)


def settled_pre_available(last_request: int, migration_start: int) -> bool:
    return migration_start - last_request >= SETTLE_NS + WINDOW_NS


def load_points(contemporaneous=CONTEMPORANEOUS, fixed: Path = FIXED,
                audit=False, raw_delta=False) -> list[dict]:
    sources = [(Path(root), CONTEMPORANEOUS_METHODS)
               for root in contemporaneous] + [(Path(fixed), {"kv_only", "replay_only"})]
    plans = [(root, json.loads((root / "plan.json").read_text()), allowed)
             for root, allowed in sources]
    profiles = {(plan["model_profile"]["path"], plan["model_profile"]["sha256"])
                for _, plan, _ in plans}
    if len(profiles) != 1:
        raise RuntimeError("cohort plans do not share one model profile")
    profile_name, profile_hash = profiles.pop()
    profile_path = ROOT.parent / profile_name
    if _hash(profile_path) != profile_hash:
        raise RuntimeError("cohort model profile changed after planning")
    curve = ModelProfile.load(profile_path).case().power_curve
    full_shed = curve.power(.4) - curve.power(0)
    rows, matches, seen = [], {method: set() for method in METHODS}, set()
    for root, plan, allowed in plans:
        source_port = json.loads((root / "run_metadata.json").read_text())[
            "config"]["src_port"]
        for scenario in plan["scenarios"]:
            method = scenario["policy"]
            if method not in allowed:
                continue
            key = method, scenario["match_id"]
            if key in seen:
                raise RuntimeError(f"duplicate cohort arm {key}")
            seen.add(key)
            scenario_root = root / "scenarios" / scenario["scenario_id"]
            result = json.loads((scenario_root / "result.json").read_text())
            if result.get("status") != "complete" \
                    or len(result.get("migrations", ())) != len(scenario["moves"]):
                raise RuntimeError(f"incomplete scenario {scenario['scenario_id']}")
            start = min(row["initial_start_ns"] for row in result["migrations"])
            switched = max(row["switch_end_ns"] for row in result["migrations"])
            last_request = _last_source_request(
                scenario_root / "events.jsonl", source_port, start)
            gap = start - last_request
            valid = settled_pre_available(last_request, start)
            if not audit and not raw_delta and not valid:
                raise RuntimeError(
                    f"no settled pre-migration power window for {scenario['scenario_id']}"
                )
            before, after, measured = (None, None, None) if audit else \
                source_power_shed(
                    scenario_root / "power.csv", result,
                    0 if raw_delta else SETTLE_NS,
                )
            matches[method].add(scenario["match_id"])
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "match_id": scenario["match_id"], "condition": scenario["condition"],
                "method": method, "repeat": scenario.get("repeat", scenario["episode"]),
                "source_gpu": 0, "pre_window_power_w": before,
                "post_window_power_w": after,
                "pre_settle_gap_s": gap / NS,
                "post_trace_after_switch_s": (result["ended_ns"] - switched) / NS,
                "valid_settled_windows": valid,
                "measurement_kind": "none" if audit else
                    "warmup_contaminated_immediate_pre" if raw_delta else "settled",
                "requested_shed_w": scenario["power_target_fraction"] * full_shed,
                "predicted_shed_w": predicted_shed(scenario, curve),
                "measured_shed_w": measured,
            })
    if not matches[METHODS[0]] or any(value != matches[METHODS[0]]
                                      for value in matches.values()):
        raise RuntimeError("policy arms do not form one complete matched cohort")
    return rows


def normalize(rows: list[dict]) -> tuple[list[dict], float]:
    if not rows or (scale := max(float(row["requested_shed_w"]) for row in rows)) <= 0:
        raise ValueError("power validation requires a positive request")
    return [{**row,
             "predicted_percent": 100 * float(row["predicted_shed_w"]) / scale,
             "measured_percent": 100 * float(row["measured_shed_w"]) / scale}
            for row in rows], scale


def write_csv(rows: list[dict], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: list[dict], _scale: float, out: Path) -> None:
    plotted = [row for row in rows if row["method"] in PLOT_METHODS]
    values = [row[key] for row in plotted
              for key in ("predicted_percent", "measured_percent")]
    lower, upper = min(-5, min(values)), max(105, max(values))
    padding = .03 * (upper - lower)
    limits = lower - padding, upper + padding
    fig, axis = plt.subplots(figsize=plot_style.FIGSIZE)
    axis.plot(limits, limits, color="black", linestyle="--", linewidth=1.5,
              zorder=1)
    for method in PLOT_METHODS:
        selected = [row for row in plotted if row["method"] == method]
        axis.scatter(
            [row["predicted_percent"] for row in selected],
            [row["measured_percent"] for row in selected],
            color=plot_style.POLICY_COLORS[method], marker=MARKERS[method],
            s=32, alpha=.45, linewidths=.8,
            label=plot_style.POLICY_NAMES[method], zorder=2,
        )
    axis.text(.03, .95, "Overshed", transform=axis.transAxes, va="top")
    axis.text(.97, .05, "Undershed", transform=axis.transAxes, ha="right")
    axis.set(xlabel="Phase-aware predicted shed (% of max prediction)",
             ylabel="Measured shed (% of max prediction)",
             xlim=limits, ylim=limits)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False,
               loc="center left", bbox_to_anchor=(.66, .54))
    fig.tight_layout(rect=(0, 0, .65, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    write_csv(rows, out.with_suffix(".csv"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contemporaneous-root", type=Path, action="append")
    parser.add_argument("--fixed-root", type=Path, default=FIXED)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--csv-only", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true")
    mode.add_argument("--raw-delta", action="store_true")
    args = parser.parse_args()
    roots = args.contemporaneous_root or CONTEMPORANEOUS
    rows = load_points(roots, args.fixed_root, args.audit_only, args.raw_delta)
    if args.audit_only:
        write_csv(rows, args.out.with_suffix(".csv"))
        return
    rows, scale = normalize(rows)
    if args.csv_only or args.raw_delta:
        write_csv(rows, args.out.with_suffix(".csv"))
    else:
        write_plot(rows, scale, args.out)


if __name__ == "__main__":
    main()
