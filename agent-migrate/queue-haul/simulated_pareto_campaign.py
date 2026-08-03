"""Run the exact, sharded full-fleet simulated Pareto sensitivity."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from destination import FluidMigrationService, dedicated_sink_architecture
from migration_profiler import file_hash, stable_seed
from planner import plan, source_power
from power_drain_experiment import build_scenario
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile
from simulate import NetworkLink, execute, step_average


ROOT = Path(__file__).parent
MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
EVIDENCE = ROOT / "outputs/policy-hardware-width8-frontier-20260730"
OUT = ROOT / "outputs/simulated-pareto-v5-20260803"
WORKLOADS = tuple(ROOT / f"profiles/{name}.json" for name in (
    "coding", "interactive_coding", "agentic_tool_loop",
))
POLICIES = (
    "queue_haul", "greedy", "isolated_fastest", "random", "replay_only", "kv_only",
)
SOLVERS = {"queue_haul": "lp_highs"}
BANDWIDTHS_MBPS = (1000, 2500, 5000, 10000)
DEADLINES_S = (30, 60, 120, 300, 900, 3600, 14400)
TARGETS = (.10, .25, .50, .75, 1.0)
ANCHORS = (1998, 4045, 8141, 16336, 31562)
SHARDS = 64
SESSIONS = 10_000
WINDOW_S = 5
SCHEMA = "queue-haul-simulated-pareto-v5"


def _csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row, key):
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"nonfinite {key}")
    return value


def fit_hardware(profile, evidence=EVIDENCE):
    scenarios = _csv(evidence / "scenarios.csv")
    stages = _csv(evidence / "migration_stages.csv")
    accepted = {
        row["scenario_id"] for row in scenarios
        if row["kind"] == "migration" and row["status"] == "complete"
        and _float(row, "bandwidth_mbps") == 10000
        and int(row["concurrency"]) == 8
    }
    replay_ids = {
        row["scenario_id"] for row in scenarios
        if row["scenario_id"] in accepted and row["method"] == "replay"
    }
    grouped = {}
    for row in stages:
        if row["scenario_id"] in replay_ids and row["method"] == "replay" \
                and row["success"].lower() == "true" and row["phase"] == "initial":
            grouped.setdefault(row["scenario_id"], []).append(row)
    case, speedups = profile.case(), []
    for rows in grouped.values():
        if len(rows) != 8:
            raise ValueError("width-8 replay fit requires eight successful migrations")
        work = sum(
            _float(row, "measured_prompt_tokens") / case.replay.rate(
                _float(row, "measured_prompt_tokens"), 1,
            ) + case.replay_completion_s for row in rows
        )
        span = (max(_float(row, "destination_ready_ns") for row in rows)
                - min(_float(row, "start_ns") for row in rows)) / 1e9
        speedups.append(work / span)
    if len(speedups) < 10:
        raise ValueError("insufficient width-8 replay evidence")
    speed = np.quantile(speedups, (.25, .5, .75))
    if np.any((speed <= 0) | (speed > 8)):
        raise ValueError("fitted replay capacity factor is outside (0, 8]")

    powers = {}
    for method in ("replay", "kv_transfer"):
        rows = [row for row in scenarios if row["scenario_id"] in accepted
                and row["method"] == method]
        if len(rows) < 10:
            raise ValueError(f"insufficient {method} action-power evidence")
        powers[method] = {
            side: np.quantile([_float(row, field) for row in rows], (.25, .5, .75)).tolist()
            for side, field in (
                ("source", "source_added_power_w"),
                ("destination", "destination_added_power_w"),
            )
        }
    cases = {}
    for name, speed_index, power_index in (
        ("conservative", 0, 2), ("central", 1, 1), ("optimistic", 2, 0),
    ):
        cases[name] = {
            "replay_speedup": float(speed[speed_index]),
            "source_power_w": {
                method: powers[method]["source"][power_index] for method in powers
            },
            "destination_power_w": {
                method: powers[method]["destination"][power_index] for method in powers
            },
        }
    return {
        "cases": cases, "replay_episodes": len(speedups),
        "source": {
            name: {"sha256": file_hash(evidence / name)}
            for name in ("scenarios.csv", "migration_stages.csv")
        },
    }


def manifest_rows():
    episodes = [
        {"episode_id": f"{path.stem}-seed-{seed}", "kind": "trace",
         "workload": str(path.relative_to(ROOT)), "seed": seed}
        for path in WORKLOADS for seed in range(3)
    ] + [
        {"episode_id": f"anchor-{context}", "kind": "anchor",
         "anchor_tokens": context, "seed": 0}
        for context in ANCHORS
    ]
    rows = []
    for episode in episodes:
        for bandwidth in BANDWIDTHS_MBPS:
            for deadline in DEADLINES_S:
                for target in TARGETS:
                    for policy in POLICIES:
                        rows.append({**episode, "case": "central",
                                     "bandwidth_mbps": bandwidth,
                                     "deadline_s": deadline,
                                     "target_fraction": target, "policy": policy})
    for path in WORKLOADS:
        episode = {"episode_id": f"{path.stem}-seed-0", "kind": "trace",
                   "workload": str(path.relative_to(ROOT)), "seed": 0}
        for case in ("conservative", "optimistic"):
            for bandwidth in (1000, 10000):
                for deadline in (60, 300, 3600, 14400):
                    for target in (.25, 1.0):
                        for policy in POLICIES:
                            rows.append({**episode, "case": case,
                                         "bandwidth_mbps": bandwidth,
                                         "deadline_s": deadline,
                                         "target_fraction": target, "policy": policy})
    for index, row in enumerate(rows):
        row["row_id"] = f"v5-{index:05d}"
        row["shard"] = index % SHARDS
    if len(rows) != 12_336:
        raise RuntimeError("unexpected campaign grid size")
    return rows


def prepare(out=OUT, model_path=MODEL):
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    profile = ModelProfile.load(model_path)
    rows = manifest_rows()
    git_sha = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ("git", "status", "--porcelain"), cwd=ROOT, check=True,
        capture_output=True, text=True,
    ).stdout)
    manifest = {
        "schema": SCHEMA, "sessions": SESSIONS, "shards": SHARDS,
        "rows": rows, "fits": fit_hardware(profile),
        "model": {"path": str(model_path.relative_to(ROOT)),
                  "sha256": file_hash(model_path)},
        "workloads": {
            str(path.relative_to(ROOT)): {"sha256": file_hash(path)}
            for path in WORKLOADS
        },
        "anchor_fit": {
            "contexts": list(ANCHORS),
            "records": "uniform sample over all three trace record sets",
            "log_bytes": "per-record trace log-bytes/context ratio",
        },
        "source": {"git_sha": git_sha, "dirty": dirty},
        "assumptions": {
            "workload": "exact idle trace snapshot; no arrivals or growth",
            "wan": "fixed site aggregate",
            "replay": "fully divisible destination-fleet fluid pool",
            "kv": "all bytes consume WAN and per-replica ingest",
            "destination": "matched idle fleet; free intra-site relocation",
            "destination_spare_fraction": 1.0,
            "evidence": "simulated sensitivity",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _profile_case(profile, case):
    return replace(profile, profile_id=f"{profile.profile_id}-{case}",
                   cases={"central": profile.case("central")})


def _workload(row):
    if row["kind"] == "trace":
        return WorkloadProfile.load(ROOT / row["workload"])
    sources = [WorkloadProfile.load(path) for path in WORKLOADS]
    base = sources[0]
    records = tuple(replace(
        record, context_tokens=row["anchor_tokens"],
        log_bytes=max(1, round(
            row["anchor_tokens"] * record.log_bytes / record.context_tokens
        )),
    ) for workload in sources for record in workload.records)
    return replace(base, profile_id=row["episode_id"], records=records)


def _architecture(profile, scenario, fit):
    destinations = tuple(
        instance.instance_id for instance in scenario.instances
        if instance.instance_id.startswith("dest-")
    )
    architecture = dedicated_sink_architecture(profile, destinations, ("wan",))
    service = FluidMigrationService(
        fit["replay_speedup"], profile.case().kv_transfer.destination_bytes_per_s,
        fit["source_power_w"], fit["destination_power_w"],
        "width8-10g-successful-replay-and-action-power",
    )
    return replace(architecture, pools=(replace(
        architecture.pools[0], fluid_migration=service,
    ),))


def attainment_time(power, target, window, end):
    if end < window:
        return None
    points = sorted({window, end} | {
        value for time, *_ in power for value in (time, time + window)
        if window <= value <= end
    })
    prior, prior_value = points[0], step_average(power, points[0], window)
    if prior_value <= target:
        return prior
    for current in points[1:]:
        value = step_average(power, current, window)
        if value <= target:
            lo, hi = prior, current
            for _ in range(60):
                mid = (lo + hi) / 2
                if step_average(power, mid, window) <= target:
                    hi = mid
                else:
                    lo = mid
            return hi
        prior, prior_value = current, value
    return None


def run_row(row, manifest):
    base = ModelProfile.load(ROOT / manifest["model"]["path"])
    if file_hash(ROOT / manifest["model"]["path"]) != manifest["model"]["sha256"]:
        raise RuntimeError("model changed after prepare")
    if any(file_hash(ROOT / path) != record["sha256"]
           for path, record in manifest["workloads"].items()):
        raise RuntimeError("workload changed after prepare")
    profile = _profile_case(base, row["case"])
    bandwidth = row["bandwidth_mbps"] * 125_000
    scenario, _ = build_scenario(
        _workload(row), profile, manifest["sessions"], row["seed"], 0,
        row["deadline_s"], row["deadline_s"], bandwidth,
        idle_snapshot=True,
    )
    scenario = replace(scenario, links=(NetworkLink("wan", bandwidth),))
    power = ExpectedPower(scenario, profile)
    initial = power.power(True)
    idle = source_power(
        scenario, profile, [session.session_id for session in scenario.sessions],
    )
    removable = initial - idle
    limit = initial - row["target_fraction"] * removable
    scenario = replace(scenario, power_limit_w=limit)
    architecture = _architecture(
        profile, scenario, manifest["fits"]["cases"][row["case"]],
    )
    seed = stable_seed(row["episode_id"], row["bandwidth_mbps"],
                       row["deadline_s"], row["target_fraction"], row["policy"],
                       row["case"])
    planned = plan(
        scenario, profile, {}, SOLVERS.get(row["policy"], row["policy"]),
        seed=seed, destination=architecture,
    )
    result = execute(scenario, profile, planned.moves, destination=architecture)
    at_deadline = step_average(result.power, row["deadline_s"], WINDOW_S)
    attained = float(np.clip((initial - at_deadline) / removable, 0, 1))
    attained_at = attainment_time(result.power, limit, WINDOW_S, row["deadline_s"])
    commits = [item.committed_s for item in result.sessions if item.committed_s is not None]
    return {
        **row, "planner_seed": seed, "sessions": len(scenario.sessions),
        "source_replicas": sum(instance.instance_id.startswith("source-")
                               for instance in scenario.instances),
        "destination_replicas": sum(instance.instance_id.startswith("dest-")
                                    for instance in scenario.instances),
        "initial_source_power_w": initial, "idle_source_power_w": idle,
        "target_power_w": limit, "power_at_deadline_w": at_deadline,
        "attained_shed_fraction": attained,
        "steady_state_shed_fraction": (initial - planned.planned_source_power_w) / removable,
        "target_attainment_s": attained_at if attained_at is not None else "",
        "target_attained": attained_at is not None,
        "censored": attained_at is None,
        "last_commit_s": max(commits) if commits else "",
        "admitted_moves": len(planned.moves), "committed_moves": len(commits),
        "replay_moves": sum(move.method == "replay" for move in planned.moves),
        "kv_moves": sum(move.method == "kv_transfer" for move in planned.moves),
        "planner_feasible": planned.feasible,
        "packing_repair_count": planned.packing_repair_count,
        "packing_repair_s": planned.packing_repair_s,
        "deadline_repair_count": planned.deadline_repair_count,
        "deadline_repair_s": planned.deadline_repair_s,
        "result_evidence": "simulated_sensitivity",
    }


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_shard(out=OUT, shard=0):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA or not 0 <= shard < manifest["shards"]:
        raise ValueError("invalid manifest or shard")
    rows = [run_row(row, manifest) for row in manifest["rows"]
            if row["shard"] == shard]
    _write_csv(out / f"shard-{shard:02d}.csv", rows)
    return rows


def pareto_flags(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["episode_id"], row["bandwidth_mbps"], row["case"]), []).append(row)
    for peers in groups.values():
        for row in peers:
            row["pareto"] = row["target_attained"] and not any(
                other["target_attained"] and (
                    not row["target_attained"]
                    or float(other["attained_shed_fraction"])
                    >= float(row["attained_shed_fraction"])
                    and float(other["target_attainment_s"])
                    <= float(row["target_attainment_s"])
                    and (float(other["attained_shed_fraction"])
                         > float(row["attained_shed_fraction"])
                         or float(other["target_attainment_s"])
                         < float(row["target_attainment_s"]))
                ) for other in peers if other is not row
            )


def _parse_rows(path):
    rows = _csv(path)
    for row in rows:
        for key in ("bandwidth_mbps", "deadline_s", "shard", "sessions"):
            row[key] = int(float(row[key]))
        for key in ("target_fraction", "attained_shed_fraction"):
            row[key] = float(row[key])
        for key in ("target_attained", "censored"):
            row[key] = row[key].lower() == "true"
    return rows


def _plot(rows, out, kind):
    selected = [row for row in rows if row["kind"] == kind and row["case"] == "central"]
    names = sorted({row["episode_id"].split("-seed-")[0] for row in selected})
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 3 * len(names)), squeeze=False)
    for axis_row, name in zip(axes, names):
        members = [row for row in selected if row["episode_id"].startswith(name)]
        for axis, regime in zip(axis_row, ((0, 300), (900, 14400))):
            for policy in POLICIES:
                points = [row for row in members if row["policy"] == policy
                          and regime[0] <= row["deadline_s"] <= regime[1]
                          and row["target_attained"]]
                axis.scatter([100 * row["attained_shed_fraction"] for row in points],
                             [float(row["target_attainment_s"]) for row in points],
                             s=8, alpha=.45, label=policy)
            axis.set(title=f"{name}: {regime[0]}–{regime[1]} s",
                     xlabel="attained shed (%)", ylabel="target attainment (s)")
            axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"pareto-{kind}.{suffix}", dpi=200)
    plt.close(fig)


def reduce(out=OUT):
    manifest = json.loads((out / "manifest.json").read_text())
    paths = [out / f"shard-{shard:02d}.csv" for shard in range(manifest["shards"])]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing shards: {', '.join(missing)}")
    rows = [row for path in paths for row in _parse_rows(path)]
    expected = {row["row_id"] for row in manifest["rows"]}
    actual = [row["row_id"] for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("shards contain duplicate or unexpected rows")
    expected_rows = {row["row_id"]: row for row in manifest["rows"]}
    for path, shard in zip(paths, range(manifest["shards"])):
        for row in _parse_rows(path):
            expected_row = expected_rows[row["row_id"]]
            if row["shard"] != shard or any(
                key not in row or str(row[key]) != str(value)
                for key, value in expected_row.items()
            ):
                raise ValueError("shard row disagrees with manifest")
    pareto_flags(rows)
    rows.sort(key=lambda row: row["row_id"])
    _write_csv(out / "pareto.csv", rows)
    summary = [{
        "policy": policy,
        "rows": len(selected := [row for row in rows if row["policy"] == policy]),
        "target_attained_fraction": float(np.mean([row["target_attained"] for row in selected])),
        "median_attained_shed_fraction": float(np.median([
            row["attained_shed_fraction"] for row in selected
        ])),
        "pareto_fraction": float(np.mean([row["pareto"] for row in selected])),
    } for policy in POLICIES]
    _write_csv(out / "policy_summary.csv", summary)
    _plot(rows, out, "trace")
    _plot(rows, out, "anchor")
    metadata = {key: manifest[key] for key in (
        "schema", "sessions", "shards", "fits", "model", "workloads",
        "anchor_fit", "source", "assumptions",
    )}
    metadata.update(rows=len(rows), evidence_label="simulated sensitivity",
                    power_metric="first trailing 5-second target window",
                    completion_metric="last commit reported separately",
                    censoring="deadline misses remain denominators and cannot dominate successes")
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = sorted(path for path in out.iterdir() if path.name != "SHA256SUMS")
    (out / "SHA256SUMS").write_text("".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
        for path in files
    ))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "reduce"):
        command = sub.add_parser(name)
        command.add_argument("--out", type=Path, default=OUT)
    command = sub.add_parser("run-shard")
    command.add_argument("--out", type=Path, default=OUT)
    command.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.out)
    elif args.command == "run-shard":
        run_shard(args.out, args.shard)
    else:
        reduce(args.out)


if __name__ == "__main__":
    main()
