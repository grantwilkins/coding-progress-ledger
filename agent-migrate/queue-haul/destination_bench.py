"""Trace-driven two-site destination pressure sensitivity bench."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from destination import (
    DESTINATION_SCHEMA,
    CompatibilityFingerprint,
    ContextRate,
    DestinationArchitecture,
    DestinationPool,
    DestinationReplica,
    DestinationType,
    LoadedCoefficients,
    MigrationComponents,
)
from planner import InstanceCapacity, plan, source_power
from profiles import ModelProfile, RateCurve, WorkloadProfile
from simulate import ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimSession


ROOT = Path(__file__).parent
DEFAULT_MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
DEFAULT_MANIFEST = ROOT / "outputs/destination-v7-20260722/content-free-manifest.json"
WORKLOADS = {
    "interactive_coding": ROOT / "profiles/interactive_coding.json",
    "coding": ROOT / "profiles/coding.json",
    "agentic_tool_loop": ROOT / "profiles/agentic_tool_loop.json",
}
CLASSES = tuple(WORKLOADS)
REFERENCE_RATE = 1 / 180
SERVICE_BOUND = .096953
KV_BLOCK_TOKENS = 16
KV_CAPACITY_TOKENS = 963_152
MIGRATION_CONTEXT = (16_384.0, 24_576.0)
MIGRATION_BANDWIDTH = (625_000_000.0, 1_250_000_000.0)


@dataclass(frozen=True)
class Shape:
    session_id: str
    turn: int
    context_tokens: int
    prompt_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class Pressure:
    arrival: float = 1
    service: float = 0
    kv: float = 0
    bandwidth_gbps: float = 10
    migration_s: float = 115


def trace_shapes(manifest: dict, job_class: str) -> tuple[Shape, ...]:
    ids = set(sum(manifest["manifest"]["splits"][job_class].values(), []))
    shapes = []
    for row in manifest["traces"]:
        context = int(row["input_tokens_total"]) - int(row["newly_append_tokens"])
        prompt, output = int(row["newly_append_tokens"]), int(row["output_tokens"])
        if row["session_id"] in ids and 3_473 <= context \
                and context + prompt + output <= 31_562:
            shapes.append(Shape(row["session_id"], int(row["turn"]), context, prompt, output))
    if not shapes:
        raise ValueError(f"no supported {job_class} trace shapes")
    return tuple(shapes)


def log_bytes_per_token(path: Path) -> float:
    records = WorkloadProfile.load(path).records
    return float(np.median([row.log_bytes / row.context_tokens for row in records]))


def code_sha256() -> str:
    return hashlib.sha256(b"".join(
        (ROOT / name).read_bytes()
        for name in ("destination.py", "pool_planner.py", "destination_bench.py")
    )).hexdigest()


def sample_sessions(shapes: tuple[Shape, ...], count: int, seed: int,
                    log_ratio: float) -> tuple[SimSession, ...]:
    if count < 1:
        raise ValueError("session count must be positive")
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(shapes), count)
    return tuple(
        SimSession(
            str(i), "unassigned", shape.context_tokens,
            shape.prompt_tokens * REFERENCE_RATE,
            shape.output_tokens * REFERENCE_RATE,
            max(1, math.ceil(shape.context_tokens * log_ratio)),
            expected_growth_tokens_per_s=(
                shape.prompt_tokens + shape.output_tokens
            ) * REFERENCE_RATE,
        )
        for i, shape in enumerate(shapes[j] for j in selected)
    )


def pack_source(sessions: tuple[SimSession, ...], profile: ModelProfile,
                horizon_s: float = 180) -> tuple[tuple[SimSession, ...], int]:
    case = profile.case()
    loads = [s.expected_f / case.F + s.expected_g / case.G for s in sessions]
    blocks = [
        math.ceil((s.context_tokens + s.expected_growth_tokens_per_s * horizon_s)
                  / KV_BLOCK_TOKENS)
        for s in sessions
    ]
    capacity = InstanceCapacity(
        [], [], profile.max_ell, KV_CAPACITY_TOKENS // KV_BLOCK_TOKENS,
    )
    assignment = np.empty(len(sessions), int)
    order = np.argsort(-np.maximum(
        np.asarray(loads) / profile.max_ell,
        np.asarray(blocks) / (KV_CAPACITY_TOKENS // KV_BLOCK_TOKENS),
    ), kind="stable")
    for i in order:
        if loads[i] > profile.max_ell or blocks[i] > capacity.max_tokens:
            raise ValueError("one session exceeds source capacity")
        assignment[i] = capacity.place(loads[i], blocks[i], grow=True)
    return tuple(
        replace(session, source_instance=f"source-{assignment[i]}")
        for i, session in enumerate(sessions)
    ), len(capacity.loads)


def _fingerprint(profile: ModelProfile) -> CompatibilityFingerprint:
    return CompatibilityFingerprint(profile.model, "gpt-oss-pinned", "source-dc-log",
                                    "lmcache-mp-v7")


def _rate(curve) -> ContextRate:
    x, y = curve.by_concurrency[1]
    return ContextRate(tuple(map(float, x)), tuple(map(float, y)))


def architecture(profile: ModelProfile, sessions: tuple[SimSession, ...], replicas: int,
                 pressure: Pressure, methods=("replay", "kv_transfer")):
    case, fp = profile.case(), _fingerprint(profile)
    loaded = LoadedCoefficients(
        (0, 1), (1, 1), MIGRATION_CONTEXT, MIGRATION_BANDWIDTH, "legacy-adapter-unused",
    )
    migration = {
        "replay": MigrationComponents(
            MIGRATION_CONTEXT, MIGRATION_BANDWIDTH, "FINDINGS.md:v7-replay",
            compute_completion_factor=.586660,
        ),
        "kv_transfer": MigrationComponents(
            MIGRATION_CONTEXT, MIGRATION_BANDWIDTH, "FINDINGS.md:v7-kv",
            residual_s=1.133822,
        ),
    }
    demand = sum((
        np.array((s.expected_f / case.prefill.rate(s.context_tokens, 1),
                  s.expected_g / case.decode.rate(s.context_tokens, 1)))
        for s in sessions
    ), start=np.zeros(2))
    direction = demand / demand.sum() if demand.sum() else np.array((.5, .5))
    baseline = tuple(SERVICE_BOUND * pressure.service * direction)
    replicas_ = tuple(
        DestinationReplica(
            f"dest-{i}", baseline,
            math.floor(KV_CAPACITY_TOKENS / KV_BLOCK_TOKENS * pressure.kv)
            * KV_BLOCK_TOKENS,
        ) for i in range(replicas)
    )
    q = DestinationType(
        "gpt-oss-20b-a100-tp1", fp, _rate(case.prefill), _rate(case.decode),
        ((1, 1),), {mode: (SERVICE_BOUND,) for mode in
                     ("normal", "emergency", "stable")},
        KV_CAPACITY_TOKENS, {"replay": loaded, "kv_transfer": loaded}, (0, 1),
        "descriptive-private-prefix-anchor:0.096953", False, KV_BLOCK_TOKENS,
        migration, "sensitivity",
    )
    pool = DestinationPool(
        "sink-a100", q.type_id, replicas_, "source-to-sink",
        ("source-egress", "wan", "destination-ingress"), tuple(methods),
    )
    return DestinationArchitecture(
        DESTINATION_SCHEMA, fp, (q,), (pool,), max(180, pressure.migration_s + 5),
    )


def scenario(profile: ModelProfile, base_sessions: tuple[SimSession, ...], replicas: int,
             pressure: Pressure) -> ExecutionScenario:
    sessions = []
    for s in base_sessions:
        context = math.ceil(
            s.context_tokens + s.expected_growth_tokens_per_s * pressure.arrival * 180
        )
        sessions.append(replace(
            s, context_tokens=context,
            expected_f=s.expected_f * pressure.arrival,
            expected_g=s.expected_g * pressure.arrival,
            log_bytes=math.ceil(s.log_bytes * context / s.context_tokens),
            expected_growth_tokens_per_s=0,
        ))
    sessions = tuple(sessions)
    node_count = math.ceil(replicas / profile.gpus_per_node)
    nodes = tuple(
        PowerNode(f"source-node-{i}", profile.gpus_per_node, True, "source")
        for i in range(node_count)
    ) + tuple(
        PowerNode(f"dest-node-{i}", profile.gpus_per_node, False, "sink")
        for i in range(node_count)
    )
    instances = tuple(
        ServingInstance(f"source-{i}", (f"source-node-{i // profile.gpus_per_node}",))
        for i in range(replicas)
    ) + tuple(
        ServingInstance(f"dest-{i}", (f"dest-node-{i // profile.gpus_per_node}",))
        for i in range(replicas)
    )
    wan = pressure.bandwidth_gbps * 125_000_000
    links = (
        NetworkLink("source-egress", wan),
        NetworkLink("wan", wan),
        NetworkLink("destination-ingress", wan),
    )
    result = ExecutionScenario(
        pressure.migration_s + profile.power_window_s,
        max(180, pressure.migration_s + profile.power_window_s),
        0, "awake", 0, nodes, instances, sessions, links,
    )
    minimum = source_power(result, profile, (s.session_id for s in sessions))
    return replace(result, power_limit_w=minimum)


def evidence(architecture_: DestinationArchitecture, sessions: tuple[SimSession, ...],
             moves, pressure: Pressure) -> tuple[str, str, float]:
    q = architecture_.types[0]
    components = q.migration or {}
    reasons = set()
    inside = 0
    by_id = {s.session_id: s for s in sessions}
    for move in moves:
        session = by_id[move.session_id]
        context = session.context_tokens + session.expected_growth_tokens_per_s \
            * pressure.migration_s
        bandwidth = pressure.bandwidth_gbps * 125_000_000
        outside = components[move.method].extrapolates(context, bandwidth)
        reasons.update(outside)
        inside += not outside
    fraction = inside / len(moves) if moves else 0
    return (
        "unsupported_extrapolation" if reasons else "sensitivity",
        ",".join(sorted(reasons)), fraction,
    )


def extrapolate_replay(profile: ModelProfile,
                       sessions: tuple[SimSession, ...], horizon: float) -> ModelProfile:
    context = math.ceil(max(
        s.context_tokens + s.expected_growth_tokens_per_s * horizon for s in sessions
    ))
    cases = {}
    for case_id, case in profile.cases.items():
        curves = {}
        for concurrency, (x, y) in case.replay.by_concurrency.items():
            curves[concurrency] = (
                np.append(x, context), np.append(y, min(y))
            ) if context > x[-1] else (x, y)
        cases[case_id] = replace(case, replay=RateCurve(curves))
    return replace(profile, cases=cases)


def evaluate(profile: ModelProfile, sessions: tuple[SimSession, ...], replicas: int,
             pressure: Pressure, seed: int, job_class: str, axis: str,
             solver="lp", methods=("replay", "kv_transfer")) -> dict:
    scenario_ = scenario(profile, sessions, replicas, pressure)
    execution_profile = extrapolate_replay(
        profile, scenario_.sessions, scenario_.deadline_s,
    )
    architecture_ = architecture(
        execution_profile, scenario_.sessions, replicas, pressure, methods,
    )
    result = plan(
        scenario_, execution_profile, {}, solver, seed=seed,
        destination=architecture_,
    )
    status, reasons, in_domain = evidence(
        architecture_, scenario_.sessions, result.moves, pressure,
    )
    counts = {method: sum(move.method == method for move in result.moves)
              for method in ("replay", "kv_transfer")}
    return {
        "workload": job_class, "seed": seed, "axis": axis, "solver": solver,
        "methods": "+".join(methods), **pressure.__dict__,
        "source_replicas": replicas, "sink_replicas": replicas,
        "sessions": len(sessions), "sessions_landed": len(result.moves),
        "all_sessions_landed": result.feasible and len(result.moves) == len(sessions),
        "initial_source_w": result.initial_source_power_w,
        "source_w_shed": result.initial_source_power_w - result.planned_source_power_w,
        "power_shortfall_w": result.power_shortfall_w,
        "replay_moves": counts["replay"], "kv_moves": counts["kv_transfer"],
        "mode": result.admission_mode, "failure": result.failure_reason,
        "bottleneck": result.bottleneck,
        "migration_makespan_s": result.predicted_migration_makespan_s,
        "packing_repairs": result.packing_repair_count, "runtime_s": result.solve_s,
        "evidence_status": status, "extrapolation_reasons": reasons,
        "in_domain_fraction": in_domain,
    }


def boundary(run, low: float, high: float, easier_high: bool,
             relative=False, iterations=8) -> tuple[float, str]:
    left, right = run(low), run(high)
    if left == right:
        return (low if left == easier_high else high), "censored"
    for _ in range(iterations):
        middle = math.sqrt(low * high) if relative else (low + high) / 2
        value = run(middle)
        if value == easier_high:
            high = middle
        else:
            low = middle
        if relative and high / low <= 1.05 or not relative and high - low <= .025:
            break
    return (high if easier_high else low), "crossing"


def pressure_search(profile, sessions, replicas, seed, job_class, iterations=8):
    rows, cache = [], {}

    def run(axis, value, base, easier_high, relative=False):
        pressure = replace(base, **{axis: value})
        key = (pressure, "lp", ("replay", "kv_transfer"))
        if key not in cache:
            cache[key] = evaluate(
                profile, sessions, replicas, pressure, seed, job_class, axis,
            )
            rows.append(cache[key])
        return cache[key]["all_sessions_landed"]

    open_ = Pressure(bandwidth_gbps=1000, migration_s=3600)
    value, state = boundary(
        lambda x: run("arrival", x, open_, False),
        .01, 1, False, iterations=iterations,
    )
    thresholds = {"arrival": {"value": value, "state": state}}
    working_arrival = max(.01, value / 2)
    definitions = {
        "service": (0.0, .99, False, False,
                    replace(open_, arrival=working_arrival)),
        "kv": (0.0, .99, False, False,
               replace(open_, arrival=working_arrival)),
        "bandwidth_gbps": (.1, 1000.0, True, True,
                           Pressure(working_arrival, migration_s=3600)),
        "migration_s": (1.0, 3600.0, True, True,
                        Pressure(working_arrival, bandwidth_gbps=1000)),
    }
    for axis, (low, high, easier_high, relative, base) in definitions.items():
        value, state = boundary(
            lambda x, a=axis, b=base, e=easier_high, r=relative:
            run(a, x, b, e, r),
            low, high, easier_high, relative, iterations,
        )
        thresholds[axis] = {"value": value, "state": state}

    reference = Pressure()
    for solver in ("lp", "greedy"):
        rows.append(evaluate(
            profile, sessions, replicas, reference, seed, job_class,
            "reference", solver,
        ))
    for axis, item in thresholds.items():
        base = open_ if axis == "arrival" else definitions[axis][4]
        pressure = replace(base, **{axis: item["value"]})
        for methods in (("replay",), ("kv_transfer",)):
            rows.append(evaluate(
                profile, sessions, replicas, pressure, seed, job_class,
                f"{axis}_control", "lp", methods,
            ))
    for service in np.clip(
        np.asarray((.8, 1, 1.25)) * thresholds["service"]["value"], 0, .99
    ):
        for kv in np.clip(
            np.asarray((.8, 1, 1.25)) * thresholds["kv"]["value"], 0, .99
        ):
            rows.append(evaluate(
                profile, sessions, replicas,
                replace(open_, arrival=working_arrival,
                        service=float(service), kv=float(kv)),
                seed, job_class, "service_x_kv",
            ))
    for bandwidth in np.asarray((.8, 1, 1.25)) * thresholds["bandwidth_gbps"]["value"]:
        for migration in np.asarray((.8, 1, 1.25)) * thresholds["migration_s"]["value"]:
            rows.append(evaluate(
                profile, sessions, replicas,
                Pressure(working_arrival, bandwidth_gbps=float(bandwidth),
                         migration_s=float(migration)),
                seed, job_class, "bandwidth_x_migration",
            ))
    return rows, thresholds


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot_thresholds(rows: list[dict], output: Path):
    thresholds = [row for row in rows if "threshold" in row]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)) if thresholds \
        else plt.subplots(figsize=(6, 4))
    axes = np.atleast_1d(axes)
    if thresholds:
        axes[0].bar(
            np.arange(len(thresholds)),
            [row["threshold"] for row in thresholds],
        )
        axes[0].set_xticks(
            np.arange(len(thresholds)),
            [f"{row['workload']}\n{row['axis']}" for row in thresholds],
            rotation=45, ha="right", fontsize=7,
        )
        axes[0].set_ylabel("transition value")
    reference = [row for row in rows if row.get("axis") == "reference"
                 and row.get("solver") == "lp"]
    axes[-1].bar(
        [row["workload"] for row in reference],
        [row["sessions_landed"] for row in reference],
    )
    axes[-1].axhline(reference[0]["sessions"] if reference else 0, color="black",
                     linestyle="--")
    axes[-1].set_ylabel("sessions landed at 10 Gbps / 115 s")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def parse_range(value: str) -> range:
    start, stop = map(int, value.split(":"))
    return range(start, stop)


def parse_classes(value: str) -> tuple[str, ...]:
    classes = tuple(value.split(","))
    if not classes or not set(classes) <= set(CLASSES):
        raise argparse.ArgumentTypeError(f"workloads must be drawn from {CLASSES}")
    return classes


def parse_solvers(value: str) -> tuple[str, ...]:
    solvers = tuple(value.split(","))
    if not solvers or not set(solvers) <= {"lp", "greedy"}:
        raise argparse.ArgumentTypeError("solvers must be lp and/or greedy")
    return solvers


def run(model_path: Path, manifest_path: Path, out: Path, sessions: int,
        seeds: range, transition_seeds: range, iterations=8, classes=CLASSES):
    profile = ModelProfile.load(model_path)
    manifest = json.loads(manifest_path.read_text())
    rows, threshold_rows = [], []
    for job_class in classes:
        shapes = trace_shapes(manifest, job_class)
        ratio = log_bytes_per_token(WORKLOADS[job_class])
        for seed in seeds:
            sampled = sample_sessions(shapes, sessions, seed, ratio)
            packed, replicas = pack_source(sampled, profile)
            found, thresholds = pressure_search(
                profile, packed, replicas, seed, job_class, iterations,
            )
            rows.extend(found)
            threshold_rows.extend({
                "workload": job_class, "seed": seed, "axis": axis,
                "threshold": item["value"], "state": item["state"],
                "source_replicas": replicas,
                "working_arrival": max(.01, thresholds["arrival"]["value"] / 2),
            } for axis, item in thresholds.items())
    medians = {
        (job_class, axis): float(np.median([
            row["threshold"] for row in threshold_rows
            if row["workload"] == job_class and row["axis"] == axis
        ]))
        for job_class in classes for axis in (
            "arrival", "service", "kv", "bandwidth_gbps", "migration_s",
        )
    }
    for job_class in classes:
        shapes = trace_shapes(manifest, job_class)
        ratio = log_bytes_per_token(WORKLOADS[job_class])
        for seed in transition_seeds:
            sampled, replicas = pack_source(
                sample_sessions(shapes, sessions, seed, ratio), profile,
            )
            for axis in ("arrival", "service", "kv", "bandwidth_gbps", "migration_s"):
                working = max(.01, medians[(job_class, "arrival")] / 2)
                base = Pressure(working, bandwidth_gbps=1000, migration_s=3600)
                rows.append(evaluate(
                    profile, sampled, replicas,
                    replace(base, **{axis: medians[(job_class, axis)]}),
                    seed, job_class, f"{axis}_transition",
                ))
    out.mkdir(parents=True, exist_ok=True)
    combined = rows + [dict(row, axis=f"threshold:{row['axis']}")
                       for row in threshold_rows]
    _write_csv(out / "results.csv", rows)
    _write_csv(out / "thresholds.csv", threshold_rows)
    metadata = {
        "schema": "queue-haul-destination-bench-v1",
        "created_local": datetime.now().astimezone().isoformat(),
        "git_sha": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "code_sha256": code_sha256(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sessions_per_workload": sessions, "primary_seeds": list(seeds),
        "transition_seeds": list(transition_seeds),
        "reference_rate_requests_per_session_s": REFERENCE_RATE,
        "deadline_s": 120, "migration_s": 115, "residency_s": 180,
        "service_bound": SERVICE_BOUND, "kv_capacity_tokens": KV_CAPACITY_TOKENS,
        "kv_block_tokens": KV_BLOCK_TOKENS,
        "claim": "sensitivity/possible; never an admission guarantee",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_thresholds(combined, out / "summary")
    return rows, threshold_rows


def run_reference(model_path: Path, manifest_path: Path, out: Path,
                  sessions: int, seed: int, classes=CLASSES,
                  solvers=("lp", "greedy")):
    profile = ModelProfile.load(model_path)
    manifest = json.loads(manifest_path.read_text())
    rows = []
    for job_class in classes:
        sampled, replicas = pack_source(sample_sessions(
            trace_shapes(manifest, job_class), sessions, seed,
            log_bytes_per_token(WORKLOADS[job_class]),
        ), profile)
        for solver in solvers:
            rows.append(evaluate(
                profile, sampled, replicas, Pressure(), seed, job_class,
                "reference", solver,
            ))
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "results.csv", rows)
    metadata = {
        "schema": "queue-haul-destination-bench-v1-reference",
        "created_local": datetime.now().astimezone().isoformat(),
        "git_sha": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "code_sha256": code_sha256(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sessions_per_workload": sessions, "seed": seed,
        "deadline_s": 120, "migration_s": 115, "residency_s": 180,
        "claim": "sensitivity/possible; never an admission guarantee",
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_thresholds(rows, out / "summary")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=10_000)
    parser.add_argument("--seeds", type=parse_range, default=range(10))
    parser.add_argument("--transition-seeds", type=parse_range, default=range(10, 30))
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--workloads", type=parse_classes, default=CLASSES)
    parser.add_argument("--solvers", type=parse_solvers, default=("lp", "greedy"))
    args = parser.parse_args()
    if args.reference_only:
        run_reference(args.model, args.manifest, args.out, args.sessions,
                      args.seeds.start, args.workloads, args.solvers)
    else:
        run(args.model, args.manifest, args.out, args.sessions, args.seeds,
            args.transition_seeds, args.iterations, args.workloads)


if __name__ == "__main__":
    main()
