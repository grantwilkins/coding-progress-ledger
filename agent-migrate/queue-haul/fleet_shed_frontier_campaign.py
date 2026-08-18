"""Run the sharded fleet-scale deadline-to-power-shed frontier."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np

from destination import (DESTINATION_SCHEMA, CompatibilityFingerprint, ContextRate,
                         DestinationArchitecture, DestinationPool, DestinationReplica,
                         DestinationType, LoadedCoefficients, MigrationComponents)
from migration_profiler import file_hash, stable_seed
from planner import InstanceCapacity, plan, source_power
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile
from simulate import (ExecutionScenario, NetworkLink, PowerNode, ServingInstance,
                      SimSession, execute, step_average)
from simulated_pareto_campaign import attainment_time

ROOT = Path(__file__).parent
MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json"
WORKLOAD = ROOT / "profiles/agentic_rps_shape.json"
ENVELOPE = ROOT / "outputs/agentic-rps-sweep-a100-pooled-p90-tpot-20260817/summary.json"
TIMING = ROOT / "outputs/timing-power-validation-20260814/timing-summary.json"
LOADED = ROOT / "outputs/loaded-service-model-20260815/model.json"
OUT = ROOT / "outputs/fleet-shed-frontier-a100-20260817"

MODEL_ID = "openai/gpt-oss-20b"
REGIONS = ("east", "germany")
SITES = {"east": "eastus2", "germany": "germanywestcentral"}
SOURCE_SITE = "swedencentral"
DEADLINES_S = (30, 60, 120, 180, 300, 450, 600, 900)
POLICIES = {
    "queue_haul": "lp_work_first", "greedy": "greedy",
    "isolated_fastest": "isolated_fastest", "kv_only": "kv_only",
    "replay_only": "replay_only",
}
ADMISSION_MODES = ("normal", "emergency")
# Requested fractions of removable source power.  Every campaign in this repo
# asks for an attainable fraction and measures attainment; asking for the idle
# floor instead drives the target-first LP into its infeasible fallback, where a
# linear credit model over a concave curve has no reason to beat greedy.
#
# The grid stops below 1.00 deliberately: requesting the whole removable band
# sets the limit exactly at the idle floor, which only a full evacuation reaches,
# so that row alone lands back in the infeasible fallback and its attainment is
# decided by float tolerance rather than by the plan.
TARGETS = (0.10, 0.25, 0.50, 0.75, 0.90)
SEEDS = (1001, 1002, 1003)
SESSIONS = 50_000
RHO_DEST = 0.45
TIERS = ("natural", "controlled_80", "controlled_40")
PROMPT, OUTPUT, REF_CONTEXT = 3920, 1024, 3920
NORMAL_TTFT_SLO_S, EMERGENCY_TTFT_SLO_S = 2.0, 10.0
WINDOW_S = 5
SHARDS = 32
SCHEMA = "queue-haul-fleet-shed-frontier-v1"
ENVELOPE_SCHEMA = "queue-haul-agentic-rps-sweep-v3"


def envelope_rps(slo_ttft_s: float) -> tuple[float, bool]:
    """Largest offered RPS below the first swept rate that violates the SLO.

    Scans in rate order and stops at the first violation, so a single passing
    repeat above a confirmed violation cannot raise the bound: the measured
    median TTFT curve is not monotone.  Both measured SLO metrics are checked.
    Returns the rate and whether it is right-censored by the swept grid.
    """
    summary = json.loads(ENVELOPE.read_text())
    if summary["schema"] != ENVELOPE_SCHEMA:
        raise RuntimeError(f"expected {ENVELOPE_SCHEMA}, got {summary['schema']}")
    model = summary["models"][MODEL_ID]
    tpot_slo = model["slo"]["p90_tpot_s"]
    passing = 0.0
    for row in sorted(model["curve"], key=lambda item: item["offered_rps"]):
        if row["p90_ttft_s_median"] > slo_ttft_s \
                or row["p90_tpot_s_median"] > tpot_slo:
            if not passing:
                raise RuntimeError(f"no measured rate meets {slo_ttft_s} s TTFT")
            return passing, False
        passing = float(row["offered_rps"])
    return passing, True


def request_work(case) -> np.ndarray:
    """Destination service work for one request of the measured shape."""
    return np.array([PROMPT / case.prefill.rate(REF_CONTEXT, 1),
                     OUTPUT / case.decode.rate(REF_CONTEXT, 1)])


def prefill_floor_factor(case, contexts) -> float:
    """Smallest replay completion factor that respects prefill throughput.

    Replay re-prefills the whole context, so its destination compute cannot run
    faster than the engine's own measured prefill rate over those tokens.  The
    regional completion factors are fitted end to end on low-concurrency
    migrations and can dip below that on this context grid; taking them at face
    value would let a saturated pool re-prefill faster than it can prefill.
    """
    return max(
        (tokens / case.prefill.rate(tokens, 1))
        / (tokens / case.replay.conservative_rate(tokens, 1)
           + case.replay_completion_s)
        for tokens in contexts)


def migration_headroom(rho: float, demand: float, replicas: int,
                       bound: float) -> float:
    """Destination envelope left free for migration ingest.

    The pools carry their own baseline ``rho`` and, at full shed, absorb the
    whole source demand as steady-state serving load.  Only what remains can pay
    for migration work without pushing served requests past the measured
    envelope.
    """
    absorbed = demand / (len(REGIONS) * replicas * bound)
    headroom = 1.0 - rho - absorbed
    if headroom <= 0:
        raise RuntimeError("destination pools cannot absorb the source fleet")
    return headroom


def build_fleet(profile, workload, sessions: int, seed: int, deadline_s: float,
                bound: float, tier: str):
    """Pack the source at min(power calibration, measured service envelope)."""
    case = profile.case()
    records = workload.sample(sessions, seed)
    ctx = np.array([r.context_tokens for r in records])
    cycles = np.array([
        r.request_gap_s + r.tool_delay_s
        + r.prompt_tokens / case.prefill.rate(r.context_tokens, 1)
        + r.output_tokens / case.decode.rate(r.context_tokens, 1)
        for r in records])
    expected_f = np.array([r.prompt_tokens for r in records]) / cycles
    expected_g = np.array([r.output_tokens for r in records]) / cycles
    work = np.stack([
        expected_f / np.array([case.prefill.rate(int(t), 1) for t in ctx]),
        expected_g / np.array([case.decode.rate(int(t), 1) for t in ctx])], 1)
    ell = expected_f / case.F + expected_g / case.G
    # The ell/work ratio spans 8.7x across the measured context grid (2.246 at
    # 4096 tokens down to 0.258 at 31562), so the power calibration binds on
    # short contexts and the service envelope on long ones; provision on both.
    # Summing per-session maxima bounds each component sum, so the packing is
    # sound and at worst opens more replicas than exact 2-D vector packing.
    load = np.maximum(ell / profile.max_power_load, work.sum(1) / bound)
    capacity = InstanceCapacity([], [], 1.0, profile.kv_capacity_tokens)
    assignment = np.empty(sessions, int)
    for j in np.argsort(-np.maximum(load, ctx / profile.kv_capacity_tokens),
                        kind="stable"):
        assignment[j] = capacity.place(float(load[j]), int(ctx[j]), grow=True)
    replicas = len(capacity.loads)
    per_node = profile.gpus_per_node
    node_count = math.ceil(replicas / per_node)
    nodes = tuple(PowerNode(f"source-node-{i}", per_node, True, SOURCE_SITE)
                  for i in range(node_count))
    instances = tuple(
        ServingInstance(f"source-{i}", (f"source-node-{i // per_node}",))
        for i in range(replicas))
    for region in REGIONS:
        nodes += tuple(
            PowerNode(f"{region}-node-{i}", per_node, False, SITES[region])
            for i in range(node_count))
        instances += tuple(
            ServingInstance(f"{region}-{i}", (f"{region}-node-{i // per_node}",))
            for i in range(replicas))
    sessions_tuple = tuple(SimSession(
        str(j), f"source-{assignment[j]}", int(ctx[j]), float(expected_f[j]),
        float(expected_g[j]), records[j].log_bytes, (), True, 0.0, state="active",
        expected_growth_tokens_per_s=0.0) for j in range(sessions))
    fits = json.loads(TIMING.read_text())["fits"]
    links = tuple(NetworkLink(
        f"pipeline/{region}",
        fits[region]["effective_pipeline_mbps"][tier] * 125_000)
        for region in REGIONS)
    scenario = ExecutionScenario(deadline_s, deadline_s, 0.0, "awake", 0.0,
                                 nodes, instances, sessions_tuple, links)
    return scenario, replicas, float(work.sum()), fits


def build_architecture(profile, replicas: int, bounds: dict, fits, rho: float,
                       headroom: float, contexts) -> DestinationArchitecture:
    case = profile.case()
    floor = prefill_floor_factor(case, contexts)
    fingerprint = CompatibilityFingerprint(profile.model, "gpt-oss-pinned",
                                           "source-dc-log", "lmcache-mp-v7")

    def rate(curve):
        return ContextRate(*(tuple(map(float, v)) for v in curve.by_concurrency[1]))

    loaded_fit = json.loads(LOADED.read_text())
    baseline = tuple(rho * bounds["normal"] / request_work(case).sum()
                     * request_work(case))
    types, pools = [], []
    for region in REGIONS:
        raw = fits[region]["migration_components"]
        factors = {
            method: max(value.get("compute_completion_factor", 1),
                        floor if method == "replay" else 0.0)
            for method, value in raw.items()}
        migration = {method: MigrationComponents(
            tuple(value["context_range"]),
            tuple(value["bandwidth_range_bytes_per_s"]),
            f"{value['provenance']}; replay floored at measured prefill throughput"
            if method == "replay" and factors[method]
            > value.get("compute_completion_factor", 1) else value["provenance"],
            factors[method], value.get("residual_s", 0),
            value.get("kv_ingest_bytes_per_s"))
            for method, value in raw.items()}
        loaded = {method: LoadedCoefficients(
            tuple(loaded_fit["rho_grid"]), tuple(loaded_fit["slowdown"][method]),
            migration[method].context_range,
            migration[method].bandwidth_range_bytes_per_s,
            f"{LOADED.relative_to(ROOT)}; normalized A100 load sensitivity")
            for method in ("replay", "kv_transfer")}
        destination_type = DestinationType(
            f"{MODEL_ID}-a100-tp1/{region}", fingerprint, rate(case.prefill),
            rate(case.decode), ((1, 1),),
            {mode: (bounds[mode],) for mode in ("normal", "emergency", "stable")},
            profile.kv_capacity_tokens, loaded, (0, 1),
            f"{ENVELOPE.relative_to(ROOT)} measured offered-RPS envelope",
            True, case.kv_transfer.block_tokens, migration)
        types.append(destination_type)
        pools.append(DestinationPool(
            f"pool/{region}", destination_type.type_id,
            tuple(DestinationReplica(f"{region}-{i}", baseline, 0)
                  for i in range(replicas)),
            f"route/{region}", (f"pipeline/{region}",),
            migration_headroom={method: headroom
                                for method in ("replay", "kv_transfer")}))
    return DestinationArchitecture(DESTINATION_SCHEMA, fingerprint, tuple(types),
                                   tuple(pools))


def run_row(row: dict, manifest: dict) -> dict:
    profile = ModelProfile.load(MODEL)
    workload = WorkloadProfile.load(WORKLOAD)
    for path, digest in manifest["inputs"].items():
        if file_hash(ROOT / path) != digest:
            raise RuntimeError(f"{path} changed after prepare")
    case = profile.case()
    bounds = {mode: manifest["envelope"][mode]["rps"] * request_work(case).sum()
              for mode in ("normal", "emergency")}
    bounds["stable"] = bounds["emergency"]
    scenario, replicas, demand, fits = build_fleet(
        profile, workload, manifest["sessions"], row["seed"], row["deadline_s"],
        bounds["normal"], row["tier"])
    contexts = sorted({record.context_tokens for record in workload.records})
    headroom = migration_headroom(row["rho"], demand, replicas, bounds["normal"])
    absorbed = 1.0 - row["rho"] - headroom
    architecture = build_architecture(profile, replicas, bounds, fits, row["rho"],
                                      headroom, contexts)
    power = ExpectedPower(scenario, profile)
    initial = power.power(True)
    idle = source_power(scenario, profile,
                        [s.session_id for s in scenario.sessions])
    removable = initial - idle
    limit = initial - row["requested_fraction"] * removable
    scenario = replace(scenario, power_limit_w=limit)
    seed = stable_seed(row["policy"], row["deadline_s"], row["mode"], row["tier"],
                       row["rho"], row["requested_fraction"], row["seed"])
    planned = plan(scenario, profile, {}, POLICIES[row["policy"]], seed=seed,
                   destination=architecture, admission_mode=row["mode"])
    shed = initial - planned.planned_source_power_w
    methods = {method: sum(m.method == method for m in planned.moves)
               for method in ("replay", "kv_transfer")}
    binding = max(planned.resource_uses, key=lambda r: r.utilization,
                  default=None)
    record = {
        **row, "planner_seed": seed, "source_replicas": replicas,
        "destination_replicas": replicas * len(REGIONS),
        "migration_headroom": headroom, "absorbed_fraction": absorbed,
        "initial_source_power_w": initial, "idle_source_power_w": idle,
        "removable_power_w": removable,
        "requested_power_w": initial - limit,
        "planned_shed_w": shed, "planned_shed_fraction": shed / removable,
        "planned_target_met": shed >= row["requested_fraction"] * removable - 1e-8,
        "moves": len(planned.moves), "replay_moves": methods["replay"],
        "kv_moves": methods["kv_transfer"], "solve_s": planned.solve_s,
        "feasible": planned.feasible, "failure_reason": planned.failure_reason or "",
        "binding_resource": binding.name if binding else "",
        "binding_utilization": binding.utilization if binding else 0.0,
    }
    if not row["headline"]:
        return {**record, "realized_shed_w": "", "realized_shed_fraction": "",
                "target_met": "", "attainment_s": "",
                "destination_offered_rps": "", "destination_rho": ""}
    result = execute(scenario, profile, planned.moves, destination=architecture)
    at_deadline = step_average(result.power, row["deadline_s"], WINDOW_S)
    realized = initial - at_deadline
    attained_at = attainment_time(result.power, limit, WINDOW_S, row["deadline_s"])
    committed = {item.session_id for item in result.sessions
                 if item.committed_s is not None}
    pool_of = {move.session_id: move.destination_pool for move in planned.moves}
    landed_work = {}
    for session in scenario.sessions:
        if session.session_id not in committed:
            continue
        pool = pool_of[session.session_id]
        landed_work[pool] = landed_work.get(pool, 0.0) + float(
            session.expected_f / case.prefill.rate(session.context_tokens, 1)
            + session.expected_g / case.decode.rate(session.context_tokens, 1))
    baseline_rps = row["rho"] * manifest["envelope"]["normal"]["rps"]
    per_request = request_work(case).sum()
    # Worst pool decides compliance; the pools are not loaded symmetrically.
    offered = max((baseline_rps + work / replicas / per_request
                   for work in landed_work.values()), default=baseline_rps)
    return {
        **record, "realized_shed_w": realized,
        "realized_shed_fraction": realized / removable,
        "landed_sessions": len(committed),
        "target_met": attained_at is not None,
        "attainment_s": attained_at if attained_at is not None else "",
        "destination_offered_rps": offered,
        "destination_rho": offered / manifest["envelope"]["normal"]["rps"],
    }


def manifest_rows() -> list[dict]:
    rows = []
    for deadline in DEADLINES_S:
        for policy in POLICIES:
            for target in TARGETS:
                for mode in ADMISSION_MODES:
                    for seed in SEEDS:
                        rows.append({
                            "deadline_s": float(deadline), "policy": policy,
                            "requested_fraction": target, "mode": mode,
                            "tier": "natural", "rho": RHO_DEST, "seed": seed,
                            "headline": True,
                        })
    for deadline in DEADLINES_S:
        for policy in POLICIES:
            for target in TARGETS:
                for tier in TIERS[1:]:
                    rows.append({
                        "deadline_s": float(deadline), "policy": policy,
                        "requested_fraction": target, "mode": "normal",
                        "tier": tier, "rho": RHO_DEST, "seed": SEEDS[0],
                        "headline": False,
                    })
                for rho in (0.30, 0.40, 0.50):
                    rows.append({
                        "deadline_s": float(deadline), "policy": policy,
                        "requested_fraction": target, "mode": "normal",
                        "tier": "natural", "rho": rho, "seed": SEEDS[0],
                        "headline": False,
                    })
    return [{**row, "row_id": i} for i, row in enumerate(rows)]


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare(out: Path) -> dict:
    shape = json.loads(ENVELOPE.read_text())["request_shape"]
    if (shape["prompt_tokens"], shape["output_tokens"]) != (PROMPT, OUTPUT):
        raise RuntimeError("request shape does not match the measured envelope")
    normal_rps, normal_censored = envelope_rps(NORMAL_TTFT_SLO_S)
    emergency_rps, emergency_censored = envelope_rps(EMERGENCY_TTFT_SLO_S)
    if emergency_rps <= normal_rps:
        raise RuntimeError("emergency envelope must exceed the normal envelope")
    git = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True,
                         text=True, cwd=ROOT, check=True).stdout.strip()
    manifest = {
        "schema": SCHEMA,
        "claim": "deadline-to-shed frontier for one shedding source site and two "
                 "equally sized destination sites held inside the measured "
                 "offered-RPS service envelope",
        "sessions": SESSIONS, "shards": SHARDS, "window_s": WINDOW_S,
        "source_site": SOURCE_SITE, "sites": SITES,
        "envelope": {
            "normal": {"rps": normal_rps, "ttft_slo_s": NORMAL_TTFT_SLO_S,
                       "right_censored": normal_censored},
            "emergency": {"rps": emergency_rps,
                          "ttft_slo_s": EMERGENCY_TTFT_SLO_S,
                          "right_censored": emergency_censored},
        },
        "inputs": {str(path.relative_to(ROOT)): file_hash(path)
                   for path in (MODEL, WORKLOAD, ENVELOPE, TIMING, LOADED)},
        "git_sha": git,
        "rows": manifest_rows(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run_shard(out: Path, shard: int) -> int:
    manifest = json.loads((out / "plan.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise RuntimeError("unexpected plan schema")
    rows = [row for row in manifest["rows"] if row["row_id"] % SHARDS == shard]
    if not rows:
        raise RuntimeError(f"shard {shard} is empty")
    write_csv(out / f"shard-{shard:02d}.csv",
              [run_row(row, manifest) for row in rows])
    return len(rows)


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def reduce(out: Path) -> dict:
    manifest = json.loads((out / "plan.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise RuntimeError("unexpected plan schema")
    rows = [row for shard in sorted(out.glob("shard-*.csv")) for row in _csv(shard)]
    if len(rows) != len(manifest["rows"]):
        raise RuntimeError("reduce requires every shard")
    headline = [row for row in rows if row["headline"] == "True"]
    envelope = manifest["envelope"]["normal"]["rps"]

    frontier, keys = [], []
    for row in headline:
        key = (float(row["deadline_s"]), row["policy"], row["mode"])
        if key not in keys:
            keys.append(key)
    for deadline, policy, mode in sorted(keys):
        group = [row for row in headline
                 if (float(row["deadline_s"]), row["policy"], row["mode"])
                 == (deadline, policy, mode)]
        met = [float(row["requested_fraction"]) for row in group
               if row["target_met"] == "True"]
        realized = [float(row["realized_shed_fraction"]) for row in group]
        frontier.append({
            "deadline_s": deadline, "policy": policy, "mode": mode,
            "cases": len(group),
            "attained_requests": len(met),
            "max_requested_met": max(met, default=0.0),
            "median_realized_shed_fraction": float(np.median(realized)),
            "max_realized_shed_fraction": max(realized, default=0.0),
            "median_realized_shed_kw": float(np.median(
                [float(row["realized_shed_w"]) for row in group])) / 1000,
        })

    compliance = [{
        "deadline_s": float(row["deadline_s"]), "policy": row["policy"],
        "mode": row["mode"], "requested_fraction": float(row["requested_fraction"]),
        "seed": int(row["seed"]),
        "destination_offered_rps": float(row["destination_offered_rps"]),
        "envelope_rps": envelope,
        "within_envelope": float(row["destination_offered_rps"]) <= envelope + 1e-9,
    } for row in headline]
    breaches = [row for row in compliance
                if row["mode"] == "normal" and not row["within_envelope"]]

    write_csv(out / "frontier.csv", frontier)
    write_csv(out / "slo_compliance.csv", compliance)
    sensitivity = [row for row in rows if row["headline"] == "False"]
    if sensitivity:
        write_csv(out / "sensitivity.csv", sensitivity)

    summary = {
        "schema": SCHEMA, "claim": manifest["claim"],
        "sessions": manifest["sessions"], "envelope": manifest["envelope"],
        "rows": len(rows), "headline_rows": len(headline),
        "normal_mode_envelope_breaches": len(breaches),
        "inputs": manifest["inputs"], "git_sha": manifest["git_sha"],
        "limitations": [
            "Power is accelerator-scoped: the sum of a measured per-GPU curve "
            "over the modeled fleet. No PUE, node, cooling, host, or network "
            "power is claimed.",
            "Sessions never end; the snapshot models an evacuation, not a "
            "drain-down.",
            "The 5 RPS normal envelope is the last swept rate whose median p90 "
            "TTFT meets the 2.0 s SLO; its worst repeat reached 2.0116 s.",
            "The emergency envelope is right-censored: no swept rate violated "
            "the 10 s tier, so 8 RPS is a lower bound set by the grid.",
            "migration_headroom is derived from the admission arithmetic, not "
            "measured; it is swept in sensitivity.csv. It scales a "
            "replica-second migration budget by a fraction of the service "
            "envelope, which is conservative only while that envelope "
            "exceeds one concurrency-1 replica-second per second.",
            "Destination pools use decoupled per-method migration budgets "
            "(coupling=0), which the planner requires before a migration "
            "headroom may be set; the hardware campaigns couple them.",
            "The stable envelope is assumed equal to the emergency envelope; "
            "no measurement separates them.",
            "The service envelope was measured on the source region's A100 and "
            "is applied to both destinations, assuming identical hardware.",
            "Replay completion factors are floored so replay compute never "
            "undercuts the measured prefill throughput for the tokens it "
            "re-prefills; the regional end-to-end fits dip below that floor on "
            "this context grid.",
            "Summed per-session marginal power under-counts a concave curve, so "
            "the target-first LP is not an upper bound on exact nonlinear shed.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    normal = [row for row in frontier if row["mode"] == "normal"]
    table = "\n".join(
        f"| {row['deadline_s']:.0f} | {row['policy']} | "
        f"{row['max_requested_met']:.0%} | "
        f"{row['median_realized_shed_kw']:.1f} |" for row in normal)
    (out / "README.md").write_text(
        f"# Fleet shed frontier, {manifest['sessions']:,} sessions\n\n"
        f"One shedding source site ({manifest['source_site']}) and two equally "
        f"sized destination sites ({', '.join(manifest['sites'].values())}), "
        f"gpt-oss-20b on A100. Each row requests a fraction of removable source "
        f"power and is scored on whether it attained that request by the "
        f"deadline.\n\n"
        f"Destination admission is capped at the measured "
        f"{envelope:g} offered RPS per replica, the last swept rate whose "
        f"median p90 TTFT meets the {manifest['envelope']['normal']['ttft_slo_s']} s "
        f"SLO. Across {len(compliance):,} headline rows, "
        f"{len(breaches)} exceeded that envelope under normal admission.\n\n"
        f"Power is accelerator-scoped: the sum of a measured per-GPU curve over "
        f"the modeled fleet. No facility, cooling, or host power is claimed.\n\n"
        f"| Deadline (s) | Policy | Max request met | Median shed (kW) |\n"
        f"|---|---|---|---|\n{table}\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run-shard", "reduce"):
        item = sub.add_parser(name)
        item.add_argument("--out", type=Path, default=OUT)
        if name == "run-shard":
            item.add_argument("--shard", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.out)
        print(f"rows={len(manifest['rows'])} out={args.out}")
    elif args.command == "run-shard":
        print(f"rows={run_shard(args.out, args.shard)} shard={args.shard}")
    else:
        summary = reduce(args.out)
        print(f"rows={summary['rows']} breaches="
              f"{summary['normal_mode_envelope_breaches']} out={args.out}")


if __name__ == "__main__":
    main()
