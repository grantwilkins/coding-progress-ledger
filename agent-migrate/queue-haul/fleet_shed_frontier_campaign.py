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
                         DestinationType, FluidMigrationService, LoadedCoefficients,
                         MigrationComponents)
from migration_profiler import file_hash, stable_seed
from planner import InstanceCapacity, plan, source_power
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile
from simulate import (ExecutionScenario, NetworkLink, PowerNode, ServingInstance,
                      SimSession, execute, step_average)

ROOT = Path(__file__).parent
MODEL = ROOT / "profiles/gpt_oss_20b_a100_tp1_azure_300w.json"
# The headline workload is the one whose context anchors straddle the measured
# replay/KV crossover (~16-24k tokens on this profile): a third of its sessions
# sit below it, where replay is the cheaper destination action, and two thirds
# above it, where KV to the faster-ingesting region is.  A mixture skewed to
# either side makes one action dominate and the action choice stops mattering,
# which is what the sensitivity workloads show.
WORKLOADS = {name: ROOT / f"profiles/{name}.json" for name in (
    "agentic_tool_loop", "agentic_rps_shape", "interactive_coding", "coding")}
HEADLINE_WORKLOAD = "agentic_tool_loop"
ENVELOPE = ROOT / "outputs/agentic-rps-sweep-a100-pooled-p90-tpot-20260817/summary.json"
TIMING = ROOT / "outputs/timing-power-validation-20260814/timing-summary.json"
LOADED = ROOT / "outputs/loaded-service-model-20260815/model.json"
OUT = ROOT / "outputs/fleet-shed-frontier-a100-20260820"

MODEL_ID = "openai/gpt-oss-20b"
REGIONS = ("east", "germany")
SITES = {"east": "eastus2", "germany": "germanywestcentral"}
SOURCE_SITE = "swedencentral"
# Dense through 180-900 s, where four of the five rho knees and the whole
# multi-action band sit, and extended to 2700 s so the scarcest rho reaches its
# convergence knee (~2200 s) inside the grid rather than being right-censored
# by it.  Below 120 s every policy is replay-only, so resolution buys nothing.
DEADLINES_S = (30, 60, 120, 180, 240, 300, 375, 450, 600, 750, 900, 1800, 2700)
POLICIES = {
    "queue_haul": "lp_work_first", "greedy": "greedy",
    "isolated_fastest": "isolated_fastest", "kv_only": "kv_only",
    "replay_only": "replay_only",
}
SINGLE_ACTION_POLICIES = ("kv_only", "replay_only")
FLEXIBLE_POLICIES = ("queue_haul", "greedy")
ADMISSION_MODES = ("normal", "emergency")
# Outcome-independent asks cover low shed through near-full evacuation.  Search
# must inspect all of them because planner composition makes contract satisfaction
# non-monotone in the requested credit.
ASK_FRACTIONS = (.005, .01, .025, .05, .10, .25, .50, .75, .90, .99)
# Destination scarcity is the headline axis: migration headroom is what is
# left of a replica after its own baseline load and the source demand it must
# absorb, so rho sets how much room the destinations have to accept migration
# work at all.  The grid stops below the absorption cap, past which the pools
# cannot hold the fleet in steady state and the scenario is infeasible by
# construction rather than by policy.
# Capped below the headline workload's absorption limit (rho 0.506 at fleet
# scale): its long contexts occupy more destination capacity than a short
# mixture, so the same rho leaves less migration headroom.  prepare() rechecks.
RHOS = (0.20, 0.30, 0.38, 0.44, 0.48)
SEEDS = (1001, 1002, 1003)
# The mechanism is per node: source packing pins ~158 sessions to each node's
# egress pipe regardless of fleet size, so executed shed is invariant above a
# few thousand sessions (measured 0.742 at 3k, 0.742 at 6k, 0.741 at 50k, with
# the KV share steady at 45-47%) while cost per cell grows ~24x over that
# range.  The headline runs at a size on that plateau and INVARIANCE_SESSIONS
# carries the fleet-scale check.
SESSIONS = 12_000
INVARIANCE_SESSIONS = 50_000
TIERS = ("natural", "controlled_80", "controlled_40")
PROMPT, OUTPUT, REF_CONTEXT = 3920, 1024, 3920
NORMAL_TTFT_SLO_S, EMERGENCY_TTFT_SLO_S = 2.0, 10.0
WINDOW_S = 5
SHARDS = 32
SCHEMA = "queue-haul-fleet-shed-frontier-v3"
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

def _planner_seed(row: dict) -> int:
    """Match solver randomness across policies in the same scenario."""
    return stable_seed(row["deadline_s"], row["mode"], row["tier"], row["rho"],
                       row["workload"], row["sessions"], row["seed"])



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


def max_shed_plan(scenario, profile, architecture, solvers, seed, mode,
                  initial: float, removable: float, evaluate):
    """Best lawful executed shed among fixed asks and solver incumbents."""
    if not solvers:
        raise ValueError("max-shed search requires a solver")
    best, probes = None, 0
    for fraction in ASK_FRACTIONS:
        ask = fraction * removable
        for solver in solvers:
            probes += 1
            planned = plan(
                replace(scenario, power_limit_w=initial - ask), profile, {},
                solver, seed=seed, destination=architecture,
                admission_mode=mode)
            outcome = evaluate(planned, ask)
            rank = (outcome["realized_shed_w"], planned.solver == solvers[0],
                    -ask)
            if outcome["within_contract"] and (best is None or rank > best[3]):
                best = planned, outcome, ask, rank
    return (planned, outcome, ask, probes) if best is None else (*best[:3], probes)


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
    # The measured effective pipeline rate is an instance-to-instance figure,
    # so one copy of it per region starves a 50k-session fleet by the node
    # count and vetoes KV transfer outright.  Each source node owns its egress
    # pipe at that measured rate, which keeps every flow inside the calibrated
    # bandwidth band while letting fleet egress scale with the fleet.
    links = tuple(NetworkLink(
        f"pipeline/{region}/node-{i}",
        fits[region]["effective_pipeline_mbps"][tier] * 125_000)
        for region in REGIONS for i in range(node_count))
    scenario = ExecutionScenario(deadline_s, deadline_s, 0.0, "awake", 0.0,
                                 nodes, instances, sessions_tuple, links)
    return scenario, replicas, float(work.sum()), fits


def build_architecture(profile, replicas: int, bounds: dict, fits, rho: float,
                       headroom: float, contexts) -> DestinationArchitecture:
    case, per_node = profile.case(), profile.gpus_per_node
    fingerprint = CompatibilityFingerprint(profile.model, "gpt-oss-pinned",
                                           "source-dc-log", "lmcache-mp-v7")

    def rate(curve):
        return ContextRate(*(tuple(map(float, v)) for v in curve.by_concurrency[1]))

    loaded_fit = json.loads(LOADED.read_text())
    source_action = {method: case.action_power_w[method].power(1, True)
                     for method in ("replay", "kv_transfer")}
    sink_action = {method: case.action_power_w[method].power(1, False)
                   for method in source_action}
    baseline = tuple(rho * bounds["normal"] / request_work(case).sum()
                     * request_work(case))
    types, pools = [], []
    for region in REGIONS:
        raw = fits[region]["migration_components"]
        factors = {method: value.get("compute_completion_factor", 1)
                   for method, value in raw.items()}
        migration = {method: MigrationComponents(
            tuple(value["context_range"]),
            tuple(value["bandwidth_range_bytes_per_s"]), value["provenance"],
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
        for node in range(math.ceil(replicas / per_node)):
            members = range(node * per_node,
                            min((node + 1) * per_node, replicas))
            pools.append(DestinationPool(
                f"pool/{region}/node-{node}", destination_type.type_id,
                tuple(DestinationReplica(f"{region}-{i}", baseline, 0)
                      for i in members),
                f"route/{region}/node-{node}",
                (f"pipeline/{region}/node-{node}",),
                migration_headroom={method: headroom
                                    for method in ("replay", "kv_transfer")},
            # Now that one migration is served at one replica, the fluid service
            # is what applies the measured loaded-service slowdown as the pools
            # fill; coupling stays off because a migration headroom requires it.
                fluid_migration=FluidMigrationService(
                    1 / factors["replay"],
                    source_action, sink_action,
                    f"{TIMING.relative_to(ROOT)} regional pipelined timing fit",
                    0, True),
                source_affinity=tuple(f"source-{i}" for i in members)))
    return DestinationArchitecture(DESTINATION_SCHEMA, fingerprint, tuple(types),
                                   tuple(pools))


def run_row(row: dict, manifest: dict) -> dict:
    profile = ModelProfile.load(MODEL)
    workload = WorkloadProfile.load(WORKLOADS[row["workload"]])
    for path, digest in manifest["inputs"].items():
        if file_hash(ROOT / path) != digest:
            raise RuntimeError(f"{path} changed after prepare")
    case = profile.case()
    bounds = {mode: manifest["envelope"][mode]["rps"] * request_work(case).sum()
              for mode in ("normal", "emergency")}
    bounds["stable"] = bounds["emergency"]
    scenario, replicas, demand, fits = build_fleet(
        profile, workload, row["sessions"], row["seed"], row["deadline_s"],
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
    seed = _planner_seed(row)
    pool_replicas = {pool.pool_id: len(pool.replicas)
                     for pool in architecture.pools}
    baseline_rps = row["rho"] * manifest["envelope"]["normal"]["rps"]
    per_request = request_work(case).sum()
    envelope = manifest["envelope"][row["mode"]]["rps"]
    work_of = {s.session_id: float(
        s.expected_f / case.prefill.rate(s.context_tokens, 1)
        + s.expected_g / case.decode.rate(s.context_tokens, 1))
        for s in scenario.sessions}

    def evaluate(planned, ask):
        result = execute(replace(scenario, power_limit_w=initial - ask),
                         profile, planned.moves, destination=architecture)
        realized = initial - step_average(result.power, row["deadline_s"],
                                          WINDOW_S)
        committed = {item.session_id for item in result.sessions
                     if item.committed_s is not None}
        landed = {}
        for move in planned.moves:
            if move.session_id in committed:
                landed[move.destination_pool] = landed.get(
                    move.destination_pool, 0.0) + work_of[move.session_id]
        # Worst pool decides compliance; the pools are not loaded symmetrically.
        offered = max((baseline_rps + work / pool_replicas[pool] / per_request
                       for pool, work in landed.items()), default=baseline_rps)
        makespan = result.migration_makespan_s
        by_region = {}
        for move in planned.moves:
            if move.session_id in committed:
                region = move.destination_pool.split("/")[1]
                by_region[region, move.method] = \
                    by_region.get((region, move.method), 0) + 1
        return {
            "realized_shed_w": realized,
            "landed_sessions": len(committed),
            "destination_offered_rps": offered,
            "within_envelope": offered <= envelope + 1e-9,
            "migration_makespan_s": makespan,
            "by_region": by_region,
            "within_contract": (
                # The plan must certify its own target, not merely survive
                # execution.  Without this the ask climbs until the target is
                # unattainable, where the deadline repair still trims the plan
                # into a lawful makespan and every policy is scored in the
                # target-first LP's fallback, which does not optimize shed --
                # the regime that makes a single-action greedy look better than
                # a policy whose action set strictly contains it.
                planned.feasible
                and makespan is not None
                and makespan <= row["deadline_s"] + 1e-9
                and offered <= envelope + 1e-9
                and all(item.within_contract for item in result.pool_service)),
        }

    native = POLICIES[row["policy"]]
    solvers = (native, *(POLICIES[p] for p in SINGLE_ACTION_POLICIES)) \
        if row["policy"] == "queue_haul" else (native,)
    planned, outcome, ask, probes = max_shed_plan(
        scenario, profile, architecture, solvers, seed, row["mode"], initial,
        removable, evaluate)
    methods = {method: sum(m.method == method for m in planned.moves)
               for method in ("replay", "kv_transfer")}
    binding = max(planned.resource_uses, key=lambda r: r.utilization,
                  default=None)
    committed_moves = sum(outcome["by_region"].values())
    return {
        **row, "git_sha": manifest["git_sha"],
        "planner_seed": seed, "source_replicas": replicas,
        "destination_replicas": replicas * len(REGIONS),
        "destination_pools": len(architecture.pools),
        "migration_headroom": headroom, "absorbed_fraction": absorbed,
        "initial_source_power_w": initial, "idle_source_power_w": idle,
        "removable_power_w": removable, "credit_target_w": ask,
        "max_shed_asks": len(ASK_FRACTIONS), "max_shed_probes": probes,
        "selected_solver": planned.solver,
        "restricted_fallback": planned.solver != native,
        "planned_shed_w": initial - planned.planned_source_power_w,
        "moves": len(planned.moves), "replay_moves": methods["replay"],
        "kv_moves": methods["kv_transfer"], "solve_s": planned.solve_s,
        "feasible": planned.feasible, "failure_reason": planned.failure_reason or "",
        "binding_resource": binding.name if binding else "",
        "binding_utilization": binding.utilization if binding else 0.0,
        "executed_shed_w": outcome["realized_shed_w"],
        "executed_shed_fraction": outcome["realized_shed_w"] / removable,
        "landed_sessions": outcome["landed_sessions"],
        "migration_makespan_s": outcome["migration_makespan_s"] or "",
        "within_contract": outcome["within_contract"],
        "within_envelope": outcome["within_envelope"],
        "destination_offered_rps": outcome["destination_offered_rps"],
        "destination_rho": outcome["destination_offered_rps"]
        / manifest["envelope"]["normal"]["rps"],
        "committed_kv_fraction": (
            sum(n for (_, m), n in outcome["by_region"].items()
                if m == "kv_transfer") / committed_moves
            if committed_moves else 0.0),
        **{f"{region}_{method}": outcome["by_region"].get((region, method), 0)
           for region in REGIONS
           for method in ("replay", "kv_transfer")},
    }


def manifest_rows() -> list[dict]:
    rows = [{
        "deadline_s": float(deadline), "policy": policy, "rho": rho,
        "mode": "normal", "tier": "natural", "workload": HEADLINE_WORKLOAD,
        "sessions": SESSIONS, "seed": seed, "headline": True,
    } for deadline in DEADLINES_S for policy in POLICIES for rho in RHOS
        for seed in SEEDS]
    # Sensitivity: the other admission mode, the throttled bandwidth tiers, and
    # every other workload mixture, which is what decides whether choosing an
    # action matters at all.
    rows += [{
        "deadline_s": float(deadline), "policy": policy, "rho": RHOS[2],
        "mode": mode, "tier": tier, "workload": workload,
        "sessions": SESSIONS, "seed": SEEDS[0], "headline": False,
    } for deadline in DEADLINES_S for policy in POLICIES
        for mode, tier, workload in
        [(m, "natural", HEADLINE_WORKLOAD) for m in ADMISSION_MODES[1:]]
        + [("normal", t, HEADLINE_WORKLOAD) for t in TIERS[1:]]
        + [("normal", "natural", w) for w in WORKLOADS if w != HEADLINE_WORKLOAD]]
    # Fleet-scale check: the same cells at the full fleet, so the headline's
    # smaller fleet is justified by measurement rather than assumed.
    rows += [{
        "deadline_s": float(deadline), "policy": policy, "rho": RHOS[2],
        "mode": "normal", "tier": "natural", "workload": HEADLINE_WORKLOAD,
        "sessions": INVARIANCE_SESSIONS, "seed": SEEDS[0], "headline": False,
    } for deadline in (300.0, 600.0) for policy in POLICIES]
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
    # Fail here, not hours into a shard: every swept rho must leave the
    # headline workload room to migrate at all.
    probe = ModelProfile.load(MODEL)
    probe_bound = normal_rps * request_work(probe.case()).sum()
    _, probe_replicas, probe_demand, _ = build_fleet(
        probe, WorkloadProfile.load(WORKLOADS[HEADLINE_WORKLOAD]),
        1500, SEEDS[0], float(DEADLINES_S[-1]), probe_bound, "natural")
    for rho in RHOS:
        migration_headroom(rho, probe_demand, probe_replicas, probe_bound)
    git = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True,
                         text=True, cwd=ROOT, check=True).stdout.strip()
    manifest = {
        "schema": SCHEMA,
        "claim": "best contract-respecting executed shed over a fixed ask "
                 "ladder for one source and two destinations; Queue-Haul is "
                 "LP-led with KV-only and replay-only incumbents, and seeds are "
                 "aggregated by the median",
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
                   for path in (MODEL, ENVELOPE, TIMING, LOADED,
                                *WORKLOADS.values())},
        "git_sha": git,
        "rows": manifest_rows(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def run_shard(out: Path, shard: int, subset: str = "all") -> int:
    manifest = json.loads((out / "plan.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise RuntimeError("unexpected plan schema")
    git = subprocess.run(("git", "rev-parse", "HEAD"), capture_output=True,
                         text=True, cwd=ROOT, check=True).stdout.strip()
    if git != manifest["git_sha"]:
        raise RuntimeError(f"shard {shard} would run on {git[:12]}, "
                           f"manifest is {manifest['git_sha'][:12]}")
    rows = [row for row in manifest["rows"] if row["row_id"] % SHARDS == shard
            and (subset == "all" or row["headline"] == (subset == "headline"))]
    if not rows:
        raise RuntimeError(f"shard {shard} is empty")
    suffix = "" if subset == "all" else f"-{subset}"
    write_csv(out / f"shard-{shard:02d}{suffix}.csv",
              [run_row(row, manifest) for row in rows])
    return len(rows)


def _csv(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _lawful_value(row: dict, key: str) -> float:
    return float(row[key]) if row["within_contract"] == "True" else 0.0


def reduce(out: Path) -> dict:
    manifest = json.loads((out / "plan.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise RuntimeError("unexpected plan schema")
    rows = [row for shard in sorted(out.glob("shard-*.csv")) for row in _csv(shard)]
    expected = {row["row_id"]: row for row in manifest["rows"]}
    seen = set()
    for row in rows:
        i = int(row["row_id"])
        if i in seen or i not in expected:
            raise RuntimeError(f"duplicate or unknown row {i}")
        seen.add(i)
        if row["git_sha"] != manifest["git_sha"]:
            raise RuntimeError(f"row {i} was produced by commit "
                               f"{row['git_sha'][:12]}, not the manifest's")
        stale = [key for key in ("deadline_s", "policy", "mode", "tier",
                                 "rho", "workload", "sessions", "seed",
                                 "headline")
                 if row[key] != str(expected[i][key])]
        if stale:
            raise RuntimeError(f"row {i} does not match the manifest: {stale}")
    # The frontier claim needs every headline row; the sensitivity block may
    # be reduced later, but never partially.
    headline_ids = {row["row_id"] for row in manifest["rows"] if row["headline"]}
    if headline_ids - seen:
        raise RuntimeError("reduce requires every headline row exactly once")
    missing = set(expected) - headline_ids - seen
    if missing and missing != set(expected) - headline_ids:
        raise RuntimeError("sensitivity shards are incomplete")
    headline = [row for row in rows if row["headline"] == "True"]
    envelope = manifest["envelope"]["normal"]["rps"]
    fields = ("deadline_s", "rho", "mode", "tier", "workload", "sessions", "seed")
    cells = {}
    for row in rows:
        cells.setdefault(tuple(row[key] for key in fields), {})[row["policy"]] = row
    for key, cell in cells.items():
        if "queue_haul" not in cell:
            continue
        qh = _lawful_value(cell["queue_haul"], "executed_shed_fraction")
        for policy in SINGLE_ACTION_POLICIES:
            if policy in cell and qh + 1e-9 < _lawful_value(
                    cell[policy], "executed_shed_fraction"):
                raise RuntimeError(
                    f"queue_haul is dominated by {policy} at {dict(zip(fields, key))}")
    for policy, rho, seed in sorted({
            (row["policy"], float(row["rho"]), row["seed"]) for row in headline}):
        curve = sorted(
            (row for row in headline
             if (row["policy"], float(row["rho"]), row["seed"])
             == (policy, rho, seed)),
            key=lambda row: float(row["deadline_s"]))
        for earlier, later in zip(curve, curve[1:]):
            if _lawful_value(later, "executed_shed_fraction") + 1e-9 \
                    < _lawful_value(earlier, "executed_shed_fraction"):
                raise RuntimeError(
                    f"{policy} at rho={rho}, seed={seed} decreases with deadline")

    # One executed number per (deadline, policy, rho): the median over seeds of
    # the largest shed that policy actually delivered inside every contract.
    frontier, keys = [], []
    for row in headline:
        key = (float(row["deadline_s"]), row["policy"], float(row["rho"]))
        if key not in keys:
            keys.append(key)
    for deadline, policy, rho in sorted(keys):
        group = [row for row in headline
                 if (float(row["deadline_s"]), row["policy"], float(row["rho"]))
                 == (deadline, policy, rho)]
        shed = [_lawful_value(row, "executed_shed_fraction")
                for row in group]
        frontier.append({
            "deadline_s": deadline, "policy": policy, "rho": rho,
            "seeds": len(group),
            "median_executed_shed_fraction": float(np.median(shed)),
            "min_executed_shed_fraction": min(shed),
            "max_executed_shed_fraction": max(shed),
            "median_executed_shed_kw": float(np.median(
                [_lawful_value(row, "executed_shed_w") for row in group])) / 1000,
            "median_committed_kv_fraction": float(np.median(
                [_lawful_value(row, "committed_kv_fraction") for row in group])),
            "contracts_met": sum(row["within_contract"] == "True"
                                 for row in group),
            "restricted_fallbacks": sum(
                row.get("restricted_fallback") == "True" for row in group),
            **{f"median_{region}_{method}": float(np.median(
                [_lawful_value(row, f"{region}_{method}") for row in group]))
               for region in REGIONS
               for method in ("replay", "kv_transfer")},
        })

    # Compare joint policies only with the two single-action restrictions.
    advantage = []
    for deadline, rho in sorted({(k[0], k[2]) for k in keys}):
        cell = {row["policy"]: row for row in frontier
                if (row["deadline_s"], row["rho"]) == (deadline, rho)}
        if not set(SINGLE_ACTION_POLICIES) <= set(cell):
            continue
        best_single = max(
            cell[p]["median_executed_shed_fraction"]
            for p in SINGLE_ACTION_POLICIES)
        flexible = {
            p: cell[p]["median_executed_shed_fraction"]
            for p in FLEXIBLE_POLICIES if p in cell}
        if not flexible:
            continue
        winner = max(flexible, key=flexible.get)
        best_flexible = flexible[winner]
        if best_flexible + 1e-9 < best_single:
            raise RuntimeError(
                f"flexible policies are dominated at deadline={deadline}, rho={rho}")
        advantage.append({
            "deadline_s": deadline, "rho": rho,
            "best_single_action": best_single,
            "best_flexible": best_flexible,
            "multi_action_gain": best_flexible - best_single,
            "best_flexible_policy": winner,
            "best_flexible_kv_fraction":
                cell[winner]["median_committed_kv_fraction"],
        })

    compliance = [{
        "deadline_s": float(row["deadline_s"]), "policy": row["policy"],
        "mode": row["mode"], "rho": float(row["rho"]), "seed": int(row["seed"]),
        "destination_offered_rps": float(row["destination_offered_rps"]),
        "envelope_rps": manifest["envelope"][row["mode"]]["rps"],
        "within_envelope": row["within_envelope"] == "True",
    } for row in headline]
    breaches = [row for row in compliance
                if row["mode"] == "normal" and not row["within_envelope"]]
    if advantage:
        write_csv(out / "multi_action_advantage.csv", advantage)
    write_csv(out / "frontier.csv", frontier)
    write_csv(out / "slo_compliance.csv", compliance)
    # The headline runs a smaller fleet than the invariance block; state the
    # agreement rather than leaving the reader to derive it.
    invariance = []
    for row in rows:
        if int(row["sessions"]) != INVARIANCE_SESSIONS:
            continue
        match = [other for other in headline
                 if (other["policy"], float(other["deadline_s"]),
                     float(other["rho"]), other["seed"])
                 == (row["policy"], float(row["deadline_s"]),
                     float(row["rho"]), row["seed"])]
        if match:
            headline_shed = _lawful_value(
                match[0], "executed_shed_fraction")
            fleet_shed = _lawful_value(row, "executed_shed_fraction")
            invariance.append({
                "deadline_s": float(row["deadline_s"]), "policy": row["policy"],
                "headline_sessions": int(match[0]["sessions"]),
                "headline_shed": headline_shed,
                "headline_within_contract":
                    match[0]["within_contract"] == "True",
                "fleet_sessions": int(row["sessions"]),
                "fleet_shed": fleet_shed,
                "fleet_within_contract": row["within_contract"] == "True",
                "delta": fleet_shed - headline_shed,
            })
    if invariance:
        write_csv(out / "fleet_invariance.csv", invariance)
    sensitivity = [row for row in rows if row["headline"] == "False"]
    if sensitivity:
        write_csv(out / "sensitivity.csv", sensitivity)

    summary = {
        "schema": SCHEMA, "claim": manifest["claim"],
        "sessions": manifest["sessions"], "envelope": manifest["envelope"],
        "rows": len(rows), "planned_rows": len(expected),
        "headline_rows": len(headline), "partial": len(seen) != len(expected),
        "missing_row_ids": sorted(set(expected) - seen),
        "rho_grid": list(RHOS),
        "headline_sessions": SESSIONS,
        "fleet_invariance_sessions": INVARIANCE_SESSIONS,
        "fleet_invariance_max_abs_delta_by_policy": {
            policy: max(abs(row["delta"]) for row in invariance
                        if row["policy"] == policy)
            for policy in POLICIES
            if any(row["policy"] == policy for row in invariance)},
        "max_multi_action_gain": max(
            (row["multi_action_gain"] for row in advantage), default=0.0),
        "normal_mode_envelope_breaches": len(breaches),
        "inputs": manifest["inputs"], "git_sha": manifest["git_sha"],
        "limitations": [
            "Power is accelerator-scoped: the sum of a measured per-GPU curve "
            "over the modeled fleet. No PUE, node, cooling, host, or network "
            "power is claimed.",
            "Sessions never end; the snapshot models an evacuation, not a "
            "drain-down.",
            "The headline workload is agentic_tool_loop, whose source is "
            "declared assumed with 50% relative error and which carries only "
            "three context anchors (14042, 30785, 31547 tokens) sampled "
            "uniformly. It was chosen because that mixture straddles the "
            "measured replay/KV crossover, not because it maximises any "
            "policy gap; sensitivity.csv reports every other workload, where "
            "a mixture skewed to one side of the crossover makes a single "
            "action dominate and the action choice stops mattering.",
            "The replay/KV crossover is a property of the measured profile, "
            "not of the planner: including the fitted per-migration residual "
            "(east 2.03 s, germany 1.07 s), replay is the cheaper destination "
            "action below about 16k tokens and KV to the faster-ingesting "
            "region is cheaper above about 24k. East never favours KV "
            "anywhere in the calibrated context range.",
            "Each source node owns one egress pipe at the measured effective "
            "pipeline rate and reaches only the destination pools on that "
            "path. That rate was measured instance-to-instance, so pooling a "
            "50k-session fleet onto one copy of it understates fleet egress "
            "by the node count; per-node pipes keep every flow inside the "
            "calibrated bandwidth band, which link-capacity scaling would "
            "not.",
            "The 5 RPS normal envelope is the last swept rate whose median p90 "
            "TTFT meets the 2.0 s SLO; its worst repeat reached 2.0116 s.",
            "The emergency envelope is right-censored: no swept rate violated "
            "the 10 s tier, so 8 RPS is a lower bound set by the grid.",
            "migration_headroom is derived from the admission arithmetic, not "
            "measured; it is swept in sensitivity.csv. It scales a "
            "replica-second migration budget by a fraction of the service "
            "envelope, which is conservative only while that envelope "
            "exceeds one concurrency-1 replica-second per second.",
            "Destination pools declare one migration headroom shared by "
            "both methods (coupling=0): the planner budgets and the executor "
            "serves replicas x headroom jointly, so a mixed plan cannot book "
            "the derived slack once per method.  The hardware campaigns "
            "couple per-method budgets instead.",
            "The stable envelope is assumed equal to the emergency envelope; "
            "no measurement separates them.",
            "The service envelope was measured on the source region's A100 and "
            "is applied to both destinations, assuming identical hardware.",
            "Replay uses the fitted regional completion factors, which "
            "reproduce the measured concurrency-1 commits to about 16% MAPE. "
            "They dip below a per-replica prefill-throughput estimate at 16384 "
            "tokens; a scalar floor was tried and rejected because it tripled "
            "the error against that same evidence file.",
            "This profile has no phase_power, so the modular credit model "
            "does not exactly rank the supermodular executed shed. Every policy "
            "therefore runs the same fixed ask ladder and is scored only on "
            "contract-respecting executed shed; this is a sampled frontier, not "
            "a proof of the global optimum between probes.",
            "queue_haul is an LP-led portfolio: the unrestricted target-first "
            "LP and the KV-only and replay-only incumbents are evaluated at "
            "every ask, and selected_solver records the lawful winner. This "
            "guarantees a tie or win over those single-action implementations; "
            "strict gains come only from the unrestricted LP.",
            "Source packing is descending-load first-fit, which gives each "
            "instance near-identical sessions and so flatters any credit-ordered "
            "selector; an arrival-order fleet is a harder case and is not swept "
            "here.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    headline_rho = RHOS[2]
    shown = [row for row in frontier if row["rho"] == headline_rho]
    table = "\n".join(
        f"| {row['deadline_s']:.0f} | {row['policy']} | "
        f"{row['median_executed_shed_fraction']:.0%} | "
        f"{row['median_executed_shed_kw']:.1f} | "
        f"{row['median_committed_kv_fraction']:.0%} |" for row in shown)
    gains = "\n".join(
        f"| {row['deadline_s']:.0f} | {row['rho']:.2f} | "
        f"{row['best_single_action']:.0%} | {row['best_flexible']:.0%} | "
        f"{row['multi_action_gain']:+.1%} |" for row in advantage
        if row["rho"] == headline_rho)
    inv = "\n".join(
        f"| {row['deadline_s']:.0f} | {row['policy']} | "
        f"{row['headline_shed']:.4f} | {row['fleet_shed']:.4f} | "
        f"{row['delta']:+.4f} | "
        f"{row['headline_within_contract'] and row['fleet_within_contract']} |"
        for row in invariance)
    status = (f"Complete: {len(rows):,}/{len(expected):,} planned rows."
              if len(rows) == len(expected) else
              f"**Partial: {len(rows):,}/{len(expected):,} planned rows.**")
    gains_section = (
        f"## What multiple actions buy at rho={headline_rho}\n\n"
        f"| Deadline (s) | rho | Best single action | Best flexible | Gain |\n"
        f"|---|---|---|---|---|\n{gains}\n\n") if gains else ""
    invariance_section = (
        "## Fleet invariance\n\n"
        "| Deadline (s) | Policy | Headline shed | Fleet shed | Delta | "
        "Both lawful |\n|---|---|---|---|---|---|\n"
        f"{inv}\n") if inv else (
        "## Fleet invariance\n\nNot included in this reduction.\n")
    (out / "README.md").write_text(
        f"# Fleet shed frontier, {manifest['sessions']:,} sessions\n\n"
        f"{status}\n\n"
        f"One shedding source site ({manifest['source_site']}) and two equally "
        f"sized destination sites ({', '.join(manifest['sites'].values())}), "
        f"gpt-oss-20b on A100. Each source node owns its measured egress pipe.\n\n"
        f"Every cell reports the best contract-respecting executed shed among "
        f"the {len(ASK_FRACTIONS)} fixed asks "
        f"({', '.join(f'{x:g}' for x in ASK_FRACTIONS)}) of removable power. "
        f"All asks are evaluated; no monotonicity is assumed. Seeds aggregate "
        f"by median. Queue-Haul is LP-led and retains KV-only and replay-only "
        f"incumbents; selected_solver records any fallback.\n\n"
        f"Destination admission is capped at {envelope:g} offered RPS per "
        f"replica. Across {len(compliance):,} headline rows, {len(breaches)} "
        f"exceeded that measured envelope. Power is accelerator-scoped.\n\n"
        f"## Executed shed at rho={headline_rho}\n\n"
        f"| Deadline (s) | Policy | Median executed shed | Median shed (kW) "
        f"| KV share of commits |\n|---|---|---|---|---|\n{table}\n\n"
        f"{gains_section}{invariance_section}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run-shard", "reduce"):
        item = sub.add_parser(name)
        item.add_argument("--out", type=Path, default=OUT)
        if name == "run-shard":
            item.add_argument("--shard", type=int, required=True)
            item.add_argument("--subset", default="all",
                              choices=("all", "headline", "sensitivity"))
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare(args.out)
        print(f"rows={len(manifest['rows'])} out={args.out}")
    elif args.command == "run-shard":
        print(f"rows={run_shard(args.out, args.shard, args.subset)} "
              f"shard={args.shard}")
    else:
        summary = reduce(args.out)
        print(f"rows={summary['rows']} breaches="
              f"{summary['normal_mode_envelope_breaches']} out={args.out}")


if __name__ == "__main__":
    main()
