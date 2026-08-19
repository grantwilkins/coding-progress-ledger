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
WORKLOAD = ROOT / "profiles/agentic_rps_shape.json"
ENVELOPE = ROOT / "outputs/agentic-rps-sweep-a100-pooled-p90-tpot-20260817/summary.json"
TIMING = ROOT / "outputs/timing-power-validation-20260814/timing-summary.json"
LOADED = ROOT / "outputs/loaded-service-model-20260815/model.json"
OUT = ROOT / "outputs/fleet-shed-frontier-a100-20260818"

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
# decided by float tolerance rather than by the plan.  It runs to 0.99 so the
# frontier top is set by capacity, not by the grid.
# Destination scarcity is the headline axis: migration headroom is what is
# left of a replica after its own baseline load and the source demand it must
# absorb, so rho sets how much room the destinations have to accept migration
# work at all.  The grid stops below the absorption cap, past which the pools
# cannot hold the fleet in steady state and the scenario is infeasible by
# construction rather than by policy.
RHOS = (0.30, 0.40, 0.45, 0.50, 0.55)
SEEDS = (1001, 1002, 1003)
SESSIONS = 50_000
TIERS = ("natural", "controlled_80", "controlled_40")
PROMPT, OUTPUT, REF_CONTEXT = 3920, 1024, 3920
NORMAL_TTFT_SLO_S, EMERGENCY_TTFT_SLO_S = 2.0, 10.0
WINDOW_S = 5
SHARDS = 32
SCHEMA = "queue-haul-fleet-shed-frontier-v2"
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


MAX_SHED_STEPS = 8


def max_shed_plan(scenario, profile, architecture, solver, seed, mode, power,
                  initial: float, evaluate):
    """Largest executed shed a policy delivers inside every contract.

    The request grid it replaces was a coarse threshold search: it reported the
    largest *asked* fraction some row happened to meet, not the shed the policy
    can actually execute.  Here the credit ask is bisected directly and every
    probe is executed, so the reported number is measured, not requested.
    ``evaluate`` returns the executed outcome for one plan and decides whether
    it honoured the deadline and the destination envelope.
    """
    low, high, best, probes = 0.0, initial, None, 0
    for probes in range(1, MAX_SHED_STEPS + 1):
        ask = (low + high) / 2
        planned = plan(replace(scenario, power_limit_w=initial - ask), profile,
                       {}, solver, seed=seed, destination=architecture,
                       admission_mode=mode)
        outcome = evaluate(planned, ask)
        if outcome["within_contract"]:
            if best is None or outcome["realized_shed_w"] > best[1]["realized_shed_w"]:
                best = planned, outcome, ask
            low = ask
        else:
            high = ask
    if best is None:
        # Nothing the policy plans at any ask executes inside the contracts.
        return planned, outcome, ask, probes
    return (*best, probes)


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
                    fits[region]["kv_ingest_lower_bound_bytes_per_s"],
                    source_action, sink_action,
                    f"{TIMING.relative_to(ROOT)} regional pipelined timing fit",
                    0, True),
                source_affinity=tuple(f"source-{i}" for i in members)))
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
    seed = stable_seed(row["policy"], row["deadline_s"], row["mode"], row["tier"],
                       row["rho"], row["seed"])
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
                makespan is not None
                and makespan <= row["deadline_s"] + 1e-9
                and offered <= envelope + 1e-9
                and all(item.within_contract for item in result.pool_service)),
        }

    planned, outcome, ask, probes = max_shed_plan(
        scenario, profile, architecture, POLICIES[row["policy"]], seed,
        row["mode"], power, initial, evaluate)
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
        "max_shed_probes": probes,
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
        "mode": "normal", "tier": "natural", "seed": seed, "headline": True,
    } for deadline in DEADLINES_S for policy in POLICIES for rho in RHOS
        for seed in SEEDS]
    rows += [{
        "deadline_s": float(deadline), "policy": policy, "rho": RHOS[2],
        "mode": mode, "tier": tier, "seed": SEEDS[0], "headline": False,
    } for deadline in DEADLINES_S for policy in POLICIES
        for mode, tier in [(m, "natural") for m in ADMISSION_MODES[1:]]
        + [("normal", t) for t in TIERS[1:]]]
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
                 "offered-RPS service envelope; the headline is each seed's "
                 "largest executed shed attained by the deadline, aggregated "
                 "over seeds by the median",
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
                                 "rho", "seed", "headline")
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
        shed = [float(row["executed_shed_fraction"])
                if row["within_contract"] == "True" else 0.0 for row in group]
        kv = [float(row["committed_kv_fraction"]) for row in group]
        frontier.append({
            "deadline_s": deadline, "policy": policy, "rho": rho,
            "seeds": len(group),
            "median_executed_shed_fraction": float(np.median(shed)),
            "min_executed_shed_fraction": min(shed),
            "max_executed_shed_fraction": max(shed),
            "median_executed_shed_kw": float(np.median(
                [float(row["executed_shed_w"]) if row["within_contract"] == "True"
                 else 0.0 for row in group])) / 1000,
            "median_committed_kv_fraction": float(np.median(kv)),
            "contracts_met": sum(row["within_contract"] == "True"
                                 for row in group),
            **{f"median_{region}_{method}": float(np.median(
                [float(row[f"{region}_{method}"]) for row in group]))
               for region in REGIONS
               for method in ("replay", "kv_transfer")},
        })

    # The multi-action claim, measured: at each (deadline, rho) how much more
    # the best flexible policy shed than the best single-action baseline.
    SINGLE = ("kv_only", "replay_only")
    advantage = []
    for deadline, rho in sorted({(k[0], k[2]) for k in keys}):
        cell = {row["policy"]: row for row in frontier
                if (row["deadline_s"], row["rho"]) == (deadline, rho)}
        if not set(SINGLE) <= set(cell):
            continue
        best_single = max(cell[p]["median_executed_shed_fraction"] for p in SINGLE)
        flexible = {p: cell[p]["median_executed_shed_fraction"]
                    for p in cell if p not in SINGLE}
        best_flexible = max(flexible.values(), default=0.0)
        advantage.append({
            "deadline_s": deadline, "rho": rho,
            "best_single_action": best_single,
            "best_flexible": best_flexible,
            "multi_action_gain": best_flexible - best_single,
            "best_flexible_policy": max(flexible, key=flexible.get)
            if flexible else "",
            "best_flexible_kv_fraction": max(
                (row["median_committed_kv_fraction"] for row in frontier
                 if (row["deadline_s"], row["rho"]) == (deadline, rho)
                 and row["policy"] not in SINGLE), default=0.0),
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
    sensitivity = [row for row in rows if row["headline"] == "False"]
    if sensitivity:
        write_csv(out / "sensitivity.csv", sensitivity)

    summary = {
        "schema": SCHEMA, "claim": manifest["claim"],
        "sessions": manifest["sessions"], "envelope": manifest["envelope"],
        "rows": len(rows), "headline_rows": len(headline),
        "rho_grid": list(RHOS),
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
            "This profile has no phase_power, so sessions are priced by "
            "their initial marginal power, which undercounts the concave "
            "removable power several-fold; the calibration ladder compensates "
            "by scaling the credit ask until the realised shed matches the "
            "request. The LP objective is modular and true shed is "
            "supermodular, so no per-session price can make the last session "
            "on an instance worth more than the first.",
            "queue_haul is the target-first LP. In a separate spot check at a "
            "300 s deadline and a 25% request it reached 0.96 of the requested "
            "shed where the Lagrangian greedy, which scores prefixes on the "
            "exact joint gain, reached 1.00. That solver is not swept here "
            "because its prefix scoring is superlinear and does not finish at "
            "fleet scale; the gap is the linear proxy, not the resource model.",
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
    (out / "README.md").write_text(
        f"# Fleet shed frontier, {manifest['sessions']:,} sessions\n\n"
        f"One shedding source site ({manifest['source_site']}) and two equally "
        f"sized destination sites ({', '.join(manifest['sites'].values())}), "
        f"gpt-oss-20b on A100. Each source node owns its egress pipe at the "
        f"measured per-pipeline rate, and reaches only the destination pools "
        f"its own path serves.\n\n"
        f"Every cell reports the **largest shed the policy actually executed** "
        f"inside every contract: committed by the deadline, destinations "
        f"within the measured offered-RPS envelope, and pool service contracts "
        f"honoured. Seeds are aggregated by the median.\n\n"
        f"Destination admission is capped at the measured "
        f"{envelope:g} offered RPS per replica, the last swept rate whose "
        f"median p90 TTFT meets the {manifest['envelope']['normal']['ttft_slo_s']} s "
        f"SLO. Across {len(compliance):,} headline rows, "
        f"{len(breaches)} exceeded that envelope under normal admission.\n\n"
        f"Power is accelerator-scoped: the sum of a measured per-GPU curve over "
        f"the modeled fleet. No facility, cooling, or host power is claimed.\n\n"
        f"## Executed shed at rho={headline_rho}\n\n"
        f"| Deadline (s) | Policy | Median executed shed | Median shed (kW) "
        f"| KV share of commits |\n|---|---|---|---|---|\n{table}\n\n"
        f"## What multiple actions buy at rho={headline_rho}\n\n"
        f"| Deadline (s) | rho | Best single action | Best flexible | Gain |\n"
        f"|---|---|---|---|---|\n{gains}\n" if gains else "")
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
