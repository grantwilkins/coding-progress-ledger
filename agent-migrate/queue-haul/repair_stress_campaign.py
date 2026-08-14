"""Preflight and run a paired, early-fault live repair experiment.

The preflight is deliberately prospective: it evaluates every registered
target, exports every result, and freezes a hardware plan only when a target
passes the common qualification rule.  Hardware execution is implemented in
``repair_hardware_campaign`` so the calibrated route and prefill controls stay
identical to the original repair experiment.
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import statistics
import time
from dataclasses import asdict, replace
from pathlib import Path

import migration_profiler as profiler
import network_campaign as network
import pool_planner
import repair_hardware_campaign as hardware
import repair_plan_shift_campaign as repair_sim
from power_model import ExpectedPower
from profiles import ModelProfile
from repair_controller import (
    Assignment,
    Attempt,
    AttemptUpdate,
    FeasibilityRepairController,
    ObservationBatch,
    ProposedDiff,
    RepairMove,
    RepairRequest,
    RevisedMaximum,
)


ROOT = Path(__file__).parent
DEFAULT_BASE_PLAN = ROOT / "outputs/repair-scheduled-hardware-20260814/plan.json"
DEFAULT_TIMING = Path(
    "/datadrive/queue-haul-repair-20260814-r3/calibration/summary.json")
DEFAULT_OUT = ROOT / "outputs/repair-stress-hardware-20260814"
SWEEP_SCHEMA = "queue-haul-repair-stress-preflight-v1"
PLAN_SCHEMA = "queue-haul-repair-stress-hardware-plan-v1"
TARGET_FRACTIONS = (0.50, 0.55, 0.60, 0.65, 0.70)
HEALTHY_EAST_LOADS = (0.50, 0.40, 0.30, 0.25)
FAULT_AT_S = 1.0
DETECTION_AT_S = 2.0
POWER_DEADLINE_S = 30.0
MIGRATION_CUTOFF_S = 25.0
REPAIR_LATEST_S = 22.0
CONTROL_MIN_SHORTFALL_FRACTION = 0.05
MIN_TARGET_TIME_GAP_S = 5.0
MAX_PREDECISION_FRACTION = 0.25
MIN_REDIRECTED_SESSIONS = 2
HARDWARE_REPEATS = 5
HOST_RETRY_S = 30.0
MOVE_CONCURRENCIES = (4, 3, 2)
CONTEXT_SEEDS = tuple(range(64))
CONTEXT_SUPPORT = (14_042, 30_785, 31_547)
FAULT_STATE = "germany"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    for candidate in (value, ROOT.parent / value, ROOT / value):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def _scenario(template: dict, plan: dict, target_fraction: float,
              healthy_east_load: float, degraded: bool) -> dict:
    scenario = json.loads(json.dumps(template))
    state = FAULT_STATE if degraded else "none"
    rates = hardware._planning_bandwidths(template, plan, state)
    scenario["background"]["east"][0] = healthy_east_load
    scenario.update({
        "design": "repair_stress_preflight",
        "condition_id": f"early-{FAULT_STATE}-joint-0.1x",
        "bandwidth": "scheduled_0.1x" if degraded else "natural",
        "bandwidth_mbps": rates,
        "requested_shed_fraction": target_fraction,
        "deadline_s": POWER_DEADLINE_S,
        "planning_deadline_s": POWER_DEADLINE_S,
        "admission_mode": "normal",
    })
    return scenario


def _intervals(durations: list[float], concurrency: int) -> list[tuple[float, float]]:
    workers = [(0.0, worker) for worker in range(concurrency)]
    rows = []
    for duration in durations:
        start, worker = heapq.heappop(workers)
        rows.append((start, start + duration))
        heapq.heappush(workers, (start + duration, worker))
    return rows


def _attainment(power: ExpectedPower, schedule: list[dict], target: float,
                cutoff_s: float) -> dict:
    credited: set[str] = set()
    curve = [{"time_s": 0.0, "shed_w": 0.0}]
    target_s = None
    for row in sorted(schedule, key=lambda value: (
            value["completion_s"], value["session_id"])):
        credited.add(row["session_id"])
        shed = float(power.drain_gain(frozenset(credited)))
        curve.append({"time_s": row["completion_s"], "shed_w": shed})
        if target_s is None and shed >= target - 1e-8:
            target_s = row["completion_s"]
    cutoff_ids = frozenset(
        row["session_id"] for row in schedule
        if row["completion_s"] <= cutoff_s)
    return {
        "target_s": target_s,
        "cutoff_shed_w": float(power.drain_gain(cutoff_ids)),
        "curve": curve,
    }


def _fixed_plan_schedule(moves, durations, concurrency: int) -> list[dict]:
    return [{
        "session_id": move.session_id,
        "status": "stable_initial_plan",
        "method": move.method,
        "destination": move.destination_instance,
        "pool": move.destination_pool,
        "scheduled_start_s": start,
        "completion_s": end,
    } for move, (start, end) in zip(
        moves, _intervals(durations, concurrency))]


def _schedule_rows(attempts, original, repair_moves, decision_s: float,
                   concurrency: int) -> list[dict]:
    """Schedule a residual plan with the declared hardware worker width."""
    rows = []
    for move in original:
        attempt = attempts.get(move.session_id)
        if attempt and attempt.status == "committed":
            rows.append({
                "session_id": move.session_id,
                "generation": attempt.generation,
                "status": "committed_before_event",
                "method": move.method,
                "destination": move.destination_instance,
                "pool": move.destination_pool,
                "completion_s": attempt.observed_s,
                "retained_work": attempt.total_work,
                "discarded_work": 0.0,
            })
    available: dict[str, float] = {}
    workers = [(decision_s, worker) for worker in range(concurrency)]
    ordered = sorted(
        enumerate(repair_moves),
        key=lambda item: (
            attempts.get(item[1].session_id) is None
            or attempts[item[1].session_id].status != "running",
            item[0],
        ),
    )
    for _, move in ordered:
        before = attempts.get(move.session_id)
        retained = 0.0
        if before and before.assignment == move.assignment:
            retained = before.completed_work
        resource = (
            f"migration:{move.assignment.pool}:{move.assignment.method}")
        worker_ready, worker = heapq.heappop(workers)
        start = max(worker_ready, available.get(resource, decision_s))
        completion = start + move.duration_s
        heapq.heappush(workers, (completion, worker))
        available[resource] = completion
        rows.append({
            "session_id": move.session_id,
            "generation": 0 if before is None else before.generation,
            "status": "scheduled_after_repair",
            "method": move.assignment.method,
            "destination": move.assignment.destination,
            "pool": move.assignment.pool,
            "scheduled_start_s": start,
            "completion_s": completion,
            "retained_work": retained,
            "discarded_work": (
                0.0 if before is None else before.completed_work - retained),
        })
    return sorted(rows, key=lambda row: (
        row["completion_s"], row["session_id"]))


def _simulate_target(parent: dict, plan: dict, timing: dict,
                     manifest: dict, profile: ModelProfile,
                     target_fraction: float,
                     healthy_east_load: float,
                     move_concurrency: int,
                     context_seed: int = 8,
                     sweep_phase: str = "resource_headroom") -> dict:
    raw_template = json.loads(json.dumps(hardware._template(parent)))
    context_rng = random.Random(context_seed)
    for session in raw_template["sessions"]:
        session["initial_tokens"] = context_rng.choice(CONTEXT_SUPPORT)
    template = hardware._promote_components(
        raw_template, plan["network_contract"])
    base_scenario = _scenario(
        raw_template, plan, target_fraction, healthy_east_load, False)
    degraded_scenario = hardware._apply_timing_fit(
        _scenario(template, plan, target_fraction, healthy_east_load, True),
        timing,
        repair_sim._affected(FAULT_STATE))
    base = network._scenario_problem(base_scenario, manifest, profile)
    changed = network._scenario_problem(degraded_scenario, manifest, profile)
    problem, architecture, routes, target, _demand = base
    changed_problem, changed_architecture, _changed_routes, changed_target, _ = changed
    if abs(target - changed_target) > 1e-8:
        raise RuntimeError("stress disturbance changed its shed target")
    initial = network.solve(
        problem, profile, routes, "lp_work_first", destination=architecture,
        admission_mode="normal")
    if not initial.moves:
        raise RuntimeError("stress preflight produced an empty initial plan")
    initial_table = pool_planner.candidate_table(
        problem, profile, architecture, "normal",
        ExpectedPower(problem, profile))
    initial_map = repair_sim._candidate_map(initial_table, architecture)
    durations = [initial_map[
        (move.session_id, move.method, move.destination_pool)
    ].duration_s for move in initial.moves]
    stable_schedule = _fixed_plan_schedule(
        initial.moves, durations, move_concurrency)

    capacities = repair_sim._prefill_observations(
        changed_architecture, FAULT_STATE)
    observed_architecture = pool_planner._repair_architecture(
        changed_architecture, capacities)
    changed_table = pool_planner.candidate_table(
        changed_problem, profile, observed_architecture, "normal",
        ExpectedPower(changed_problem, profile))
    changed_map = repair_sim._candidate_map(
        changed_table, observed_architecture)
    intervals = _intervals(durations, move_concurrency)
    attempts, continuations, rate_by_session = [], [], {}
    for move, total, (start_s, completion_s) in zip(
            initial.moves, durations, intervals):
        assignment = Assignment(
            move.method, move.destination_instance, move.destination_pool)
        replacement = changed_map.get((
            move.session_id, move.method, move.destination_pool))
        rate = total / replacement.duration_s if replacement else 1.0
        if replacement is None and move.destination_instance == FAULT_STATE:
            rate *= hardware.CUT_SCALE
        completed = min(total, max(0.0, FAULT_AT_S - start_s))
        status = ("committed" if completed >= total else
                  "running" if start_s <= FAULT_AT_S else "pending")
        attempt = Attempt(
            move.session_id, 0, assignment, status, total, completed,
            FAULT_AT_S, completion_s, rate=rate,
            repairable=status != "running")
        attempts.append(attempt)
        if status != "committed":
            continuations.append(RepairMove(
                move.session_id, assignment,
                (total - completed) / max(rate, 1e-12), total))
        rate_by_session[move.session_id] = rate

    attempt_map = {attempt.session_id: attempt for attempt in attempts}
    control_schedule = _schedule_rows(
        attempt_map, initial.moves, tuple(continuations),
        FAULT_AT_S, move_concurrency)
    planned_commits = {
        row["session_id"]: row["completion_s"] for row in control_schedule
        if row["status"] == "scheduled_after_repair"
    }
    attempts = [replace(
        attempt,
        planned_commit_s=planned_commits.get(
            attempt.session_id, attempt.planned_commit_s),
    ) for attempt in attempts]
    power = ExpectedPower(replace(
        problem, final_state="awake", assumed_shutdown_s=None), profile)
    controller = FeasibilityRepairController(
        tuple(attempts), {session.session_id for session in problem.sessions},
        float(target), MIGRATION_CUTOFF_S, 0, power.drain_gain)
    route_rates = tuple((link.link_id, link.bytes_per_s)
                        for link in changed_problem.links)
    first = controller.observe(ObservationBatch(
        1, FAULT_AT_S, route_rates=route_rates,
        prefill_capacities=capacities))
    if first is not None:
        raise RuntimeError("stress repair fired before its second sample")
    updates = tuple(AttemptUpdate(
        attempt.session_id, attempt.generation,
        "committed" if attempt.completed_work + rate_by_session[
            attempt.session_id] * (DETECTION_AT_S - FAULT_AT_S)
        >= attempt.total_work else attempt.status,
        attempt.total_work,
        min(attempt.total_work, attempt.completed_work + rate_by_session[
            attempt.session_id] * (DETECTION_AT_S - FAULT_AT_S)),
    ) for attempt in attempts if attempt.status == "running")
    decision = controller.observe(ObservationBatch(
        2, DETECTION_AT_S, attempts=updates, route_rates=route_rates,
        prefill_capacities=capacities))
    request = decision if isinstance(decision, RepairRequest) else None
    repair_result = None
    for _ in range(2):
        if not isinstance(decision, RepairRequest):
            break
        repair_result = pool_planner.repair_destination(
            changed_problem, profile, changed_architecture, decision, "normal")
        decision = controller.complete_repair(repair_result, DETECTION_AT_S)
    observed_attempts = dict(controller.attempts)
    if isinstance(decision, ProposedDiff):
        outcome, repair_moves = "applied", decision.moves
    elif isinstance(decision, RevisedMaximum):
        outcome = "revised_maximum"
        repair_moves = repair_sim._remaining_moves(observed_attempts)
    else:
        outcome = "unchanged"
        repair_moves = repair_sim._remaining_moves(observed_attempts)
    repair_schedule = _schedule_rows(
        observed_attempts, initial.moves, repair_moves,
        DETECTION_AT_S, move_concurrency)
    repair_final = tuple(
        move for move in initial.moves
        if observed_attempts[move.session_id].status == "committed") \
        + repair_sim._planned_moves(repair_moves, changed_architecture)
    diff = repair_sim._diff(initial.moves, repair_final)

    stable = _attainment(power, stable_schedule, float(target), REPAIR_LATEST_S)
    repaired = _attainment(
        power, repair_schedule, float(target), MIGRATION_CUTOFF_S)
    control = _attainment(
        power, control_schedule, float(target), MIGRATION_CUTOFF_S)
    predecision_ids = frozenset(
        row["session_id"] for row in control_schedule
        if row["completion_s"] <= DETECTION_AT_S)
    predecision_shed = float(power.drain_gain(predecision_ids))
    gap = (None if control["target_s"] is None or repaired["target_s"] is None
           else control["target_s"] - repaired["target_s"])
    no_extrapolation = all(
        not value.allow_extrapolation
        for dtype in observed_architecture.types
        for value in dtype.migration.values())
    shortfall_fraction = max(
        0.0, (float(target) - control["cutoff_shed_w"]) / float(target))
    qualifies = (
        outcome == "applied"
        and stable["target_s"] is not None
        and stable["target_s"] <= REPAIR_LATEST_S
        and repaired["target_s"] is not None
        and repaired["target_s"] <= REPAIR_LATEST_S
        and shortfall_fraction >= CONTROL_MIN_SHORTFALL_FRACTION
        and (gap is None or gap >= MIN_TARGET_TIME_GAP_S)
        and predecision_shed <= MAX_PREDECISION_FRACTION * float(target)
        and diff["redirected_sessions"] >= MIN_REDIRECTED_SESSIONS
        and no_extrapolation
    )
    return {
        "target_fraction": target_fraction,
        "healthy_east_load": healthy_east_load,
        "move_concurrency": move_concurrency,
        "context_seed": context_seed,
        "sweep_phase": sweep_phase,
        "requested_shed_w": float(target),
        "fault_at_s": FAULT_AT_S,
        "detection_at_s": DETECTION_AT_S,
        "power_deadline_s": POWER_DEADLINE_S,
        "migration_cutoff_s": MIGRATION_CUTOFF_S,
        "outcome": outcome,
        "stable_target_s": stable["target_s"],
        "repair_target_s": repaired["target_s"],
        "repair_cutoff_shed_w": repaired["cutoff_shed_w"],
        "control_target_s": control["target_s"],
        "control_cutoff_shed_w": control["cutoff_shed_w"],
        "control_shortfall_fraction": shortfall_fraction,
        "target_time_gap_s": gap,
        "predecision_shed_w": predecision_shed,
        "predecision_fraction": predecision_shed / float(target),
        "diff": diff,
        "no_extrapolation": no_extrapolation,
        "qualifies": qualifies,
        "initial_moves": [asdict(move) for move in initial.moves],
        "repair_moves": [asdict(move) for move in repair_final],
        "stable_schedule": stable_schedule,
        "repair_schedule": repair_schedule,
        "control_schedule": control_schedule,
        "repair_curve": repaired["curve"],
        "control_curve": control["curve"],
        "repair_result": None if repair_result is None else asdict(repair_result),
    }


def preflight(base_plan_path: Path, timing_path: Path, out: Path) -> dict:
    plan = json.loads(base_plan_path.read_text())
    hardware.validate_plan(plan)
    timing = json.loads(timing_path.read_text())
    if timing.get("schema") != "queue-haul-repair-10x-timing-fit-v3" \
            or not timing.get("passed"):
        raise ValueError("stress preflight requires the passing live 0.1x fit")
    parent_path = _resolve(plan["parent"]["path"])
    parent = json.loads(parent_path.read_text())
    manifest_path = _resolve(plan["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    profile = ModelProfile.load(network.MODEL_PATH)
    resource_cells = [_simulate_target(
        parent, plan, timing, manifest, profile, target, east_load,
        move_concurrency, 8, "resource_headroom")
        for move_concurrency in MOVE_CONCURRENCIES
        for east_load in HEALTHY_EAST_LOADS for target in TARGET_FRACTIONS]
    workload_cells = [_simulate_target(
        parent, plan, timing, manifest, profile, target, 0.50, 4,
        context_seed, "workload_bootstrap")
        for context_seed in CONTEXT_SEEDS for target in TARGET_FRACTIONS]
    cells = resource_cells + workload_cells
    qualifiers = [row for row in workload_cells if row["qualifies"]]
    selected = min(qualifiers, key=lambda row: (
        -row["target_fraction"], row["context_seed"])) \
        if qualifiers else None
    summaries = [{
        key: value for key, value in row.items()
        if key not in {
            "initial_moves", "repair_moves", "stable_schedule",
            "repair_schedule", "control_schedule", "repair_curve",
            "control_curve", "repair_result",
        }
    } for row in cells]
    bundle = {
        "schema": SWEEP_SCHEMA,
        "semantics": "prospective full target sweep with fixed early fault",
        "base_plan": {"path": str(base_plan_path),
                      "sha256": profiler.file_hash(base_plan_path)},
        "timing": {"path": str(timing_path),
                   "sha256": profiler.file_hash(timing_path),
                   "schema": timing["schema"]},
        "parent": {"path": str(parent_path),
                   "sha256": profiler.file_hash(parent_path)},
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "model_profile": {"path": str(network.MODEL_PATH),
                          "sha256": profiler.file_hash(network.MODEL_PATH)},
        "targets": list(TARGET_FRACTIONS),
        "healthy_east_loads": list(HEALTHY_EAST_LOADS),
        "move_concurrencies": list(MOVE_CONCURRENCIES),
        "context_seeds": list(CONTEXT_SEEDS),
        "context_support": list(CONTEXT_SUPPORT),
        "qualification": {
            "repair_latest_s": REPAIR_LATEST_S,
            "control_min_shortfall_fraction": CONTROL_MIN_SHORTFALL_FRACTION,
            "minimum_target_time_gap_s": MIN_TARGET_TIME_GAP_S,
            "maximum_predecision_fraction": MAX_PREDECISION_FRACTION,
            "minimum_redirected_sessions": MIN_REDIRECTED_SESSIONS,
            "requires_no_extrapolation": True,
        },
        "cells": summaries,
        "selected_cell": selected,
        "qualified_targets": sorted({
            row["target_fraction"] for row in qualifiers}),
        "qualified_cells": [{
            "target_fraction": row["target_fraction"],
            "healthy_east_load": row["healthy_east_load"],
            "move_concurrency": row["move_concurrency"],
            "context_seed": row["context_seed"],
        } for row in qualifiers],
        "selected_target": None if selected is None else selected[
            "target_fraction"],
        "selected_healthy_east_load": None if selected is None else selected[
            "healthy_east_load"],
        "selected_move_concurrency": None if selected is None else selected[
            "move_concurrency"],
        "selected_context_seed": None if selected is None else selected[
            "context_seed"],
        "passed": selected is not None,
    }
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "preflight.json", bundle)
    profiler.write_csv(out / "preflight.csv", [{
        key: value for key, value in row.items()
        if not isinstance(value, (dict, list))
    } for row in summaries])
    return bundle


def _selected_cell(preflight_bundle: dict) -> dict:
    if not preflight_bundle.get("passed"):
        raise ValueError("stress hardware plan requires a passing preflight")
    selected = preflight_bundle.get("selected_cell")
    if not selected or not selected.get("qualifies") \
            or selected["sweep_phase"] != "workload_bootstrap" \
            or selected["target_fraction"] \
            != preflight_bundle["selected_target"] \
            or selected["context_seed"] \
            != preflight_bundle["selected_context_seed"]:
        raise ValueError("stress preflight selected cell is not unique")
    return selected


def make_hardware_plan(base_plan_path: Path, preflight_path: Path) -> dict:
    base = json.loads(base_plan_path.read_text())
    hardware.validate_plan(base)
    sweep = json.loads(preflight_path.read_text())
    if sweep.get("schema") != SWEEP_SCHEMA:
        raise ValueError("invalid stress preflight schema")
    selected = _selected_cell(sweep)
    initial_hash = profiler.object_hash(selected["initial_moves"])
    pair_order = list(range(HARDWARE_REPEATS))
    order_rng = random.Random(20260814)
    order_rng.shuffle(pair_order)
    episodes = []
    for pair in pair_order:
        policies = [hardware.APPLY_POLICY, hardware.CONTROL_POLICY]
        order_rng.shuffle(policies)
        for policy in policies:
            episodes.append({
                "episode_id": profiler.object_hash((
                    "repair-stress", pair, policy, selected["context_seed"],
                    selected["target_fraction"]))[:16],
                "pair": pair,
                "policy": policy,
                "bandwidth_state": FAULT_STATE,
                "prefill_state": FAULT_STATE,
                "cut_scale": hardware.CUT_SCALE,
                "fault_at_s": FAULT_AT_S,
                "detection_at_s": DETECTION_AT_S,
                "power_deadline_s": POWER_DEADLINE_S,
                "migration_cutoff_s": MIGRATION_CUTOFF_S,
                "target_shed_fraction": selected["target_fraction"],
                "healthy_east_load": selected["healthy_east_load"],
                "move_concurrency": selected["move_concurrency"],
                "context_seed": selected["context_seed"],
                "context_support": list(CONTEXT_SUPPORT),
                "expected_initial_moves_sha256": initial_hash,
                "frozen_initial_moves": selected["initial_moves"],
            })
    implementation_paths = tuple(dict.fromkeys(
        hardware.IMPLEMENTATION_FILES + ("repair_stress_campaign.py",)))
    git_sha, dirty = profiler.git_state(True)
    plan = {
        "schema": PLAN_SCHEMA,
        "semantics": (
            "five randomized interleaved early-fault repair/control pairs; "
            "control dispatches continuously during detection"),
        "base_plan": {"path": str(base_plan_path.resolve()),
                      "sha256": profiler.file_hash(base_plan_path)},
        "preflight": {"path": str(preflight_path.resolve()),
                      "sha256": profiler.file_hash(preflight_path),
                      "schema": SWEEP_SCHEMA},
        "timing": sweep["timing"],
        "parent": base["parent"],
        "cluster": base["cluster"],
        "cluster_input": base["cluster_input"],
        "manifest": base["manifest"],
        "model_profile": base["model_profile"],
        "network_contract": base["network_contract"],
        "selection": {
            key: selected[key] for key in (
                "sweep_phase", "target_fraction", "healthy_east_load",
                "move_concurrency", "context_seed", "requested_shed_w",
                "stable_target_s", "repair_target_s", "control_target_s",
                "control_shortfall_fraction", "predecision_fraction",
                "diff", "no_extrapolation")
        },
        "qualification": sweep["qualification"],
        "pair_order_seed": 20260814,
        "repeats": HARDWARE_REPEATS,
        "episodes": episodes,
        "implementation": {
            "git_sha": git_sha,
            "dirty": dirty,
            "files": [{
                "path": str((ROOT / name).resolve()),
                "sha256": profiler.file_hash(ROOT / name),
            } for name in implementation_paths],
        },
    }
    validate_hardware_plan(plan)
    return plan


def validate_hardware_plan(plan: dict) -> None:
    episodes = plan.get("episodes", ())
    pairs = {row.get("pair") for row in episodes}
    if plan.get("schema") != PLAN_SCHEMA \
            or plan.get("repeats") != HARDWARE_REPEATS \
            or len(episodes) != 2 * HARDWARE_REPEATS \
            or pairs != set(range(HARDWARE_REPEATS)):
        raise ValueError("invalid stress hardware plan shape")
    for pair in pairs:
        rows = [row for row in episodes if row["pair"] == pair]
        if {row["policy"] for row in rows} != {
                hardware.APPLY_POLICY, hardware.CONTROL_POLICY}:
            raise ValueError("stress pair does not contain both policies")
        frozen = {row["expected_initial_moves_sha256"] for row in rows}
        if len(frozen) != 1:
            raise ValueError("stress pair does not freeze one initial plan")
    if any(row["cut_scale"] != 0.1
           or row["fault_at_s"] != FAULT_AT_S
           or row["detection_at_s"] != DETECTION_AT_S
           or row["migration_cutoff_s"] != MIGRATION_CUTOFF_S
           for row in episodes):
        raise ValueError("stress hardware disturbance changed")


def prepare_hardware(base_plan_path: Path, preflight_path: Path,
                     out: Path) -> dict:
    plan = make_hardware_plan(base_plan_path, preflight_path)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", plan)
    return plan


def _p90(values: list[float]) -> float | None:
    return (sorted(values)[int(0.9 * (len(values) - 1))]
            if values else None)


def reduce_hardware(plan: dict, run_root: Path) -> dict:
    validate_hardware_plan(plan)
    results = []
    for episode in plan["episodes"]:
        path = run_root / "episodes" / episode["episode_id"] / "result.json"
        if path.is_file():
            results.append({**episode, **json.loads(path.read_text())})
    episode_rows = []
    ttft_rows = []
    for row in results:
        shortfall = max(0.0, (
            row["requested_shed_w"] - row["cutoff_shed_w"]
        ) / row["requested_shed_w"])
        episode_rows.append({
            "episode_id": row["episode_id"], "pair": row["pair"],
            "policy": row["policy"], "event_s": row["event_s"],
            "decision_s": row["decision_s"], "proposal_s": row["proposal_s"],
            "apply_s": row["apply_s"], "repair_outcome": row["repair_outcome"],
            "redirected_sessions": row["redirected_sessions"],
            "requested_shed_w": row["requested_shed_w"],
            "cutoff_shed_w": row["cutoff_shed_w"],
            "cutoff_shortfall_fraction": shortfall,
            "target_met_by_cutoff": row["target_met_by_cutoff"],
            "time_to_target_s": row["time_to_target_s"],
            "predecision_shed_w": row["predecision_shed_w"],
            "initial_moves_sha256": row["initial_moves_sha256"],
            "requests": len(row["requests"]),
            "ttft_recorded": row["ttft_recorded"],
        })
        ttft_rows.extend({
            "episode_id": row["episode_id"], "pair": row["pair"],
            "policy": row["policy"], "session_id": request["session_id"],
            "method": request["method"],
            "destination": request["destination_instance"],
            "start_ns": request["request"].get("start_ns"),
            "first_token_ns": request["request"].get("first_byte_ns"),
            "end_ns": request["request"].get("end_ns"),
            "ttft_s": request["ttft_s"],
            "http_status": request["request"].get("status_code"),
        } for request in row["requests"])
    if episode_rows:
        profiler.write_csv(run_root / "episode_summary.csv", episode_rows)
    if ttft_rows:
        profiler.write_csv(run_root / "ttft.csv", ttft_rows)
    common_rows = []
    pair_checks = []
    for pair in range(HARDWARE_REPEATS):
        pair_results = [row for row in results if row["pair"] == pair]
        by_policy = {row["policy"]: row for row in pair_results}
        repair = by_policy.get(hardware.APPLY_POLICY)
        control = by_policy.get(hardware.CONTROL_POLICY)
        if repair and control:
            repair_ttft = {row["session_id"]: row["ttft_s"]
                           for row in repair["requests"]}
            control_ttft = {row["session_id"]: row["ttft_s"]
                            for row in control["requests"]}
            for session_id in sorted(repair_ttft.keys() & control_ttft.keys()):
                common_rows.append({
                    "pair": pair, "session_id": session_id,
                    "repair_ttft_s": repair_ttft[session_id],
                    "control_ttft_s": control_ttft[session_id],
                    "control_minus_repair_ttft_s": (
                        control_ttft[session_id] - repair_ttft[session_id]),
                })
        requested = repair["requested_shed_w"] if repair else None
        pair_checks.append({
            "pair": pair,
            "complete": repair is not None and control is not None,
            "initial_plan_matched": bool(repair and control and
                repair["initial_moves_sha256"] == control["initial_moves_sha256"]
                == repair["expected_initial_moves_sha256"]),
            "repair_applied": bool(repair and repair["repair_outcome"] == "applied"),
            "repair_target_by_cutoff": bool(
                repair and repair["target_met_by_cutoff"]),
            "control_missed_cutoff": bool(
                control and not control["target_met_by_cutoff"]),
            "control_shortfall_fraction": (None if not control else max(
                0.0, (control["requested_shed_w"] - control["cutoff_shed_w"])
                / control["requested_shed_w"])),
            "predecision_fraction": (None if not repair else
                repair["predecision_shed_w"] / requested),
            "redirected_sessions": (None if not repair else
                repair["redirected_sessions"]),
            "actual_timestamps_recorded": bool(repair and
                repair["solver_timings"] and repair["proposal_s"] is not None
                and repair["apply_s"] is not None),
            "http_and_ttft": bool(repair and control and all(
                request["request"].get("status_code") == 200
                and request.get("ttft_s") is not None
                for row in (repair, control) for request in row["requests"])),
        })
    if common_rows:
        profiler.write_csv(run_root / "common_session_ttft.csv", common_rows)
    for check in pair_checks:
        check["passed"] = (
            check["complete"] and check["initial_plan_matched"]
            and check["repair_applied"] and check["repair_target_by_cutoff"]
            and check["control_missed_cutoff"]
            and check["control_shortfall_fraction"] is not None
            and check["control_shortfall_fraction"]
            >= CONTROL_MIN_SHORTFALL_FRACTION
            and check["predecision_fraction"] is not None
            and check["predecision_fraction"] <= MAX_PREDECISION_FRACTION
            and check["redirected_sessions"] is not None
            and check["redirected_sessions"] >= MIN_REDIRECTED_SESSIONS
            and check["actual_timestamps_recorded"] and check["http_and_ttft"])
    ttfts = [row["ttft_s"] for row in ttft_rows]
    common_deltas = [row["control_minus_repair_ttft_s"] for row in common_rows]
    summary = {
        "schema": "queue-haul-repair-stress-hardware-validation-v1",
        "semantics": (
            "primary endpoint is completion-credited modeled shed by the "
            "25-second migration cutoff; raw A100 traces are diagnostic"),
        "expected_episodes": 2 * HARDWARE_REPEATS,
        "completed_episodes": len(results),
        "pair_checks": pair_checks,
        "repair_target_by_cutoff": sum(
            bool(row.get("repair_target_by_cutoff")) for row in pair_checks),
        "control_target_by_cutoff": sum(
            not bool(row.get("control_missed_cutoff"))
            for row in pair_checks if row["complete"]),
        "ttft_rows": len(ttft_rows),
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p90_s": _p90(ttfts),
        "ttft_max_s": max(ttfts) if ttfts else None,
        "common_session_ttft_rows": len(common_rows),
        "common_control_minus_repair_ttft_p50_s": (
            statistics.median(common_deltas) if common_deltas else None),
        "passed": len(results) == 2 * HARDWARE_REPEATS
        and all(row["passed"] for row in pair_checks),
    }
    profiler.write_json(run_root / "validation.json", summary)
    return summary


def run_hardware(plan_path: Path, key: Path, run_root: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_hardware_plan(plan)
    pinned = ("base_plan", "preflight", "timing", "parent", "cluster_input",
              "manifest", "model_profile")
    for name in pinned:
        item = plan[name]
        path = _resolve(item["path"])
        if profiler.file_hash(path) != item["sha256"]:
            raise RuntimeError(f"stress hardware input changed: {name}")
    for row in plan["implementation"]["files"]:
        if profiler.file_hash(_resolve(row["path"])) != row["sha256"]:
            raise RuntimeError(f"stress implementation changed: {row['path']}")
    parent = json.loads(_resolve(plan["parent"]["path"]).read_text())
    manifest = json.loads(_resolve(plan["manifest"]["path"]).read_text())
    timing = json.loads(_resolve(plan["timing"]["path"]).read_text())
    if not timing.get("passed"):
        raise RuntimeError("pinned live 0.1x timing gate no longer passes")
    profile = ModelProfile.load(_resolve(plan["model_profile"]["path"]))
    cluster = network.Cluster.parse(plan["cluster"])
    run_root.mkdir(parents=True, exist_ok=True)
    existing_plan_path = run_root / "plan.json"
    if existing_plan_path.is_file() \
            and json.loads(existing_plan_path.read_text()) != plan:
        raise RuntimeError("stress run root belongs to a different plan")
    profiler.write_json(run_root / "plan.json", plan)
    host_attempt = 0
    while True:
        host_attempt += 1
        profiler.write_json(run_root / "status.json", {
            "schema": "queue-haul-repair-stress-status-v1",
            "phase": "host_preflight", "attempt": host_attempt,
            "completed": 0, "expected": len(plan["episodes"]),
        })
        try:
            host_reports = network.host_check(cluster, key)
        except Exception as error:
            profiler.write_json(run_root / "status.json", {
                "schema": "queue-haul-repair-stress-status-v1",
                "phase": "waiting_for_hosts", "attempt": host_attempt,
                "completed": 0, "expected": len(plan["episodes"]),
                "retry_in_s": HOST_RETRY_S,
                "error": f"{type(error).__name__}: {error}",
            })
            time.sleep(HOST_RETRY_S)
            continue
        profiler.write_json(run_root / "host_reports.json", host_reports)
        break
    stack_root = run_root / f"stack-{profiler.object_hash(str(run_root))[:16]}"
    stack = network.start_cluster(
        cluster, key, plan["network_contract"], "natural", stack_root,
        power_interval_s=.1)
    completed = 0
    try:
        for episode in plan["episodes"]:
            result_path = (run_root / "episodes" / episode["episode_id"]
                           / "result.json")
            if result_path.is_file():
                result = json.loads(result_path.read_text())
            else:
                episode_plan = {**plan, "apply_policy": episode["policy"]}
                result = hardware._run_episode(
                    stack, episode_plan, parent, manifest, profile, timing,
                    episode, result_path.parent)
            completed += 1
            profiler.write_json(run_root / "status.json", {
                "schema": "queue-haul-repair-stress-status-v1",
                "phase": "episodes", "completed": completed,
                "expected": len(plan["episodes"]),
                "latest_episode_id": episode["episode_id"],
                "latest_pair": episode["pair"],
                "latest_policy": episode["policy"],
                "latest_target_met_by_cutoff": result["target_met_by_cutoff"],
                "latest_ttft_max_s": max(
                    row["ttft_s"] for row in result["requests"]),
            })
    except Exception as error:
        profiler.write_json(run_root / "status.json", {
            "schema": "queue-haul-repair-stress-status-v1",
            "phase": "failed", "completed": completed,
            "expected": len(plan["episodes"]),
            "error": f"{type(error).__name__}: {error}",
        })
        raise
    finally:
        network.stop_cluster(stack)
    summary = reduce_hardware(plan, run_root)
    profiler.write_json(run_root / "status.json", {
        "schema": "queue-haul-repair-stress-status-v1",
        "phase": "complete", "completed": completed,
        "expected": len(plan["episodes"]),
        "validation_passed": summary["passed"],
    })
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("preflight")
    command.add_argument("--base-plan", type=Path, default=DEFAULT_BASE_PLAN)
    command.add_argument("--timing", type=Path, default=DEFAULT_TIMING)
    command.add_argument("--out", type=Path, default=DEFAULT_OUT)
    command = sub.add_parser("prepare-hardware")
    command.add_argument("--base-plan", type=Path, default=DEFAULT_BASE_PLAN)
    command.add_argument("--preflight", type=Path,
                         default=DEFAULT_OUT / "preflight.json")
    command.add_argument("--out", type=Path, default=DEFAULT_OUT)
    command = sub.add_parser("run-hardware")
    command.add_argument("--plan", type=Path,
                         default=DEFAULT_OUT / "plan.json")
    command.add_argument("--ssh-key", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command = sub.add_parser("validate-hardware")
    command.add_argument("--plan", type=Path,
                         default=DEFAULT_OUT / "plan.json")
    command.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.base_plan, args.timing, args.out)
        print(json.dumps({
            "schema": result["schema"],
            "cells": len(result["cells"]),
            "qualified_cells": result["qualified_cells"],
            "selected_target": result["selected_target"],
            "selected_context_seed": result["selected_context_seed"],
            "passed": result["passed"],
        }, indent=2))
    elif args.command == "prepare-hardware":
        plan = prepare_hardware(args.base_plan, args.preflight, args.out)
        print(json.dumps({
            "schema": plan["schema"], "episodes": len(plan["episodes"]),
            "selection": plan["selection"],
            "plan": str(args.out / "plan.json"),
        }, indent=2))
    elif args.command == "run-hardware":
        print(json.dumps(run_hardware(
            args.plan, args.ssh_key.expanduser(), args.run_root), indent=2))
    else:
        plan = json.loads(args.plan.read_text())
        summary = reduce_hardware(plan, args.run_root)
        print(json.dumps(summary, indent=2))
        if not summary["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
