"""Run a ledger-driven scheduled repair grid with calibrated regional timing."""

from __future__ import annotations

import argparse
import heapq
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import migration_profiler as profiler
import network_campaign as network
import plot_style
import pool_planner
from planner import _expected_scenario
from pool_planner import candidate_table
from power_model import ExpectedPower
from profiles import ModelProfile
from repair_controller import (
    Assignment,
    Attempt,
    AttemptUpdate,
    FeasibilityRepairController,
    ObservationBatch,
    PrefillCapacity,
    ProposedDiff,
    RepairMove,
    RepairRequest,
    RevisedMaximum,
)
from simulate import PlannedMove, predict


ROOT = Path(__file__).parent
DEFAULT_PARENT = (
    ROOT / "outputs/timing-power-validation-20260814/"
    "separation-regional-timing-v2.json"
)
CUT_SCALE = 0.1
TRIGGER_WORK_FRACTION = 0.25
DETECTION_SAMPLES = 2
LOCATION_STATES = ("none", "east", "germany", "both")
LOCATIONS = ("east", "germany")
REFERENCE_CONTEXT_TOKENS = network.SINK_LOAD_PREFILL_TOKENS
TARGET_SHED_FRACTION = 0.5
MOVE_CONCURRENCY = 4
plot_style.apply()


def _affected(state: str) -> frozenset[str]:
    if state == "none":
        return frozenset()
    if state == "both":
        return frozenset(LOCATIONS)
    if state not in LOCATIONS:
        raise ValueError(f"unknown disturbance location {state!r}")
    return frozenset((state,))


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    candidates = (value, ROOT.parent / value, ROOT / value)
    resolved = [candidate for candidate in candidates if candidate.is_file()]
    if not resolved:
        raise FileNotFoundError(path)
    return resolved[0]


def _scenario(template: dict, parent: dict, bandwidth_state: str,
              prefill_state: str) -> dict:
    scenario = json.loads(json.dumps(template))
    bandwidth = _affected(bandwidth_state)
    rates = network._bandwidths(parent["network_contract"], "natural")
    for node in bandwidth:
        rates[node] *= CUT_SCALE
    components = scenario.get("migration_components", {})
    for node in bandwidth:
        for value in components[node].values():
            value["allow_extrapolation"] = True
            value["provenance"] += "; uncalibrated 0.1x bandwidth sensitivity"
    scenario.update({
        "design": "scheduled_repair_simulation",
        "condition_id": f"bandwidth-{bandwidth_state}__prefill-{prefill_state}",
        "bandwidth": "scheduled_0.1x" if bandwidth else "natural",
        "bandwidth_mbps": rates,
        "admission_mode": "normal",
        "full_horizon_s": network.ORACLE_STALE_HORIZON_S,
        "requested_shed_fraction": TARGET_SHED_FRACTION,
    })
    return scenario


def _prefill_observations(architecture, state: str) -> tuple[PrefillCapacity, ...]:
    affected = _affected(state)
    return tuple(
        PrefillCapacity(
            pool.pool_id,
            REFERENCE_CONTEXT_TOKENS,
            architecture.type_by_id[pool.type_id].prefill.at(
                REFERENCE_CONTEXT_TOKENS) * CUT_SCALE,
        )
        for pool in architecture.pools
        if pool.pool_id.rsplit("/", 1)[-1] in affected
    )


def _candidate_map(table, architecture) -> dict[tuple[str, str, str], object]:
    return {
        (
            table.sessions[candidate.session].session_id,
            candidate.method,
            architecture.pools[candidate.pool].pool_id,
        ): candidate
        for candidate in table.candidates
    }


def _initial_schedule(durations: list[float],
                      concurrency: int = MOVE_CONCURRENCY
                      ) -> tuple[list[tuple[float, float]], float]:
    if not durations or min(durations) <= 0:
        raise ValueError("scheduled repair requires positive planned work")
    if concurrency < 1:
        raise ValueError("scheduled repair concurrency must be positive")
    workers = [(0.0, worker) for worker in range(concurrency)]
    intervals = []
    for duration in durations:
        start, worker = heapq.heappop(workers)
        end = start + duration
        intervals.append((start, end))
        heapq.heappush(workers, (end, worker))
    target = TRIGGER_WORK_FRACTION * sum(durations)
    lo, hi = 0.0, max(end for _, end in intervals)
    for _ in range(80):
        mid = (lo + hi) / 2
        progress = sum(min(duration, max(0.0, mid - start))
                       for duration, (start, _end)
                       in zip(durations, intervals))
        if progress < target:
            lo = mid
        else:
            hi = mid
    return intervals, hi


def _diff(original, moves):
    signature = lambda move: (
        move.method, move.destination_pool, move.destination_instance)
    before = {move.session_id: signature(move) for move in original}
    after = {move.session_id: signature(move) for move in moves}
    changed = {session for session in before.keys() | after.keys()
               if before.get(session) != after.get(session)}
    return {
        "changed_sessions": len(changed),
        "redirected_sessions": sum(session in before and session in after
                                   for session in changed),
        "added_sessions": len(after.keys() - before.keys()),
        "removed_sessions": len(before.keys() - after.keys()),
        "unchanged_sessions": sum(before.get(session) == action
                                  for session, action in after.items()),
    }


def _planned_moves(repair_moves, architecture) -> tuple[PlannedMove, ...]:
    pools = {pool.pool_id: pool for pool in architecture.pools}
    return tuple(
        PlannedMove(
            move.session_id,
            move.assignment.destination,
            move.assignment.method,
            order,
            pools[move.assignment.pool].route,
            destination_pool=move.assignment.pool,
        )
        for order, move in enumerate(sorted(
            repair_moves, key=lambda row: row.session_id))
    )


def _remaining_moves(attempts) -> tuple[RepairMove, ...]:
    return tuple(
        RepairMove(
            attempt.session_id, attempt.assignment,
            max(0.0, (attempt.total_work - attempt.completed_work)
                / max(attempt.rate or 0.0, 1e-12)),
            attempt.total_work,
        )
        for attempt in attempts.values()
        if attempt.status in {"pending", "running"}
    )


def _schedule_resources(moves, horizon_s: float) -> dict:
    if horizon_s <= 0:
        raise ValueError("repair schedule horizon must be positive")
    used = {}
    for move in moves:
        key = f"migration:{move.assignment.pool}:{move.assignment.method}"
        used[key] = used.get(key, 0.0) + move.duration_s
    rows = [{
        "resource": resource,
        "used_s": value,
        "capacity_s": horizon_s,
        "utilization": value / horizon_s,
        "slack_s": horizon_s - value,
    } for resource, value in sorted(used.items())]
    worker_used = sum(move.duration_s for move in moves)
    rows.append({
        "resource": "migration-workers",
        "used_s": worker_used,
        "capacity_s": MOVE_CONCURRENCY * horizon_s,
        "utilization": worker_used / (MOVE_CONCURRENCY * horizon_s),
        "slack_s": MOVE_CONCURRENCY * horizon_s - worker_used,
    })
    return {
        "resources": rows,
        "maximum_utilization": max(row["utilization"] for row in rows),
        "minimum_slack_s": min(row["slack_s"] for row in rows),
    }


def _schedule_rows(attempts, original, repair_moves, trigger_s, decision_s):
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
    workers = [(decision_s, worker) for worker in range(MOVE_CONCURRENCY)]
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
        resources = (f"migration:{move.assignment.pool}:{move.assignment.method}",)
        worker_ready, worker = heapq.heappop(workers)
        start = max(worker_ready, *(available.get(resource, decision_s)
                                    for resource in resources))
        completion = start + move.duration_s
        heapq.heappush(workers, (completion, worker))
        for resource in resources:
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
            "discarded_work": 0.0 if before is None else before.completed_work - retained,
        })
    return sorted(rows, key=lambda row: (row["completion_s"], row["session_id"]))


def _run_cell(base, degraded, profile, manifest, original_result,
              original_table, bandwidth_state, prefill_state):
    problem, architecture, routes, target, _ = base
    changed_problem, changed_architecture, _, changed_target, _ = degraded
    if abs(target - changed_target) > 1e-8:
        raise RuntimeError("disturbance changed the requested shed target")
    original_map = _candidate_map(original_table, architecture)
    durations = [original_map[
        (move.session_id, move.method, move.destination_pool)
    ].duration_s for move in original_result.moves]
    initial_schedule, trigger_s = _initial_schedule(durations)
    capacities = _prefill_observations(changed_architecture, prefill_state)
    rate_by_session = {}
    changed_architecture_with_capacity = pool_planner._repair_architecture(
        changed_architecture, capacities)
    changed_table = candidate_table(
        changed_problem, profile, changed_architecture_with_capacity, "normal",
        ExpectedPower(changed_problem, profile),
    )
    changed_map = _candidate_map(changed_table, changed_architecture_with_capacity)
    bandwidth_affected = _affected(bandwidth_state)
    prefill_affected = _affected(prefill_state)
    attempts = []
    continuations = []
    for move, total, (start_s, completion_s) in zip(
            original_result.moves, durations, initial_schedule):
        assignment = Assignment(
            move.method, move.destination_instance, move.destination_pool)
        key = (move.session_id, move.method, move.destination_pool)
        replacement = changed_map.get(key)
        if replacement:
            rate = total / replacement.duration_s
        else:
            rate = 1.0
            if move.method == "kv_transfer" \
                    and move.destination_instance in bandwidth_affected:
                rate *= CUT_SCALE
            if move.method == "replay" \
                    and move.destination_instance in prefill_affected:
                rate *= CUT_SCALE
        completed = min(total, max(0.0, trigger_s - start_s))
        status = ("committed" if completed >= total else
                  "running" if start_s <= trigger_s else "pending")
        attempts.append(Attempt(
            move.session_id, 0, assignment,
            status, total, completed, trigger_s, completion_s, rate=rate,
            repairable=status != "running",
        ))
        if status != "committed":
            continuations.append(RepairMove(
                move.session_id, assignment,
                (total - completed) / rate, total,
            ))
        rate_by_session[move.session_id] = rate
    attempt_map = {attempt.session_id: attempt for attempt in attempts}
    continuation_schedule = _schedule_rows(
        attempt_map, original_result.moves, tuple(continuations), trigger_s,
        trigger_s)
    planned_commits = {
        row["session_id"]: row["completion_s"] for row in continuation_schedule
        if row["status"] == "scheduled_after_repair"
    }
    attempts = [replace(
        attempt,
        planned_commit_s=planned_commits.get(
            attempt.session_id, attempt.planned_commit_s),
        observed_s=(completion_s if attempt.status == "committed" else trigger_s),
    ) for attempt, (_start_s, completion_s) in zip(attempts, initial_schedule)]
    power = ExpectedPower(replace(problem, final_state="awake",
                                  assumed_shutdown_s=None), profile)
    credit_deadline = problem.deadline_s - profile.power_window_s
    controller = FeasibilityRepairController(
        tuple(attempts), {session.session_id for session in problem.sessions},
        float(target), credit_deadline, 0.0, power.drain_gain,
    )
    route_rates = tuple((link.link_id, link.bytes_per_s)
                        for link in changed_problem.links)
    first = controller.observe(ObservationBatch(
        1, trigger_s, route_rates=route_rates,
        prefill_capacities=capacities,
    ))
    if first is not None:
        raise RuntimeError("repair fired without two miss samples")
    decision_s = trigger_s + 1
    updates = tuple(
        AttemptUpdate(
            attempt.session_id, attempt.generation,
            "committed" if min(
                attempt.total_work,
                attempt.completed_work + rate_by_session[attempt.session_id],
            ) >= attempt.total_work else attempt.status,
            attempt.total_work,
            min(attempt.total_work,
                attempt.completed_work + rate_by_session[attempt.session_id]),
        )
        for attempt in attempts if attempt.status == "running"
    )
    decision = controller.observe(ObservationBatch(
        2, decision_s, attempts=updates, route_rates=route_rates,
        prefill_capacities=capacities,
    ))
    observed_attempts = dict(controller.attempts)
    request = decision if isinstance(decision, RepairRequest) else None
    repair_result = proposal = None
    for _ in range(2):
        if not isinstance(decision, RepairRequest):
            break
        repair_result = pool_planner.repair_destination(
            changed_problem, profile, changed_architecture,
            decision, "normal",
        )
        decision = controller.complete_repair(repair_result, decision_s)
    if isinstance(decision, ProposedDiff):
        proposal = decision
        controller.acknowledge(proposal.proposal_id, "applied", decision_s)
        outcome = "applied"
        repair_moves = proposal.moves
    elif isinstance(decision, RevisedMaximum):
        outcome = "revised_maximum"
        repair_moves = _remaining_moves(observed_attempts)
    elif decision is None:
        outcome = "unchanged"
        repair_moves = _remaining_moves(observed_attempts)
    else:
        raise RuntimeError(f"unexpected repair decision {decision!r}")
    committed_ids = {session_id for session_id, attempt in observed_attempts.items()
                     if attempt.status == "committed"}
    committed_moves = tuple(move for move in original_result.moves
                            if move.session_id in committed_ids)
    final_moves = committed_moves + _planned_moves(
        repair_moves, changed_architecture)
    schedule = _schedule_rows(
        observed_attempts, original_result.moves, repair_moves, trigger_s,
        decision_s)
    credited = {row["session_id"] for row in schedule
                if row["completion_s"] <= credit_deadline}
    forecast_w = float(power.drain_gain(frozenset(credited)))
    target_met = forecast_w >= target - 1e-8
    if outcome == "applied" and not target_met:
        raise RuntimeError("applied repair schedule does not restore the target")
    diff = _diff(original_result.moves, final_moves)
    bandwidth_impaired = _affected(bandwidth_state)
    prefill_impaired = _affected(prefill_state)
    def impairment(destination, method):
        return int(destination in bandwidth_impaired) \
            + int(destination in prefill_impaired and method == "replay")
    before_assignments = {
        move.session_id: (move.destination_instance, move.method)
        for move in original_result.moves
    }
    after_assignments = {
        move.session_id: (move.destination_instance, move.method)
        for move in final_moves
    }
    direction = {
        "reduced_impaired_actions": sum(
            impairment(*before_assignments[session])
            > impairment(*after_assignments[session])
            for session in before_assignments.keys() & after_assignments
            if before_assignments[session] != after_assignments[session]
        ),
        "increased_impaired_actions": sum(
            (impairment(*before_assignments[session])
             if session in before_assignments else 0)
            < impairment(*after_assignments[session])
            for session in after_assignments
            if before_assignments.get(session) != after_assignments[session]
        ),
        "removed_from_impaired": sum(
            impairment(*before_assignments[session]) > 0
            for session in before_assignments.keys() - after_assignments.keys()
        ),
    }
    counts = network._constraint_action_counts(final_moves)
    return {
        "bandwidth_state": bandwidth_state,
        "prefill_state": prefill_state,
        "condition": f"bandwidth-{bandwidth_state}__prefill-{prefill_state}",
        "trigger_work_fraction": TRIGGER_WORK_FRACTION,
        "move_concurrency": MOVE_CONCURRENCY,
        "trigger_s": trigger_s,
        "decision_s": decision_s,
        "cut_scale": CUT_SCALE,
        "bandwidth_mbps": {
            link.link_id.rsplit("/", 1)[-1]: link.bytes_per_s / 125_000
            for link in changed_problem.links
        },
        "prefill_capacities": [asdict(row) for row in capacities],
        "repair_requested": request is not None,
        "outcome": outcome,
        "request": None if request is None else {
            "request_id": request.request_id,
            "trigger": request.trigger,
            "budget_version": request.snapshot.budget_version,
        },
        "result": None if repair_result is None else {
            "attainable_watts": repair_result.attainable_watts,
            "reaches_target": repair_result.reaches_target,
        },
        "requested_shed_w": float(target),
        "forecast_shed_w": forecast_w,
        "target_met": target_met,
        "action_mix": counts,
        "diff": diff,
        "repair_direction": direction,
        "resource_utilization": {
            "before": _schedule_resources(
                continuations, credit_deadline - trigger_s),
            "after": _schedule_resources(
                repair_moves, credit_deadline - decision_s),
        },
        "schedule": schedule,
        "moves": [asdict(move) for move in final_moves],
        "timing_evidence": (
            "calibrated" if bandwidth_state == "none" else
            "sensitivity_pending_0.1x_hardware_calibration"
        ),
    }


def _plot(rows, path):
    order = tuple(plot_style.REPAIR_NAMES)
    values = np.empty((len(LOCATION_STATES), len(LOCATION_STATES)))
    changed = np.empty_like(values)
    by_key = {(row["prefill_state"], row["bandwidth_state"]): row
              for row in rows}
    for y, prefill in enumerate(LOCATION_STATES):
        for x, bandwidth in enumerate(LOCATION_STATES):
            row = by_key[prefill, bandwidth]
            values[y, x] = order.index(row["outcome"])
            changed[y, x] = row["diff"]["changed_sessions"]
    fig, axis = plt.subplots(figsize=(6.8, 5.4))
    axis.imshow(values, cmap=ListedColormap([
        plot_style.REPAIR_COLORS[value] for value in order]),
        vmin=-.5, vmax=len(order) - .5)
    axis.set(
        xticks=range(len(LOCATION_STATES)),
        yticks=range(len(LOCATION_STATES)),
        xticklabels=LOCATION_STATES,
        yticklabels=LOCATION_STATES,
        xlabel="Bandwidth cut location",
        ylabel="Prefill cut location",
    )
    for y in range(len(LOCATION_STATES)):
        for x in range(len(LOCATION_STATES)):
            axis.text(x, y, f"{int(changed[y, x])} changed",
                      ha="center", va="center", color="white", fontsize=9)
    fig.legend(
        handles=[Patch(facecolor=plot_style.REPAIR_COLORS[value],
                       label=plot_style.REPAIR_NAMES[value]) for value in order],
        frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, .01),
    )
    fig.subplots_adjust(left=.2, right=.97, bottom=.22, top=.97)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def run(out: Path, parent_path: Path = DEFAULT_PARENT):
    out.mkdir(parents=True, exist_ok=True)
    parent = json.loads(parent_path.read_text())
    manifest_path = _resolve(parent["manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    profile = ModelProfile.load(network.MODEL_PATH)
    template = next(row for row in parent["scenarios"]
                    if row["condition_id"] == "joint-shaped"
                    and row["repeat"] == 0 and row["policy"] == "queue_haul")
    original_scenario = _scenario(template, parent, "none", "none")
    base = network._scenario_problem(original_scenario, manifest, profile)
    problem, architecture, routes, target, _ = base
    original_result = network.solve(
        problem, profile, routes, "lp_work_first", destination=architecture,
        admission_mode="normal")
    original_table = candidate_table(
        problem, profile, architecture, "normal", ExpectedPower(problem, profile))
    original_execution = predict(
        _expected_scenario(problem, original_result.moves), profile,
        original_result.moves, destination=architecture)
    rows = []
    for bandwidth_state in LOCATION_STATES:
        for prefill_state in LOCATION_STATES:
            scenario = _scenario(
                template, parent, bandwidth_state, prefill_state)
            degraded = network._scenario_problem(scenario, manifest, profile)
            rows.append(_run_cell(
                base, degraded, profile, manifest, original_result,
                original_table, bandwidth_state, prefill_state))
    mixes = [{
        "bandwidth_state": row["bandwidth_state"],
        "prefill_state": row["prefill_state"],
        "outcome": row["outcome"],
        "target_met": row["target_met"],
        **row["action_mix"],
    } for row in rows]
    diffs = [{
        "bandwidth_state": row["bandwidth_state"],
        "prefill_state": row["prefill_state"],
        "outcome": row["outcome"],
        **row["diff"],
    } for row in rows]
    sha, dirty = profiler.git_state(True)
    bundle = {
        "schema": "queue-haul-scheduled-repair-simulation-v2",
        "semantics": "one initial plan, ledger observations, residual repair",
        "parent_plan": str(parent_path),
        "parent_plan_sha256": profiler.file_hash(parent_path),
        "manifest": {"path": str(manifest_path),
                     "sha256": profiler.file_hash(manifest_path)},
        "model_profile": {"path": str(network.MODEL_PATH),
                          "sha256": profiler.file_hash(network.MODEL_PATH)},
        "git_sha": sha,
        "dirty": dirty,
        "grid": {
            "bandwidth_states": list(LOCATION_STATES),
            "prefill_states": list(LOCATION_STATES),
            "cut_scale": CUT_SCALE,
            "trigger_work_fraction": TRIGGER_WORK_FRACTION,
            "target_shed_fraction": TARGET_SHED_FRACTION,
            "move_concurrency": MOVE_CONCURRENCY,
        },
        "original": {
            "requested_shed_w": float(target),
            "planned_shed_w": float(original_result.initial_source_power_w
                                     - original_result.planned_source_power_w),
            "simulated_shed_w": float(original_result.initial_source_power_w
                                       - original_execution.modeled_source_power_at_deadline_w),
            "deadline_met": bool(original_execution.deadline_met),
            "moves": [asdict(move) for move in original_result.moves],
        },
        "cells": rows,
    }
    profiler.write_json(out / "plans.json", bundle)
    profiler.write_csv(out / "action_mix.csv", mixes)
    profiler.write_csv(out / "plan_diffs.csv", diffs)
    _plot(rows, out / "repair_grid.png")
    control = next(row for row in rows
                   if row["bandwidth_state"] == row["prefill_state"] == "none")
    passed = len(rows) == 16 and control["outcome"] == "unchanged" \
        and control["target_met"] \
        and all(row["outcome"] in plot_style.REPAIR_NAMES for row in rows) \
        and all(row["target_met"] or row["outcome"] == "revised_maximum"
                for row in rows) \
        and all(
            row["repair_direction"]["increased_impaired_actions"] == 0
            and (row["repair_direction"]["reduced_impaired_actions"]
                 + row["repair_direction"]["removed_from_impaired"] > 0)
            for row in rows if row["outcome"] == "applied"
        )
    report = {
        "schema": "queue-haul-scheduled-repair-validation-v2",
        "cells": len(rows),
        "applied": sum(row["outcome"] == "applied" for row in rows),
        "revised_maximum": sum(row["outcome"] == "revised_maximum" for row in rows),
        "sensitivity_cells": sum(row["timing_evidence"].startswith("sensitivity")
                                 for row in rows),
        "passed": passed,
    }
    profiler.write_json(out / "validation.json", report)
    if not passed:
        raise RuntimeError("scheduled repair simulation failed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    args = parser.parse_args(argv)
    run(args.out, args.parent)


if __name__ == "__main__":
    main()
