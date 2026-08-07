"""Fixed-30-second destination-load and effective-goodput capacity sweeps."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import migration_profiler as profiler
from destination import dedicated_sink_architecture
from planner import _expected_scenario, plan, source_power
from profiles import ModelProfile
from simulate import (ExecutionScenario, NetworkLink, PowerNode,
                      ServingInstance, SimSession, predict)


ROOT = Path(__file__).parent
DEFAULT_PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
SCHEMA = "queue-haul-capacity-sweep-v1"
COMMIT_DEADLINE_S = 30
PLANNER_DEADLINE_S = 30
RUN_TIMEOUT_S = 180
CONTEXTS = (2048, 4096, 4096, 8192, 8192, 12288, 12288, 14336)
GOODPUT_CAPS_MBPS = (1000, 1600, 2500, 4000, 5000, 7000, 10000)
DEFAULT_TEMPLATE = ROOT / "outputs/policy-hardware-width8-packing-plan/plan.json"
DEFAULT_BUNDLE = ROOT / "outputs/destination-v7-20260722/content-free-manifest.json"
DEFAULT_SERVICE_PROFILE = ROOT / "outputs/destination-v7-20260722/baseline-profile.json"
LOAD_BASE_FRACTIONS = (0, .25, .5, .65, .75, .8, .85, .875,
                       .9, .925, .95, .975)
FULL_DRAIN_LOADS = (.85, .875, .8875, .9, .9125,
                    .925, .9375, .95, .9625, .975)
FULL_DRAIN_BANDWIDTHS_MBPS = (1000, 2500, 5000, 10000)
FULL_DRAIN_POLICIES = ("replay_only", "kv_only")
LIVE_REPEATS = 10
MAX_LIVE_REPEATS = 30
CI_WIDTH_W = 5
FULL_SHED_TOLERANCE_W = 1e-6
LOAD_WARMUP_S = LOAD_WINDOW_S = 30
POLICIES = {
    "lp": "lp_work_first", "greedy": "greedy",
    "replay_only": "replay_only", "kv_only": "kv_only",
}
LABELS = {
    "lp": "Queue-Haul LP", "greedy": "Queue-Haul Greedy",
    "replay_only": "Replay only", "kv_only": "KV only",
}
COLORS = {"lp": "#B1040E", "greedy": "#008566",
          "replay_only": "#E98300", "kv_only": "#006CB8"}


def source_session_rates(sessions: int) -> tuple[float, float]:
    if sessions < 1:
        raise ValueError("source needs at least one session")
    return 4 * 128 / sessions, 4 * 2 / sessions


def credited_sessions(rows, deadline_s: float = COMMIT_DEADLINE_S) -> set[str]:
    return {row["session_id"] for row in rows
            if row.get("committed_s") is not None
            and row.get("first_token_s") is not None
            and max(float(row["committed_s"]), float(row["first_token_s"]))
            <= deadline_s}


def full_drain_times(rows) -> tuple[float, float, float]:
    route = max(float(row["committed_s"]) for row in rows)
    token = max(float(row["first_token_s"]) for row in rows)
    return route, token, max(route, token)


def shapley_watts(value) -> tuple[float, float]:
    empty, replay, kv, both = (value(groups) for groups in
                               ((), ("replay",), ("kv",), ("replay", "kv")))
    return (.5 * ((replay - empty) + (both - kv)),
            .5 * ((kv - empty) + (both - replay)))


def adaptive_load_fractions(lp_watts, target_w: float) -> tuple[float, ...]:
    if len(lp_watts) != len(LOAD_BASE_FRACTIONS) or target_w <= 0:
        raise ValueError("load adaptation needs the dense grid and a target")
    extra = {(a + b) / 2 for a, b, x, y in zip(
        LOAD_BASE_FRACTIONS, LOAD_BASE_FRACTIONS[1:], lp_watts, lp_watts[1:])
             if abs(x - y) > 5}
    return tuple(sorted({*LOAD_BASE_FRACTIONS, *extra}))


def knee_indices(watts, target_w: float, feasible_first: bool) -> tuple[int, int]:
    if len(watts) < 2:
        raise ValueError("knee selection needs two cells")
    if feasible_first:
        crossing = next((i for i in range(1, len(watts))
                         if watts[i - 1] >= target_w > watts[i]), None)
        return (crossing - 1, crossing) if crossing is not None else (
            (len(watts) - 2, len(watts) - 1) if watts[-1] >= target_w else (0, 1))
    crossing = next((i for i in range(1, len(watts))
                     if watts[i - 1] < target_w <= watts[i]), None)
    return (crossing - 1, crossing) if crossing is not None else (
        (len(watts) - 2, len(watts) - 1) if watts[-1] < target_w else (0, 1))


def _architecture(profile, load_fraction: float):
    architecture = dedicated_sink_architecture(profile, "destination", ("link",))
    case = profile.case()
    work = (2048 / case.prefill.rate(2048, 1),
            32 / case.decode.rate(2048, 1))
    total = sum(work)
    baseline = tuple(load_fraction * value / total for value in work)
    pool = architecture.pools[0]
    pool = replace(pool, replicas=(replace(pool.replicas[0],
                                           baseline_work=baseline),))
    return replace(architecture, pools=(pool,),
                   residency_horizon_s=COMMIT_DEADLINE_S)


def _scenario(profile, goodput_mbps: float) -> ExecutionScenario:
    expected_f, expected_g = source_session_rates(len(CONTEXTS))
    sessions = tuple(SimSession(
        f"s{index}", "source", context, expected_f, expected_g, 2 * context,
    ) for index, context in enumerate(CONTEXTS))
    case = profile.case()
    return ExecutionScenario(
        PLANNER_DEADLINE_S, PLANNER_DEADLINE_S, case.power_curve.power(0),
        "awake", 0,
        (PowerNode("source-node", 1, True),
         PowerNode("destination-node", 1, False)),
        (ServingInstance("source", ("source-node",)),
         ServingInstance("destination", ("destination-node",))),
        sessions, (NetworkLink("link", goodput_mbps * 125_000),),
    )


def _model_cell(profile, policy: str, load_fraction: float,
                configured_mbps: float, measured_mbps: float) -> dict:
    scenario = _scenario(profile, measured_mbps)
    architecture = _architecture(profile, load_fraction)
    result = plan(
        scenario, profile, {("source", "destination"): ("link",)},
        POLICIES[policy], destination=architecture,
    )
    execution = predict(
        _expected_scenario(scenario, result.moves), profile, result.moves,
        destination=architecture,
    )
    method = {move.session_id: move.method for move in result.moves}
    timings = [{"session_id": row.session_id, "committed_s": row.committed_s,
                "first_token_s": row.committed_s, "method": method[row.session_id]}
               for row in execution.sessions]
    credited = credited_sessions(timings)
    initial = source_power(scenario, profile)

    def value(groups):
        names = {"replay" if group == "replay" else "kv_transfer"
                 for group in groups}
        moved = {session_id for session_id in credited
                 if method[session_id] in names}
        return initial - source_power(scenario, profile, moved)

    achieved = value(("replay", "kv"))
    replay_w, kv_w = shapley_watts(value)
    target = initial - source_power(
        scenario, profile, (row.session_id for row in scenario.sessions))
    return {
        "policy": policy, "load_fraction": load_fraction,
        "configured_goodput_mbps": configured_mbps,
        "measured_goodput_mbps": measured_mbps,
        "initial_source_power_w": initial, "requested_shed_w": target,
        "achieved_shed_w": achieved, "replay_w": replay_w, "kv_w": kv_w,
        "unmet_w": max(0, target - achieved),
        "credited_sessions": len(credited), "planned_sessions": len(result.moves),
        "moves": [asdict(move) for move in result.moves], "timings": timings,
        "planner_feasible": bool(result.feasible),
        "planner_shortfall_w": result.power_shortfall_w,
        "planner_makespan_s": result.predicted_migration_makespan_s,
        "binding_resources": list(result.binding_resources),
    }


def _goodput_cells(calibration: dict | None):
    measured = {int(row["configured_mbps"]): float(row["median_mbps"])
                for row in (calibration or {}).get("cells", [])}
    return [(cap, measured.get(cap, cap)) for cap in GOODPUT_CAPS_MBPS]


def _load_calibration(calibration: dict | None) -> dict:
    service = json.loads(DEFAULT_SERVICE_PROFILE.read_text())
    rates = {metric: float(np.interp(
        2048, *zip(*service["cases"]["central"][f"{metric}_tps"]["1"])
    )) for metric in ("prefill", "decode")}
    times = {"prefill": 2048 / rates["prefill"],
             "decode": 32 / rates["decode"]}
    return {**(calibration or {}), "service_calibration": {
        "path": str(DEFAULT_SERVICE_PROFILE),
        "sha256": profiler.file_hash(DEFAULT_SERVICE_PROFILE),
        "prefill_tokens_per_s": rates["prefill"],
        "decode_tokens_per_s": rates["decode"],
        "prefill_s": times["prefill"], "decode_s": times["decode"],
        "total_s": sum(times.values()),
    }}


def arrival_trace(rho: float, repeat: int, calibration: dict,
                  horizon_s: float = LOAD_WARMUP_S + LOAD_WINDOW_S) -> dict:
    if not 0 <= rho < 1 or not 0 <= repeat < MAX_LIVE_REPEATS:
        raise ValueError("stationary load trace must have rho < 1 and a valid repeat")
    if horizon_s < LOAD_WARMUP_S + LOAD_WINDOW_S:
        raise ValueError("arrival horizon must include warmup and measurement")
    service = calibration["service_calibration"]
    total = float(service["total_s"])
    offsets = []
    if rho:
        interval = total / rho
        block, within = divmod(repeat, LIVE_REPEATS)
        phase = ((within + .5) / LIVE_REPEATS + block / MAX_LIVE_REPEATS) % 1
        value = phase * interval
        while value < horizon_s:
            offsets.append(value); value += interval
    measured = sum(LOAD_WARMUP_S <= value < LOAD_WARMUP_S + LOAD_WINDOW_S
                   for value in offsets)
    prefill = measured * float(service["prefill_s"]) / LOAD_WINDOW_S
    decode = measured * float(service["decode_s"]) / LOAD_WINDOW_S
    trace = {"offsets_s": offsets, "rho_prefill": prefill,
             "rho_decode": decode, "rho": prefill + decode}
    trace["trace_id"] = profiler.object_hash([rho, repeat, trace])[:16]
    return trace


def make_campaign(kind: str, calibration: dict | None = None,
                  profile_path: Path = DEFAULT_PROFILE) -> dict:
    if kind not in {"load", "goodput"}:
        raise ValueError("campaign must be load or goodput")
    profile = ModelProfile.load(profile_path)
    if kind == "load":
        calibration = _load_calibration(calibration)
        cells = [(fraction, 10_000, 10_000)
                 for fraction in LOAD_BASE_FRACTIONS]
    else:
        cells = [(0, configured, measured)
                 for configured, measured in _goodput_cells(calibration)]
    rows = [_model_cell(profile, policy, load, configured, measured)
            for load, configured, measured in cells for policy in POLICIES]
    lp = [row for row in rows if row["policy"] == "lp"]
    knees = knee_indices(
        [row["achieved_shed_w"] for row in lp], lp[0]["requested_shed_w"],
        feasible_first=kind == "load",
    )
    return {
        "schema": SCHEMA, "campaign": kind,
        "commit_deadline_s": COMMIT_DEADLINE_S,
        "planner_power_deadline_s": PLANNER_DEADLINE_S,
        "hardware_timeout_s": RUN_TIMEOUT_S,
        "source": {"sessions": len(CONTEXTS), "rps": 4,
                   "input_tokens": 128, "output_tokens": 2,
                   "contexts": list(CONTEXTS), "final_state": "awake"},
        "destination_request": {"input_tokens": 2048, "output_tokens": 32},
        "profile": {"path": str(profile_path),
                    "sha256": profiler.file_hash(profile_path)},
        "calibration": calibration,
        "live_validation": {"repeats": LIVE_REPEATS if kind == "load" else 3,
                            "lp_knee_indices": list(knees),
                            "policies": list(POLICIES)},
        "rows": rows,
    }


def make_full_drain_campaign(bandwidth_mbps: int,
                             calibration: dict | None = None,
                             profile_path: Path = DEFAULT_PROFILE) -> dict:
    if bandwidth_mbps not in FULL_DRAIN_BANDWIDTHS_MBPS:
        raise ValueError("unsupported full-drain bandwidth")
    return {
        "schema": SCHEMA, "campaign": "full_drain",
        "commit_deadline_s": COMMIT_DEADLINE_S,
        "hardware_timeout_s": RUN_TIMEOUT_S,
        "trace_horizon_s": LOAD_WARMUP_S + RUN_TIMEOUT_S,
        "bandwidth_mbps": bandwidth_mbps,
        "loads": list(FULL_DRAIN_LOADS), "repeats": LIVE_REPEATS,
        "policies": list(FULL_DRAIN_POLICIES),
        "profile": {"path": str(profile_path),
                    "sha256": profiler.file_hash(profile_path)},
        "calibration": _load_calibration(calibration),
    }


def _frozen_sessions(template: dict) -> list[dict]:
    frozen = next((row["sessions"] for row in template["scenarios"]
                   if tuple(item["initial_tokens"] for item in row["sessions"])
                   == CONTEXTS), None)
    if frozen is None:
        raise ValueError("template does not contain the frozen context pack")
    return frozen


def make_full_drain_plan(campaign: dict, template: dict) -> dict:
    if campaign.get("schema") != SCHEMA \
            or campaign.get("campaign") != "full_drain":
        raise ValueError("invalid full-drain campaign")
    frozen = _frozen_sessions(template)
    bandwidth = campaign["bandwidth_mbps"]
    scenarios = []
    for load in FULL_DRAIN_LOADS:
        for repeat in range(LIVE_REPEATS):
            trace = arrival_trace(
                load, repeat, campaign["calibration"],
                LOAD_WARMUP_S + RUN_TIMEOUT_S,
            )
            match_id = profiler.object_hash(
                ["full_drain", bandwidth, load, repeat])[:16]
            for policy in FULL_DRAIN_POLICIES:
                method = "replay" if policy == "replay_only" else "kv_transfer"
                scenarios.append({
                    "scenario_id": "drain-" + profiler.object_hash(
                        [match_id, policy])[:16],
                    "match_id": match_id, "kind": "migration",
                    "campaign": "capacity_full_drain", "split": "measurement",
                    "condition": "full_drain", "policy": policy,
                    "method": method, "load_fraction": load,
                    "arrival_trace": trace,
                    "configured_goodput_mbps": bandwidth,
                    "planned_measured_goodput_mbps": bandwidth,
                    "bandwidth_mbps": bandwidth,
                    "required_deadline_s": COMMIT_DEADLINE_S,
                    "deadline_s": RUN_TIMEOUT_S, "repeat": repeat,
                    "sessions": [{**item, "source_index": i}
                                 for i, item in enumerate(frozen)],
                    "moves": [{"session_id": item["session_id"],
                               "method": method, "order": i}
                              for i, item in enumerate(frozen)],
                    "allow_partial_moves": False,
                    "concurrency": len(frozen), "move_concurrency": len(frozen),
                    "serving_concurrency": 1, "activity": "none",
                    "activity_tokens": 0, "request_schedule": [],
                    "copy_policy": "initial_final", "final_state": "awake",
                    "reset_caches": True, "verify_continuations": True,
                    "wait_cache_idle": True, "prestage_all": True,
                    "warm_concurrency": len(frozen), "power_interval_s": .1,
                })
    random.Random(0).shuffle(scenarios)
    return {
        "schema": profiler.PLAN_SCHEMA, "manifest": template["manifest"],
        "profile": campaign["profile"], "capacity_campaign": "full_drain",
        "calibration": campaign["calibration"],
        "commit_deadline_s": COMMIT_DEADLINE_S,
        "trace_horizon_s": LOAD_WARMUP_S + RUN_TIMEOUT_S,
        "scenarios": scenarios,
    }


def make_live_plan(campaign: dict, template: dict, repeat_map: dict | None = None,
                   phase: str = "base", prior_plan_hashes=(),
                   selection: dict | None = None) -> dict:
    if campaign.get("schema") != SCHEMA:
        raise ValueError("invalid capacity campaign")
    frozen = _frozen_sessions(template)
    by_index = {f"s{i}": row for i, row in enumerate(frozen)}
    lp = [row for row in campaign["rows"] if row["policy"] == "lp"]
    if repeat_map is None:
        chosen = (range(len(lp)) if campaign["campaign"] == "load"
                  else campaign["live_validation"]["lp_knee_indices"])
        keys = {(lp[i]["load_fraction"], lp[i]["configured_goodput_mbps"],
                 lp[i]["measured_goodput_mbps"]) for i in chosen}
        repeats = {key: tuple(range(campaign["live_validation"]["repeats"]))
                   for key in keys}
    else:
        if campaign["campaign"] != "load":
            raise ValueError("adaptive repeats require a load campaign")
        keys = {(row["load_fraction"], row["configured_goodput_mbps"],
                 row["measured_goodput_mbps"]) for row in lp
                if row["load_fraction"] in repeat_map}
        if {key[0] for key in keys} != set(repeat_map):
            raise ValueError("adaptive schedule contains an unmodeled load")
        repeats = {key: tuple(repeat_map[key[0]]) for key in keys}
        if any(not values or len(values) != len(set(values)) or
               not set(values) <= set(range(MAX_LIVE_REPEATS))
               for values in repeats.values()):
            raise ValueError("adaptive repeat schedule is invalid")
    selected = [row for row in campaign["rows"]
                if (row["load_fraction"], row["configured_goodput_mbps"],
                    row["measured_goodput_mbps"]) in keys]
    scenarios = []
    for row in selected:
        key = (row["load_fraction"], row["configured_goodput_mbps"],
               row["measured_goodput_mbps"])
        for repeat in repeats[key]:
            moves = [{"session_id": by_index[move["session_id"]]["session_id"],
                      "method": move["method"], "order": order}
                     for order, move in enumerate(row["moves"])]
            if not moves:
                raise ValueError("live cell has no executable moves")
            sessions = [{**item, "source_index": i}
                        for i, item in enumerate(frozen)]
            trace = (arrival_trace(row["load_fraction"], repeat,
                                   campaign["calibration"])
                     if campaign["campaign"] == "load" else {
                         "offsets_s": [], "rho_prefill": 0, "rho_decode": 0,
                         "rho": 0, "trace_id": profiler.object_hash(
                             ["goodput", row["configured_goodput_mbps"], repeat]
                         )[:16],
                     })
            methods = {move["method"] for move in moves}
            cell = (row["load_fraction"], row["configured_goodput_mbps"])
            match_id = profiler.object_hash([campaign["campaign"], cell, repeat])[:16]
            scenario_id = profiler.object_hash([match_id, row["policy"]])[:16]
            scenarios.append({
                "scenario_id": f"cap-{scenario_id}", "match_id": match_id,
                "kind": "migration", "campaign": f"capacity_{campaign['campaign']}",
                "split": "measurement", "condition": campaign["campaign"],
                "policy": row["policy"],
                "method": next(iter(methods)) if len(methods) == 1 else "mixed",
                "load_fraction": row["load_fraction"], "arrival_trace": trace,
                "configured_goodput_mbps": row["configured_goodput_mbps"],
                "planned_measured_goodput_mbps": row["measured_goodput_mbps"],
                "bandwidth_mbps": row["configured_goodput_mbps"],
                "required_deadline_s": COMMIT_DEADLINE_S,
                "deadline_s": RUN_TIMEOUT_S, "repeat": repeat,
                "sessions": sessions, "moves": moves,
                "allow_partial_moves": True,
                "concurrency": len(moves), "move_concurrency": len(moves),
                "serving_concurrency": 1, "activity": "none",
                "activity_tokens": 0, "request_schedule": [],
                "copy_policy": "initial_final", "final_state": "awake",
                "reset_caches": True, "verify_continuations": True,
                "wait_cache_idle": True, "prestage_all": True,
                "warm_concurrency": len(moves), "power_interval_s": .1,
            })
    expected = sum(map(len, repeats.values())) * len(POLICIES)
    if len(scenarios) != expected:
        raise ValueError("live validation matrix is incomplete")
    random.Random(0).shuffle(scenarios)
    result = {"schema": profiler.PLAN_SCHEMA, "manifest": template["manifest"],
              "profile": campaign["profile"],
              "capacity_campaign": campaign["campaign"],
              "calibration": campaign.get("calibration"),
              "commit_deadline_s": COMMIT_DEADLINE_S, "scenarios": scenarios}
    if repeat_map is not None:
        result.update({"adaptive_phase": phase,
                       "prior_plan_sha256": list(prior_plan_hashes),
                       "adaptive_selection": selection,
                       "load_repeat_schedule": [
                           {"load_fraction": load, "repeats": list(repeat_map[load])}
                           for load in sorted(repeat_map)]})
    return result


def write_campaign(kind: str, out: Path, calibration_path: Path | None = None,
                   profile_path: Path = DEFAULT_PROFILE) -> dict:
    calibration = json.loads(calibration_path.read_text()) if calibration_path else None
    campaign = make_campaign(kind, calibration, profile_path)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", campaign)
    profiler.write_csv(out / "modeled_capacity.csv", campaign["rows"])
    plot_campaign(campaign, out)
    profiler.write_json(out / "summary.json", {
        key: campaign[key] for key in ("schema", "campaign", "commit_deadline_s",
                                       "planner_power_deadline_s", "live_validation")
    })
    return campaign

def plot_campaign(campaign: dict, out: Path) -> None:
    rows, kind = campaign["rows"], campaign["campaign"]
    xfield = "load_fraction" if kind == "load" else "measured_goodput_mbps"
    xlabel = ("Destination offered load / measured stable load"
              if kind == "load" else "Measured effective goodput (Mbit/s)")
    target = rows[0]["requested_shed_w"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for policy in POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        ax.plot([row[xfield] for row in selected],
                [row["achieved_shed_w"] for row in selected], marker="o",
                color=COLORS[policy], label=LABELS[policy])
    ax.axhline(target, color="black", linestyle="--", label="Requested shed")
    if kind == "goodput":
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel="Maximum executable shed by 30 s (W)")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{kind}_capacity.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    for ax, policy in zip(axes, ("lp", "greedy")):
        selected = [row for row in rows if row["policy"] == policy]
        x = [row[xfield] for row in selected]
        bottom = [0] * len(x)
        for field, label, color in (("replay_w", "Replay", "#E98300"),
                                    ("kv_w", "KV transfer", "#006CB8"),
                                    ("unmet_w", "Unmet", "#999999")):
            values = [row[field] for row in selected]
            width = .06 if kind == "load" else [value * .12 for value in x]
            ax.bar(x, values, bottom=bottom, width=width,
                   color=color, label=label)
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_title(LABELS[policy]); ax.set_ylabel("Power (W)"); ax.grid(axis="y", alpha=.25)
        if kind == "goodput":
            ax.set_xscale("log")
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel(xlabel)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{kind}_capacity_stack.{suffix}", dpi=220)
    plt.close(fig)


def write_live_plan(campaign: dict, template_path: Path, out: Path,
                    repeat_map: dict | None = None, phase: str = "base",
                    prior_plan_hashes=(), selection=None) -> dict:
    template = json.loads(template_path.read_text())
    plan_ = make_live_plan(campaign, template, repeat_map, phase,
                           prior_plan_hashes, selection)
    profiler.write_json(out / "live_plan.json", plan_)
    return plan_


def write_full_drain_campaign(bandwidth_mbps: int, template_path: Path,
                              out: Path, calibration_path: Path | None = None,
                              profile_path: Path = DEFAULT_PROFILE) -> dict:
    calibration = json.loads(calibration_path.read_text()) if calibration_path else None
    campaign = make_full_drain_campaign(
        bandwidth_mbps, calibration, profile_path)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", campaign)
    profiler.write_json(out / "summary.json", campaign)
    template = json.loads(template_path.read_text())
    profiler.write_json(out / "live_plan.json",
                        make_full_drain_plan(campaign, template))
    return campaign


def make_adaptive_campaign(prior: dict, repeat_map: dict, phase: str,
                           selection: dict) -> dict:
    if not prior.get("capacity_campaign") == "load" or not prior.get("calibration"):
        raise ValueError("adaptive phase requires a calibrated load plan")
    profile_path = Path(prior["profile"]["path"])
    profile = ModelProfile.load(profile_path)
    rows = [_model_cell(profile, policy, load, 10_000, 10_000)
            for load in sorted(repeat_map) for policy in POLICIES]
    return {
        "schema": SCHEMA, "campaign": "load",
        "commit_deadline_s": COMMIT_DEADLINE_S,
        "planner_power_deadline_s": PLANNER_DEADLINE_S,
        "hardware_timeout_s": RUN_TIMEOUT_S,
        "profile": prior["profile"], "calibration": prior["calibration"],
        "live_validation": {"policies": list(POLICIES), "adaptive_phase": phase},
        "adaptive_selection": selection, "rows": rows,
    }


def write_adaptive_phase(phase: str, prior_roots, template_path: Path,
                         out: Path) -> tuple[dict, dict]:
    expected = 1 if phase == "phase2a" else 2
    roots = [root.resolve() for root in prior_roots]
    if len(roots) != expected or len(set(roots)) != expected:
        raise ValueError(f"{phase} requires {expected} distinct prior run roots")
    parts = [read_live_results(root) for root in roots]
    if {kind for kind, _ in parts} != {"load"}:
        raise ValueError("adaptive phase requires load results")
    rows = merge_live_rows([rows for _, rows in parts])
    if phase == "phase2a":
        repeat_map, selection = phase2a_schedule(rows)
    else:
        repeat_map = phase2b_schedule(rows)
        selection = {"wide_loads": sorted(repeat_map)}
    if not repeat_map:
        raise ValueError(f"{phase} selected no cells")
    plan_paths = [root / "plan.json" for root in roots]
    plans = [json.loads(path.read_text()) for path in plan_paths]
    signatures = {(profiler.object_hash(plan["profile"]),
                   profiler.object_hash(plan.get("calibration"))) for plan in plans}
    if len(signatures) != 1:
        raise ValueError("prior plans disagree on profile or calibration")
    hashes = [profiler.file_hash(path) for path in plan_paths]
    selection.update({"ci_width_threshold_w": CI_WIDTH_W,
                      "full_shed_tolerance_w": FULL_SHED_TOLERANCE_W})
    campaign = make_adaptive_campaign(plans[0], repeat_map, phase, selection)
    campaign["prior_plan_sha256"] = hashes
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", campaign)
    profiler.write_csv(out / "modeled_capacity.csv", campaign["rows"])
    profiler.write_json(out / "summary.json", {
        "schema": SCHEMA, "campaign": "load", "adaptive_phase": phase,
        "selection": selection, "prior_plan_sha256": hashes,
    })
    live = write_live_plan(campaign, template_path, out, repeat_map, phase,
                           hashes, selection)
    return campaign, live


def execute_live(plan_path: Path, run_root: Path, allow_dirty: bool = False) -> None:
    import destination_runner as destination
    import migration_testbed as testbed

    plan_ = json.loads(plan_path.read_text())
    calibration = plan_.get("calibration") or {}
    service_calibration = calibration.get("service_calibration")
    if plan_["capacity_campaign"] in {"load", "full_drain"} \
            and not service_calibration:
        raise ValueError("loaded hardware run requires independent service calibration")
    bundle = json.loads(DEFAULT_BUNDLE.read_text())
    service = json.loads(DEFAULT_SERVICE_PROFILE.read_text())
    background = sum((
        destination.manifest_sessions(bundle, "agentic_tool_loop", split, 201088, 7)
        for split in ("validation", "tune")
    ), [])
    background = [replace(row, prefix_tokens=1, append_tokens=2048,
                          output_tokens=32) for row in background]
    rates = (
        float(service_calibration["prefill_tokens_per_s"]),
        float(service_calibration["decode_tokens_per_s"]),
    ) if service_calibration else (
        destination.profile_rate(service, "prefill", 2048),
        destination.profile_rate(service, "decode", 2048),
    )
    work = float(service_calibration["total_s"]) if service_calibration else 1
    original = profiler.run_scenario

    def run_loaded(stack, cfg, manifest, scenario, root, run_id, **kwargs):
        fraction = float(scenario["load_fraction"])
        trace = scenario["arrival_trace"]
        load = None if not fraction else destination.DestinationLoad(
            cfg.host, cfg.sink_port, cfg.model, background, fraction, *rates,
            root / "destination_load", 1000 + scenario["repeat"],
            normal_bound=1, rps=fraction / work, max_inflight=256,
            bypass_lmcache=True, chat=True,
            arrival_schedule=tuple(trace["offsets_s"]),
            warmup_s=LOAD_WARMUP_S, measurement_s=LOAD_WINDOW_S,
            schedule_horizon_s=plan_.get("trace_horizon_s"),
        )
        return original(stack, cfg, manifest, scenario, root, run_id,
                        destination_load=load, **kwargs)

    profiler.run_scenario = run_loaded
    try:
        profiler.run_plan(plan_path, run_root, testbed.Config(), allow_dirty, [],
                          fail_fast=True, stack_scenarios=64)
    finally:
        profiler.run_scenario = original


def median_ci(values, seed: int = 0,
              samples: int = 4000) -> tuple[float, float, float]:
    if not values or samples < 1:
        raise ValueError("median confidence interval needs observations")
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.median(rng.choice(values, (samples, len(values))), axis=1)
    low, high = np.quantile(draws, (.025, .975))
    return float(np.median(values)), float(low), float(high)


def validate_live_rows(rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError("live results are empty")
    triples = [(row["load_fraction"], row["repeat"], row["policy"])
               for row in rows]
    if len(triples) != len(set(triples)):
        raise RuntimeError("duplicate load/repeat/policy result")
    for load, repeat in sorted({pair[:2] for pair in triples}):
        cell = [row for row in rows if row["load_fraction"] == load
                and row["repeat"] == repeat]
        if len(cell) != len(POLICIES) or {row["policy"] for row in cell} != set(POLICIES):
            raise RuntimeError("incomplete common-trace cell")
        if len({row["trace_id"] for row in cell}) != 1:
            raise RuntimeError("policies did not share an arrival trace")


def validate_full_drain_rows(rows: list[dict], complete: bool = False) -> None:
    if not rows:
        raise RuntimeError("full-drain results are empty")
    fields = ("configured_goodput_mbps", "load_fraction", "repeat", "policy")
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate full-drain result")
    cells = {(row["configured_goodput_mbps"], row["load_fraction"], row["repeat"])
             for row in rows}
    for bandwidth, load, repeat in cells:
        cell = [row for row in rows
                if (row["configured_goodput_mbps"], row["load_fraction"], row["repeat"])
                == (bandwidth, load, repeat)]
        if {row["policy"] for row in cell} != set(FULL_DRAIN_POLICIES) \
                or len({row["trace_id"] for row in cell}) != 1:
            raise RuntimeError("incomplete full-drain common-trace cell")
        if any(row["planned_sessions"] != len(CONTEXTS)
               for row in cell):
            raise RuntimeError("full drain must attempt all eight sessions")
    if complete:
        bandwidths = {row["configured_goodput_mbps"] for row in rows}
        expected = {(bandwidth, load, repeat) for bandwidth in bandwidths
                    for load in FULL_DRAIN_LOADS for repeat in range(LIVE_REPEATS)}
        if cells != expected:
            raise RuntimeError("full-drain grid is incomplete")


def load_statistics(rows: list[dict]) -> dict:
    validate_live_rows(rows)
    return {load: {policy: dict(zip(("median", "low", "high"), median_ci(
        [row["achieved_shed_w"] for row in rows
         if row["load_fraction"] == load and row["policy"] == policy],
        int(profiler.object_hash([policy, load])[:8], 16))))
        for policy in POLICIES}
        for load in sorted({row["load_fraction"] for row in rows})}


def phase2a_schedule(rows: list[dict]) -> tuple[dict, dict]:
    stats = load_statistics(rows)
    loads = tuple(stats)
    if loads != LOAD_BASE_FRACTIONS or any(
            {row["repeat"] for row in rows if row["load_fraction"] == load}
            != set(range(LIVE_REPEATS)) for load in loads):
        raise ValueError("phase2a requires the complete base grid")
    targets = {float(row["requested_shed_w"]) for row in rows}
    if len(targets) != 1:
        raise ValueError("live results disagree on requested shed")
    target = targets.pop()
    full = [index for index, load in enumerate(loads)
            if stats[load]["lp"]["median"] >= target - FULL_SHED_TOLERANCE_W]
    if not full or max(full) == len(loads) - 1:
        raise ValueError("base grid does not bracket the LP full-shed knee")
    knee = {loads[max(full)], loads[max(full) + 1]}
    wide = {load for load in loads if any(
        stats[load][policy]["high"] - stats[load][policy]["low"] > CI_WIDTH_W
        for policy in POLICIES)}
    intervals = {(left, right) for left, right in zip(loads, loads[1:]) if any(
        abs(stats[left][policy]["median"] - stats[right][policy]["median"])
        > CI_WIDTH_W for policy in POLICIES)}
    midpoints = {(left + right) / 2 for left, right in intervals}
    schedule = {load: tuple(range(10, 20)) for load in sorted(knee | wide)}
    schedule.update({load: tuple(range(10)) for load in sorted(midpoints)})
    return schedule, {
        "knee_loads": sorted(knee), "wide_loads": sorted(wide),
        "midpoint_loads": sorted(midpoints),
        "midpoint_intervals": [list(pair) for pair in sorted(intervals)],
    }


def phase2b_schedule(rows: list[dict]) -> dict:
    stats = load_statistics(rows)
    return {load: tuple(range(20, 30)) for load in stats if any(
        stats[load][policy]["high"] - stats[load][policy]["low"] > CI_WIDTH_W
        for policy in POLICIES)}


def merge_live_rows(parts) -> list[dict]:
    rows = [row for part in parts for row in part]
    validate_live_rows(rows)
    return rows


def plot_live(rows: list[dict], out: Path, kind: str) -> None:
    if kind != "load":
        return
    loads = sorted({row["load_fraction"] for row in rows})
    validate_live_rows(rows)
    target = rows[0]["requested_shed_w"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for policy in POLICIES:
        x, center, low, high = [], [], [], []
        for load in loads:
            selected = [row for row in rows if row["policy"] == policy
                        and row["load_fraction"] == load]
            x.append(statistics.median(row["offered_rho"] for row in selected))
            seed = int(profiler.object_hash([policy, load])[:8], 16)
            a, b, c = median_ci(
                [row["achieved_shed_w"] for row in selected], seed)
            center.append(a); low.append(b); high.append(c)
        ax.plot(x, center, marker="o", color=COLORS[policy], label=LABELS[policy])
        ax.fill_between(x, low, high, color=COLORS[policy], alpha=.16)
    ax.axhline(target, color="black", linestyle="--",
               label=f"Requested shed ({target:.1f} W)")
    ax.set(xlabel="Trace-derived normalized offered load",
           ylabel="Median executable shed by 30 s (W)")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"load_capacity_live.{suffix}", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True)
    for ax, policy in zip(axes, ("lp", "greedy")):
        selected = [row for row in rows if row["policy"] == policy]
        x = [statistics.median(row["offered_rho"] for row in selected
                              if row["load_fraction"] == load) for load in loads]
        for field, label, color in (("replay_w", "Replay", "#E98300"),
                                    ("kv_w", "KV transfer", "#006CB8"),
                                    ("unmet_w", "Unmet", "#999999")):
            ax.plot(x, [statistics.median(row[field] for row in selected
                                          if row["load_fraction"] == load)
                        for load in loads], marker="o", label=label, color=color)
        ax.set_title(LABELS[policy]); ax.set_ylabel("Power (W)"); ax.grid(alpha=.25)
    axes[0].legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("Trace-derived normalized offered load")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"load_capacity_components_live.{suffix}", dpi=220)
    plt.close(fig)


def read_live_results(run_root: Path) -> tuple[str, list[dict]]:
    live = json.loads((run_root / "plan.json").read_text())
    kind = live["capacity_campaign"]
    profile = ModelProfile.load(Path(live["profile"]["path"]))
    scenario = _scenario(profile, 10_000)
    initial = source_power(scenario, profile)
    target = initial - source_power(
        scenario, profile, (row.session_id for row in scenario.sessions))
    rows = []
    for spec in live["scenarios"]:
        path = run_root / "scenarios" / spec["scenario_id"] / "result.json"
        if not path.exists():
            raise RuntimeError(f"missing live result {spec['scenario_id']}")
        result = json.loads(path.read_text())
        if result.get("status") != "complete":
            raise RuntimeError(f"invalid live result {spec['scenario_id']}")
        raw = result["migrations"]
        if kind == "full_drain" and (
                not result.get("full_drain") or len(raw) != len(CONTEXTS)
                or any(row.get("error") for row in raw)
                or not profiler.valid_continuations(result, len(CONTEXTS))):
            raise RuntimeError(f"incomplete full drain {spec['scenario_id']}")
        epoch = result["started_ns"]
        continuation = {row["session_id"]: row for row in result["continuations"]}
        timings = [{
            "session_id": row["move"]["session_id"],
            "method": row["move"]["method"],
            "committed_s": (row["switch_end_ns"] - epoch) / 1e9,
            "first_token_s":
                (continuation[row["move"]["session_id"]]["first_byte_ns"] - epoch) / 1e9,
        } for row in raw]
        credited = credited_sessions(timings)
        last_route, last_token, drain = full_drain_times(timings)
        if kind == "full_drain" and drain >= RUN_TIMEOUT_S:
            raise RuntimeError(f"full drain exceeded timeout {spec['scenario_id']}")
        source_ids = {item["session_id"]: f"s{item['source_index']}"
                      for item in spec["sessions"]}
        methods = {source_ids[row["session_id"]]: row["method"] for row in timings}

        def value(groups):
            names = {"replay" if group == "replay" else "kv_transfer"
                     for group in groups}
            moved = {source_ids[session_id] for session_id in credited
                     if methods[source_ids[session_id]] in names}
            return initial - source_power(scenario, profile, moved)

        replay_w, kv_w = shapley_watts(value)
        achieved = value(("replay", "kv"))
        start = min(row["initial_start_ns"] for row in raw)
        end = max(row["switch_end_ns"] for row in raw)
        network = profiler.network_measurements(
            path.parent / "proxy_bytes.csv", start, end)
        load = result.get("destination_load") or {}
        trace = spec["arrival_trace"]
        rho_prefill = load.get("offered_rho_prefill", trace["rho_prefill"])
        rho_decode = load.get("offered_rho_decode", trace["rho_decode"])
        rows.append({
            "scenario_id": spec["scenario_id"], "policy": spec["policy"],
            "repeat": spec["repeat"], "load_fraction": spec["load_fraction"],
            "trace_id": trace["trace_id"],
            "offered_rho_prefill": rho_prefill,
            "offered_rho_decode": rho_decode,
            "offered_rho": rho_prefill + rho_decode,
            "configured_goodput_mbps": spec["configured_goodput_mbps"],
            "measured_goodput_mbps": network["measured_kv_throughput_mbps"],
            "requested_shed_w": target, "achieved_shed_w": achieved,
            "replay_w": replay_w, "kv_w": kv_w,
            "unmet_w": max(0, target - achieved),
            "credited_sessions": len(credited), "planned_sessions": len(raw),
            "last_route_commit_s": last_route,
            "last_first_token_s": last_token,
            "full_drain_s": drain,
            "episode_elapsed_s": result["elapsed_s"],
            "queue_at_start": load.get("queue_at_start"),
            "deadline_miss_sessions": len(raw) - len(credited),
            "right_censored": drain >= RUN_TIMEOUT_S,
            "destination_load": load or None,
        })
    if kind == "goodput":
        measured = {
            cap: statistics.median(row["measured_goodput_mbps"] for row in rows
                                   if row["configured_goodput_mbps"] == cap
                                   and row["measured_goodput_mbps"] > 0)
            for cap in {row["configured_goodput_mbps"] for row in rows}
        }
        for row in rows:
            row["measured_goodput_mbps"] = measured[row["configured_goodput_mbps"]]
    return kind, rows


def plot_full_drain(rows: list[dict], out: Path) -> None:
    bandwidths = sorted({row["configured_goodput_mbps"] for row in rows})
    fig, axes = plt.subplots(2, len(bandwidths), figsize=(3.5 * len(bandwidths), 6),
                             sharex="col", squeeze=False)
    target = rows[0]["requested_shed_w"]
    for column, bandwidth in enumerate(bandwidths):
        selected = [row for row in rows
                    if row["configured_goodput_mbps"] == bandwidth]
        loads = sorted({row["load_fraction"] for row in selected})
        for policy in FULL_DRAIN_POLICIES:
            cells = [[row for row in selected if row["policy"] == policy
                      and row["load_fraction"] == load] for load in loads]
            x = [statistics.median(row["offered_rho"] for row in cell)
                 for cell in cells]
            for axis, field in zip(axes[:, column],
                                   ("full_drain_s", "achieved_shed_w")):
                ci = [median_ci(
                    [row[field] for row in cell],
                    int(profiler.object_hash([bandwidth, load, policy, field])[:8], 16),
                ) for load, cell in zip(loads, cells)]
                axis.plot(x, [row[0] for row in ci], marker="o",
                          color=COLORS[policy], label=LABELS[policy])
                axis.fill_between(x, [row[1] for row in ci], [row[2] for row in ci],
                                  color=COLORS[policy], alpha=.16)
        axes[0, column].axhline(COMMIT_DEADLINE_S, color="black", linestyle="--")
        axes[1, column].axhline(target, color="black", linestyle="--")
        axes[0, column].set_title(f"{bandwidth:g} Mbit/s")
        axes[1, column].set_xlabel("Trace-derived offered load")
        for axis in axes[:, column]:
            axis.grid(alpha=.25)
    axes[0, 0].set_ylabel("Full drain time (s)")
    axes[1, 0].set_ylabel("Executable shed by 30 s (W)")
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"full_drain_capacity.{suffix}", dpi=220)
    plt.close(fig)


def write_full_drain_results(rows: list[dict], out: Path) -> None:
    validate_full_drain_rows(rows, complete=True)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_csv(out / "full_drain_capacity.csv", rows)
    plot_full_drain(rows, out)
    bandwidths = sorted({row["configured_goodput_mbps"] for row in rows})
    profiler.write_json(out / "full_drain_summary.json", {
        "schema": SCHEMA, "campaign": "full_drain", "episodes": len(rows),
        "bandwidths_mbps": bandwidths, "load_points": len(FULL_DRAIN_LOADS),
        "repeats_per_cell": LIVE_REPEATS,
        "deadline_credited_sessions": sum(row["credited_sessions"] for row in rows),
        "deadline_miss_sessions": sum(row["deadline_miss_sessions"] for row in rows),
        "right_censored_episodes": sum(row["right_censored"] for row in rows),
    })


def write_live_results(rows: list[dict], out: Path, kind: str) -> None:
    if kind == "full_drain":
        write_full_drain_results(rows, out)
        return
    if kind == "load":
        validate_live_rows(rows)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_csv(out / "live_capacity.csv", rows)
    plot_live(rows, out, kind)
    repeats = {str(load): len({row["repeat"] for row in rows
                              if row["load_fraction"] == load})
               for load in sorted({row["load_fraction"] for row in rows})}
    counts = list(repeats.values())
    profiler.write_json(out / "live_summary.json", {
        "schema": SCHEMA, "campaign": kind,
        "episodes": len(rows), "complete": len(rows),
        "load_points": len(repeats), "repeats_by_load": repeats,
        "min_repeats_per_cell": min(counts),
        "max_repeats_per_cell": max(counts),
        "deadline_credited_sessions": sum(row["credited_sessions"] for row in rows),
        "right_censored_episodes": sum(row["right_censored"] for row in rows),
    })


def reduce_live(run_root: Path, campaign: dict, out: Path) -> list[dict]:
    kind, rows = read_live_results(run_root)
    write_live_results(rows, out, kind)
    return rows


def merge_live(run_roots, out: Path) -> list[dict]:
    parts = [read_live_results(root) for root in run_roots]
    if {kind for kind, _ in parts} != {"load"}:
        raise ValueError("adaptive merge requires load campaigns")
    rows = merge_live_rows([rows for _, rows in parts])
    write_live_results(rows, out, "load")
    summary_path = out / "live_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source_runs"] = [
        {"run_root": str(root), "plan_sha256":
         profiler.file_hash(root / "plan.json")} for root in run_roots]
    profiler.write_json(summary_path, summary)
    return rows


def merge_full_drain(run_roots, out: Path) -> list[dict]:
    parts = [read_live_results(root) for root in run_roots]
    if {kind for kind, _ in parts} != {"full_drain"}:
        raise ValueError("full-drain merge requires full-drain campaigns")
    rows = [row for _, part in parts for row in part]
    if {row["configured_goodput_mbps"] for row in rows} \
            != set(FULL_DRAIN_BANDWIDTHS_MBPS):
        raise RuntimeError("full-drain bandwidth grid is incomplete")
    write_full_drain_results(rows, out)
    summary_path = out / "full_drain_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["source_runs"] = [
        {"run_root": str(root), "plan_sha256": profiler.file_hash(root / "plan.json")}
        for root in run_roots]
    profiler.write_json(summary_path, summary)
    return rows

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", choices=("load", "goodput", "full-drain"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--live-template", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--adaptive-stage", choices=("phase2a", "phase2b"))
    parser.add_argument("--prior-run-root", type=Path, nargs="+")
    parser.add_argument("--merge-run-root", type=Path, nargs="+")
    parser.add_argument("--bandwidth-mbps", type=int,
                        choices=FULL_DRAIN_BANDWIDTHS_MBPS)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.campaign == "full-drain":
        if args.adaptive_stage or args.prior_run_root:
            raise ValueError("full drain does not use adaptive phases")
        if args.merge_run_root:
            if args.run_root or args.live_template or args.bandwidth_mbps:
                raise ValueError("full-drain merge accepts only roots and output")
            merge_full_drain(args.merge_run_root, args.out)
            return
        existing = args.run_root and (args.out / "plan.json").exists()
        if existing:
            campaign = json.loads((args.out / "plan.json").read_text())
        else:
            if not args.live_template or args.bandwidth_mbps is None:
                raise ValueError("full drain requires a template and bandwidth")
            campaign = write_full_drain_campaign(
                args.bandwidth_mbps, args.live_template, args.out,
                args.calibration, args.profile)
        if args.run_root:
            if args.out.resolve() == args.run_root.resolve():
                raise ValueError("plan and run roots must be distinct")
            execute_live(args.out / "live_plan.json", args.run_root, args.allow_dirty)
            reduce_live(args.run_root, campaign, args.out)
        return
    if args.merge_run_root:
        if not args.campaign == "load" or args.adaptive_stage or args.prior_run_root \
                or args.run_root or args.live_template:
            raise ValueError("merge accepts only load, --out, and --merge-run-root")
        roots = [root.resolve() for root in args.merge_run_root]
        if len(roots) != len(set(roots)) or args.out.resolve() in roots:
            raise ValueError("merge roots and output must be distinct")
        merge_live(roots, args.out)
        return
    if args.adaptive_stage:
        if not args.campaign == "load" or not args.prior_run_root \
                or not args.live_template:
            raise ValueError("adaptive phase requires load, prior roots, and template")
        roots = {root.resolve() for root in args.prior_run_root}
        current = {args.out.resolve()}
        if args.run_root:
            current.add(args.run_root.resolve())
        if len(current) != 1 + bool(args.run_root) or current & roots:
            raise ValueError("adaptive plan, run, and prior roots must be distinct")
        campaign, _ = write_adaptive_phase(
            args.adaptive_stage, args.prior_run_root, args.live_template, args.out)
    else:
        if args.prior_run_root:
            raise ValueError("--prior-run-root requires --adaptive-stage")
        campaign = json.loads((args.out / "plan.json").read_text()) \
            if args.run_root and (args.out / "plan.json").exists() \
            else write_campaign(args.campaign, args.out, args.calibration, args.profile)
        if args.live_template:
            write_live_plan(campaign, args.live_template, args.out)
    if args.run_root:
        if not args.live_template and not (args.out / "live_plan.json").exists():
            raise ValueError("hardware execution requires --live-template or live_plan.json")
        execute_live(args.out / "live_plan.json", args.run_root, args.allow_dirty)
        reduce_live(args.run_root, campaign, args.out)


if __name__ == "__main__":
    main()
