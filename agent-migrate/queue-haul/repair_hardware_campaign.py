"""Prepare and run the guarded three-region scheduled-repair hardware grid."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import migration_profiler as profiler
import network_campaign as network
import pool_planner
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
from repair_plan_shift_campaign import (
    CUT_SCALE,
    DEFAULT_PARENT,
    LOCATION_STATES,
    MOVE_CONCURRENCY,
    REFERENCE_CONTEXT_TOKENS,
    TARGET_SHED_FRACTION,
    TRIGGER_WORK_FRACTION,
    _affected,
    _candidate_map,
    _planned_moves,
    _prefill_observations,
    _resolve,
    _schedule_rows,
)
from simulate import PlannedMove


ROOT = Path(__file__).parent
SCHEMA = "queue-haul-scheduled-repair-hardware-plan-v5"
CONTROL_SCHEMA = "queue-haul-scheduled-repair-disabled-control-plan-v1"
RESULT_SCHEMA = "queue-haul-scheduled-repair-hardware-result-v2"
APPLY_POLICY = "shadow_validate_then_apply_pending_only"
CONTROL_POLICY = "shadow_validate_but_keep_original_pending"
REPEATS = 3
CALIBRATION_CONTEXTS = (1536, 7680, 32256)
CALIBRATION_METHODS = ("replay", "kv_transfer")
TIMING_RELATIVE_ERROR_GATE = 0.15
TIMING_ABSOLUTE_ERROR_GATE_S = 1.0
IMPLEMENTATION_FILES = (
    "destination.py", "migration_profiler.py", "migration_testbed.py",
    "network_campaign.py", "pool_planner.py", "prefill_gateway.py",
    "repair_controller.py", "repair_hardware_campaign.py",
    "repair_plan_shift_campaign.py",
)


def _hash(*values) -> str:
    return profiler.object_hash(values)[:16]


def _ttft_s(request: dict) -> float | None:
    """Return client-observed time to first content token in seconds."""
    start_ns = request.get("start_ns")
    first_byte_ns = request.get("first_byte_ns")
    if start_ns is None or first_byte_ns is None:
        return None
    return max(0.0, (int(first_byte_ns) - int(start_ns)) / 1e9)


def _impairment_score(destination: str, method: str,
                      bandwidth_nodes: tuple[str, ...],
                      prefill_nodes: tuple[str, ...]) -> int:
    """Count only controls that can slow this concrete action."""
    return int(destination in bandwidth_nodes and method == "kv_transfer") \
        + int(destination in prefill_nodes and method == "replay")


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.parent))
    except ValueError:
        return str(path.resolve())


def _template(parent: dict) -> dict:
    return next(row for row in parent["scenarios"]
                if row["condition_id"] == "joint-shaped"
                and row["repeat"] == 0 and row["policy"] == "queue_haul")


def make_plan(parent_path: Path, cluster_path: Path,
              calibration_path: Path) -> dict:
    parent = json.loads(parent_path.read_text())
    if parent.get("schema") != network.PLAN_SCHEMA:
        raise ValueError("hardware repair parent is not a network plan")
    manifest_path = _resolve(parent.get("manifest", {}).get("path", ""))
    if not manifest_path.is_file() \
            or profiler.file_hash(manifest_path) \
            != parent.get("manifest", {}).get("sha256"):
        raise ValueError("hardware repair parent manifest is missing or changed")
    timing_summary_path = parent_path.with_name("timing-summary.json")
    timing_summary = json.loads(timing_summary_path.read_text()) \
        if timing_summary_path.is_file() else {}
    if parent.get("design") != "separation" \
            or parent.get("timing_calibration", {}).get("schema") \
            != "queue-haul-regional-timing-fit-v2" \
            or not timing_summary.get("migration_gate_passed"):
        raise ValueError("hardware repair requires the passing regional timing plan")
    cluster = network.Cluster.load(cluster_path)
    if {node.id for node in cluster.destinations} != {"east", "germany"}:
        raise ValueError("hardware repair requires East and Germany destinations")
    calibration = json.loads(calibration_path.read_text())
    contract = network.freeze_contract(calibration)
    template = _template(parent)
    sessions = template["sessions"]
    calibration_cells = []
    for node in ("east", "germany"):
        for method in CALIBRATION_METHODS:
            for context in CALIBRATION_CONTEXTS:
                closest = min(
                    sessions, key=lambda row: abs(row["initial_tokens"] - context))
                for repeat in range(REPEATS):
                    calibration_cells.append({
                        "cell_id": _hash("calibration", node, method, context, repeat),
                        "node": node, "method": method,
                        "context_tokens": context, "repeat": repeat,
                        "session": {**closest, "initial_tokens": context},
                        "bandwidth_mbps": contract["paths"][node]["natural_mbps"]
                        * CUT_SCALE,
                    })
    episodes = []
    for bandwidth_state in LOCATION_STATES:
        for prefill_state in LOCATION_STATES:
            for repeat in range(REPEATS):
                episodes.append({
                    "episode_id": _hash(
                        "scheduled-repair", bandwidth_state, prefill_state, repeat),
                    "bandwidth_state": bandwidth_state,
                    "prefill_state": prefill_state,
                    "repeat": repeat,
                    "cut_scale": CUT_SCALE,
                    "trigger_work_fraction": TRIGGER_WORK_FRACTION,
                    "target_shed_fraction": TARGET_SHED_FRACTION,
                    "move_concurrency": MOVE_CONCURRENCY,
                })
    git_sha, dirty = profiler.git_state(True)
    return {
        "schema": SCHEMA,
        "parent": {"path": _portable(parent_path),
                   "sha256": profiler.file_hash(parent_path)},
        "cluster": cluster.as_dict(),
        "cluster_input": {"path": _portable(cluster_path),
                          "sha256": profiler.file_hash(cluster_path)},
        "calibration": {"path": _portable(calibration_path),
                        "sha256": profiler.file_hash(calibration_path)},
        "manifest": parent["manifest"],
        "model_profile": {"path": _portable(network.MODEL_PATH),
                          "sha256": profiler.file_hash(network.MODEL_PATH)},
        "network_contract": contract,
        "timing_calibration": parent["timing_calibration"],
        "timing_summary": {"path": _portable(timing_summary_path),
                           "sha256": profiler.file_hash(timing_summary_path)},
        "calibration_gate": {
            "relative_error": TIMING_RELATIVE_ERROR_GATE,
            "absolute_error_s": TIMING_ABSOLUTE_ERROR_GATE_S,
            "error_rule": "absolute_or_relative",
            "contexts": list(CALIBRATION_CONTEXTS),
            "repeats": REPEATS,
        },
        "calibration_cells": calibration_cells,
        "episodes": episodes,
        "repeats": REPEATS,
        "apply_policy": APPLY_POLICY,
        "implementation": {
            "git_sha": git_sha, "dirty": dirty,
            "files": [{
                "path": _portable(ROOT / name),
                "sha256": profiler.file_hash(ROOT / name),
            } for name in IMPLEMENTATION_FILES],
        },
    }


def validate_plan(plan: dict) -> None:
    is_control = plan.get("schema") == CONTROL_SCHEMA
    expected_episode_count = REPEATS if is_control else 16 * REPEATS
    if plan.get("schema") not in {SCHEMA, CONTROL_SCHEMA} \
            or plan.get("repeats") != REPEATS \
            or len(plan.get("calibration_cells", ())) != 36 \
            or len(plan.get("episodes", ())) != expected_episode_count \
            or len(plan.get("implementation", {}).get("files", ())) \
            != len(IMPLEMENTATION_FILES):
        raise ValueError("invalid scheduled repair hardware plan shape")
    if len({row["cell_id"] for row in plan["calibration_cells"]}) != 36 \
            or len({row["episode_id"] for row in plan["episodes"]}) \
            != expected_episode_count:
        raise ValueError("scheduled repair IDs are not unique")
    grid = {(row["bandwidth_state"], row["prefill_state"], row["repeat"])
            for row in plan["episodes"]}
    expected = ({("germany", "germany", repeat)
                 for repeat in range(REPEATS)} if is_control else {
        (bandwidth, prefill, repeat)
        for bandwidth in LOCATION_STATES
        for prefill in LOCATION_STATES for repeat in range(REPEATS)})
    calibration_grid = {
        (row["node"], row["method"], row["context_tokens"], row["repeat"])
        for row in plan["calibration_cells"]
    }
    expected_calibration = {
        (node, method, context, repeat)
        for node in ("east", "germany") for method in CALIBRATION_METHODS
        for context in CALIBRATION_CONTEXTS for repeat in range(REPEATS)
    }
    implementation = {
        Path(row["path"]).name for row in plan["implementation"]["files"]}
    if grid != expected or calibration_grid != expected_calibration \
            or implementation != set(IMPLEMENTATION_FILES) \
            or plan.get("apply_policy") \
            != (CONTROL_POLICY if is_control else APPLY_POLICY) \
            or plan.get("calibration_gate", {}).get("error_rule") \
            != "absolute_or_relative" \
            or any(row["cut_scale"] != CUT_SCALE
                               or row["trigger_work_fraction"]
                               != TRIGGER_WORK_FRACTION
                               or row["target_shed_fraction"]
                               != TARGET_SHED_FRACTION
                               or row["move_concurrency"] != MOVE_CONCURRENCY
                               for row in plan["episodes"]):
        raise ValueError("scheduled repair grid changed")
    if is_control and (
            not plan.get("control_of", {}).get("sha256")
            or not plan.get("paired_hardware_run", {}).get("validation", {}).get(
                "sha256")
            or any(not row.get("paired_repair_episode_id")
                   or not row.get("paired_result", {}).get("sha256")
                   or not row.get("expected_initial_moves_sha256")
                   for row in plan["episodes"])):
        raise ValueError("repair-disabled control provenance is incomplete")


def make_control_plan(base_plan_path: Path, paired_run_root: Path) -> dict:
    """Pair a no-apply control with the accepted Germany repair episodes."""
    base = json.loads(base_plan_path.read_text())
    validate_plan(base)
    if base["schema"] != SCHEMA or base["apply_policy"] != APPLY_POLICY:
        raise ValueError("control requires a full applied-repair hardware plan")
    validation_path = paired_run_root / "validation.json"
    validation = json.loads(validation_path.read_text())
    if not validation.get("passed"):
        raise ValueError("paired hardware run did not pass validation")
    episodes = []
    for original in base["episodes"]:
        if (original["bandwidth_state"], original["prefill_state"]) \
                != ("germany", "germany"):
            continue
        result_path = (paired_run_root / "episodes" /
                       original["episode_id"] / "result.json")
        result = json.loads(result_path.read_text())
        if result.get("status") != "complete" \
                or result.get("repair_outcome") != "applied" \
                or not result.get("target_met"):
            raise ValueError(
                f"paired episode is not an applied success: {result_path}")
        episodes.append({
            **original,
            "episode_id": _hash(
                "repair-disabled-control", original["episode_id"]),
            "paired_repair_episode_id": original["episode_id"],
            "paired_result": {
                "path": str(result_path.resolve()),
                "sha256": profiler.file_hash(result_path),
            },
            "expected_initial_moves_sha256": profiler.object_hash(
                result["initial_moves"]),
        })
    git_sha, dirty = profiler.git_state(True)
    control = {
        **base,
        "schema": CONTROL_SCHEMA,
        "episodes": episodes,
        "apply_policy": CONTROL_POLICY,
        "control_of": {
            "path": _portable(base_plan_path),
            "sha256": profiler.file_hash(base_plan_path),
        },
        "paired_hardware_run": {
            "path": str(paired_run_root.resolve()),
            "validation": {
                "path": str(validation_path.resolve()),
                "sha256": profiler.file_hash(validation_path),
            },
        },
        "implementation": {
            "git_sha": git_sha, "dirty": dirty,
            "files": [{
                "path": _portable(ROOT / name),
                "sha256": profiler.file_hash(ROOT / name),
            } for name in IMPLEMENTATION_FILES],
        },
    }
    validate_plan(control)
    return control


def _write_run_script(out: Path) -> None:
    script = out / "run.sh"
    script.write_text("""#!/usr/bin/env bash
set -euo pipefail
: "${QH_AZURE_SSH_KEY:?set QH_AZURE_SSH_KEY}"
: "${QH_REPAIR_RUN_ROOT:?set QH_REPAIR_RUN_ROOT}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/../../.."
export QH_LMCACHE_MODE="${QH_LMCACHE_MODE:-mp}"
uv run python queue-haul/repair_hardware_campaign.py run \
  --plan "$script_dir/plan.json" --ssh-key "$QH_AZURE_SSH_KEY" \
  --run-root "$QH_REPAIR_RUN_ROOT"
uv run python queue-haul/repair_hardware_campaign.py validate \
  --plan "$script_dir/plan.json" --run-root "$QH_REPAIR_RUN_ROOT"
""")
    script.chmod(0o755)


def prepare_control(base_plan_path: Path, paired_run_root: Path,
                    out: Path) -> dict:
    plan = make_control_plan(base_plan_path, paired_run_root)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", plan)
    _write_run_script(out)
    return plan


def prepare(parent_path: Path, cluster_path: Path, calibration_path: Path,
            out: Path) -> dict:
    plan = make_plan(parent_path, cluster_path, calibration_path)
    validate_plan(plan)
    out.mkdir(parents=True, exist_ok=True)
    profiler.write_json(out / "plan.json", plan)
    _write_run_script(out)
    return plan


def _promote_components(template: dict, contract: dict) -> dict:
    value = json.loads(json.dumps(template))
    for node in ("east", "germany"):
        cut = contract["paths"][node]["natural_mbps"] * CUT_SCALE * 125_000
        for component in value["migration_components"][node].values():
            component["bandwidth_range_bytes_per_s"][0] = min(
                component["bandwidth_range_bytes_per_s"][0], cut)
            component["allow_extrapolation"] = False
            component["provenance"] += "; pending live 0.1x route calibration"
    return value


def _scenario(template: dict, plan: dict, episode: dict) -> dict:
    scenario = json.loads(json.dumps(template))
    rates = _planning_bandwidths(
        template, plan, episode["bandwidth_state"])
    scenario.update({
        "design": "scheduled_repair_hardware",
        "scenario_id": episode["episode_id"],
        "condition_id": (
            f"bandwidth-{episode['bandwidth_state']}__"
            f"prefill-{episode['prefill_state']}"),
        "bandwidth": "scheduled_0.1x",
        "bandwidth_mbps": rates,
        "requested_shed_fraction": episode.get(
            "target_shed_fraction", TARGET_SHED_FRACTION),
        "deadline_s": episode.get("power_deadline_s", scenario["deadline_s"]),
        "planning_deadline_s": episode.get(
            "power_deadline_s", scenario.get(
                "planning_deadline_s", scenario["deadline_s"])),
        "full_horizon_s": episode.get(
            "observation_horizon_s", scenario.get("full_horizon_s")),
        "admission_mode": "normal",
    })
    if "healthy_east_load" in episode:
        scenario["background"]["east"][0] = episode["healthy_east_load"]
    return scenario


def _planning_bandwidths(template: dict, plan: dict,
                         bandwidth_state: str) -> dict[str, float]:
    """Use effective timing ceilings unless a path is live-shaped to 0.1x."""
    raw = network._bandwidths(plan["network_contract"], "natural")
    affected = _affected(bandwidth_state)
    return {
        node: (raw[node] * CUT_SCALE if node in affected else min(
            raw[node],
            min(component["bandwidth_range_bytes_per_s"][1]
                for component in template["migration_components"][node].values())
            / 125_000,
        )) for node in raw
    }


def _p90(values: list[float]) -> float | None:
    return sorted(values)[int(.9 * (len(values) - 1))] if values else None


def _timing_terms(problem, architecture, profile, node: str,
                  method: str) -> dict[str, float]:
    """Return the calibrated formula's fixed and fitted timing terms."""
    session = next(row for row in problem.sessions
                   if row.source_instance == "source")
    pool = next(row for row in architecture.pools
                if row.pool_id == f"pool/{node}")
    component = architecture.type_by_id[pool.type_id].migration[method]
    links = {link.link_id: link.bytes_per_s for link in problem.links}
    horizon = problem.deadline_s - problem.controller_delay_s \
        - profile.power_window_s

    def duration(value):
        return pool_planner._destination_duration(
            session, method, profile.case("central"), pool.route, links,
            horizon, value)

    original = duration(component)
    if method == "replay":
        one = duration(replace(component, compute_completion_factor=1))
        two = duration(replace(component, compute_completion_factor=2))
        scale = two - one
        base = one - scale
    else:
        base = duration(replace(component, residual_s=0))
        scale = 1.0
    if scale <= 0 or base < 0:
        raise RuntimeError("invalid regional timing fit terms")
    return {
        "original_predicted_s": original,
        "fit_base_s": base,
        "fit_scale_s": scale,
    }


def _timing_summary(rows: list[dict], gate: dict, template: dict) -> dict:
    """Fit small contexts and gate exclusively on the 32k holdout cells."""
    contexts = tuple(gate["contexts"])
    training_contexts, holdout_context = contexts[:-1], contexts[-1]
    fits, predictions = {}, []
    for node in ("east", "germany"):
        components = {}
        for method in CALIBRATION_METHODS:
            training = [row for row in rows
                        if row["node"] == node and row["method"] == method
                        and row["context_tokens"] in training_contexts
                        and row["status"] == "complete"]
            if method == "replay":
                denominator = sum(row["fit_scale_s"] ** 2 for row in training)
                parameter = max(1e-9, sum(
                    row["fit_scale_s"]
                    * (row["observed_s"] - row["fit_base_s"])
                    for row in training) / denominator) if denominator else None
            else:
                parameter = max(0.0, statistics.mean(
                    row["observed_s"] - row["fit_base_s"]
                    for row in training)) if training else None
            component = json.loads(json.dumps(
                template["migration_components"][node][method]))
            if parameter is not None:
                if method == "replay":
                    component["compute_completion_factor"] = parameter
                else:
                    component["residual_s"] = parameter
            bandwidth = [row["bandwidth_mbps"] * 125_000 for row in rows
                         if row["node"] == node]
            if bandwidth:
                supported = component["bandwidth_range_bytes_per_s"]
                component["bandwidth_range_bytes_per_s"] = [
                    min(supported[0], *bandwidth),
                    max(supported[1], *bandwidth),
                ]
            component["allow_extrapolation"] = False
            component["provenance"] += (
                "; live 0.1x fit on contexts "
                f"{list(training_contexts)}, holdout {holdout_context}")
            components[method] = component
            for row in rows:
                if row["node"] != node or row["method"] != method \
                        or row["status"] != "complete" or parameter is None:
                    continue
                predicted = row["fit_base_s"] + parameter * row["fit_scale_s"]
                error = abs(row["observed_s"] - predicted)
                predictions.append({
                    "cell_id": row["cell_id"], "node": node,
                    "method": method,
                    "context_tokens": row["context_tokens"],
                    "repeat": row["repeat"],
                    "split": ("holdout_context"
                              if row["context_tokens"] == holdout_context
                              else "training"),
                    "observed_s": row["observed_s"],
                    "predicted_s": predicted,
                    "error_s": error,
                    "relative_error": error / max(row["observed_s"], 1e-12),
                })
        fits[node] = {"migration_components": components}
    holdout = [row for row in predictions
               if row["split"] == "holdout_context"]
    errors = [row["error_s"] for row in holdout]
    relative = [row["relative_error"] for row in holdout]
    tolerance_ratios = [
        row["error_s"] / max(
            gate["absolute_error_s"],
            gate["relative_error"] * row["observed_s"],
        ) for row in holdout
    ]
    ttfts = [row["ttft_s"] for row in rows if row.get("ttft_s") is not None]
    expected_rows = 2 * len(CALIBRATION_METHODS) * len(contexts) \
        * gate["repeats"]
    expected_holdout = 2 * len(CALIBRATION_METHODS) * gate["repeats"]
    passed = len(rows) == expected_rows \
        and all(row["status"] == "complete" for row in rows) \
        and len(ttfts) == expected_rows \
        and len(holdout) == expected_holdout \
        and _p90(tolerance_ratios) is not None \
        and _p90(tolerance_ratios) <= 1 \
        and all(row.get("kv_verified", True) for row in rows)
    return {
        "schema": "queue-haul-repair-10x-timing-fit-v3",
        "rows": len(rows),
        "training_contexts": list(training_contexts),
        "holdout_context": holdout_context,
        "held_out_rows": len(holdout),
        "held_out_median_relative_error": (
            statistics.median(relative) if relative else None),
        "held_out_p90_relative_error": _p90(relative),
        "held_out_p90_absolute_error_s": _p90(errors),
        "held_out_p90_tolerance_ratio": _p90(tolerance_ratios),
        "error_rule": gate["error_rule"],
        "ttft_rows": len(ttfts),
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p90_s": _p90(ttfts),
        "ttft_max_s": max(ttfts) if ttfts else None,
        "fits": fits,
        "predictions": predictions,
        "passed": passed,
    }


def _apply_timing_fit(scenario: dict, timing: dict,
                      nodes: tuple[str, ...]) -> dict:
    value = json.loads(json.dumps(scenario))
    for node in nodes:
        value["migration_components"][node] = json.loads(json.dumps(
            timing["fits"][node]["migration_components"]))
    return value


def _run_calibration(stack, plan, parent, manifest, profile, root: Path) -> dict:
    rows = []
    template = _template(parent)
    for cell in plan["calibration_cells"]:
        cell_root = root / cell["cell_id"]
        result_path = cell_root / "result.json"
        if result_path.exists():
            rows.append(json.loads(result_path.read_text()))
            continue
        bandwidth_ack = network.set_live_bandwidth(
            stack, {cell["node"]: cell["bandwidth_mbps"]})
        network.set_live_prefill(stack, {"east": None, "germany": None})
        move = {
            "session_id": cell["session"]["session_id"],
            "destination_instance": cell["node"],
            "destination_pool": f"pool/{cell['node']}",
            "method": cell["method"], "order": 0,
            "path": [f"link/{cell['node']}"],
            "rate_limit_bytes_per_s": None, "quiesce_s": None,
        }
        scenario = {
            **template,
            "design": "calibration", "scenario_id": cell["cell_id"],
            "sessions": [cell["session"]], "moves": [move],
            "background": {"east": [0, 0], "germany": [0, 0]},
            "source_load": 0,
            "deadline_s": network.ORACLE_STALE_HORIZON_S,
            "load_warmup_s": 0,
        }
        raw_path = cell_root / "raw" / "result.json"
        raw = json.loads(raw_path.read_text()) if raw_path.exists() else \
            network.run_network_scenario(
                stack, manifest, scenario, cell_root / "raw", profile.case().F)
        modeled = _scenario(_promote_components(template, plan["network_contract"]),
                            plan, {"episode_id": cell["cell_id"],
                            "bandwidth_state": cell["node"],
                            "prefill_state": "none"})
        modeled["sessions"] = [cell["session"]]
        modeled["background"] = {"east": [0, 0], "germany": [0, 0]}
        modeled["source_load"] = 0
        problem, architecture, _, _, _ = network._scenario_problem(
            modeled, manifest, profile)
        terms = _timing_terms(
            problem, architecture, profile, cell["node"], cell["method"])
        row = {
            **cell, **terms,
            "schema": "queue-haul-repair-10x-timing-row-v2",
            "status": raw["status"], "observed_s": raw["migration_s"],
            "ttft_s": _ttft_s(raw["requests"][0]["request"]),
            "bandwidth_control_ack": bandwidth_ack,
            "kv_verified": cell["method"] != "kv_transfer" or bool(
                raw["requests"][0]["request"].get("cached_tokens", 0)),
        }
        profiler.write_json(result_path, row)
        rows.append(row)
        profiler.write_json(root / "progress.json", {
            "schema": "queue-haul-repair-calibration-progress-v1",
            "completed": len(rows),
            "expected": len(plan["calibration_cells"]),
            "latest_cell_id": cell["cell_id"],
            "latest_ttft_s": row["ttft_s"],
        })
    profiler.write_csv(root / "timing_rows.csv", rows)
    summary = _timing_summary(rows, plan["calibration_gate"], template)
    profiler.write_csv(root / "timing_predictions.csv", summary["predictions"])
    profiler.write_json(root / "summary.json", summary)
    return summary


def _run_episode(stack, plan, parent, manifest, profile, timing, episode,
                 root: Path):
    """Run a guarded repair or its paired no-apply hardware control."""
    root.mkdir(parents=True, exist_ok=False)
    network._clear_cluster(stack)
    network.set_live_bandwidth(stack, {})
    network.set_live_prefill(stack, {"east": None, "germany": None})
    natural_template = json.loads(json.dumps(_template(parent)))
    if "context_seed" in episode:
        context_rng = random.Random(episode["context_seed"])
        support = tuple(episode.get(
            "context_support", (14_042, 30_785, 31_547)))
        for session in natural_template["sessions"]:
            session["initial_tokens"] = context_rng.choice(support)
    template = _promote_components(
        natural_template, plan["network_contract"])
    scenario = _apply_timing_fit(
        _scenario(template, plan, episode), timing,
        _affected(episode["bandwidth_state"]))
    profiler.write_json(root / "scenario.json", scenario)
    records = network.scenario_records(manifest, scenario)
    messages = {row["session_id"]: profiler.calibration_messages(
        records[row["session_id"]], row["initial_tokens"])
        for row in scenario["sessions"]}
    loads = {}
    rates = (profile.case().prefill.rate(network.SINK_LOAD_PREFILL_TOKENS, 1),
             profile.case().decode.rate(network.SINK_LOAD_PREFILL_TOKENS, 1))
    for node in stack.cluster.destinations:
        rho = scenario["background"][node.id][0]
        if rho:
            loads[node.id] = network.SinkLoad(
                stack.cfg, stack.ports[node.id]["api"], rates[0], rho,
                root / f"sink_load_{node.id}.jsonl", rates[1], "background")
            loads[node.id].start()
    try:
        time.sleep(scenario.get("load_warmup_s", 5))
        snapshots = {node.id: network.destination_metrics(
            stack, node.id, scenario["background"][node.id][1])
            for node in stack.cluster.destinations}
        demand = network.agentic_demand(
            records, scenario["sessions"], profile, scenario["source_load"])
        natural = {
            **scenario,
            "bandwidth_mbps": _planning_bandwidths(
                natural_template, plan, "none"),
            "migration_components": natural_template["migration_components"],
        }
        problem, architecture, routes, target = network.joint_problem(
            natural, snapshots, profile, demand)
        result = network.solve(
            problem, profile, routes, "lp_work_first", destination=architecture,
            admission_mode="normal")
        live_initial_moves = [asdict(move) for move in result.moves]
        snapshot_warnings = {
            node: bool(snapshot.get("warning"))
            for node, snapshot in snapshots.items()
        }
        if episode.get("frozen_initial_moves"):
            result = replace(result, moves=tuple(PlannedMove(
                **{**move, "path": tuple(move["path"])})
                for move in episode["frozen_initial_moves"]))
        if not result.moves:
            raise RuntimeError(
                "initial hardware plan has no moves: "
                f"{result.failure_reason or 'unknown'}; "
                f"power shortfall {result.power_shortfall_w:.6f} W")
        initial_moves = [asdict(move) for move in result.moves]
        if episode.get("expected_initial_moves_sha256") \
                and profiler.object_hash(initial_moves) \
                != episode["expected_initial_moves_sha256"]:
            raise RuntimeError(
                "hardware initial plan differs from the frozen preflight plan")
        table = pool_planner.candidate_table(
            problem, profile, architecture, "normal", ExpectedPower(problem, profile))
        candidates = _candidate_map(table, architecture)
        missing_frozen_candidates = sorted(
            move.session_id for move in result.moves
            if (move.session_id, move.method, move.destination_pool)
            not in candidates)
        pool_planner.validate_destination_execution(
            problem, architecture, result.moves)
        initial_power = ExpectedPower(
            replace(problem, final_state="awake", assumed_shutdown_s=None),
            profile)
        frozen_shed_w = initial_power.drain_gain(frozenset(
            move.session_id for move in result.moves))
        live_plan_validation = {
            "passed": bool(
                result.admission_mode == "normal"
                and result.failure_reason is None
                and result.power_shortfall_w <= 1e-8
                and not any(snapshot_warnings.values())
                and not missing_frozen_candidates
                and frozen_shed_w >= float(target) - 1e-8),
            "live_solver_feasible": result.failure_reason is None,
            "live_solver_admission_mode": result.admission_mode,
            "live_solver_power_shortfall_w": result.power_shortfall_w,
            "live_solver_moves_sha256": profiler.object_hash(live_initial_moves),
            "frozen_moves_sha256": profiler.object_hash(initial_moves),
            "live_solver_matches_frozen": live_initial_moves == initial_moves,
            "snapshot_warnings": snapshot_warnings,
            "missing_frozen_candidates": missing_frozen_candidates,
            "frozen_shed_w": frozen_shed_w,
            "requested_shed_w": float(target),
        }
        if not live_plan_validation["passed"]:
            raise RuntimeError(
                "frozen initial plan failed live architecture validation: "
                f"{live_plan_validation}")
        durations = {move.session_id: candidates[
            (move.session_id, move.method, move.destination_pool)].duration_s
            for move in result.moves}
        # Prestage the initial plan; repair may switch these pending moves to KV.
        prestaged = {move.session_id for move in result.moves}
        for session_id in sorted(prestaged):
            network._warm(stack, messages[session_id],
                          records[session_id]["state_code"],
                          network.REQUEST_TIMEOUT_S)
        started_ns = time.monotonic_ns()
        execution: dict[str, dict] = {}
        active: dict[Future, object] = {}
        submitted_ns: dict[str, int] = {}
        pending = list(sorted(result.moves, key=lambda move: move.order))
        move_concurrency = episode.get("move_concurrency", MOVE_CONCURRENCY)
        executor = ThreadPoolExecutor(max_workers=move_concurrency)

        def reconstruct(move):
            began = time.monotonic_ns()
            request = network._chat(
                stack.cfg, stack.ports[move.destination_instance]["api"],
                messages[move.session_id], records[move.session_id]["state_code"],
                network.REQUEST_TIMEOUT_S, move.method == "replay", move.method)
            return {**asdict(move), "request": request,
                    "ttft_s": _ttft_s(request),
                    "started_ns": began, "ended_ns": time.monotonic_ns()}

        def submit() -> None:
            while pending and len(active) < move_concurrency:
                move = pending.pop(0)
                submitted_ns[move.session_id] = time.monotonic_ns()
                active[executor.submit(reconstruct, move)] = move

        def collect() -> None:
            for future in list(active):
                if future.done():
                    move = active.pop(future)
                    execution[move.session_id] = future.result()

        submit()
        total_work = sum(durations.values())
        while True:
            collect()
            now = time.monotonic_ns()
            progress = sum(durations[session] for session in execution)
            progress += sum(min(
                durations[move.session_id],
                (now - submitted_ns[move.session_id]) / 1e9)
                for move in active.values())
            elapsed_s = (now - started_ns) / 1e9
            fixed_fault_s = episode.get("fault_at_s")
            triggered = (
                elapsed_s >= fixed_fault_s if fixed_fault_s is not None else
                progress / total_work >= episode.get(
                    "trigger_work_fraction", TRIGGER_WORK_FRACTION))
            if triggered:
                break
            submit()
            if not active and not pending:
                raise RuntimeError("episode completed before its repair trigger")
            time.sleep(.05)
        event_s = (time.monotonic_ns() - started_ns) / 1e9
        fault_apply_started_s = (time.monotonic_ns() - started_ns) / 1e9
        bandwidth_nodes = _affected(episode["bandwidth_state"])
        cut_rates = {node: plan["network_contract"]["paths"][node]["natural_mbps"]
                     * CUT_SCALE for node in bandwidth_nodes}
        bandwidth_ack = network.set_live_bandwidth(stack, cut_rates)
        bandwidth_ack_s = (time.monotonic_ns() - started_ns) / 1e9
        prefill_nodes = _affected(episode["prefill_state"])
        gateway_rates = {node: (rates[0] * CUT_SCALE
                                if node in prefill_nodes else None)
                         for node in ("east", "germany")}
        prefill_ack = network.set_live_prefill(stack, gateway_rates)
        prefill_ack_s = (time.monotonic_ns() - started_ns) / 1e9
        fault_applied_s = prefill_ack_s
        changed_problem, changed_architecture, _, changed_target = network.joint_problem(
            scenario, snapshots, profile, demand)
        if abs(changed_target - target) > 1e-8:
            raise RuntimeError("scheduled disturbance changed the shed target")
        capacities = _prefill_observations(
            changed_architecture, episode["prefill_state"])
        observed_architecture = pool_planner._repair_architecture(
            changed_architecture, capacities)
        changed_table = pool_planner.candidate_table(
            changed_problem, profile, observed_architecture, "normal",
            ExpectedPower(changed_problem, profile),
        )
        changed_candidates = _candidate_map(
            changed_table, observed_architecture)
        attempts = []
        continuations = []
        active_sessions = {move.session_id for move in active.values()}
        for move in result.moves:
            total = durations[move.session_id]
            if move.session_id in execution:
                completed, status = total, "committed"
            elif move.session_id in active_sessions:
                completed = min(
                    total,
                    (time.monotonic_ns() - submitted_ns[move.session_id]) / 1e9)
                status = "running"
            else:
                completed, status = 0.0, "pending"
            replacement = changed_candidates.get((
                move.session_id, move.method, move.destination_pool))
            rate = total / replacement.duration_s if replacement else 1.0
            if replacement is None \
                    and move.method == "kv_transfer" \
                    and move.destination_instance in bandwidth_nodes:
                rate *= CUT_SCALE
            if replacement is None \
                    and move.method == "replay" \
                    and move.destination_instance in prefill_nodes:
                rate *= CUT_SCALE
            attempts.append(Attempt(
                move.session_id, 0, Assignment(
                    move.method, move.destination_instance, move.destination_pool),
                status, total, completed, event_s, total, rate=rate,
                repairable=status != "running"))
            if status != "committed":
                continuations.append(RepairMove(
                    move.session_id, Assignment(
                        move.method, move.destination_instance,
                        move.destination_pool),
                    (total - completed) / rate, total,
                ))
        attempt_map = {attempt.session_id: attempt for attempt in attempts}
        continuation_schedule = _schedule_rows(
            attempt_map, result.moves, tuple(continuations), event_s, event_s)
        planned_commits = {
            row["session_id"]: row["completion_s"]
            for row in continuation_schedule
            if row["status"] == "scheduled_after_repair"
        }
        attempts = [replace(
            attempt,
            planned_commit_s=planned_commits.get(
                attempt.session_id, attempt.planned_commit_s),
        ) for attempt in attempts]
        power = ExpectedPower(replace(problem, final_state="awake",
                                      assumed_shutdown_s=None), profile)
        controller = FeasibilityRepairController(
            tuple(attempts), {session.session_id for session in problem.sessions},
            float(target), problem.deadline_s - profile.power_window_s, 0,
            power.drain_gain)
        route_rates = tuple((link.link_id, link.bytes_per_s)
                            for link in changed_problem.links)
        controller.observe(ObservationBatch(
            1, event_s, route_rates=route_rates,
            prefill_capacities=capacities))
        detection_deadline = (
            started_ns / 1e9
            + episode.get("detection_at_s", fault_applied_s + 1.0))
        while time.monotonic() < detection_deadline:
            collect()
            if plan["apply_policy"] == CONTROL_POLICY:
                submit()
            time.sleep(min(.05, max(0.0, detection_deadline - time.monotonic())))
        collect()
        decision_s = (time.monotonic_ns() - started_ns) / 1e9
        updates = tuple(AttemptUpdate(
            attempt.session_id, 0,
            "committed" if attempt.session_id in execution else attempt.status,
            attempt.total_work,
            attempt.total_work if attempt.session_id in execution else min(
                attempt.total_work,
                attempt.completed_work
                + (decision_s - event_s) * (attempt.rate or 0)),
        )
            for attempt in attempts if attempt.status == "running")
        decision = controller.observe(ObservationBatch(
            2, decision_s, attempts=updates, route_rates=route_rates,
            prefill_capacities=capacities))
        repair_result = proposal = None
        solver_timings = []
        for _ in range(2):
            if not isinstance(decision, RepairRequest):
                break
            solver_started_s = (time.monotonic_ns() - started_ns) / 1e9
            repair_result = pool_planner.repair_destination(
                changed_problem, profile, changed_architecture, decision, "normal")
            solver_ended_s = (time.monotonic_ns() - started_ns) / 1e9
            solver_timings.append({
                "request_id": decision.request_id,
                "started_s": solver_started_s,
                "ended_s": solver_ended_s,
                "duration_s": solver_ended_s - solver_started_s,
            })
            decision = controller.complete_repair(repair_result, solver_ended_s)
        proposal_s = ((time.monotonic_ns() - started_ns) / 1e9
                      if isinstance(decision, ProposedDiff) else None)
        apply_s = None
        shadow_guard = {"passed": False, "reason": "no target-restoring proposal"}
        if isinstance(decision, ProposedDiff):
            changed = {row.session_id for row in decision.changes}
            forbidden = changed & active_sessions
            before = {move.session_id: (
                move.destination_instance, move.method)
                      for move in result.moves}
            reduced_impaired = sum(
                row.session_id in before
                and row.assignment is not None
                and _impairment_score(
                    *before[row.session_id], bandwidth_nodes, prefill_nodes)
                > _impairment_score(
                    row.assignment.destination, row.assignment.method,
                    bandwidth_nodes, prefill_nodes)
                for row in decision.changes)
            increased_impaired = sum(
                row.assignment is not None
                and (_impairment_score(
                    *before[row.session_id], bandwidth_nodes, prefill_nodes)
                     if row.session_id in before else 0) < _impairment_score(
                    row.assignment.destination, row.assignment.method,
                    bandwidth_nodes, prefill_nodes)
                for row in decision.changes)
            removed_from_impaired = sum(
                row.session_id in before
                and _impairment_score(
                    *before[row.session_id], bandwidth_nodes, prefill_nodes) > 0
                and row.assignment is None for row in decision.changes)
            unsafe_kv = sorted(
                row.session_id for row in decision.changes
                if row.assignment is not None
                and row.assignment.method == "kv_transfer"
                and row.session_id not in prestaged)
            shadow_guard = {
                "passed": not forbidden and not unsafe_kv
                and increased_impaired == 0
                and reduced_impaired + removed_from_impaired > 0,
                "reason": (
                    f"proposal changes active sessions: {sorted(forbidden)}"
                    if forbidden else
                    f"KV state was not prestaged: {unsafe_kv}"
                    if unsafe_kv else
                    "proposal increases impaired-resource work"
                    if increased_impaired else
                    "proposal does not reduce impaired-resource work"
                    if not reduced_impaired + removed_from_impaired else "passed"
                ),
                "budget_version": controller.budget_version,
                "reduced_impaired_actions": reduced_impaired,
                "increased_impaired_actions": increased_impaired,
                "removed_from_impaired": removed_from_impaired,
                "unsafe_unstaged_kv": unsafe_kv,
            }
            if shadow_guard["passed"] \
                    and plan["apply_policy"] == APPLY_POLICY:
                proposal = decision
                apply_s = (time.monotonic_ns() - started_ns) / 1e9
                controller.acknowledge(proposal.proposal_id, "applied", apply_s)
                repaired = _planned_moves(proposal.moves, changed_architecture)
                pending = [move for move in repaired
                           if move.session_id not in active_sessions
                           and move.session_id not in execution]
        elif isinstance(decision, RevisedMaximum):
            shadow_guard["attainable_watts"] = decision.attainable_watts

        def model_target_reached() -> bool:
            return power.drain_gain(frozenset(execution)) >= float(target) - 1e-8

        submit()
        while active or pending:
            collect()
            elapsed_s = (time.monotonic_ns() - started_ns) / 1e9
            if model_target_reached() or elapsed_s >= episode.get(
                    "observation_horizon_s", float("inf")):
                pending.clear()
            submit()
            if active:
                time.sleep(.05)
        executor.shutdown()
        ended_ns = time.monotonic_ns()
        request_rows = list(execution.values())
        outcomes = network.diagnostic_outcomes(
            scenario, request_rows, demand, profile, started_ns)
        cutoff_s = episode.get(
            "migration_cutoff_s", scenario["deadline_s"] - profile.power_window_s)
        cutoff_outcomes = network.diagnostic_outcomes(
            {**scenario, "deadline_s": cutoff_s}, request_rows,
            demand, profile, started_ns)
        predecision_outcomes = network.diagnostic_outcomes(
            {**scenario, "deadline_s": decision_s}, request_rows,
            demand, profile, started_ns)
        direct_power_rows = []
        power_path = stack.run_root / "power.csv"
        if power_path.is_file():
            direct_power_rows = [{
                **row,
                "elapsed_s": (row["monotonic_ns"] - started_ns) / 1e9,
            } for row in profiler.power_rows(power_path)
                if started_ns <= row["monotonic_ns"] <= ended_ns]
            if direct_power_rows:
                profiler.write_csv(root / "source_power.csv", direct_power_rows)
        proposal_changes = ([] if proposal is None else [
            asdict(change) for change in proposal.changes])
        initial_assignments = {move["session_id"]: {
            "method": move["method"],
            "destination": move["destination_instance"],
            "pool": move["destination_pool"],
        } for move in initial_moves}

        def is_impaired(assignment: dict) -> bool:
            return (
                assignment["method"] == "kv_transfer"
                and assignment["destination"] in bandwidth_nodes
            ) or (
                assignment["method"] == "replay"
                and assignment["destination"] in prefill_nodes
            )

        causal_changes = [change for change in proposal_changes
                          if change.get("assignment") is not None
                          and change["session_id"] in initial_assignments
                          and is_impaired(initial_assignments[
                              change["session_id"]])
                          and not is_impaired(change["assignment"])]
        output = {
            "schema": RESULT_SCHEMA, "status": "complete",
            "episode_id": episode["episode_id"],
            "event_s": event_s, "decision_s": decision_s,
            "fault_apply_started_s": fault_apply_started_s,
            "bandwidth_ack_s": bandwidth_ack_s,
            "prefill_ack_s": prefill_ack_s,
            "fault_applied_s": fault_applied_s,
            "proposal_s": proposal_s, "apply_s": apply_s,
            "solver_timings": solver_timings,
            "bandwidth_control": cut_rates,
            "bandwidth_control_ack": bandwidth_ack,
            "prefill_control": gateway_rates,
            "prefill_control_ack": prefill_ack,
            "live_plan_validation": live_plan_validation,
            "shadow_guard": shadow_guard,
            "repair_outcome": (
                "disabled" if plan["apply_policy"] == CONTROL_POLICY else
                "applied" if proposal else
                "revised_maximum" if isinstance(decision, RevisedMaximum) else
                "unchanged"),
            "apply_policy": plan["apply_policy"],
            "paired_repair_episode_id": episode.get(
                "paired_repair_episode_id"),
            "repair_result": None if repair_result is None else asdict(repair_result),
            "initial_moves": initial_moves,
            "initial_moves_sha256": profiler.object_hash(initial_moves),
            "proposal_changes": proposal_changes,
            "redirected_sessions": sum(
                change.get("assignment") is not None
                and initial_assignments.get(change["session_id"])
                != change["assignment"]
                for change in proposal_changes),
            "causal_redirected_sessions": len(causal_changes),
            "causal_method_switches": sum(
                initial_assignments[change["session_id"]]["method"]
                != change["assignment"]["method"]
                for change in causal_changes),
            "causal_destination_switches": sum(
                initial_assignments[change["session_id"]]["destination"]
                != change["assignment"]["destination"]
                for change in causal_changes),
            "requests": request_rows,
            "ttft_recorded": all(row["ttft_s"] is not None
                                 for row in request_rows),
            "started_ns": started_ns, "ended_ns": ended_ns,
            "migration_cutoff_s": cutoff_s,
            "cutoff_shed_w": cutoff_outcomes["realized_shed_w"],
            "target_met_by_cutoff": cutoff_outcomes["target_met"],
            "predecision_shed_w": predecision_outcomes["realized_shed_w"],
            "direct_source_power": {
                "path": "source_power.csv",
                "samples": len(direct_power_rows),
                "semantics": (
                    "direct A100 board-power trace for diagnosis; shed and "
                    "attainment use completion-credited workload-model watts"),
            },
            "attainment_semantics": (
                "modeled source watts credited only when a successful "
                "migration request completes"),
            **outcomes,
        }
        profiler.write_json(root / "result.json", output)
        return output
    finally:
        for load in loads.values():
            load.close()


def run(plan_path: Path, key: Path, run_root: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    validate_plan(plan)
    parent_path = _resolve(plan["parent"]["path"])
    calibration_path = _resolve(plan["calibration"]["path"])
    timing_summary_path = _resolve(plan["timing_summary"]["path"])
    manifest_path = _resolve(plan["manifest"]["path"])
    cluster_input_path = _resolve(plan["cluster_input"]["path"])
    if profiler.file_hash(parent_path) != plan["parent"]["sha256"] \
            or profiler.file_hash(calibration_path) != plan["calibration"]["sha256"] \
            or profiler.file_hash(timing_summary_path) \
            != plan["timing_summary"]["sha256"] \
            or profiler.file_hash(manifest_path) != plan["manifest"]["sha256"] \
            or profiler.file_hash(cluster_input_path) \
            != plan["cluster_input"]["sha256"] \
            or profiler.file_hash(network.MODEL_PATH) != plan["model_profile"]["sha256"]:
        raise RuntimeError("scheduled repair plan input changed")
    if plan["apply_policy"] == CONTROL_POLICY:
        control_of = _resolve(plan["control_of"]["path"])
        validation = Path(
            plan["paired_hardware_run"]["validation"]["path"])
        paired_results = [Path(row["paired_result"]["path"])
                          for row in plan["episodes"]]
        if profiler.file_hash(control_of) != plan["control_of"]["sha256"] \
                or profiler.file_hash(validation) \
                != plan["paired_hardware_run"]["validation"]["sha256"] \
                or any(profiler.file_hash(path) != row["paired_result"]["sha256"]
                       for path, row in zip(paired_results, plan["episodes"])):
            raise RuntimeError("paired repair hardware evidence changed")
    for row in plan["implementation"]["files"]:
        if profiler.file_hash(_resolve(row["path"])) != row["sha256"]:
            raise RuntimeError(
                f"scheduled repair implementation changed: {row['path']}")
    parent = json.loads(parent_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    profile = ModelProfile.load(network.MODEL_PATH)
    cluster = network.Cluster.parse(plan["cluster"])
    network.host_check(cluster, key)
    run_root.mkdir(parents=True, exist_ok=True)
    profiler.write_json(run_root / "plan.json", plan)
    stack_id = _hash(str(run_root.resolve()),
                     plan["implementation"]["git_sha"])
    stack_root = run_root / f"stack-{stack_id}"
    stack = network.start_cluster(
        cluster, key, plan["network_contract"], "natural",
        stack_root, power_interval_s=.1)
    try:
        profiler.write_json(run_root / "status.json", {
            "schema": "queue-haul-repair-hardware-status-v1",
            "phase": "calibration", "completed": 0,
            "expected": len(plan["calibration_cells"]),
        })
        timing = _run_calibration(
            stack, plan, parent, manifest, profile, run_root / "calibration")
        profiler.write_json(run_root / "status.json", {
            "schema": "queue-haul-repair-hardware-status-v1",
            "phase": "calibration_complete",
            "completed": len(plan["calibration_cells"]),
            "expected": len(plan["calibration_cells"]),
            "gate_passed": timing["passed"],
        })
        if not timing["passed"]:
            raise RuntimeError("0.1x timing calibration gate failed; main grid not run")
        results = []
        for episode in plan["episodes"]:
            path = run_root / "episodes" / episode["episode_id"] / "result.json"
            if path.exists():
                results.append(json.loads(path.read_text()))
            else:
                results.append(_run_episode(
                    stack, plan, parent, manifest, profile, timing,
                    episode, path.parent))
            profiler.write_json(run_root / "status.json", {
                "schema": "queue-haul-repair-hardware-status-v1",
                "phase": "episodes",
                "completed": len(results),
                "expected": len(plan["episodes"]),
                "latest_episode_id": episode["episode_id"],
            })
    except Exception as error:
        profiler.write_json(run_root / "status.json", {
            "schema": "queue-haul-repair-hardware-status-v1",
            "phase": "failed", "error": f"{type(error).__name__}: {error}",
        })
        raise
    finally:
        network.stop_cluster(stack)
    summary = reduce(plan, run_root)
    if not summary["passed"]:
        raise RuntimeError("scheduled repair hardware validation failed")
    profiler.write_json(run_root / "status.json", {
        "schema": "queue-haul-repair-hardware-status-v1",
        "phase": "complete", "validation_passed": True,
        "completed": len(plan["episodes"]),
        "expected": len(plan["episodes"]),
    })
    return summary


def reduce(plan: dict, run_root: Path) -> dict:
    results = []
    for episode in plan["episodes"]:
        path = run_root / "episodes" / episode["episode_id"] / "result.json"
        if path.exists():
            results.append({**episode, **json.loads(path.read_text())})
    rows = [{
        "episode_id": row["episode_id"],
        "bandwidth_state": row["bandwidth_state"],
        "prefill_state": row["prefill_state"], "repeat": row["repeat"],
        "repair_outcome": row["repair_outcome"],
        "shadow_guard_passed": row["shadow_guard"]["passed"],
        "would_repair": bool((row.get("repair_result") or {}).get(
            "reaches_target", False)) and row["shadow_guard"]["passed"],
        "target_met": row["target_met"],
    } for row in results]
    if rows:
        profiler.write_csv(run_root / "repair_episodes.csv", rows)
    ttft_rows = [{
        "episode_id": row["episode_id"],
        "bandwidth_state": row["bandwidth_state"],
        "prefill_state": row["prefill_state"],
        "repeat": row["repeat"],
        "session_id": request["session_id"],
        "method": request["method"],
        "destination_instance": request["destination_instance"],
        "start_ns": request["request"].get("start_ns"),
        "first_token_ns": request["request"].get("first_byte_ns"),
        "end_ns": request["request"].get("end_ns"),
        "ttft_s": request.get("ttft_s", _ttft_s(request["request"])),
    } for row in results for request in row["requests"]]
    if ttft_rows:
        profiler.write_csv(run_root / "repair_ttft.csv", ttft_rows)
    ttfts = [row["ttft_s"] for row in ttft_rows
             if row["ttft_s"] is not None]
    is_control = plan["apply_policy"] == CONTROL_POLICY
    baseline = [row for row in rows if row["bandwidth_state"] == "none"
                and row["prefill_state"] == "none"]
    request_rows = [request for row in results for request in row["requests"]]
    requests_passed = all(
        request["request"].get("status_code") == 200
        for request in request_rows)
    common_passed = len(rows) == len(plan["episodes"]) \
        and bool(ttft_rows) and len(ttfts) == len(ttft_rows) \
        and requests_passed
    passed = (common_passed
              and all(row["repair_outcome"] == "disabled"
                      and row["would_repair"] for row in rows)) \
        if is_control else (common_passed
              and all(row["target_met"] for row in baseline)
              and any(row["repair_outcome"] == "applied" for row in rows)
              and all(row["target_met"] for row in rows
                      if row["repair_outcome"] == "applied")
              and all(row["repair_outcome"] != "applied"
                      or row["shadow_guard_passed"] for row in rows))
    summary = {
        "schema": (
            "queue-haul-repair-disabled-control-validation-v1"
            if is_control else
            "queue-haul-scheduled-repair-hardware-validation-v2"),
        "expected": len(plan["episodes"]), "completed": len(rows),
        "applied": sum(row["repair_outcome"] == "applied" for row in rows),
        "revised_maximum": sum(row["repair_outcome"] == "revised_maximum"
                               for row in rows),
        "disabled": sum(row["repair_outcome"] == "disabled" for row in rows),
        "would_repair": sum(row["would_repair"] for row in rows),
        "target_met": sum(row["target_met"] for row in rows),
        "http_200": sum(
            request["request"].get("status_code") == 200
            for request in request_rows),
        "ttft_rows": len(ttfts),
        "ttft_p50_s": statistics.median(ttfts) if ttfts else None,
        "ttft_p90_s": sorted(ttfts)[int(.9 * (len(ttfts) - 1))]
        if ttfts else None,
        "ttft_max_s": max(ttfts) if ttfts else None,
        "passed": passed,
    }
    profiler.write_json(run_root / "validation.json", summary)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    command.add_argument("--cluster", type=Path, required=True)
    command.add_argument("--calibration", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("prepare-control")
    command.add_argument("--base-plan", type=Path, required=True)
    command.add_argument("--paired-run-root", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = sub.add_parser("run")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--ssh-key", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command = sub.add_parser("validate")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        prepare(args.parent, args.cluster, args.calibration, args.out)
    elif args.command == "prepare-control":
        prepare_control(args.base_plan, args.paired_run_root, args.out)
    elif args.command == "run":
        print(json.dumps(run(
            args.plan, args.ssh_key.expanduser(), args.run_root), indent=2))
    else:
        plan = json.loads(args.plan.read_text())
        validate_plan(plan)
        value = reduce(plan, args.run_root)
        print(json.dumps(value, indent=2))
        if not value["passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
