"""Conservative workload-sensitivity action mixes over a resource factorial."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bootstrap_action_adaptation import (
    read_csv, stratified_timing_bootstrap, timing_fit,
)
from destination import (
    DestinationArchitecture, DestinationPool, DestinationReplica,
    FluidMigrationService, LoadedCoefficients, MigrationComponents,
    dedicated_sink_architecture,
)
from planner import plan, source_power
import plot_style
from loaded_service_model import validate_model as validate_loaded_service_model
from pool_planner import candidate_table, fractional_power_opportunity
from power_model import ExpectedPower
from profiles import ModelProfile
from simulate import (
    ExecutionScenario, NetworkLink, PowerNode, ServingInstance, SimSession,
)
from stress_frontier_campaign import state_profile


ROOT = Path(__file__).parent
PROFILE = ROOT / "outputs/azure-compact-calibration-20260813/gpt_oss_20b_a100_tp1_azure_300w_phase.json"
MANIFEST = ROOT / "outputs/destination-v7-20260722/content-free-manifest.json"
LOCAL_TIMING = ROOT / "outputs/coding-run"
WIDTH8_TIMING = ROOT / "outputs/policy-hardware-width8-frontier-20260730"
TIMING = ROOT / "outputs/timing-power-validation-20260814/migrations.csv"
TIMING_SUMMARY = ROOT / "outputs/timing-power-validation-20260814/timing-summary.json"
TIMING_PARENT = ROOT / "outputs/timing-power-validation-20260814/separation-regional-timing-v2.json"
LOADED_SERVICE = ROOT / "outputs/loaded-service-model-20260815/model.json"
NETWORK_CALIBRATION = ROOT / "outputs/east-germany-frontier-20260808/control/calibration-east-germany-frontier-001.json"
OUT = ROOT / "outputs/workload-action-adaptation-20260814"
BASE_PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
WIDTH8_PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
FACTORS = ("hbm", "bandwidth", "dest_compute")
ACTIVATION_GATED_FACTORS = ("hbm", "dest_compute")
ORDER = (
    frozenset(("hbm",)), frozenset(("bandwidth",)), frozenset(("dest_compute",)),
    frozenset(("hbm", "bandwidth")), frozenset(("hbm", "dest_compute")),
    frozenset(("bandwidth", "dest_compute")), frozenset(FACTORS), frozenset(),
)
LABELS = {
    frozenset(("hbm",)): "HBM", frozenset(("bandwidth",)): "Bandwidth",
    frozenset(("dest_compute",)): "Dest. compute",
    frozenset(("hbm", "bandwidth")): "HBM + bandwidth",
    frozenset(("hbm", "dest_compute")): "HBM + compute",
    frozenset(("bandwidth", "dest_compute")): "Bandwidth + compute",
    frozenset(FACTORS): "All bound", frozenset(): "None bound",
}
LEVELS = {"hbm": (0.0, .98), "bandwidth": ("natural", "bottleneck_1g"),
          "dest_compute": (.25, .95)}
MIN_ACTION_RESPONSE_RATE = .1
REGIONS = ("east", "germany")
ACTIONS = ("replay", "kv_transfer", "not_moved")
DISPLAY_CASES = (
    "hbm", "bandwidth", "dest_compute", "bandwidth-dest_compute-hbm", "none",
)
ACTION_BOXPLOT_QUANTILES = (.05, .25, .5, .75, .95)
POWER_TOLERANCE_W = 1e-6
MIGRATION_HORIZON_S = 41
BANDWIDTH_BOTTLENECK_MBPS = 1000
PREFILL_REFERENCE_TOKENS = 7680
OAT_BANDWIDTH_LOWER_MBPS = 100
OAT_BANDWIDTH_UPPER_MBPS = 15_000
OAT_PREFILL_UPPER_TPS = 15_000
OAT_DEST_COMPUTE = (0.05, 0.99)
SOURCE_LOAD = .4
DEFAULT_SEED = 1001
plot_style.apply()


@dataclass(frozen=True)
class Shape:
    template_id: str
    family: str
    context_tokens: int
    prompt_tokens: int
    output_tokens: int


def factorial_cases():
    generated = {
        frozenset(factor for factor, enabled in zip(FACTORS, flags) if enabled)
        for flags in product((False, True), repeat=len(FACTORS))
    }
    if generated != set(ORDER):
        raise RuntimeError("constraint factorial is incomplete")
    return tuple(("-".join(sorted(case)) or "none", LABELS[case], case)
                 for case in ORDER)


def load_templates(path: Path, profile: ModelProfile):
    raw = json.loads(path.read_text())
    family = {session_id: name
              for name, splits in raw["manifest"]["splits"].items()
              for session_id in sum(splits.values(), [])}
    case = profile.case()
    prefill = case.prefill.by_concurrency[1][0]
    decode = case.decode.by_concurrency[1][0]
    x = (max(1536, prefill[0], decode[0]),
         min(32256, prefill[-1], decode[-1]))
    phase = case.phase_power
    if phase is None:
        raise ValueError("workload adaptation requires phase-power support")
    templates, excluded, phase_excluded, unselected = {}, 0, 0, 0
    for row in raw["traces"]:
        if row["session_id"] not in family:
            unselected += 1
            continue
        context = int(row["input_tokens_total"]) - int(row["newly_append_tokens"])
        if not x[0] <= context <= x[1]:
            excluded += 1
            continue
        shape = Shape(
            row["session_id"], family[row["session_id"]], context,
            int(row["newly_append_tokens"]), int(row["output_tokens"]),
        )
        phase_rate = max(shape.prompt_tokens, shape.output_tokens)
        if phase_rate <= 0:
            raise ValueError("workload state has no phase work")
        scale = 1e-3 / phase_rate
        if not phase.contains(scale * shape.prompt_tokens,
                              scale * shape.output_tokens):
            phase_excluded += 1
            continue
        templates.setdefault(shape.template_id, []).append(shape)
    if not templates:
        raise ValueError("no workload templates inside timing support")
    return {key: tuple(value) for key, value in templates.items()}, {
        "manifest_templates": len(family), "supported_templates": len(templates),
        "supported_states": sum(map(len, templates.values())),
        "excluded_states": excluded,
        "phase_direction_excluded_states": phase_excluded,
        "unselected_states": unselected,
    }


def sample_pack(templates, count: int, seed: int):
    if count < 1:
        raise ValueError("session count must be positive")
    rng, keys = np.random.default_rng(seed), tuple(sorted(templates))
    sessions = []
    for index, key in enumerate(keys[i] for i in rng.integers(0, len(keys), count)):
        shape = templates[key][int(rng.integers(len(templates[key])))]
        sessions.append(SimSession(
            f"pack-{seed}-{index}", "source", shape.context_tokens,
            shape.prompt_tokens, shape.output_tokens,
            2 * shape.context_tokens,
        ))
    return tuple(sessions)


def normalize_pack(profile, sessions, load=SOURCE_LOAD):
    case = profile.case()
    total = sum(session.expected_f / case.F + session.expected_g / case.G
                for session in sessions)
    if not 0 < load <= 1 or total <= 0:
        raise ValueError("invalid source workload normalization")
    scale = load / total
    return tuple(replace(session, expected_f=session.expected_f * scale,
                         expected_g=session.expected_g * scale)
                 for session in sessions)


def state_values(constraints):
    return {factor: LEVELS[factor][factor in constraints] for factor in FACTORS}


@lru_cache(maxsize=1)
def loaded_service_model(path=LOADED_SERVICE):
    value = json.loads(Path(path).read_text())
    value = validate_loaded_service_model(value)
    if value["bootstrap_samples"] != 1000 or value["bootstrap_seed"] != 1:
        raise ValueError("loaded-service artifact is not the canonical fit")
    return value


@lru_cache(maxsize=1)
def physical_route_mbps(path=NETWORK_CALIBRATION):
    raw = json.loads(Path(path).read_text())
    expected = json.loads(TIMING_PARENT.read_text())["calibration"]
    if raw.get("schema") != "queue-haul-network-calibration-v1" \
            or set(raw.get("paths", {})) != set(REGIONS) \
            or file_hash(Path(path)) != expected["sha256"]:
        raise ValueError("regional network calibration does not match timing evidence")
    values = {region: float(np.median(raw["paths"][region]["simultaneous_mbps"]))
              for region in REGIONS}
    if min(values.values()) <= BANDWIDTH_BOTTLENECK_MBPS:
        raise ValueError("bandwidth bottleneck must be below natural route capacity")
    return values


def central_timing_fits():
    profile = ModelProfile.load(PROFILE)
    fits = timing_fit(
        profile, json.loads(TIMING_PARENT.read_text()), read_csv(TIMING), str(TIMING),
    )
    summary = json.loads(TIMING_SUMMARY.read_text())
    expected = summary["fits"]
    if not summary["migration_gate_passed"] or any(
            evidence["coverage"] != 1 for evidence in summary["held_out"].values()):
        raise RuntimeError("regional timing validation gate failed")
    for region in REGIONS:
        for key in ("replay_compute_completion_factor", "kv_residual_s",
                    "kv_ingest_lower_bound_bytes_per_s"):
            if not np.isclose(fits[region][key], expected[region][key]):
                raise RuntimeError("central timing refit does not reproduce summary")
        for label, value in expected[region]["effective_pipeline_mbps"].items():
            if not np.isclose(fits[region]["effective_pipeline_mbps"][label], value):
                raise RuntimeError("central route refit does not reproduce summary")
    return fits


def ordered_timing_fit(profile, parent, rows, rng):
    fits = timing_fit(
        profile, parent, stratified_timing_bootstrap(rows, rng), str(TIMING),
    )
    projected = 0
    for region in REGIONS:
        rates = fits[region]["effective_pipeline_mbps"]
        if rates["controlled_40"] > rates["natural"]:
            rates["controlled_40"] = rates["natural"]
            projected += 1
    return fits, projected


def sample_draw(profile, templates, timing_rows, parent, rng, replicate, seed,
                sessions):
    pack = normalize_pack(profile, sample_pack(
        templates, sessions, seed + replicate,
    ))
    fits, projected_regions = ordered_timing_fit(
        profile, parent, timing_rows, rng,
    )
    timing_hash = hashlib.sha256(
        json.dumps(fits, sort_keys=True).encode()
    ).hexdigest()
    phase = profile.case().phase_power
    if phase is None or not phase.bootstrap:
        raise ValueError("workload adaptation requires phase-power bootstrap draws")
    power_index = int(rng.integers(len(phase.bootstrap)))
    sampled_profile = state_profile(profile, {
        "power_bootstrap_index": power_index, "service_multiplier": 1,
        "replay_multiplier": 1, "kv_multiplier": 1,
    })
    return (sampled_profile, pack, fits, power_index, timing_hash,
            projected_regions)


def build_problem(profile, sessions, constraints, target_fraction, fits, *,
                  bandwidth_mbps=None, prefill_tps=None):
    values, case = state_values(constraints), profile.case()
    constrained = values["bandwidth"] == "bottleneck_1g"
    timing_condition = "controlled_40" if constrained else "natural"
    physical = physical_route_mbps()
    if bandwidth_mbps is None:
        physical_bandwidths = {region: (
            BANDWIDTH_BOTTLENECK_MBPS if constrained else physical[region]
        ) * 125_000 for region in REGIONS}
        pipeline_bandwidths = {region: fits[region]["effective_pipeline_mbps"][
            timing_condition] * 125_000 for region in REGIONS}
    else:
        if constraints:
            raise ValueError("absolute bandwidth with factorial constraints")
        if bandwidth_mbps <= 0:
            raise ValueError("bandwidth must be positive")
        physical_bandwidths, pipeline_bandwidths = {}, {}
        for region in REGIONS:
            cap = min(bandwidth_mbps, physical[region])
            fraction = np.clip(
                (cap - BANDWIDTH_BOTTLENECK_MBPS) / (
                    physical[region] - BANDWIDTH_BOTTLENECK_MBPS), 0, 1)
            rates = fits[region]["effective_pipeline_mbps"]
            physical_bandwidths[region] = cap * 125_000
            pipeline_bandwidths[region] = (
                rates["controlled_40"]
                + fraction * (rates["natural"] - rates["controlled_40"])
            ) * 125_000
    paths = {region: (f"link/{region}", f"pipeline/{region}")
             for region in REGIONS}
    scenario = ExecutionScenario(
        41, 41, 0, "awake", 0,
        (PowerNode("source-node", 1, True), *(PowerNode(
            f"{region}-node", 1, False) for region in REGIONS)),
        (ServingInstance("source", ("source-node",)), *(ServingInstance(
            region, (f"{region}-node",)) for region in REGIONS)),
        sessions, tuple(link for region in REGIONS for link in (
            NetworkLink(f"link/{region}", physical_bandwidths[region]),
            NetworkLink(f"pipeline/{region}", pipeline_bandwidths[region]),
        )),
    )
    phase = case.phase_power
    if phase is None or not phase.contains(
        sum(session.expected_f for session in sessions),
        sum(session.expected_g for session in sessions),
    ):
        raise ValueError("sampled source pack is outside phase-power support")
    initial = source_power(scenario, profile)
    minimum = source_power(
        scenario, profile, (session.session_id for session in sessions),
    )
    target = target_fraction * (initial - minimum)
    scenario = replace(scenario, power_limit_w=initial - target)

    architecture = dedicated_sink_architecture(
        profile, REGIONS[0], (f"link/{REGIONS[0]}",),
    )
    q = architecture.types[0]
    if prefill_tps is not None:
        rate = q.prefill.at(PREFILL_REFERENCE_TOKENS)
        values["dest_compute"] = max(0, 1 - prefill_tps / rate)
    demand = sum((q.work(
        session.expected_f, session.expected_g, session.context_tokens, True,
    ) for session in sessions), start=np.zeros(2))
    direction = demand / max(
        np.asarray(q.normals) @ demand / np.asarray(q.bounds["normal"])
    )
    source_action = {method: case.action_power_w[method].power(1, True)
                     for method in ("replay", "kv_transfer")}
    sink_action = {method: case.action_power_w[method].power(1, False)
                   for method in source_action}
    load = loaded_service_model()
    if BANDWIDTH_BOTTLENECK_MBPS != load["validation_bandwidth_mbps"][0]:
        raise ValueError("bandwidth bottleneck must match measured validation support")
    types, pools = [], []
    for region in REGIONS:
        raw = fits[region]["migration_components"]
        path_bandwidth = min(
            physical_bandwidths[region], pipeline_bandwidths[region],
        )
        migration = {}
        for method, value in raw.items():
            bandwidth_range = tuple(value["bandwidth_range_bytes_per_s"])
            provenance = value["provenance"]
            if path_bandwidth < bandwidth_range[0]:
                bandwidth_range = (path_bandwidth, bandwidth_range[1])
                provenance += f"; {LOADED_SERVICE.relative_to(ROOT)} 1-Gbit/s validation"
            migration[method] = MigrationComponents(
                tuple(value["context_range"]), bandwidth_range, provenance,
                value.get("compute_completion_factor", 1),
                value.get("residual_s", 0),
                value.get("kv_ingest_bytes_per_s"),
            )
        loaded = {method: LoadedCoefficients(
            tuple(load["rho_grid"]), tuple(load["slowdown"][method]),
            migration[method].context_range,
            migration[method].bandwidth_range_bytes_per_s,
            f"{LOADED_SERVICE.relative_to(ROOT)}; normalized A100 load sensitivity",
        ) for method in ("replay", "kv_transfer")}
        dtype = replace(
            q, type_id=f"{q.type_id}/{region}", migration=migration,
            loaded=loaded,
        )
        service = FluidMigrationService(
            1 / fits[region]["replay_compute_completion_factor"],
            source_action, sink_action,
            "regional-c1 timing; pipelined route and shared endpoint work",
            1, True,
        )
        resident = math.floor(
            values["hbm"] * q.kv_capacity_tokens / q.kv_block_tokens
        ) * q.kv_block_tokens
        types.append(dtype)
        pools.append(DestinationPool(
            f"pool/{region}", dtype.type_id,
            (DestinationReplica(
                region, tuple(values["dest_compute"] * direction), resident,
            ),), f"route/{region}", paths[region],
            fluid_migration=service,
        ))
    architecture = DestinationArchitecture(
        architecture.schema, architecture.source_compatibility,
        tuple(types), tuple(pools),
    )
    routes = {("source", region): paths[region] for region in REGIONS}
    return scenario, architecture, routes, target


def run_case(profile, sessions, case_id, label, constraints, replicate,
             target_fraction, fits, power_index, timing_hash, projected_regions,
             *, bandwidth_mbps=None, prefill_tps=None, return_result=False):
    scenario, architecture, routes, target = build_problem(
        profile, sessions, constraints, target_fraction, fits,
        bandwidth_mbps=bandwidth_mbps, prefill_tps=prefill_tps,
    )
    result = plan(
        scenario, profile, routes, "lp_highs", seed=replicate,
        destination=architecture, admission_mode="normal",
    )
    power = ExpectedPower(scenario, profile)
    table = candidate_table(scenario, profile, architecture, "normal", power)
    fractional_opportunity = fractional_power_opportunity(
        table, power,
    )
    dominance = method_dominance(table, len(architecture.pools))
    counts = {method: sum(move.method == method for move in result.moves)
              for method in ("replay", "kv_transfer")}
    counts["not_moved"] = len(sessions) - len(result.moves)
    if min(counts.values()) < 0 or sum(counts.values()) != len(sessions):
        raise RuntimeError("action counts do not conserve sessions")
    phase_shares = phase_load_shares(sessions, result.moves, power)
    usage = {row.name: row.utilization for row in result.resource_uses}
    def max_usage(prefix):
        return max((value for name, value in usage.items()
                    if name.startswith(prefix)), default=0.0)
    moved = {move.session_id for move in result.moves}
    remaining = [session for session in sessions if session.session_id not in moved]
    phase = profile.case().phase_power
    if phase is None or not phase.contains(
        sum(session.expected_f for session in remaining),
        sum(session.expected_g for session in remaining),
    ):
        raise RuntimeError("planned source state is outside phase-power support")
    row = {
        "case_id": case_id, "bound_constraint": label,
        "constraints": "+".join(sorted(constraints)) or "none",
        "replicate": replicate, "sessions": len(sessions), "target_w": target,
        "power_bootstrap_index": power_index,
        "timing_fit_sha256": timing_hash,
        "bandwidth_projection_regions": projected_regions,
        "fractional_lp_opportunity_w": fractional_opportunity,
        "target_met": result.feasible and result.power_shortfall_w <= POWER_TOLERANCE_W,
        "feasible": result.feasible, "power_shortfall_w": result.power_shortfall_w,
        "planned_shed_w": result.initial_source_power_w - result.planned_source_power_w,
        "predicted_migration_makespan_s": result.predicted_migration_makespan_s,
        **loaded_transport_counts(scenario, architecture, table, result.moves),
        **dominance,
        **{f"{action}_count": counts[action] for action in ACTIONS},
        **{action: counts[action] / len(sessions) for action in ACTIONS},
        **{f"{action}_phase_load": phase_shares[action] for action in ACTIONS},
        "route_utilization": max_usage("route:"),
        "service_utilization": max_usage("service:"),
        "hbm_utilization": max_usage("kv:"),
        "migration_utilization": max_usage("migration:"),
        "binding_resources": ";".join(result.binding_resources),
        "failure": result.failure_reason or "",
    }
    return (row, result) if return_result else row


def phase_load_shares(sessions, moves, power):
    selected = {move.session_id: move.method for move in moves}
    if len(selected) != len(moves) or not set(selected) <= {
            session.session_id for session in sessions}:
        raise RuntimeError("invalid selected-session action accounting")
    total = sum(power.ell[session.session_id] for session in sessions)
    if total <= 0:
        raise RuntimeError("action mix requires positive source phase load")
    shares = {
        method: sum(power.ell[session_id] for session_id, action in selected.items()
                    if action == method) / total
        for method in ("replay", "kv_transfer")
    }
    shares["not_moved"] = 1 - sum(shares.values())
    if min(shares.values()) < -1e-12 or not np.isclose(sum(shares.values()), 1):
        raise RuntimeError("action phase-load shares do not conserve source load")
    return shares


def loaded_transport_counts(scenario, architecture, table, moves):
    """Count uses outside the fixed-pack load-factor evidence."""
    low, high = loaded_service_model()["fit_context_tokens"]
    min_bw, max_bw = (value * 125_000 for value in
                      loaded_service_model()["validation_bandwidth_mbps"])
    links = {link.link_id: link.bytes_per_s for link in scenario.links}
    sessions = {session.session_id: session for session in table.sessions}

    def bandwidth(path):
        return min(links[link] for link in path)

    def context_counts(items):
        values = [item.context_tokens for item in items]
        return sum(value < low for value in values), sum(value > high for value in values)

    def bandwidth_counts(paths):
        values = [bandwidth(path) for path in paths]
        return sum(value < min_bw for value in values), sum(value > max_bw for value in values)

    candidate_sessions = [table.sessions[item.session] for item in table.candidates]
    selected_sessions = [sessions[move.session_id] for move in moves]
    result = {
        "loaded_candidate_count": len(table.candidates),
        "loaded_pool_count": len(architecture.pools),
        "loaded_selected_count": len(moves),
    }
    for scope, items in (("session", table.sessions),
                         ("candidate", candidate_sessions),
                         ("selected", selected_sessions)):
        below, above = context_counts(items)
        result[f"loaded_context_below_{scope}_count"] = below
        result[f"loaded_context_above_{scope}_count"] = above
    for scope, paths in (("pool", [pool.route for pool in architecture.pools]),
                         ("candidate", [item.path for item in table.candidates]),
                         ("selected", [move.path for move in moves])):
        below, above = bandwidth_counts(paths)
        result[f"loaded_bandwidth_below_{scope}_count"] = below
        result[f"loaded_bandwidth_above_{scope}_count"] = above
    for method in ("replay", "kv_transfer"):
        selected = [move for move in moves if move.method == method]
        below_context, above_context = context_counts(
            [sessions[move.session_id] for move in selected])
        below_bandwidth, above_bandwidth = bandwidth_counts(
            [move.path for move in selected])
        result.update({
            f"loaded_context_below_{method}_selected_count": below_context,
            f"loaded_context_above_{method}_selected_count": above_context,
            f"loaded_bandwidth_below_{method}_selected_count": below_bandwidth,
            f"loaded_bandwidth_above_{method}_selected_count": above_bandwidth,
        })
    return result


def method_dominance(table, pools):
    """Classify eligible Replay/KV choices by gain, work, and resources."""
    choices = {(candidate.session, candidate.pool, candidate.method): i
               for i, candidate in enumerate(table.candidates)}
    if len(choices) != len(table.candidates):
        raise RuntimeError("duplicate migration candidate")
    columns = table.resources.toarray()
    result = {name: 0 for name in (
        "candidate_matched_pairs", "candidate_replay_only", "candidate_kv_only",
        "candidate_neither", "candidate_replay_dominates",
        "candidate_kv_dominates", "candidate_equivalent",
        "candidate_incomparable",
    )}

    def dominates(first, second):
        a, b = table.candidates[first], table.candidates[second]
        return a.gain_w + 1e-9 >= b.gain_w \
            and a.objective_cost_s <= b.objective_cost_s + 1e-9 \
            and np.all(columns[:, first] <= columns[:, second] + 1e-9)

    for session, pool in product(range(len(table.sessions)), range(pools)):
        replay = choices.get((session, pool, "replay"))
        kv = choices.get((session, pool, "kv_transfer"))
        if replay is None or kv is None:
            name = "candidate_neither" if replay is None and kv is None else \
                "candidate_replay_only" if replay is not None else "candidate_kv_only"
            result[name] += 1
            continue
        result["candidate_matched_pairs"] += 1
        replay_wins, kv_wins = dominates(replay, kv), dominates(kv, replay)
        name = "candidate_equivalent" if replay_wins and kv_wins else \
            "candidate_replay_dominates" if replay_wins else \
            "candidate_kv_dominates" if kv_wins else "candidate_incomparable"
        result[name] += 1
    return result


def simulate(samples=1000, seed=DEFAULT_SEED, sessions=28, target_fraction=2 / 3,
             profile_path=PROFILE, manifest_path=MANIFEST,
             loaded_context_only=False):
    if samples < 1 or sessions < 1 or not 0 < target_fraction <= 1:
        raise ValueError("invalid workload-adaptation simulation controls")
    profile = ModelProfile.load(profile_path)
    templates, workload = load_templates(manifest_path, profile)
    low, high = loaded_service_model()["fit_context_tokens"]
    original_states = sum(map(len, templates.values()))
    fit_templates = {key: selected for key, values in templates.items()
                     if (selected := tuple(value for value in values
                                           if low <= value.context_tokens <= high))}
    if loaded_context_only:
        templates = fit_templates
        if not templates:
            raise ValueError("no workload templates inside loaded-factor context support")
    workload.update({
        "loaded_factor_context_only": loaded_context_only,
        "loaded_factor_context_tokens": [low, high],
        "loaded_factor_in_context_templates": len(fit_templates),
        "loaded_factor_in_context_states": sum(map(len, fit_templates.values())),
        "loaded_factor_outside_context_states":
            original_states - sum(map(len, fit_templates.values())),
    })
    timing_rows, parent = read_csv(TIMING), json.loads(TIMING_PARENT.read_text())
    central_timing_fits()
    rng, rows = np.random.default_rng(seed), []
    for replicate in range(samples):
        sampled_profile, pack, fits, power_index, timing_hash, projected_regions = \
            sample_draw(
                profile, templates, timing_rows, parent, rng, replicate, seed,
                sessions,
            )
        paired_target = None
        for case_id, label, constraints in factorial_cases():
            row = run_case(
                sampled_profile, pack, case_id, label, constraints, replicate,
                target_fraction, fits, power_index, timing_hash, projected_regions,
            )
            paired_target = row["target_w"] if paired_target is None else paired_target
            if not np.isclose(row["target_w"], paired_target):
                raise RuntimeError("factorial states do not share one paired target")
            rows.append(row)
    if len(rows) != samples * 8:
        raise RuntimeError("factorial simulation is incomplete")
    if any(len({row["timing_fit_sha256"] for row in rows
                if row["replicate"] == replicate}) != 1
           for replicate in range(samples)):
        raise RuntimeError("factorial states do not share one timing draw")
    return rows, workload


def oat_design(profile, levels=20):
    if levels < 2:
        raise ValueError("OAT sweep requires at least two levels")
    q = dedicated_sink_architecture(profile, REGIONS[0], ("link/east",)).types[0]
    rate = q.prefill.at(PREFILL_REFERENCE_TOKENS)
    bandwidths = np.linspace(
        OAT_BANDWIDTH_LOWER_MBPS, OAT_BANDWIDTH_UPPER_MBPS, levels)
    prefills = np.linspace(
        (1 - OAT_DEST_COMPUTE[1]) * rate, OAT_PREFILL_UPPER_TPS, levels)
    return bandwidths, prefills, float(np.median(bandwidths)), \
        float(np.median(prefills)), rate


def simulate_oat(samples=1000, seed=DEFAULT_SEED, sessions=28,
                 target_fraction=1.0, levels=20, profile_path=PROFILE,
                 manifest_path=MANIFEST):
    if samples < levels or samples % levels:
        raise ValueError("OAT samples must be a multiple of levels")
    profile = ModelProfile.load(profile_path)
    templates, _ = load_templates(manifest_path, profile)
    timing_rows, parent = read_csv(TIMING), json.loads(TIMING_PARENT.read_text())
    central_timing_fits()
    bandwidths, prefills, fixed_bandwidth, fixed_prefill, rate = \
        oat_design(profile, levels)
    rng, cells = np.random.default_rng(seed), {}
    for repeat in range(samples // levels):
        sampled_profile, pack, fits, power_index, timing_hash, projected = \
            sample_draw(profile, templates, timing_rows, parent, rng, repeat,
                        seed, sessions)
        for sweep, values in (("bandwidth", bandwidths), ("prefill", prefills)):
            for level, value in enumerate(values):
                bandwidth = value if sweep == "bandwidth" else fixed_bandwidth
                prefill = fixed_prefill if sweep == "bandwidth" else value
                row = run_case(
                    sampled_profile, pack, f"oat_{sweep}", f"OAT {sweep}",
                    frozenset(), repeat, target_fraction, fits, power_index,
                    timing_hash, projected, bandwidth_mbps=bandwidth,
                    prefill_tps=prefill,
                )
                cells.setdefault((sweep, level), []).append(row)
    rows = []
    for sweep, values in (("bandwidth", bandwidths), ("prefill", prefills)):
        for level, value in enumerate(values):
            selected = cells[sweep, level]
            total = len(selected) * sessions
            common = {
                "sweep": sweep, "level": level,
                "bandwidth_cap_gbps": (value if sweep == "bandwidth"
                                        else fixed_bandwidth) / 1000,
                "prefill_available_tps": (fixed_prefill if sweep == "bandwidth"
                                           else value),
                "plans": len(selected),
                "target_met_rate": np.mean([row["target_met"] for row in selected]),
            }
            rows.extend({
                **common, "action": action,
                "session_count": int(sum(row[f"{action}_count"]
                                         for row in selected)),
                "session_share": sum(row[f"{action}_count"]
                                     for row in selected) / total,
            } for action in ACTIONS)
    if any(not np.isclose(sum(row["session_share"] for row in rows
                              if row["sweep"] == sweep and row["level"] == level), 1)
           for sweep in ("bandwidth", "prefill") for level in range(levels)):
        raise RuntimeError("OAT action shares do not conserve sessions")
    return rows, {
        "samples_per_sweep": samples, "paired_draws": samples // levels,
        "levels": levels, "sessions_per_pack": sessions,
        "prefill_reference_tokens": PREFILL_REFERENCE_TOKENS,
        "prefill_full_rate_tps": rate,
        "bandwidth_cap_gbps": [bandwidths[0] / 1000, bandwidths[-1] / 1000],
        "fixed_bandwidth_cap_gbps": fixed_bandwidth / 1000,
        "prefill_available_tps": [prefills[0], prefills[-1]],
        "fixed_prefill_available_tps": fixed_prefill,
        "action_metric": "source-session share at each OAT level",
        "pipeline_interpolation":
            "regional controlled-40 to natural endpoint interpolation",
    }


def transport_summary(rows):
    first_case = factorial_cases()[0][0]
    pack_rows = [row for row in rows if row["case_id"] == first_case]

    def summarize_scope(selected, denominator, prefix, scope):
        total = sum(row[denominator] for row in selected)
        return {
            "denominator": total,
            **{side: sum(row[f"loaded_{prefix}_{side}_{scope}_count"]
                         for row in selected)
               for side in ("below", "above")},
        }

    context = {
        "sessions": summarize_scope(pack_rows, "sessions", "context", "session"),
        "candidates": summarize_scope(rows, "loaded_candidate_count", "context",
                                      "candidate"),
        "selected": summarize_scope(rows, "loaded_selected_count", "context",
                                    "selected"),
    }
    bandwidth = {
        "pools": summarize_scope(rows, "loaded_pool_count", "bandwidth", "pool"),
        "candidates": summarize_scope(rows, "loaded_candidate_count", "bandwidth",
                                      "candidate"),
        "selected": summarize_scope(rows, "loaded_selected_count", "bandwidth",
                                    "selected"),
    }
    for values in (*context.values(), *bandwidth.values()):
        values["outside"] = values["below"] + values["above"]
        values["outside_rate"] = values["outside"] / values["denominator"] \
            if values["denominator"] else 0.0
    selected_by_method = {}
    for method in ("replay", "kv_transfer"):
        denominator = sum(row[f"{method}_count"] for row in rows)
        selected_by_method[method] = {
            "denominator": denominator,
            "context_outside": sum(
                row[f"loaded_context_{side}_{method}_selected_count"]
                for row in rows for side in ("below", "above")),
            "bandwidth_outside": sum(
                row[f"loaded_bandwidth_{side}_{method}_selected_count"]
                for row in rows for side in ("below", "above")),
        }
        for axis in ("context", "bandwidth"):
            selected_by_method[method][f"{axis}_outside_rate"] = \
                selected_by_method[method][f"{axis}_outside"] / denominator \
                if denominator else 0.0
    return {"context": context, "bandwidth": bandwidth,
            "selected_by_method": selected_by_method}


def factor_checks(rows):
    by_case = {constraints: case_id for case_id, _, constraints in factorial_cases()}
    utilization = {"hbm": "hbm_utilization", "bandwidth": "route_utilization",
                   "dest_compute": "service_utilization"}
    checks = []
    for replicate in sorted({row["replicate"] for row in rows}):
        selected = {row["case_id"]: row for row in rows
                    if row["replicate"] == replicate}
        for case_id, label, constraints in factorial_cases():
            constrained = selected[case_id]
            for factor in sorted(constraints):
                released = selected[by_case[constraints - {factor}]]
                actions = tuple(f"{action}_count" for action in ACTIONS)
                changed = any(constrained[name] != released[name] for name in actions)
                improved = released["power_shortfall_w"] + POWER_TOLERANCE_W \
                    < constrained["power_shortfall_w"]
                shortfall_change = released["power_shortfall_w"] \
                    - constrained["power_shortfall_w"]
                opportunity_change = released["fractional_lp_opportunity_w"] \
                    - constrained["fractional_lp_opportunity_w"]
                pressure = constrained[utilization[factor]]
                checks.append({
                    "replicate": replicate, "case_id": case_id,
                    "bound_constraint": label, "factor": factor,
                    "utilization": pressure, "action_changed_on_release": changed,
                    "shortfall_improved_on_release": improved,
                    "planner_shortfall_change_w": shortfall_change,
                    "planner_shortfall_worsened_on_release":
                        shortfall_change > POWER_TOLERANCE_W,
                    "target_met_lost_on_release":
                        constrained["target_met"] and not released["target_met"],
                    "fractional_lp_opportunity_change_w": opportunity_change,
                    "fractional_opportunity_worsened_on_release":
                        opportunity_change < -POWER_TOLERANCE_W,
                    "resource_near_capacity": pressure >= .9,
                    "opportunity_reduced":
                        opportunity_change > POWER_TOLERANCE_W,
                    "active": pressure >= .9 \
                        or opportunity_change > POWER_TOLERANCE_W,
                })
    return checks


def summarize(rows):
    result = []
    for case_id, label, _ in factorial_cases():
        selected = [row for row in rows if row["case_id"] == case_id]
        for action in ACTIONS:
            values = np.asarray([row[f"{action}_phase_load"] for row in selected])
            q = np.quantile(values, (.05, .5, .95))
            result.append({
                "case_id": case_id, "bound_constraint": label, "action": action,
                "mean": values.mean(), "p05": q[0], "median": q[1], "p95": q[2],
                "session_share_mean": np.mean([row[action] for row in selected]),
                "target_met_rate": np.mean([row["target_met"] for row in selected]),
            })
    return result


def _surface_row(source, split, scenario, replay_s, kv_s, route_s, coupling):
    predicted = max(
        route_s,
        replay_s + coupling * kv_s,
        coupling * replay_s + kv_s,
    )
    measured = float(scenario.migration_s)
    return {
        "source": source, "split": split, "scenario_id": scenario.scenario_id,
        "session_set": scenario.session_set, "horizon_s": MIGRATION_HORIZON_S,
        "method": scenario.method, "bandwidth_mbps": scenario.bandwidth_mbps,
        "concurrency": int(scenario.concurrency), "replay_work_s": replay_s,
        "kv_work_s": kv_s, "route_s": route_s, "predicted_s": predicted,
        "measured_s": measured, "predicted_over_measured": predicted / measured,
        "false_feasible": predicted <= MIGRATION_HORIZON_S < measured,
        "false_infeasible": measured <= MIGRATION_HORIZON_S < predicted,
    }


def validate_surface():
    rows = []
    profile = ModelProfile.load(BASE_PROFILE).case()
    scenarios = pd.read_csv(LOCAL_TIMING / "scenarios.csv")
    migrations = pd.read_csv(LOCAL_TIMING / "migrations.csv")
    selected = scenarios[(scenarios.kind == "migration")
                         & (scenarios.status == "complete")
                         & (scenarios.activity == "none")]
    for scenario in selected.itertuples():
        group = migrations[(migrations.scenario_id == scenario.scenario_id)
                           & migrations.success]
        replay = kv = 0.0
        if scenario.method == "replay":
            replay = sum(
                row.measured_prompt_tokens / profile.replay.rate(
                    row.measured_prompt_tokens, 1,
                ) + profile.replay_completion_s for row in group.itertuples()
            )
            route_bytes = 2 * group.measured_prompt_tokens.sum()
        else:
            kv = sum(
                row.measured_kv_bytes / profile.kv_transfer.destination_bytes_per_s
                + profile.kv_transfer.initial_completion_s
                for row in group.itertuples()
            )
            route_bytes = group.measured_kv_bytes.sum()
        digest = int(hashlib.sha256(scenario.session_set.encode()).hexdigest(), 16)
        split = "grouped-audit" if digest % 3 == 0 else "development"
        rows.append(_surface_row(
            "coding-c1-c4", split, scenario, replay, kv,
            route_bytes / (scenario.bandwidth_mbps * 125_000), 1,
        ))

    profile = ModelProfile.load(WIDTH8_PROFILE).case()
    scenarios = pd.read_csv(WIDTH8_TIMING / "scenarios.csv")
    stages = pd.read_csv(WIDTH8_TIMING / "migration_stages.csv")
    selected = scenarios[(scenarios.kind == "migration")
                         & (scenarios.status == "complete")]
    for scenario in selected.itertuples():
        group = stages[(stages.scenario_id == scenario.scenario_id)
                       & stages.success & (stages.phase == "initial")]
        if len(group) != 8:
            raise RuntimeError("width-8 validation requires eight successful stages")
        replay = kv = 0.0
        for row in group.itertuples():
            if row.method == "replay":
                replay += row.measured_prompt_tokens / profile.replay.rate(
                    row.measured_prompt_tokens, 1,
                ) + profile.replay_completion_s
            else:
                kv += row.logical_body_bytes \
                    / profile.kv_transfer.destination_bytes_per_s \
                    + profile.kv_transfer.initial_completion_s
        digest = int(hashlib.sha256(scenario.session_set.encode()).hexdigest(), 16)
        split = "grouped-audit" if digest % 3 == 0 else "development"
        rows.append(_surface_row(
            "width8", split, scenario, replay, kv,
            group.wire_bytes.sum() / (scenario.bandwidth_mbps * 125_000), 1,
        ))
    if any(row["predicted_s"] <= 0 or row["measured_s"] <= 0 for row in rows):
        raise RuntimeError("timing validation contains a nonpositive span")
    if any(row["source"] == "width8" and row["false_feasible"] for row in rows):
        raise RuntimeError("pipeline model is false-feasible on width-8 evidence")
    return rows


def validation_summary(rows):
    output = {}
    groups = {(row["source"], row["split"]) for row in rows} | {
        (row["source"], row["split"], row["method"]) for row in rows
    }
    for key in sorted(groups):
        selected = [row for row in rows if tuple(row[name] for name in (
            "source", "split", "method")[:len(key)]) == key]
        ratio = np.asarray([row["predicted_over_measured"] for row in selected])
        output["/".join(key)] = {
            "scenarios": len(selected), "median_predicted_over_measured": float(np.median(ratio)),
            "p90_absolute_relative_error": float(np.quantile(abs(ratio - 1), .9)),
            "underprediction_rate": float(np.mean(ratio < 1)),
            "false_feasible_at_25s": int(sum(row["false_feasible"]
                                              for row in selected)),
            "false_infeasible_at_25s": int(sum(row["false_infeasible"]
                                                for row in selected)),
        }
    return output


def surface_scope_limitation(rows):
    local = [row for row in rows if row["source"] == "coding-c1-c4"]
    grouped = [row for row in local if row["split"] == "grouped-audit"]
    width8 = [row for row in rows if row["source"] == "width8"]
    return (
        "route overlap is a regional modeled sensitivity, not a generic "
        f"deadline guarantee: {sum(row['false_feasible'] for row in local)}/"
        f"{len(local)} local c1-c4 cases and "
        f"{sum(row['false_feasible'] for row in grouped)}/{len(grouped)} grouped "
        f"audit cases are false-feasible at 25 s, versus "
        f"{sum(row['false_feasible'] for row in width8)}/{len(width8)} width-8 cases"
    )


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def action_mix_means(rows):
    groups = [[row for row in rows if row["case_id"] == case_id]
              for case_id in DISPLAY_CASES]
    if any(not group for group in groups):
        raise RuntimeError("action plot is missing a displayed constraint case")
    values = np.asarray([[np.mean([row[f"{action}_phase_load"] for row in group])
                          for group in groups] for action in ACTIONS])
    if not np.allclose(values.sum(0), 1):
        raise RuntimeError("mean action mix does not conserve phase load")
    return values


def plot(rows, path):
    fig, axis = plt.subplots(figsize=(3.85, 2.5))
    labels = {case_id: label for case_id, label, _ in factorial_cases()}
    values_by_action = action_mix_means(rows)
    left = np.zeros(len(DISPLAY_CASES))
    for action, fractions in zip(ACTIONS, values_by_action):
        values = fractions * 100
        axis.barh(
            range(len(DISPLAY_CASES)), values, left=left,
            color=plot_style.ACTION_COLORS[action],
            hatch=plot_style.ACTION_HATCHES[action], edgecolor="white", linewidth=1.2,
            label=plot_style.ACTION_NAMES[action],
        )
        left += values
    axis.set(
        yticks=range(len(DISPLAY_CASES)),
        yticklabels=[labels[case_id] for case_id in DISPLAY_CASES],
        xlim=(0, 100), xlabel="Modeled source phase-load share (%)",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.tick_params(labelsize=10)
    axis.xaxis.label.set_size(10)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(.58, .01), fontsize=9, handlelength=1.8,
    )
    fig.subplots_adjust(left=.34, right=.95, bottom=.36, top=.98)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def action_boxplot_statistics(rows):
    output = []
    labels = {case_id: label for case_id, label, _ in factorial_cases()}
    for case_id in DISPLAY_CASES:
        selected = [row for row in rows if row["case_id"] == case_id]
        if not selected:
            raise RuntimeError(f"boxplot case has no draws: {case_id}")
        for action in ACTIONS:
            values = np.asarray([
                100 * row[f"{action}_count"] / row["sessions"]
                for row in selected
            ])
            if not np.isfinite(values).all():
                raise RuntimeError("boxplot session share is not finite")
            quantiles = np.quantile(values, ACTION_BOXPLOT_QUANTILES)
            output.append({
                "case_id": case_id, "bound_constraint": labels[case_id],
                "action": action,
                **{name: float(value) for name, value in zip(
                    ("p05", "p25", "median", "p75", "p95"), quantiles,
                )},
            })
    return output


def plot_action_boxplot(rows, path):
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    statistics = action_boxplot_statistics(rows)
    positions = np.arange(len(DISPLAY_CASES))
    fig, axis = plt.subplots(figsize=(5.5, 3))
    for action, offset in zip(ACTIONS, (-.24, 0, .24)):
        selected = [row for row in statistics if row["action"] == action]
        artists = axis.bxp([
            {
                "label": row["bound_constraint"], "whislo": row["p05"],
                "q1": row["p25"], "med": row["median"], "q3": row["p75"],
                "whishi": row["p95"], "fliers": (),
            } for row in selected
        ], positions=positions + offset, widths=.2, patch_artist=True,
           showfliers=False, manage_ticks=False,
           boxprops={"facecolor": plot_style.ACTION_COLORS[action],
                     "edgecolor": "white", "linewidth": 1.2,
                     "hatch": plot_style.ACTION_HATCHES[action]},
           medianprops={"color": "#222222", "linewidth": 1.3},
           whiskerprops={"color": "#444444", "linewidth": 1},
           capprops={"color": "#444444", "linewidth": 1})
        if len(artists["boxes"]) != len(DISPLAY_CASES):
            raise RuntimeError("boxplot omitted a constraint case")
    labels = {case_id: label for case_id, label, _ in factorial_cases()}
    tick_labels = [labels[case].replace("Dest. compute", "Dest.\ncompute")
                   .replace("All bound", "All\nbound")
                   .replace("None bound", "None\nbound")
                   for case in DISPLAY_CASES]
    axis.set(
        xticks=positions, xticklabels=tick_labels, xlim=(-.55, 4.55),
        ylim=(0, 100), yticks=(0, 25, 50, 75, 100), ylabel="Sessions (%)",
    )
    axis.yaxis.set_major_formatter(PercentFormatter(100))
    axis.grid(axis="y", alpha=.2)
    axis.tick_params(labelsize=10)
    axis.yaxis.label.set_size(12)
    axis.text(
        1, 1.02, "Box: 25-75%; whiskers: 5-95%", transform=axis.transAxes,
        ha="right", va="bottom", fontsize=9,
    )
    fig.legend(
        [Patch(facecolor=plot_style.ACTION_COLORS[action], edgecolor="white",
               hatch=plot_style.ACTION_HATCHES[action]) for action in ACTIONS],
        [plot_style.ACTION_NAMES[action] for action in ACTIONS],
        frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.56, .01),
        fontsize=10, handlelength=1.8,
    )
    fig.subplots_adjust(left=.15, right=.98, bottom=.34, top=.88)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def plot_oat(rows, path):
    from matplotlib.ticker import PercentFormatter

    fs = plot_style.HALF_COLUMN_FONT_SIZE
    legend_fs = plot_style.HALF_COLUMN_LEGEND_FONT_SIZE
    left, axes_w, axes_h, bottom, top = .47, .82, .92, .27, .06
    legend_w, right = .64, .09
    for sweep in ("bandwidth", "prefill"):
        wide = sweep == "prefill"
        fig_w = left + axes_w + (legend_w if wide else right)
        fig_h = bottom + axes_h + top
        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_axes((left / fig_w, bottom / fig_h,
                           axes_w / fig_w, axes_h / fig_h))
        selected = [row for row in rows if row["sweep"] == sweep]
        levels = sorted({row["level"] for row in selected})
        by_action = {action: {row["level"]: row for row in selected
                              if row["action"] == action} for action in ACTIONS}
        points = [by_action[ACTIONS[0]][level] for level in levels]
        x = [row["bandwidth_cap_gbps"] if sweep == "bandwidth"
             else row["prefill_available_tps"] / 1000 for row in points]
        artists = ax.stackplot(
            x, *([by_action[action][level]["session_share"] for level in levels]
                 for action in ACTIONS),
            colors=[plot_style.ACTION_COLORS[action] for action in ACTIONS],
            labels=[plot_style.ACTION_NAMES[action] for action in ACTIONS],
        )
        for artist in artists:
            artist.set_edgecolor("white")
            artist.set_linewidth(.3)
        ax.set(
            xlabel=("Bandwidth (Gbit/s)" if sweep == "bandwidth"
                    else "Prefill (k·token/s)"),
            xlim=(x[0], x[-1]), ylim=(0, 1),
        )
        ax.margins(0)
        ax.set_ylabel("Sessions (%)")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_locator(plt.MultipleLocator(.25))
        ax.xaxis.set_major_locator(plt.MultipleLocator(5))
        plot_style.half_column(ax)
        ax.xaxis.label.set_size(fs)
        ax.yaxis.label.set_size(fs)
        ax.xaxis.labelpad = ax.yaxis.labelpad = 2
        if wide:
            ax.legend(*ax.get_legend_handles_labels(), fontsize=legend_fs,
                      loc="center left", bbox_to_anchor=(1.02, .5),
                      handlelength=.9, handletextpad=.3, labelspacing=.4,
                      borderaxespad=0, borderpad=0, frameon=False)
        for suffix in ("png", "pdf"):
            fig.savefig(path.parent / f"action_choice_oat_{sweep}.{suffix}",
                        dpi=plot_style.SAVE_DPI)
        plt.close(fig)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sessions", type=int, default=28)
    parser.add_argument("--target", type=float, default=2 / 3)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    rows, workload = simulate(
        args.samples, args.seed, args.sessions, args.target,
    )
    in_context_rows, in_context_workload = simulate(
        args.samples, args.seed, args.sessions, args.target,
        loaded_context_only=True,
    )
    oat_rows, oat_design_metadata = simulate_oat(
        args.samples, args.seed, args.sessions, 1.0,
    )
    if any((row["replicate"], row["case_id"], row["timing_fit_sha256"],
            row["power_bootstrap_index"]) !=
           (robust["replicate"], robust["case_id"], robust["timing_fit_sha256"],
            robust["power_bootstrap_index"])
           for row, robust in zip(rows, in_context_rows)):
        raise RuntimeError("in-context robustness is not calibration-paired")
    summary, validation, checks = (summarize(rows), validate_surface(),
                                   factor_checks(rows))
    in_context_checks = factor_checks(in_context_rows)
    pairs = {(row["case_id"], row["factor"]) for row in checks}
    def rates(field):
        return {(case, factor): np.mean([
            row[field] for row in checks
            if row["case_id"] == case and row["factor"] == factor
        ]) for case, factor in pairs}
    activation, response = rates("active"), rates("action_changed_on_release")
    near_capacity, opportunity_reduced = (
        rates("resource_near_capacity"), rates("opportunity_reduced"),
    )
    singles = {case_id for case_id, _, constraints in factorial_cases()
               if len(constraints) == 1}
    if any(row["fractional_opportunity_worsened_on_release"] for row in checks):
        raise RuntimeError("constraint release reduced the fractional opportunity set")
    if any(row["fractional_opportunity_worsened_on_release"]
           for row in in_context_checks):
        raise RuntimeError("in-context constraint release reduced the opportunity set")
    if any(rate < .9 for (case, factor), rate in activation.items()
           if case in singles and factor in ACTIVATION_GATED_FACTORS):
        raise RuntimeError("labeled constraints failed activation gates")
    if any(rate < MIN_ACTION_RESPONSE_RATE
           for (case, _factor), rate in response.items() if case in singles):
        raise RuntimeError("single-factor intervention did not change enough plans")
    write_csv(args.out / "action_mix.csv", rows)
    write_csv(args.out / "action_mix_summary.csv", summary)
    write_csv(args.out / "action_mix_support_restricted.csv", in_context_rows)
    write_csv(args.out / "action_mix_support_restricted_summary.csv",
              summarize(in_context_rows))
    write_csv(args.out / "surface_validation.csv", validation)
    write_csv(args.out / "factor_checks.csv", checks)
    write_csv(args.out / "action_choice_oat.csv", oat_rows)
    plot(rows, args.out / "action_mix")
    plot_action_boxplot(rows, args.out / "action_mix_boxplot")
    plot_oat(oat_rows, args.out / "action_choice_oat")
    timing_evidence = json.loads(TIMING_SUMMARY.read_text())
    loaded_evidence = loaded_service_model()
    timing_loads = sorted({float(row["destination_prefill_load"])
                           for row in read_csv(TIMING)})
    selected = {action: int(sum(row[f"{action}_count"] for row in rows))
                for action in ACTIONS}
    dominance = {name: int(sum(row[name] for row in rows)) for name in (
        "candidate_matched_pairs", "candidate_replay_only", "candidate_kv_only",
        "candidate_neither", "candidate_replay_dominates",
        "candidate_kv_dominates", "candidate_equivalent",
        "candidate_incomparable",
    )}
    regressions = np.asarray([
        row["planner_shortfall_change_w"] for row in checks
        if row["planner_shortfall_worsened_on_release"]
    ])
    metadata = {
        "schema": "queue-haul-workload-adaptation-v8",
        "claim": "modeled regional phase-load-weighted action mix and predicted target-attainment sensitivity with exact nonlinear one-source power targets",
        "samples": args.samples, "sessions_per_pack": args.sessions,
        "seed": args.seed, "target_fraction": args.target,
        "source_load": SOURCE_LOAD,
        "oat_action_sweeps": oat_design_metadata,
        "source_load_definition": "sum(f/F + g/G); distinct from sampled phase load z=af+bg",
        "plotted_constraint_states": list(DISPLAY_CASES),
        "action_boxplot_cases": list(DISPLAY_CASES),
        "action_boxplot_metric": "per-draw percentage of source sessions assigned to each action",
        "stacked_action_metric": "mean modeled source phase-load share across paired draws",
        "action_boxplot_quantiles": list(ACTION_BOXPLOT_QUANTILES),
        "power_target": "invert sampled monotone phase power once and constrain additive removed phase load; verify exact nonlinear watts after packing",
        "power_scope": "steady awake source-region power; destination power excluded",
        "unique_timing_draws": len({row["timing_fit_sha256"] for row in rows}),
        "cross_method_coupling": 1,
        "bandwidth_projection_region_rate": float(np.mean([
            row["bandwidth_projection_regions"] / len(REGIONS)
            for row in rows[::len(ORDER)]
        ])),
        "route_compute_overlap": True,
        "route_endpoint_composition":
            "max(route time, fully shared Replay/KV endpoint work)",
        "planner_objective_cost":
            "sum of isolated candidate durations; endpoint replica-seconds remain a separate physical capacity row",
        "migration_horizon_s": MIGRATION_HORIZON_S,
        "regional_timing_destination_prefill_loads": timing_loads,
        "selected_session_totals": selected,
        "mean_phase_load_share": {
            action: float(np.mean([row[f"{action}_phase_load"] for row in rows]))
            for action in ACTIONS
        },
        "candidate_method_dominance": dominance,
        "candidate_method_dominance_definition": {
            "dominates": "weakly greater gain, weakly lower predicted duration, "
                         "and no greater normalized LP resource coefficient",
            "slot_partition": ["candidate_matched_pairs", "candidate_replay_only",
                               "candidate_kv_only", "candidate_neither"],
            "matched_pair_partition": ["candidate_replay_dominates",
                                       "candidate_kv_dominates",
                                       "candidate_equivalent", "candidate_incomparable"],
        },
        "factor_levels": LEVELS, "workload": workload,
        "bandwidth_bottleneck": {
            "physical_route_mbps": BANDWIDTH_BOTTLENECK_MBPS,
            "natural_physical_route_mbps": physical_route_mbps(),
            "pipeline_timing_condition": "controlled_40",
            "selection_basis": "lowest bandwidth in the existing 1-10-Gbit/s A100 loaded-migration validation",
        },
        "target_met_rate": {
            label: float(np.mean([row["target_met"] for row in rows
                                  if row["case_id"] == case_id]))
            for case_id, label, _ in factorial_cases()
        },
        "surface_validation": validation_summary(validation),
        "regional_timing_validation": {
            "migration_gate_passed": timing_evidence["migration_gate_passed"],
            "held_out": timing_evidence["held_out"],
        },
        "loaded_service_model": {
            "equation": loaded_evidence["equation"],
            "selected_commit_log_slope_per_rho": loaded_evidence[
                "selected_commit_log_slope_per_rho"],
            "slowdown_at_rho_0_95": loaded_evidence["slowdown_at_rho_0_95"],
            "fit_context_tokens": loaded_evidence["fit_context_tokens"],
            "training_bandwidth_mbps": loaded_evidence[
                "training_bandwidth_mbps"],
            "validation_bandwidth_mbps": loaded_evidence[
                "validation_bandwidth_mbps"],
            "bootstrap_samples": loaded_evidence["bootstrap_samples"],
            "bootstrap_seed": loaded_evidence["bootstrap_seed"],
            "width8_relative_factor_validation": loaded_evidence[
                "width8_relative_factor_validation"],
        },
        "loaded_factor_transport": transport_summary(rows),
        "support_restricted_workload_sensitivity": {
            "comparison": "2,048-14,336-token workload resample with paired timing and power draws; not a within-pack counterfactual",
            "workload": in_context_workload,
            "selected_session_totals": {
                action: int(sum(row[f"{action}_count"] for row in in_context_rows))
                for action in ACTIONS
            },
            "target_met_rate": {
                label: float(np.mean([row["target_met"] for row in in_context_rows
                                      if row["case_id"] == case_id]))
                for case_id, label, _ in factorial_cases()
            },
            "loaded_factor_transport": transport_summary(in_context_rows),
        },
        "factor_activation_rate": {f"{case}/{factor}": float(rate)
                                   for (case, factor), rate in sorted(activation.items())},
        "factor_action_response_rate": {f"{case}/{factor}": float(rate)
                                        for (case, factor), rate in sorted(response.items())},
        "factor_resource_near_capacity_rate": {
            f"{case}/{factor}": float(rate)
            for (case, factor), rate in sorted(near_capacity.items())
        },
        "factor_opportunity_reduction_rate": {
            f"{case}/{factor}": float(rate)
            for (case, factor), rate in sorted(opportunity_reduced.items())
        },
        "minimum_single_factor_action_response_rate": MIN_ACTION_RESPONSE_RATE,
        "activation_gated_factors": list(ACTIVATION_GATED_FACTORS),
        "planner_release_audit": {
            "comparisons": len(checks),
            "shortfall_regression_count": int(sum(
                row["planner_shortfall_worsened_on_release"] for row in checks
            )),
            "shortfall_regression_rate": float(np.mean([
                row["planner_shortfall_worsened_on_release"] for row in checks
            ])),
            "shortfall_regression_median_w": float(np.median(regressions))
                if regressions.size else 0,
            "shortfall_regression_p90_w": float(np.quantile(regressions, .9))
                if regressions.size else 0,
            "shortfall_regression_max_w": float(regressions.max())
                if regressions.size else 0,
            "target_loss_count": int(sum(
                row["target_met_lost_on_release"] for row in checks
            )),
            "target_loss_rate": float(np.mean([
                row["target_met_lost_on_release"] for row in checks
            ])),
            "fractional_opportunity_violation_count": int(sum(
                row["fractional_opportunity_worsened_on_release"] for row in checks
            )),
        },
        "inputs": {str(path.relative_to(ROOT)): file_hash(path) for path in (
            PROFILE, MANIFEST, TIMING, TIMING_SUMMARY, TIMING_PARENT,
            LOADED_SERVICE, NETWORK_CALIBRATION,
            LOCAL_TIMING / "scenarios.csv",
            LOCAL_TIMING / "migrations.csv", WIDTH8_TIMING / "scenarios.csv",
            WIDTH8_TIMING / "migration_stages.csv",
        )},
        "limitations": [
            "workload packs resample measured conversation templates and are sensitivity draws, not independent observations",
            "two bytes per resident token and equal unnormalized turn opportunity are declared workload assumptions",
            "each sampled pack is normalized to the pooled campaign's 0.4 source load",
            "the 0.4 source load is service-normalized load, not sampled phase-power load",
            "phase-load action shares are additive optimizer weights, not a per-action attribution of nonlinear watts",
            "reported shed is awake source-region power rather than net fleet power or energy",
            "two timing-supported trace states are excluded because their prefill/decode direction falls outside the phase-power calibration cone",
            "Replay endpoint work uses a measured prefill-heavy relative load factor while the regional zero-load anchor is unchanged",
            "route and endpoint stages overlap under the calibrated effective pipeline rate; Replay and KV endpoint work remain fully shared",
            surface_scope_limitation(validation),
            "timing audits are grouped retrospective checks, not untouched validation sets",
            "mixed timing evidence is KV-majority, so partial Replay/KV overlap is not used",
            "short-context Replay inside regional support uses a constant minimum-base-rate sensitivity extension",
            "the relative load factor was measured on a fixed width-eight pack; transport to regional concurrency-one timing and other contexts is a sensitivity",
            "all uses outside the 2,048-14,336-token fit pack or 1-10-Gbit/s validation routes are counted, with a support-restricted workload sensitivity",
            "the support-restricted workload resamples a narrower state population and is not a within-pack counterfactual",
            "the action ensemble fixes the loaded-service slope at its central fit rather than propagating its bootstrap",
            "the relative-factor check does not directly validate the deployed regional concurrency-one loaded model",
            "the load campaign identifies prefill-heavy normalized load, not a separate decode-load coefficient",
            "resume TTFT under loaded migration remains diagnostic because its 1-Gbit/s validation has false-feasible cases",
            "the destination envelope combines measured timing with synthetic 98%-occupied HBM and service-pressure levels",
            "the bandwidth state caps both physical destination routes at the predeclared 1-Gbit/s lower boundary of existing A100 loaded-migration validation",
            "the OAT action sweeps interpolate regional pipeline fits between the measured controlled-40 and natural endpoints; their action shares are modeled planner outcomes, not hardware action observations",
            "East's fitted controlled pipeline remains below 1 Gbit/s, so its loaded-factor transport is counted outside the validation bandwidth range",
            "HBM is method-independent because Replay and KV transfer leave the same resident KV state",
            "each constraint state is planned independently; rounded-LP release regressions are reported",
            "fractional opportunity monotonicity does not prove integer packability",
            "controlled-40 timing draws above natural bandwidth are projected down; the rate is recorded",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
