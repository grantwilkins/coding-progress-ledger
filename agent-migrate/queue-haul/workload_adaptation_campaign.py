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
    FluidMigrationService, MigrationComponents, dedicated_sink_architecture,
)
from planner import plan, source_power
import plot_style
from pool_planner import candidate_table, phase_one_capacity_duals
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
OUT = ROOT / "outputs/workload-action-adaptation-20260814"
BASE_PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1.json"
WIDTH8_PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
FACTORS = ("hbm", "bandwidth", "dest_compute")
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
LEVELS = {"hbm": (0.0, .9), "bandwidth": ("natural", "controlled_40"),
          "dest_compute": (.25, .95)}
REGIONS = ("east", "germany")
ACTIONS = ("replay", "kv_transfer", "not_moved")
POWER_TOLERANCE_W = 1e-6
MIGRATION_HORIZON_S = 25
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
    templates, excluded, unselected = {}, 0, 0
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
        templates.setdefault(shape.template_id, []).append(shape)
    if not templates:
        raise ValueError("no workload templates inside timing support")
    return {key: tuple(value) for key, value in templates.items()}, {
        "manifest_templates": len(family), "supported_templates": len(templates),
        "supported_states": sum(map(len, templates.values())),
        "excluded_states": excluded, "unselected_states": unselected,
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


def build_problem(profile, sessions, constraints, target_fraction, fits):
    values, case = state_values(constraints), profile.case()
    bandwidths = {region: fits[region]["effective_pipeline_mbps"][
        values["bandwidth"]] * 125_000 for region in REGIONS}
    scenario = ExecutionScenario(
        30, 30, 0, "awake", 0,
        (PowerNode("source-node", 1, True), *(PowerNode(
            f"{region}-node", 1, False) for region in REGIONS)),
        (ServingInstance("source", ("source-node",)), *(ServingInstance(
            region, (f"{region}-node",)) for region in REGIONS)),
        sessions, tuple(NetworkLink(f"link/{region}", bandwidths[region])
                        for region in REGIONS),
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
    types, pools = [], []
    for region in REGIONS:
        raw = fits[region]["migration_components"]
        migration = {method: MigrationComponents(
            tuple(value["context_range"]),
            tuple(value["bandwidth_range_bytes_per_s"]), value["provenance"],
            value.get("compute_completion_factor", 1), value.get("residual_s", 0),
            value.get("kv_ingest_bytes_per_s"),
        ) for method, value in raw.items()}
        dtype = replace(q, type_id=f"{q.type_id}/{region}", migration=migration)
        service = FluidMigrationService(
            1 / fits[region]["replay_compute_completion_factor"],
            fits[region]["kv_ingest_lower_bound_bytes_per_s"],
            source_action, sink_action,
            "regional-c1 timing; conservative shared-work envelope",
            1, False,
        )
        resident = math.floor(
            values["hbm"] * q.kv_capacity_tokens / q.kv_block_tokens
        ) * q.kv_block_tokens
        types.append(dtype)
        pools.append(DestinationPool(
            f"pool/{region}", dtype.type_id,
            (DestinationReplica(
                region, tuple(values["dest_compute"] * direction), resident,
            ),), f"route/{region}", (f"link/{region}",),
            fluid_migration=service,
        ))
    architecture = DestinationArchitecture(
        architecture.schema, architecture.source_compatibility,
        tuple(types), tuple(pools),
    )
    routes = {("source", region): (f"link/{region}",) for region in REGIONS}
    return scenario, architecture, routes, target


def run_case(profile, sessions, case_id, label, constraints, replicate,
             target_fraction, fits, power_index, timing_hash, projected_regions):
    scenario, architecture, routes, target = build_problem(
        profile, sessions, constraints, target_fraction, fits,
    )
    result = plan(
        scenario, profile, routes, "lp_highs", seed=replicate,
        destination=architecture, admission_mode="normal",
    )
    table = candidate_table(
        scenario, profile, architecture, "normal", ExpectedPower(scenario, profile),
    )
    fractional_opportunity, _ = phase_one_capacity_duals(table)
    counts = {method: sum(move.method == method for move in result.moves)
              for method in ("replay", "kv_transfer")}
    counts["not_moved"] = len(sessions) - len(result.moves)
    if min(counts.values()) < 0 or sum(counts.values()) != len(sessions):
        raise RuntimeError("action counts do not conserve sessions")
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
    return {
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
        **{f"{action}_count": counts[action] for action in ACTIONS},
        **{action: counts[action] / len(sessions) for action in ACTIONS},
        "route_utilization": max_usage("route:"),
        "service_utilization": max_usage("service:"),
        "hbm_utilization": max_usage("kv:"),
        "migration_utilization": max_usage("migration:"),
        "binding_resources": ";".join(result.binding_resources),
        "failure": result.failure_reason or "",
    }


def simulate(samples=1000, seed=DEFAULT_SEED, sessions=28, target_fraction=2 / 3,
             profile_path=PROFILE, manifest_path=MANIFEST):
    if samples < 1 or sessions < 1 or not 0 < target_fraction <= 1:
        raise ValueError("invalid workload-adaptation simulation controls")
    profile = ModelProfile.load(profile_path)
    templates, workload = load_templates(manifest_path, profile)
    timing_rows, parent = read_csv(TIMING), json.loads(TIMING_PARENT.read_text())
    central_timing_fits()
    rng, rows = np.random.default_rng(seed), []
    for replicate in range(samples):
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


def factor_checks(rows):
    by_case = {constraints: case_id for case_id, _, constraints in factorial_cases()}
    utilization = {"hbm": "hbm_utilization", "bandwidth": "migration_utilization",
                   "dest_compute": "service_utilization"}
    checks = []
    for replicate in sorted({row["replicate"] for row in rows}):
        selected = {row["case_id"]: row for row in rows
                    if row["replicate"] == replicate}
        for case_id, label, constraints in factorial_cases():
            constrained = selected[case_id]
            for factor in constraints:
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
                    "active": pressure >= .9 \
                        or opportunity_change > POWER_TOLERANCE_W,
                })
    return checks


def summarize(rows):
    result = []
    for case_id, label, _ in factorial_cases():
        selected = [row for row in rows if row["case_id"] == case_id]
        for action in ACTIONS:
            values = np.asarray([row[action] for row in selected])
            q = np.quantile(values, (.05, .5, .95))
            result.append({
                "case_id": case_id, "bound_constraint": label, "action": action,
                "mean": values.mean(), "p05": q[0], "median": q[1], "p95": q[2],
                "target_met_rate": np.mean([row["target_met"] for row in selected]),
            })
    return result


def _surface_row(source, split, scenario, replay_s, kv_s, route_s, coupling):
    predicted = route_s + max(
        replay_s + coupling * kv_s, coupling * replay_s + kv_s,
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
    return rows


def validation_summary(rows):
    output = {}
    for key in sorted({(row["source"], row["split"]) for row in rows}):
        selected = [row for row in rows if (row["source"], row["split"]) == key]
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


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path):
    from matplotlib.lines import Line2D

    fig, axis = plt.subplots(figsize=(5.5, 3))
    groups = [[row for row in rows if row["case_id"] == case_id]
              for case_id, _, _ in factorial_cases()]
    replay = np.asarray([np.median([row["replay"] for row in group])
                         for group in groups])
    moved = np.asarray([np.median([row["replay"] + row["kv_transfer"]
                                  for row in group]) for group in groups])
    values_by_action = (replay, moved - replay, 1 - moved)
    left = np.zeros(8)
    for action, fractions in zip(ACTIONS, values_by_action):
        values = fractions * 100
        axis.barh(
            range(8), values, left=left, color=plot_style.ACTION_COLORS[action],
            hatch=plot_style.ACTION_HATCHES[action], edgecolor="white", linewidth=1.2,
            label=plot_style.ACTION_NAMES[action],
        )
        left += values
    for y, group in enumerate(groups):
        for values in ([row["replay"] for row in group],
                       [row["replay"] + row["kv_transfer"] for row in group]):
            low, middle, high = 100 * np.quantile(values, (.05, .5, .95))
            axis.hlines(y, low, high, color="#333333", linewidth=1, zorder=4)
            axis.plot(middle, y, marker="|", color="#333333", markersize=7,
                      zorder=5)
    axis.set(
        yticks=range(8), yticklabels=[label for _, label, _ in factorial_cases()],
        xlim=(0, 100), xlabel="Source-session share (%)",
    )
    axis.invert_yaxis()
    axis.grid(axis="x", alpha=.2)
    axis.tick_params(labelsize=11)
    axis.xaxis.label.set_size(12)
    handles, labels = axis.get_legend_handles_labels()
    handles.append(Line2D((0,), (0,), color="#333333", marker="|", linewidth=1))
    labels.append("5-95% boundary")
    fig.legend(
        handles, labels, frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(.58, .01), fontsize=10, handlelength=1.8,
    )
    fig.subplots_adjust(left=.34, right=.98, bottom=.36, top=.98)
    for suffix in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
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
    summary, validation, checks = (summarize(rows), validate_surface(),
                                   factor_checks(rows))
    activation = {(case, factor): np.mean([
        row["active"] for row in checks
        if row["case_id"] == case and row["factor"] == factor
    ]) for case, factor in {(row["case_id"], row["factor"]) for row in checks}}
    singles = {case_id for case_id, _, constraints in factorial_cases()
               if len(constraints) == 1}
    if any(row["fractional_opportunity_worsened_on_release"] for row in checks):
        raise RuntimeError("constraint release reduced the fractional opportunity set")
    if any(rate < .9 for (case, _), rate in activation.items()
           if case in singles) or any(rate == 0 for rate in activation.values()):
        raise RuntimeError("labeled constraints failed activation gates")
    write_csv(args.out / "action_mix.csv", rows)
    write_csv(args.out / "action_mix_summary.csv", summary)
    write_csv(args.out / "surface_validation.csv", validation)
    write_csv(args.out / "factor_checks.csv", checks)
    plot(rows, args.out / "action_mix")
    timing_evidence = json.loads(TIMING_SUMMARY.read_text())
    regressions = np.asarray([
        row["planner_shortfall_change_w"] for row in checks
        if row["planner_shortfall_worsened_on_release"]
    ])
    metadata = {
        "schema": "queue-haul-workload-adaptation-v1",
        "claim": "conservative measurement-calibrated workload sensitivity",
        "samples": args.samples, "sessions_per_pack": args.sessions,
        "seed": args.seed, "target_fraction": args.target,
        "source_load": SOURCE_LOAD,
        "unique_timing_draws": len({row["timing_fit_sha256"] for row in rows}),
        "cross_method_coupling": 1,
        "bandwidth_projection_region_rate": float(np.mean([
            row["bandwidth_projection_regions"] / len(REGIONS)
            for row in rows[::len(ORDER)]
        ])),
        "route_compute_overlap": False,
        "migration_horizon_s": MIGRATION_HORIZON_S,
        "factor_levels": LEVELS, "workload": workload,
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
        "factor_activation_rate": {f"{case}/{factor}": float(rate)
                                   for (case, factor), rate in sorted(activation.items())},
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
            LOCAL_TIMING / "scenarios.csv",
            LOCAL_TIMING / "migrations.csv", WIDTH8_TIMING / "scenarios.csv",
            WIDTH8_TIMING / "migration_stages.csv",
        )},
        "limitations": [
            "workload packs resample measured conversation templates and are sensitivity draws, not independent observations",
            "two bytes per resident token and equal unnormalized turn opportunity are declared workload assumptions",
            "each sampled pack is normalized to the pooled campaign's 0.4 source load",
            "destination compute pressure consumes shared prefill/decode service headroom; load-dependent migration slowdown is not modeled",
            "route plus shared destination work is a conservative no-overlap envelope",
            "timing audits are grouped retrospective checks, not untouched validation sets",
            "mixed timing evidence is KV-majority, so partial Replay/KV overlap is not used",
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
