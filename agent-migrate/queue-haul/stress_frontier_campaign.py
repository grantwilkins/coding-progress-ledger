"""Fixed-suite 90%-coverage deadline--shed simulation campaign."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import replace
from pathlib import Path

import numpy as np

import network_campaign as network
import evidence_catalog as evidence
import plot_style
from planner import _expected_scenario, plan as solve, source_power
from profiles import ModelProfile, RateCurve
from simulate import predict


SCHEMA = "queue-haul-stress-frontier-plan-v1"
POLICIES = ("queue_haul", "greedy", "replay_only", "kv_only",
            "isolated_fastest", "queue_haul_power_blind",
            network.DEADLINE_BLIND_POLICY)
DEADLINES = tuple(range(10, 61, 5))
GREEDY_SOLVER = "greedy_best_effort"
LP_SOLVER = "lp_work_first_best_effort"
POWER_BLIND_SOLVER = "lp_power_blind_best_effort"
CONFIDENCE_DRAWS = 10_000
CONFIDENCE_SEED = 1
REGIMES = (
    ("jointly-binding", .75, .90, "controlled_40"),
    ("bandwidth-only", .25, 0, "controlled_40"),
    ("east-kv-only", .25, .90, "natural"),
    ("germany-service-only", .75, 0, "natural"),
    ("slack", .25, 0, "natural"),
)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if value.exists():
        return value.resolve()
    if value.parts and value.parts[0] == "queue-haul":
        local = Path(__file__).parent.joinpath(*value.parts[1:])
        if local.exists():
            return local.resolve()
    raise FileNotFoundError(value)


def _latin(rng, count: int) -> list[float]:
    values = (np.arange(count) + .5) / count
    rng.shuffle(values)
    return values.tolist()


def stress_states(profile: ModelProfile, seed: int = 1) -> list[dict]:
    phase = profile.case().phase_power
    if phase is None:
        raise ValueError("stress frontier requires a phase-aware profile")
    power_draws = max(len(getattr(phase, "measured_power_bootstrap", ())),
                      len(phase.bootstrap))
    if not power_draws:
        raise ValueError("stress frontier requires power bootstrap draws")
    rng, power_rng, states = np.random.default_rng(seed), np.random.default_rng(seed + 1), []
    errors = profile.sources
    for regime, germany, east_kv, bandwidth in REGIMES:
        axes = [_latin(rng, 8) for _ in range(4)]
        for index in range(8):
            states.append({
                "state_id": f"{regime}-{index}", "regime": regime,
                "germany_service_load": germany,
                "east_kv_fraction": east_kv, "bandwidth": bandwidth,
                "bandwidth_multiplier": .95 + .1 * axes[0][index],
                "service_multiplier": 1 + errors["service"].relative_error
                * (2 * axes[1][index] - 1),
                "replay_multiplier": 1 + errors["replay"].relative_error
                * (2 * axes[2][index] - 1),
                "kv_multiplier": 1 + errors["kv_transfer"].relative_error
                * (2 * axes[3][index] - 1),
                "power_bootstrap_index": int(power_rng.integers(power_draws)),
                "weight": 1 / 40,
            })
    return states


def prepare(parent_path: Path, profile_path: Path, out: Path, seed: int = 1) -> dict:
    parent_path, profile_path = _resolve(parent_path), _resolve(profile_path)
    parent, profile = json.loads(parent_path.read_text()), ModelProfile.load(profile_path)
    if parent.get("schema") != network.PLAN_SCHEMA or parent.get("design") != "separation":
        raise ValueError("stress campaign needs a separation parent plan")
    template = next(row for row in parent["scenarios"]
                    if row["condition_id"] == "joint-shaped" and row["repeat"] == 0
                    and row["policy"] == "queue_haul")
    plan = {
        "schema": SCHEMA, "seed": seed,
        "parent": {"path": str(parent_path), "sha256": network.profiler.file_hash(parent_path)},
        "profile": {"path": str(profile_path), "sha256": network.profiler.file_hash(profile_path)},
        "manifest": {"path": str(_resolve(parent["manifest"]["path"])),
                     "sha256": parent["manifest"]["sha256"]}, "template": template,
        "states": stress_states(profile, seed), "deadlines_s": list(DEADLINES),
        "policies": list(POLICIES),
        "coverage": {"states": 40, "required": 36, "order_statistic": 5},
        "pack": "recorded-28-seed-8",
    }
    out.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return plan


def _scale_curve(curve: RateCurve, multiplier: float) -> RateCurve:
    return RateCurve({key: (x.copy(), y / multiplier)
                      for key, (x, y) in curve.by_concurrency.items()})


def state_profile(profile: ModelProfile, state: dict) -> ModelProfile:
    case, phase = profile.case(), profile.case().phase_power
    if phase is None:
        raise ValueError("state profile requires phase power")
    if phase.measured_power_bootstrap:
        curve = phase.measured_power_bootstrap[
            state["power_bootstrap_index"] % len(phase.measured_power_bootstrap)]
        phase = replace(phase, measured_power_curve=curve)
    elif phase.bootstrap:
        p0, delta, a, b = phase.bootstrap[state["power_bootstrap_index"]]
        phase = replace(phase, p0_w=p0, delta_w=delta,
                        a_s_per_prefill_token=a, b_s_per_decode_token=b)
    service, replay, kv = (state[key] for key in (
        "service_multiplier", "replay_multiplier", "kv_multiplier"))
    transfer = case.kv_transfer
    transfer = replace(
        transfer, setup_s=transfer.setup_s * kv,
        destination_bytes_per_s=transfer.destination_bytes_per_s / kv,
        initial_completion_s=transfer.initial_completion_s * kv,
        catch_up_fixed_s=transfer.catch_up_fixed_s * kv,
        tail_replay_tps=transfer.tail_replay_tps / kv,
    )
    varied = replace(
        case, phase_power=phase, F=case.F / service, G=case.G / service,
        prefill=_scale_curve(case.prefill, service),
        decode=_scale_curve(case.decode, service),
        replay=_scale_curve(case.replay, replay),
        replay_completion_s=case.replay_completion_s * replay,
        kv_transfer=transfer,
    )
    max_power_load = max(phase.load(*point) for point in phase.valid_hull)
    return replace(profile, cases={"central": varied}, max_power_load=max_power_load)


def fifth_smallest(values) -> float:
    values = sorted(float(value) for value in values)
    if len(values) != 40:
        raise ValueError("90% suite coverage requires exactly 40 states")
    return values[4]


def normalized_confidence_intervals(values, regimes, draws=CONFIDENCE_DRAWS):
    values, regimes = np.asarray(values, float), np.asarray(regimes)
    groups = [np.flatnonzero(regimes == name) for name, *_ in REGIMES]
    if values.ndim != 2 or values.shape[1] != 40 \
            or any(len(group) != 8 for group in groups) or draws < 1:
        raise ValueError("confidence intervals require eight states per regime")
    rng = np.random.default_rng(CONFIDENCE_SEED)
    samples = np.concatenate([
        group[rng.integers(0, len(group), size=(draws, len(group)))]
        for group in groups
    ], axis=1)
    bootstrap = np.partition(values[:, samples], 4, axis=2)[:, :, 4]
    maximum = bootstrap.max(axis=0)
    if np.any(maximum <= 0):
        raise ValueError("confidence intervals require positive attainment")
    return np.quantile(bootstrap / maximum, (.025, .975), axis=1).T


def _best_outcome(outcomes):
    return min(outcomes, key=lambda row: row[1].modeled_source_power_at_deadline_w)


def _scenario(template: dict, state: dict, deadline: int, contract: dict) -> dict:
    bandwidth = network._bandwidths(contract, state["bandwidth"])
    bandwidth = {key: value * state["bandwidth_multiplier"]
                 for key, value in bandwidth.items()}
    return {
        **template, "design": "hardware_gap", "condition_id": state["state_id"],
        "deadline_s": deadline, "planning_deadline_s": deadline,
        "full_horizon_s": network.ORACLE_STALE_HORIZON_S,
        "requested_shed_fraction": 1.0, "bandwidth": state["bandwidth"],
        "bandwidth_mbps": bandwidth,
        "background": {"east": (.25, state["east_kv_fraction"]),
                       "germany": (state["germany_service_load"], 0)},
        "background_kv_headroom_tokens": {
            "east": network.HARDWARE_GAP_BACKGROUND_KV_TOKENS,
            "germany": network.HARDWARE_GAP_BACKGROUND_KV_TOKENS,
        },
        "migration_components": {
            node: contract["paths"][node]["migration_components"]
            for node in bandwidth
            if "migration_components" in contract["paths"][node]
        },
        "admission_mode": "normal", "objective": "max_shed",
    }


def run(plan_path: Path, out: Path, shard: int = 0, shards: int = 1) -> list[dict]:
    plan = json.loads(plan_path.read_text())
    if plan.get("schema") != SCHEMA or len(plan.get("states", ())) != 40:
        raise ValueError("invalid stress frontier plan")
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError("invalid shard")
    profile_path = _resolve(plan["profile"]["path"])
    if network.profiler.file_hash(profile_path) != plan["profile"]["sha256"]:
        raise RuntimeError("phase-aware profile changed")
    parent = json.loads(_resolve(plan["parent"]["path"]).read_text())
    manifest = json.loads(_resolve(plan["manifest"]["path"]).read_text())
    base_profile, rows = ModelProfile.load(profile_path), []
    records = network.scenario_records(manifest, plan["template"])
    base_demand = network.agentic_demand(
        records, plan["template"]["sessions"], base_profile,
        plan["template"]["source_load"])
    tasks = [(deadline, state) for deadline in plan["deadlines_s"]
             for state in plan["states"]]
    for task_index, (deadline, state) in enumerate(tasks):
        if task_index % shards != shard:
            continue
        profile = state_profile(base_profile, state)
        scenario = _scenario(plan["template"], state, deadline,
                             parent["network_contract"])
        snapshots = {node: {"kv_fraction": values[1]}
                     for node, values in scenario["background"].items()}
        snapshots = network._hardware_gap_snapshots(scenario, snapshots, profile)
        problem, architecture, routes, _target = network.joint_problem(
            scenario, snapshots, profile, base_demand)
        initial = source_power(problem, profile)
        for policy in plan["policies"]:
            planning = problem
            solver = ({"queue_haul": LP_SOLVER, "greedy": GREEDY_SOLVER,
                       "queue_haul_power_blind": POWER_BLIND_SOLVER,
                       network.DEADLINE_BLIND_POLICY: LP_SOLVER}
                      .get(policy, network.joint_solver(policy)))
            if policy == network.DEADLINE_BLIND_POLICY:
                planning = replace(problem, deadline_s=network.ORACLE_STALE_HORIZON_S,
                                   end_s=network.ORACLE_STALE_HORIZON_S)
            outcomes = []
            for candidate_solver in ((solver, GREEDY_SOLVER)
                                     if policy == "queue_haul" else (solver,)):
                result = solve(planning, profile, routes, candidate_solver,
                               seed=plan["seed"], destination=architecture,
                               admission_mode="normal")
                execution = predict(_expected_scenario(problem, result.moves),
                                    profile, result.moves, destination=architecture)
                outcomes.append((result, execution))
            result, execution = _best_outcome(outcomes)
            rows.append({
                "deadline_s": deadline, "state_id": state["state_id"],
                "regime": state["regime"], "policy": policy,
                "shed_by_deadline_w": initial - execution.modeled_source_power_at_deadline_w,
                "deadline_met": execution.deadline_met,
                "selected_sessions": len(result.moves),
                "migration_makespan_s": execution.migration_makespan_s,
                "profile_case": "bootstrap", "power_evidence_kind": "model_credited",
            })
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader(); writer.writerows(rows)
    return rows


def promotion_report(profile: ModelProfile, power_summary: dict,
                     destination_summary: dict, trailing_rows: list[dict],
                     provenance: dict) -> dict:
    differences = [abs(float(row["measured_trailing_shed_w"])
                       - float(row["modeled_deadline_shed_w"]))
                   for row in trailing_rows]
    if provenance.get("schema") != "queue-haul-evidence-catalog-v1":
        raise ValueError("invalid evidence catalog")
    root = Path(provenance["root"])
    for path in {row["power_path"] for row in trailing_rows}:
        evidence.verify(provenance, root, Path(path))
    checks = {
        "phase_power_gate": power_summary.get("gate_passed") is True,
        "service_gate": destination_summary.get("service_gate_passed") is True,
        "migration_gate": destination_summary.get("migration_gate_passed") is True,
        "operational_gate": destination_summary.get("operational_gate_passed") is True,
        "false_feasible_gate": destination_summary.get("false_feasible") == 0,
        "correctness_gate": destination_summary.get("correctness_failures") == 0,
        "hardware_window_gate": len(trailing_rows) >= 15
        and statistics.fmean(differences) <= 5,
        "provenance_gate": True,
        "transition_gate": profile.sources["transitions"].kind != "assumed",
    }
    return {"checks": checks, "passed": all(checks.values()),
            "hardware_window_mae_w": (statistics.fmean(differences)
                                      if differences else None)}


def reduce(results_paths: list[Path], out: Path, promotion: dict | None = None) -> dict:
    rows = []
    for results_path in results_paths:
        with results_path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    grouped, attained_by_state, regimes, frontier, best = {}, {}, {}, [], {}
    empirical = bool(promotion and promotion.get("passed"))
    for row in sorted(rows, key=lambda value: int(value["deadline_s"])):
        state_id, policy = row["state_id"], row["policy"]
        if state_id in regimes and regimes[state_id] != row["regime"]:
            raise ValueError("a stress state changed regime")
        regimes[state_id] = row["regime"]
        state = state_id, policy
        attained = max(float(row["shed_by_deadline_w"]),
                       best.get(state, float("-inf")))
        best[state] = attained
        cell = int(row["deadline_s"]), policy
        grouped.setdefault(cell, []).append(attained)
        attained_by_state[cell + (state_id,)] = attained
    expected = {(deadline, policy) for deadline in DEADLINES
                for policy in POLICIES}
    if set(grouped) != expected:
        raise ValueError("incomplete stress frontier results")
    states, cells = sorted(regimes), sorted(grouped)
    intervals = dict(zip(cells, normalized_confidence_intervals(
        [[attained_by_state[cell + (state,)] for state in states] for cell in cells],
        [regimes[state] for state in states],
    )))
    for deadline, policy in cells:
        row = {"deadline_s": deadline, "policy": policy,
               "coverage_90_shed_w": fifth_smallest(grouped[deadline, policy]),
               "states": 40, "claim": ("empirical deadline-shed frontier"
                                        if empirical else
                                        "modeled stress-suite sensitivity")}
        row["normalized_coverage_90_ci_low"], \
            row["normalized_coverage_90_ci_high"] = intervals[deadline, policy]
        frontier.append(row)
    value = {"schema": "queue-haul-stress-frontier-v1", "empirical": empirical,
             "confidence": {"level": .95, "draws": CONFIDENCE_DRAWS,
                            "method": "regime-stratified trajectory bootstrap",
                            "seed": CONFIDENCE_SEED},
             "frontier": frontier}
    if promotion is not None:
        value["promotion"] = promotion
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    _plot(frontier, out.with_suffix(""))
    return value


def _plot(frontier: list[dict], stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_style.apply()
    fig, ax = plt.subplots(figsize=plot_style.WIDE_FIGSIZE)
    maximum = max(row["coverage_90_shed_w"] for row in frontier
                  if row["policy"] in POLICIES)
    for policy in POLICIES:
        selected = [row for row in frontier if row["policy"] == policy]
        deadlines = [row["deadline_s"] for row in selected]
        ax.fill_between(
            deadlines,
            [row["normalized_coverage_90_ci_low"] for row in selected],
            [row["normalized_coverage_90_ci_high"] for row in selected],
            color=plot_style.POLICY_COLORS[policy], alpha=.1, linewidth=0,
        )
        ax.plot(deadlines,
                [row["coverage_90_shed_w"] / maximum for row in selected],
                **plot_style.policy_style(policy))
    ax.set(xlabel="Deadline (s)", ylabel="Normalized Power Shed", ylim=(0, 1.02))
    ax.tick_params(labelsize=plot_style.LARGE_FONT_SIZE)
    ax.xaxis.label.set_size(plot_style.LARGE_FONT_SIZE)
    ax.yaxis.label.set_size(plot_style.LARGE_FONT_SIZE)
    ax.grid(alpha=.25)
    ax.legend(loc="lower right", framealpha=1, facecolor="white",
              edgecolor="none", fontsize=plot_style.LARGE_LEGEND_FONT_SIZE)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parent", type=Path, required=True); p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True); p.add_argument("--seed", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True); r.add_argument("--out", type=Path, required=True)
    r.add_argument("--shard", type=int, default=0); r.add_argument("--shards", type=int, default=1)
    d = sub.add_parser("reduce")
    d.add_argument("--results", type=Path, nargs="+", required=True); d.add_argument("--out", type=Path, required=True)
    d.add_argument("--profile", type=Path); d.add_argument("--power-summary", type=Path)
    d.add_argument("--destination-summary", type=Path); d.add_argument("--trailing-power", type=Path)
    d.add_argument("--catalog", type=Path); d.add_argument("--modeled-only", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.parent, args.profile, args.out, args.seed)
    elif args.command == "run": run(args.plan, args.out, args.shard, args.shards)
    else:
        evidence_paths = (args.profile, args.power_summary, args.destination_summary,
                          args.trailing_power, args.catalog)
        if args.modeled_only:
            if any(evidence_paths):
                raise ValueError("modeled-only reduction does not accept promotion evidence")
            reduce(args.results, args.out)
            return
        if not all(evidence_paths):
            raise ValueError("empirical reduction requires all promotion evidence")
        with args.trailing_power.open(newline="") as handle:
            trailing = list(csv.DictReader(handle))
        provenance = json.loads(args.catalog.read_text())
        evidence.verify(provenance, Path(provenance["root"]), args.trailing_power)
        promotion = promotion_report(
            ModelProfile.load(args.profile), json.loads(args.power_summary.read_text()),
            json.loads(args.destination_summary.read_text()), trailing,
            provenance)
        reduce(args.results, args.out, promotion)


if __name__ == "__main__":
    main()
