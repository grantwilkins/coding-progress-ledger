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
REFERENCE = plot_style.REFERENCE
DEADLINES = tuple(range(10, 61, 5))
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
    rng, states = np.random.default_rng(seed), []
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
                "power_bootstrap_index": int(rng.integers(
                    0, max(1, len(getattr(phase, "measured_power_bootstrap", ())),
                           len(phase.bootstrap)))),
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
        "policies": list(POLICIES), "reference": REFERENCE,
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
        for policy in (*POLICIES, REFERENCE):
            planning = problem
            solver = "max_shed" if policy == REFERENCE else network.joint_solver(
                policy, scenario["objective"])
            if policy == network.DEADLINE_BLIND_POLICY:
                planning = replace(problem, deadline_s=network.ORACLE_STALE_HORIZON_S,
                                   end_s=network.ORACLE_STALE_HORIZON_S)
            result = solve(planning, profile, routes, solver,
                           seed=plan["seed"], destination=architecture,
                           admission_mode="normal")
            execution = predict(_expected_scenario(problem, result.moves),
                                profile, result.moves, destination=architecture)
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
    grouped, frontier = {}, []
    empirical = bool(promotion and promotion.get("passed"))
    for row in rows:
        grouped.setdefault((int(row["deadline_s"]), row["policy"]), []).append(
            float(row["shed_by_deadline_w"]))
    expected = {(deadline, policy) for deadline in DEADLINES
                for policy in (*POLICIES, REFERENCE)}
    if set(grouped) != expected:
        raise ValueError("incomplete stress frontier results")
    for deadline, policy in sorted(grouped):
        frontier.append({"deadline_s": deadline, "policy": policy,
                         "coverage_90_shed_w": fifth_smallest(grouped[deadline, policy]),
                         "states": 40, "claim": ("empirical deadline-shed frontier"
                                                  if empirical else
                                                  "modeled stress-suite sensitivity")})
    value = {"schema": "queue-haul-stress-frontier-v1", "empirical": empirical,
             "reference_label": "exact modeled MILP optimum", "frontier": frontier}
    if promotion is not None:
        value["promotion"] = promotion
    out.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    _plot(frontier, out.with_suffix(""), empirical)
    return value


def _plot(frontier: list[dict], stem: Path, empirical: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_style.apply()
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for policy in (*POLICIES, REFERENCE):
        selected = [row for row in frontier if row["policy"] == policy]
        ax.plot([row["deadline_s"] for row in selected],
                [row["coverage_90_shed_w"] for row in selected],
                **plot_style.policy_style(policy))
    ax.set(xlabel="Deadline (s)", ylabel="90%-coverage trailing-window shed (W)",
           title=("Empirical deadline–shed frontier" if empirical else
                  "Modeled stress-suite sensitivity"))
    ax.legend(frameon=False, fontsize=8, ncol=2)
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
