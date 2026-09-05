"""Compile six profile-selected boundary cases for a 90-episode live test."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import constrained_state_campaign as campaign
from destination import dedicated_sink_architecture
from profiles import ModelProfile


def prepare(source: dict, out: Path) -> dict:
    campaign._verify_plan(source)
    frozen = copy.deepcopy(source["frozen"])
    inputs = frozen["inputs"]
    pack = next(row for row in inputs["packs"] if row["pack_id"] == "large-r0")
    live = inputs["background_manifest"]["live"]
    profile = ModelProfile.load(Path(live["model_profile_path"]))
    if json.loads(Path(live["model_profile_path"]).read_text()) != inputs["profile"]:
        raise ValueError("profile differs from frozen source")
    destination_type = dedicated_sink_architecture(profile, "destination", ("link",)).types[0]
    service_work = {row["session_id"]: float(sum(destination_type.work(
        row["expected_f"], row["expected_g"], row["initial_tokens"]))) * campaign.D_S
        for row in pack["sessions"]}
    for row in inputs["action_demands"]:
        if row["pack_id"] == pack["pack_id"]:
            row["d_service"] = service_work[row["session_id"]]
    frozen = campaign.freeze_inputs(inputs, frozen["seed"])
    reference = max(inputs["wan_mbps"])
    control = next(row for row in source["states"] if
                   row["wan_mbps"] == reference and not any(
                       row[f"n_{axis}"] for axis in campaign.AXES))
    base = control["capacity_inputs"]
    slopes = {}
    for axis in ("prefill", "serving"):
        row = next(row for row in source["states"] if
                   row["wan_mbps"] == reference and row[f"n_{axis}"] == 1)
        slopes[axis] = row["capacity_inputs"]["baseline_work"]
    held = next(row for row in source["states"] if row["n_hbm"] == 1)
    tokens_per_unit = (base["kv_capacity_tokens"] -
                       held["capacity_inputs"]["kv_capacity_tokens"])
    demands = [row for row in inputs["action_demands"]
               if row["pack_id"] == pack["pack_id"]]
    total = lambda resource, action: sum(row[f"d_{resource}"] for row in demands
                                        if row["action"] == action)
    # Fixed physical margin, chosen before observing any live policy outcomes.
    margin = .95
    wan = margin * total("wan", "kv_transfer") / (125_000 * campaign.D_S)
    prefill = (1 - margin * total("prefill", "replay") / campaign.D_S) / slopes["prefill"][0]
    serving = (1 - margin * total("service", "replay") / campaign.D_S) / sum(slopes["serving"])
    hbm = (base["kv_capacity_tokens"] - margin * total("hbm", "replay")) / tokens_per_unit
    cases = [("control", reference, 0, 0, 0),
             ("wan", wan, 0, 0, 0),
             ("prefill", reference, prefill, 0, 0),
             ("hbm", reference, 0, hbm, 0),
             ("serving", reference, 0, 0, serving),
             ("wan_prefill", wan, prefill, 0, 0)]
    states, decisions, schedule = [], [], []
    for family, rate, n_prefill, n_hbm, n_serving in cases:
        baseline = [n_prefill * slopes["prefill"][index] +
                    n_serving * slopes["serving"][index] for index in range(2)]
        kv_capacity = round(base["kv_capacity_tokens"] - n_hbm * tokens_per_unit)
        capacity = {"baseline_work": baseline, "baseline_kv_tokens": 0,
                    "kv_capacity_tokens": kv_capacity,
                    "resources": {"wan": rate * 125_000 * campaign.D_S,
                                  "prefill": max(0, 1 - baseline[0]) * campaign.D_S,
                                  "service": max(0, 1 - sum(baseline)) * campaign.D_S,
                                  "hbm": kv_capacity}}
        state = {"state_id": campaign.digest([family, rate, n_prefill, n_hbm, n_serving])[:16],
                 "family": family, "wan_mbps": rate, "n_prefill": n_prefill,
                 "n_hbm": n_hbm, "n_serving": n_serving,
                 "capacity_inputs": capacity, "operational": True,
                 "capacity_source": "offline estimate; remeasured before each live policy"}
        states.append(state)
        schedule.extend(campaign.execution_schedule([state], [pack], frozen["seed"], repeats=3))
        for policy in campaign.POLICIES:
            moves = campaign._planner_decisions(inputs, state, pack, policy, profile, capacity)
            counts = Counter(move["action"] for move in moves)
            decisions.append({"state_id": state["state_id"], "family": family,
                              "pack_id": pack["pack_id"], "policy": policy,
                              **{action: counts[action] for action in campaign.ACTIONS},
                              "admitted": len(moves), "moves": moves})
    for policy in ("queue_haul", "greedy"):
        rows = [row for row in decisions if row["policy"] == policy]
        if len({tuple(row[action] for action in campaign.ACTIONS) for row in rows}) < 2:
            raise ValueError(f"{policy} has no predicted action shift")
        coupled = next(row for row in rows if row["family"] == "wan_prefill")
        if not all(coupled[action] for action in campaign.ACTIONS) or coupled["admitted"] != 8:
            raise ValueError(f"{policy} does not admit a full mixed coupled plan")
    annotations = campaign.oracle_annotations(states, [pack], demands, inputs["target"])
    coupled = next(row for row in annotations if row["state_id"] == states[-1]["state_id"])
    if not coupled["mixed_full"]:
        raise ValueError("coupled boundary lacks mixed-only full feasibility")
    schedule.sort(key=lambda row: row["repeat"])
    plan = {"frozen": frozen, "states": states, "schedule": schedule,
            "annotations": annotations, "offline_decisions": decisions,
            "design": {"source_plan_sha256": campaign._verify_plan(source),
                       "residual_demand_fraction": margin, "repeats": 3,
                       "selected_pack": pack["pack_id"], "episode_count": len(schedule),
                       "selection": "physical full-pack demand boundaries; no live outcome selection"}}
    plan["sha256"] = campaign.digest(plan)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    campaign.write_csv(out / "offline_decisions.csv", decisions)
    campaign.write_csv(out / "background_states.csv", states)
    campaign.write_csv(out / "execution_schedule.csv", schedule)
    campaign.write_csv(out / "oracle_annotations.csv", annotations)
    return plan


def summarize(plan: dict, run_root: Path) -> dict:
    campaign._verify_plan(plan)
    raw = [json.loads(line) for line in
           (run_root / "raw_episodes.jsonl").read_text().splitlines()]
    inputs, schedule = plan["frozen"]["inputs"], plan["schedule"]
    if len(raw) > len(schedule):
        raise ValueError("raw episodes exceed schedule")
    episodes = campaign.normalize_episodes(
        raw, schedule[:len(raw)], inputs["packs"], inputs["target"])
    summary = []
    for state in plan["states"]:
        for policy in campaign.POLICIES:
            rows = sorted((row for row in episodes if
                           row["state_id"] == state["state_id"] and
                           row["policy"] == policy), key=lambda row: row["repeat"])
            summary.append({"family": state["family"], "policy": policy,
                            "repeats": [row["repeat"] for row in rows],
                            "kv": [row["kv_count"] for row in rows],
                            "replay": [row["replay_count"] for row in rows],
                            "admitted": [row["kv_count"] + row["replay_count"] for row in rows],
                            "attained": sum(row["target_attained"] for row in rows),
                            "relief_w": [row["achieved_relief_w"] for row in rows],
                            "target_time_s": [row["target_time_s"] for row in rows]})
    checks = []
    for policy in ("queue_haul", "greedy"):
        for repeat in range(3):
            rows = [row for row in episodes if
                    row["policy"] == policy and row["repeat"] == repeat]
            checks.append({"policy": policy, "repeat": repeat,
                           "cases_completed": len(rows),
                           "action_shift": len({row["kv_count"] /
                                                (row["kv_count"] + row["replay_count"])
                                                for row in rows
                                                if row["kv_count"] + row["replay_count"]}) > 1,
                           "mixed_observed": any(bool(row["kv_count"] and row["replay_count"])
                                                 for row in rows
                                                 if row["family"] == "wan_prefill")})
    validation = {"complete": len(raw) == len(schedule),
                  "completed_episodes": len(raw), "scheduled_episodes": len(schedule),
                  "three_repeats_per_case_policy": all(
                      row["repeats"] == [0, 1, 2] for row in summary),
                  "action_checks": checks}
    campaign.write_csv(run_root / "summary.csv", summary)
    (run_root / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if validation["complete"] and (not validation["three_repeats_per_case_policy"] or
                                      not all(row["action_shift"] and row["mixed_observed"]
                                              for row in checks)):
        raise ValueError("live run lacks repeats, QH action shifts or mixed actions; see validation.json")
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source-plan", type=Path, required=True)
    prepare_parser.add_argument("--out", type=Path, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--plan", type=Path, required=True)
    summary_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(json.loads(args.source_plan.read_text()), args.out)
        print(f"Prepared {len(plan['states'])} states and {len(plan['schedule'])} episodes")
    else:
        print(json.dumps(summarize(json.loads(args.plan.read_text()), args.run_root)))


if __name__ == "__main__":
    main()
