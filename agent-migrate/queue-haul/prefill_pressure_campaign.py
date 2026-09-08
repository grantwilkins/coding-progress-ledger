"""Freeze four prefill-pressure cases at one WAN rate and a 30-second deadline."""

import argparse
import copy
import json
import statistics
from pathlib import Path

import constrained_state_campaign as campaign
import contention_campaign as contention
from destination import dedicated_sink_architecture
from profiles import ModelProfile


def background_response(plan: dict, raw: list[dict]) -> dict:
    campaign._verify_plan(plan)
    jobs = {row["episode_id"]: row for row in plan["schedule"]}
    prefill = [row for row in raw if row["capacity_inputs"].get("background_kind") == "prefill"]
    slopes = [statistics.mean(row["capacity_inputs"]["baseline_work"][index] /
                             row["capacity_inputs"]["background_rps"] for row in prefill)
              for index in range(2)]
    pure = []
    for row in raw:
        job, moves = jobs[row["episode_id"]], row["decisions"]
        if len(moves) != 8 or len({move["action"] for move in moves}) != 1 or job["n_hbm"] or job["n_serving"]:
            continue
        if any(move.get("error") or move.get("completion_s") is None for move in moves):
            raise ValueError("background calibration migration failed")
        pure.append({"episode_id": row["episode_id"], "action": moves[0]["action"],
                     "wan_mbps": job["wan_mbps"], "rho": row["capacity_inputs"]["baseline_work"][0],
                     "elapsed_s": max(move["completion_s"] for move in moves)})
    response = {}
    for action in campaign.ACTIONS:
        coefficients = []
        for row in pure:
            if row["action"] != action or not row["rho"]:
                continue
            idle = statistics.mean(other["elapsed_s"] for other in pure if
                                   other["action"] == action and not other["rho"] and
                                   other["wan_mbps"] == row["wan_mbps"])
            difference = row["elapsed_s"] / idle - 1 if action == "replay" else row["elapsed_s"] - idle
            coefficients.append(difference / row["rho"])
        response[action] = statistics.mean(coefficients)
    if min(slopes + list(response.values())) < 0:
        raise ValueError("background calibration has negative pressure response")
    return {"baseline_work_per_rps": slopes, "replay_duration_per_rho": response["replay"],
            "kv_tail_s_per_rho": response["kv_transfer"], "evidence": pure,
            "source_plan_sha256": campaign._verify_plan(plan), "raw_sha256": campaign.digest(raw)}


def prepare(out: Path, source_plan=None, calibration_plan=None, calibration_raw=None,
            background_plan=None, background_raw=None) -> dict:
    def read(directory, name):
        path = campaign.ROOT / "outputs" / directory / name
        return ([json.loads(line) for line in path.read_text().splitlines()]
                if path.suffix == ".jsonl" else json.loads(path.read_text()))
    source_plan = source_plan if source_plan is not None else read("constrained-resource-a100-20260905", "prepared/plan.json")
    calibration_plan = calibration_plan if calibration_plan is not None else read("contention-a100-20260905", "prepared/plan.json")
    calibration_raw = calibration_raw if calibration_raw is not None else read("contention-a100-20260905", "run/raw_episodes.jsonl")
    background_plan = background_plan if background_plan is not None else read("marginal-resource-a100-20260905", "prepared/plan.json")
    background_raw = background_raw if background_raw is not None else read("marginal-resource-a100-20260905", "run/raw_episodes.jsonl")
    campaign._verify_plan(source_plan)
    calibration, response = contention.calibrate(calibration_plan, calibration_raw), background_response(background_plan, background_raw)
    inputs = copy.deepcopy(source_plan["frozen"]["inputs"])
    profile_path = Path(inputs["background_manifest"]["live"]["model_profile_path"])
    profile = ModelProfile.load(profile_path)
    if json.loads(profile_path.read_text()) != inputs["profile"] or any(
            plan["frozen"]["inputs"]["profile"] != inputs["profile"] for plan in (calibration_plan, background_plan)):
        raise ValueError("calibration and frozen planner profiles differ")
    pack = next(row for row in inputs["packs"] if row["pack_id"] == "large-r0")
    old_id, pack["pack_id"] = pack["pack_id"], "prefill-pressure-27360"
    for row in pack["sessions"]:
        row.update(initial_tokens=27360, log_bytes=54720)
    case = profile.case()
    destination_type = dedicated_sink_architecture(profile, "destination", ("link",)).types[0]
    for row in inputs["action_demands"]:
        if row["pack_id"] != old_id:
            continue
        session = next(s for s in pack["sessions"] if s["session_id"] == row["session_id"])
        replay = row["action"] == "replay"
        duration = 27360 / case.replay.rate(27360, 1) + case.replay_completion_s if replay else case.kv_transfer.setup_s + case.kv_transfer.initial_completion_s
        row.update(pack_id=pack["pack_id"], duration_s=duration + case.switch_s,
                   d_wan=54720 if replay else case.kv_transfer.sealed_bytes(27360),
                   d_prefill=duration if replay else 0, d_hbm=profile.kv_admission_tokens(27360),
                   d_service=float(sum(destination_type.work(session["expected_f"], session["expected_g"], 27360))) * 30)
    inputs["wan_mbps"] = [4000]
    frozen = campaign.freeze_inputs(inputs, source_plan["frozen"]["seed"])
    states, predictions = [], []
    for units in (0, .4, .5, .6):
        rate = units * inputs["background_manifest"]["live"]["prefill_unit_rps"]
        baseline = [rate * value for value in response["baseline_work_per_rps"]]
        capacity = {"baseline_work": baseline, "baseline_kv_tokens": 0,
                    "kv_capacity_tokens": profile.kv_capacity_tokens,
                    "resources": {"wan": 4000 * 125_000 * 30, "prefill": (1 - baseline[0]) * 30,
                                  "service": (1 - sum(baseline)) * 30, "hbm": profile.kv_capacity_tokens}}
        state = {"state_id": campaign.digest([pack["pack_id"], 4000, 30, units])[:16],
                 "family": "prefill_pressure" if units else "control", "wan_mbps": 4000,
                 "deadline_s": 30, "n_prefill": units, "n_hbm": 0, "n_serving": 0,
                 "operational": True, "capacity_inputs": capacity}
        states.append(state)
        fit = {**calibration, "replay_speedup": calibration["replay_speedup"] /
               (1 + response["replay_duration_per_rho"] * baseline[0]),
               "kv_tail_s": calibration["kv_tail_s"] + response["kv_tail_s_per_rho"] * baseline[0]}
        for policy in campaign.POLICIES:
            moves = campaign._planner_decisions(inputs, state, pack, policy, profile, capacity)
            for scale in (.85, 1., 1.15):
                predictions.append({"state_id": state["state_id"], "n_prefill": units, "policy": policy,
                                    "duration_scale": scale, "moves": moves,
                                    **contention.simulate(pack, moves, 4000, 30, profile, fit, scale)})
    schedule = campaign.execution_schedule(states, [pack], frozen["seed"], repeats=3)
    schedule.sort(key=lambda row: row["repeat"])
    plan = {"frozen": frozen, "states": states, "schedule": schedule, "annotations": [],
            "calibration": calibration, "background_response": response,
            "design": {"repeats": 3, "selected_pack": pack["pack_id"], "episode_count": len(schedule),
                       "scope": "prefill pressure at fixed 4 Gbps; all KV fits nominally; narrow unvalidated calibration",
                       "selection": "fixed control and three prefill brackets; retain every case regardless of prediction",
                       "metric": "full-target attainment primary; five-second windowed relief secondary"}}
    plan["sha256"] = campaign.digest(plan)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    campaign.write_csv(out / "simulation.csv", predictions)
    campaign.write_csv(out / "execution_schedule.csv", schedule)
    return plan


def summarize(plan: dict, run_root: Path) -> dict:
    campaign._verify_plan(plan)
    raw = [json.loads(line) for line in (run_root / "raw_episodes.jsonl").read_text().splitlines()]
    inputs = plan["frozen"]["inputs"]
    rows = campaign.normalize_episodes(raw, plan["schedule"][:len(raw)], inputs["packs"], inputs["target"])
    paired = {(row["state_id"], row["repeat"], row["policy"]): row["target_attained"] for row in rows}
    policies = ("queue_haul", "greedy", "per_session_greedy")
    control = next(state for state in plan["states"] if not state["n_prefill"])
    control_full = all(paired.get((control["state_id"], repeat, policy), False)
                       for repeat in range(3) for policy in policies)
    cases = []
    for state in plan["states"]:
        selected = [row for row in rows if row["state_id"] == state["state_id"]]
        transitions = [repeat for repeat in range(3) if state["n_prefill"] and
                       all(paired.get((state["state_id"], repeat, policy), False) for policy in policies[:2])
                       and paired.get((state["state_id"], repeat, policies[-1])) is False]
        cases.append({"state_id": state["state_id"], "n_prefill": state["n_prefill"],
                      "attained": {policy: sum(row["target_attained"] for row in selected if row["policy"] == policy)
                                   for policy in campaign.POLICIES},
                      "repeats": {policy: sorted(row["repeat"] for row in selected if row["policy"] == policy)
                                  for policy in campaign.POLICIES}, "transition_repeats": transitions})
    validation = {"complete": len(raw) == len(plan["schedule"]), "control_full": control_full,
                  "cases": cases, "genuine_prefill_transition": control_full and any(row["transition_repeats"] for row in cases),
                  "robust": control_full and any(row["transition_repeats"] == [0, 1, 2] for row in cases)}
    (run_root / "prefill_validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    contention.summarize(plan, run_root)
    if validation["complete"] and not validation["genuine_prefill_transition"]:
        raise ValueError("no prefill-induced transition observed; all control and loaded outcomes retained")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("prepare", "summarize"), default="prepare")
    for name in ("out", "plan", "run-root"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.out is None:
            parser.error("prepare requires --out")
        prepare(args.out)
    else:
        if args.plan is None or args.run_root is None:
            parser.error("summarize requires --plan and --run-root")
        summarize(json.loads(args.plan.read_text()), args.run_root)
