"""Select a small live test using calibrated concurrent migration execution."""

import argparse
import copy
import json
import statistics
from collections import Counter
from pathlib import Path

import constrained_state_campaign as campaign
from profiles import ModelProfile
from simulate import fluid_service_completion


def calibrate(plan: dict, raw: list[dict]) -> dict:
    campaign._verify_plan(plan)
    inputs = plan["frozen"]["inputs"]
    profile = ModelProfile.load(Path(inputs["background_manifest"]["live"]["model_profile_path"]))
    jobs = {row["episode_id"]: row for row in plan["schedule"]}
    packs = {row["pack_id"]: row for row in inputs["packs"]}
    replay, kv, evidence = [], [], []
    case = profile.case()
    for row in raw:
        job = jobs[row["episode_id"]]
        moves = row["decisions"]
        if any(job[f"n_{axis}"] for axis in campaign.AXES) or not moves or \
                job["policy"] not in ("replay_only", "kv_only"):
            continue
        if any(move.get("error") or move.get("completion_s") is None for move in moves):
            raise ValueError("calibration migration failed")
        sessions = {s["session_id"]: s for s in packs[job["pack_id"]]["sessions"]}
        tokens = [sessions[move["session_id"]]["initial_tokens"] for move in moves]
        elapsed = max(move["completion_s"] for move in moves)
        value = (sum(t / case.replay.rate(t, 1) + case.replay_completion_s for t in tokens)
                 / elapsed if job["policy"] == "replay_only" else elapsed -
                 sum(case.kv_transfer.sealed_bytes(t) for t in tokens) / (job["wan_mbps"] * 125_000))
        (replay if job["policy"] == "replay_only" else kv).append(value)
        evidence.append({"episode_id": job["episode_id"], "policy": job["policy"],
                         "value": value, "completion_s": elapsed, "sessions": len(moves)})
    if not replay or not kv or min(replay + kv) <= 0:
        raise ValueError("positive replay and KV calibration measurements required")
    return {"replay_speedup": statistics.median(replay), "kv_tail_s": statistics.median(kv),
            "replay_speedup_range": [min(replay), max(replay)],
            "kv_tail_range_s": [min(kv), max(kv)], "evidence": evidence,
            "source_plan_sha256": campaign._verify_plan(plan),
            "raw_sha256": campaign.digest(raw),
            "scope": "idle destination; calibrated at eight equal 16K contexts; heterogeneous execution is a prediction"}


def simulate(pack: dict, moves: list[dict], wan_mbps: float, deadline_s: float,
             profile, calibration: dict, scale: float = 1) -> dict:
    sessions = {row["session_id"]: row for row in pack["sessions"]}
    case = profile.case()
    tokens = [sessions[move["session_id"]]["initial_tokens"] for move in moves]
    network = fluid_service_completion([
        (2 * t if move["action"] == "replay" else case.kv_transfer.sealed_bytes(t))
        / (wan_mbps * 125_000) for move, t in zip(moves, tokens)], 1)
    replay = [i for i, move in enumerate(moves) if move["action"] == "replay"]
    completed = {moves[i]["session_id"]: float(network[i] + calibration["kv_tail_s"] * scale)
                 for i in range(len(moves)) if i not in replay}
    times = fluid_service_completion([
        (tokens[i] / case.replay.rate(tokens[i], 1) + case.replay_completion_s)
        * scale / calibration["replay_speedup"] for i in replay], 1,
        [network[i] for i in replay])
    completed.update({moves[i]["session_id"]: float(at) for i, at in zip(replay, times)})
    events = [(at, session) for session, at in completed.items()]
    return {"relief_w": campaign.window_relief(pack, events, deadline_s),
            "target_time_s": campaign.target_time(pack, events, pack["power_gains"][-1], deadline_s),
            "kv_count": len(moves) - len(replay), "replay_count": len(replay),
            "admitted": len(moves), "completion_times": completed}


def prepare(source_plan: dict, calibration_plan: dict, calibration_raw: list[dict], out: Path) -> dict:
    campaign._verify_plan(source_plan)
    calibration = calibrate(calibration_plan, calibration_raw)
    inputs = copy.deepcopy(source_plan["frozen"]["inputs"])
    profile = ModelProfile.load(Path(inputs["background_manifest"]["live"]["model_profile_path"]))
    if json.loads(Path(inputs["background_manifest"]["live"]["model_profile_path"]).read_text()) != inputs["profile"]:
        raise ValueError("planner profile changed")
    template = next(row for row in inputs["packs"] if row["pack_id"] == "large-r0")
    contexts = [(4096, 8192, 12288, 16384), (8192, 12288, 14336, 16384),
                (8192, 16384, 24576, 31562), (16384, 24576, 28672, 31562),
                (4096, 8192, 24576, 31562), (12288, 16384, 24576, 31562)]
    outcomes, candidates = [], []
    for context in contexts:
        pack = copy.deepcopy(template)
        pack["pack_id"] = campaign.digest(context)[:16]
        for row, tokens in zip(pack["sessions"], sorted(context * 2)):
            row.update(initial_tokens=tokens, log_bytes=2 * tokens)
        for wan in (2000, 3000, 4000, 5000, 6000, 8000, 10000):
            for deadline in (12, 13, 14, 15, 16, 18, 20, 22, 25):
                state = {"state_id": campaign.digest([context, wan, deadline])[:16],
                         "family": "contention", "wan_mbps": wan, "deadline_s": deadline,
                         "n_prefill": 0, "n_hbm": 0, "n_serving": 0, "operational": True,
                         "capacity_inputs": {"baseline_work": [0, 0], "baseline_kv_tokens": 0,
                             "kv_capacity_tokens": profile.kv_capacity_tokens,
                             "resources": {"wan": wan * 125_000 * deadline, "prefill": deadline,
                                           "service": deadline, "hbm": profile.kv_capacity_tokens}}}
                rows, plans = [], {}
                for policy in campaign.POLICIES:
                    moves = campaign._planner_decisions(inputs, state, pack, policy, profile, state["capacity_inputs"])
                    plans[policy] = moves
                    for scale in (.85, 1., 1.15):
                        rows.append({"state_id": state["state_id"], "contexts": list(context),
                                     "wan_mbps": wan, "deadline_s": deadline, "policy": policy,
                                     "duration_scale": scale,
                                     **simulate(pack, moves, wan, deadline, profile, calibration, scale)})
                outcomes.extend(rows)
                by = {(row["policy"], row["duration_scale"]): row for row in rows}
                gain = min(by[policy, 1.]["relief_w"] - by["per_session_greedy", 1.]["relief_w"]
                           for policy in ("queue_haul", "greedy"))
                robust = all(by[policy, scale]["relief_w"] + 1e-8 >= by[baseline, scale]["relief_w"]
                             for policy in ("queue_haul", "greedy") for baseline in campaign.POLICIES[2:]
                             for scale in (.85, 1., 1.15))
                candidates.append({"pack": pack, "state": state, "moves": plans,
                                   "robust": robust, "gain": gain,
                                   "full": all(by[policy, 1.]["target_time_s"] is not None
                                               for policy in ("queue_haul", "greedy")),
                                   "full_robust": all(by[policy, scale]["target_time_s"] is not None
                                               for policy in ("queue_haul", "greedy")
                                               for scale in (.85, 1., 1.15))})
    out.mkdir(parents=True, exist_ok=True)
    campaign.write_csv(out / "simulation.csv", outcomes)
    (out / "calibration.json").write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n")
    eligible = sorted((row for row in candidates if row["robust"] and row["full"] and row["gain"] > 1e-6),
                      key=lambda row: (not row["full_robust"], -row["gain"], row["state"]["state_id"]))
    if not eligible:
        raise ValueError("no robust QH improvement found; all simulation outcomes retained")
    winner = eligible[0]
    nearby = sorted((row for row in candidates if row["pack"]["pack_id"] == winner["pack"]["pack_id"] and
                     row["robust"] and row["full"] and row["state"]["state_id"] != winner["state"]["state_id"]),
                    key=lambda row: (abs(row["state"]["deadline_s"] - winner["state"]["deadline_s"]) +
                                     abs(row["state"]["wan_mbps"] - winner["state"]["wan_mbps"]) / 1000,
                                     -row["gain"]))
    slack = next(row for row in candidates if row["pack"]["pack_id"] == winner["pack"]["pack_id"] and
                 row["state"]["wan_mbps"] == 10000 and row["state"]["deadline_s"] == 25)
    if len(nearby) < 3 or not slack["robust"]:
        raise ValueError("selected candidate lacks three robust neighbors or a robust slack control")
    selected = [winner, *nearby[:3]]
    if slack not in selected:
        selected.append(slack)
    pack = winner["pack"]
    from destination import dedicated_sink_architecture
    destination_type = dedicated_sink_architecture(profile, "destination", ("link",)).types[0]
    inputs["packs"] = [pack if row["pack_id"] == template["pack_id"] else row for row in inputs["packs"]]
    inputs["wan_mbps"] = sorted({row["state"]["wan_mbps"] for row in selected})
    for row in inputs["action_demands"]:
        if row["pack_id"] == template["pack_id"]:
            session = next(s for s in pack["sessions"] if s["session_id"] == row["session_id"])
            tokens, case = session["initial_tokens"], profile.case()
            replay = row["action"] == "replay"
            duration = tokens / case.replay.rate(tokens, 1) + case.replay_completion_s if replay else case.kv_transfer.setup_s + case.kv_transfer.initial_completion_s
            row.update(pack_id=pack["pack_id"], duration_s=duration + case.switch_s,
                       d_wan=2 * tokens if replay else case.kv_transfer.sealed_bytes(tokens),
                       d_prefill=duration if replay else 0, d_hbm=profile.kv_admission_tokens(tokens),
                       d_service=float(sum(destination_type.work(session["expected_f"],
                                       session["expected_g"], tokens))) * campaign.D_S)
    frozen = campaign.freeze_inputs(inputs, source_plan["frozen"]["seed"])
    states = [row["state"] for row in selected]
    slack["state"]["family"] = "slack"
    schedule = campaign.execution_schedule(states, [pack], frozen["seed"], repeats=3)
    schedule.sort(key=lambda row: (row["wan_mbps"], row["repeat"]))
    plan = {"frozen": frozen, "states": states, "schedule": schedule, "annotations": [],
            "calibration": calibration, "offline_decisions": [{"state_id": row["state"]["state_id"],
                "policy": policy, "moves": moves} for row in selected for policy, moves in row["moves"].items()],
            "design": {"repeats": 3, "selected_pack": pack["pack_id"], "episode_count": len(schedule),
                       "selection": "both QH variants attain full target; prefer attainment at all +/-15% duration sensitivities, then maximize relief gain under robust dominance; three nearest full-attainment robust neighbors and slack; all candidates retained",
                       "planner_profile": "unchanged", "metric": "full-target attainment primary; five-second windowed relief secondary"}}
    plan["sha256"] = campaign.digest(plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    campaign.write_csv(out / "execution_schedule.csv", schedule)
    return plan


def summarize(plan: dict, run_root: Path) -> list[dict]:
    campaign._verify_plan(plan)
    raw = [json.loads(line) for line in (run_root / "raw_episodes.jsonl").read_text().splitlines()]
    inputs, schedule = plan["frozen"]["inputs"], plan["schedule"]
    episodes = campaign.normalize_episodes(raw, schedule[:len(raw)], inputs["packs"], inputs["target"])
    packs = {row["pack_id"]: row for row in inputs["packs"]}
    paired = {}
    for row in episodes:
        decisions = json.loads(row["decisions"])
        completions = [(move["completion_s"], move["session_id"]) for move in decisions
                       if move.get("completion_s") is not None and not move.get("error")]
        row["window_relief_w"] = campaign.window_relief(packs[row["pack_id"]], completions, row["deadline_s"])
        paired[row["state_id"], row["repeat"], row["policy"]] = row
    comparisons = []
    for (state, repeat, policy), row in paired.items():
        if policy not in ("queue_haul", "greedy"):
            continue
        for baseline in campaign.POLICIES[2:]:
            other = paired.get((state, repeat, baseline))
            if other is not None:
                delta = row["window_relief_w"] - other["window_relief_w"]
                comparisons.append({"state_id": state, "repeat": repeat, "policy": policy,
                                    "baseline": baseline, "relief_delta_w": delta,
                                    "target_attained": row["target_attained"],
                                    "baseline_target_attained": other["target_attained"],
                                    "attainment_delta": int(row["target_attained"]) - int(other["target_attained"]),
                                    "result": "win" if delta > 1e-8 else "loss" if delta < -1e-8 else "tie"})
    campaign.write_csv(run_root / "episodes.csv", episodes)
    campaign.write_csv(run_root / "comparisons.csv", comparisons)
    counts = Counter(row["result"] for row in comparisons)
    validation = {"complete": len(raw) == len(schedule), "completed_episodes": len(raw),
                  "scheduled_episodes": len(schedule), "comparison_counts": dict(counts),
                  "three_repeats_per_case_policy": all(
                      sorted(row["repeat"] for row in episodes if row["state_id"] == state["state_id"]
                             and row["policy"] == policy) == [0, 1, 2]
                      for state in plan["states"] for policy in campaign.POLICIES),
                  "all_QH_no_losses": counts["loss"] == 0,
                  "no_attainment_losses": all(row["attainment_delta"] >= 0 for row in comparisons),
                  "strict_attainment_win_vs_per_session": all(any(row["policy"] == policy and
                      row["baseline"] == "per_session_greedy" and row["attainment_delta"] > 0
                      for row in comparisons) for policy in ("queue_haul", "greedy")),
                  "strict_win_vs_per_session": all(any(row["policy"] == policy and
                      row["baseline"] == "per_session_greedy" and row["result"] == "win"
                      for row in comparisons) for policy in ("queue_haul", "greedy"))}
    (run_root / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    if validation["complete"] and not all(validation[key] for key in
            ("three_repeats_per_case_policy", "no_attainment_losses", "strict_attainment_win_vs_per_session")):
        raise ValueError("live comparison did not meet the declared criterion; all outcomes retained")
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    for name in ("source-plan", "calibration-plan", "calibration-raw", "out"):
        prepare_parser.add_argument(f"--{name}", type=Path, required=True)
    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--plan", type=Path, required=True)
    summary_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(json.loads(args.source_plan.read_text()), json.loads(args.calibration_plan.read_text()),
                [json.loads(line) for line in args.calibration_raw.read_text().splitlines()], args.out)
    else:
        summarize(json.loads(args.plan.read_text()), args.run_root)


if __name__ == "__main__":
    main()
