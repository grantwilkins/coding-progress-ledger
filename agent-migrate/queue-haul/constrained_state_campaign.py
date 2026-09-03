"""Run the small two-A100 constrained-resource experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_style


D_S = 30
W_S = 5
REPEATS = 2
SCHEMA = "queue-haul-constrained-resource-v1"
AXES = ("prefill", "hbm", "serving")
RESOURCES = ("wan", "service", "prefill", "hbm")
ACTIONS = ("kv_transfer", "replay")
POLICIES = ("queue_haul", "greedy", "kv_only", "replay_only",
            "per_session_greedy")
ROOT = Path(__file__).parent
DEFAULT_TEMPLATE = ROOT / "outputs/policy-hardware-width8-packing-plan/plan.json"
DEFAULT_PROFILE = ROOT / "profiles/gpt_oss_20b_a100_tp1_crossover.json"
DEFAULT_MANIFEST = ROOT / "outputs/coding-manifest.json"
DEFAULT_BUNDLE = ROOT / "outputs/destination-v7-20260722/content-free-manifest.json"
DEFAULT_SERVICE = ROOT / "outputs/destination-v7-20260722/baseline-profile.json"


def digest(value) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _number(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid {name}")
    return value


def freeze_inputs(inputs: dict, seed: int) -> dict:
    """Freeze the packs, physics, shaped WAN grid, and background units."""
    required = {"packs", "profile", "action_demands", "target", "wan_mbps",
                "background_manifest"}
    if not required <= inputs.keys():
        raise ValueError(f"missing frozen inputs: {sorted(required - inputs.keys())}")
    packs = inputs["packs"]
    if len(packs) != 10 or any(len(row.get("sessions", ())) != 8 for row in packs):
        raise ValueError("campaign requires ten eight-session packs")
    if len({row["pack_id"] for row in packs}) != 10:
        raise ValueError("pack IDs must be unique")
    target, expected = _number(inputs["target"], "target"), set()
    for pack in packs:
        sessions = [row["session_id"] for row in pack["sessions"]]
        gains = [_number(value, "power gain") for value in pack["power_gains"]]
        if len(set(sessions)) != 8 or len(gains) != 256 or gains[0] != 0 \
                or not math.isclose(gains[-1], target):
            raise ValueError("invalid frozen pack")
        if any(gain + 1e-9 >= target for gain in gains[:-1]):
            raise ValueError("full-shed target must require all eight sessions")
        if any(gains[mask] > gains[mask | 1 << bit] + 1e-9
               for mask in range(256) for bit in range(8)
               if not mask & 1 << bit):
            raise ValueError("power gains must be monotone")
        expected |= {(pack["pack_id"], session, action)
                     for session in sessions for action in ACTIONS}
    demands = inputs["action_demands"]
    actual = {(row["pack_id"], row["session_id"], row["action"])
              for row in demands}
    if len(demands) != len(actual) or actual != expected:
        raise ValueError("action-demand table must contain both actions once")
    for row in demands:
        for name in ("duration_s", *(f"d_{resource}" for resource in RESOURCES)):
            _number(row[name], name)
    wan = [float(value) for value in inputs["wan_mbps"]]
    if not wan or wan != sorted(set(wan)) or min(wan) <= 0:
        raise ValueError("WAN settings must be unique positive shaped rates")
    if not isinstance(seed, int) or not isinstance(inputs["profile"], dict) \
            or not inputs["profile"] or not isinstance(
                inputs["background_manifest"], dict):
        raise ValueError("profile, background manifest, and integer seed are required")
    frozen = {"schema": SCHEMA,
              "constants": {"deadline_s": D_S, "power_window_s": W_S,
                            "repeats": REPEATS},
              "seed": seed, "inputs": inputs}
    frozen["sha256"] = digest(frozen)
    return frozen


def verify_frozen(frozen: dict) -> None:
    if freeze_inputs(frozen["inputs"], frozen["seed"]) != frozen:
        raise ValueError("frozen inputs changed")


def default_inputs(template_path: Path = DEFAULT_TEMPLATE,
                   profile_path: Path = DEFAULT_PROFILE) -> dict:
    """Reuse two frozen repeats of each existing width-eight context pack."""
    from profiles import ModelProfile

    template = json.loads(template_path.read_text())
    profile_raw = json.loads(profile_path.read_text())
    profile, packs = ModelProfile.load(profile_path), []
    case, load = profile.case(), .4 / 8
    by_pack = {}
    for row in template["scenarios"]:
        repeat = row.get("repeat", -1) % 3
        if row.get("kind") == "migration" and repeat in (0, 1) \
                and row.get("context_profile") in {
                    "tiny", "small", "medium", "mixed", "large"}:
            by_pack.setdefault((row["context_profile"], repeat),
                               row["sessions"])
    expected = {(name, repeat) for name in
                ("tiny", "small", "medium", "mixed", "large")
                for repeat in (0, 1)}
    if set(by_pack) != expected:
        raise ValueError("existing template lacks the ten fixed packs")
    for name, repeat in sorted(expected):
        sessions = [{**row, "expected_f": case.F * load, "expected_g": 0.0,
                     "log_bytes": 2 * int(row["initial_tokens"])}
                    for row in by_pack[name, repeat]]
        gains = [case.power(.4) - case.power(
            .4 - load * sum(bool(mask & 1 << bit) for bit in range(8)))
                 for mask in range(256)]
        packs.append({"pack_id": f"{name}-r{repeat}", "sessions": sessions,
                      "power_gains": gains})
    demands = []
    for pack in packs:
        for session in pack["sessions"]:
            tokens = int(session["initial_tokens"])
            for action in ACTIONS:
                replay = action == "replay"
                migration_s = (tokens / case.replay.rate(tokens, 1)
                               + case.replay_completion_s if replay else
                               case.kv_transfer.setup_s
                               + case.kv_transfer.initial_completion_s)
                demands.append({
                    "pack_id": pack["pack_id"],
                    "session_id": session["session_id"], "action": action,
                    "duration_s": migration_s + case.switch_s,
                    "d_wan": (2 * tokens if replay else
                              case.kv_transfer.sealed_bytes(tokens)),
                    "d_service": load * D_S,
                    "d_prefill": migration_s if replay else 0.0,
                    "d_hbm": profile.kv_admission_tokens(tokens),
                })
    return {
        "packs": packs, "profile": profile_raw, "action_demands": demands,
        "target": packs[0]["power_gains"][-1],
        "wan_mbps": [1000, 2500, 5000, 10000],
        "background_manifest": {
            "seed": 7,
            "live": {
                "manifest_path": str(DEFAULT_MANIFEST.resolve()),
                "model_profile_path": str(profile_path.resolve()),
                "bundle_path": str(DEFAULT_BUNDLE.resolve()),
                "service_profile_path": str(DEFAULT_SERVICE.resolve()),
                "prefill_unit_rps": 1.0, "serving_unit_rps": .25,
                "hbm_sessions_per_unit": 8,
                "max_units": {"prefill": 4, "serving": 10, "hbm": 10},
                "warmup_s": 30,
            },
        },
    }


def discovery_limits(rows: list[dict]) -> tuple[dict[str, int], list[dict]]:
    """Keep every consecutive operational rung on each physical axis."""
    limits, retained = {}, []
    for axis in AXES:
        selected = sorted((row for row in rows if row.get("axis") == axis),
                          key=lambda row: row["count"])
        if not selected or [row["count"] for row in selected] \
                != list(range(len(selected))) or "telemetry" not in selected[0]:
            raise ValueError(f"{axis} discovery must start at zero and be consecutive")
        stopped = False
        for row in selected:
            if "telemetry" not in row:
                raise ValueError("discovery telemetry is required")
            operational = bool(row["operational"])
            if operational and stopped:
                raise ValueError(f"{axis} became operational after stopping")
            stopped |= not operational
            if operational:
                retained.append(row)
        if not retained or not any(row["axis"] == axis for row in retained):
            raise ValueError(f"{axis} zero rung is not operational")
        limits[axis] = max(row["count"] for row in retained
                           if row["axis"] == axis)
    return limits, retained


def state_grid(limits: dict[str, int], wan_mbps) -> list[dict]:
    """Return WAN×prefill plus reference-WAN HBM and serving ladders."""
    if set(limits) != set(AXES) or min(limits.values()) < 0:
        raise ValueError("invalid discovered limits")
    wan = sorted({float(value) for value in wan_mbps})
    if not wan or min(wan) <= 0:
        raise ValueError("invalid shaped WAN grid")
    rows = [("wan_prefill", rate, prefill, 0, 0)
            for rate in wan for prefill in range(limits["prefill"] + 1)]
    rows += [("hbm", wan[-1], 0, count, 0)
             for count in range(1, limits["hbm"] + 1)]
    rows += [("serving", wan[-1], 0, 0, count)
             for count in range(1, limits["serving"] + 1)]
    return [{"state_id": digest(row)[:16], "family": row[0],
             "wan_mbps": row[1], "n_prefill": row[2], "n_hbm": row[3],
             "n_serving": row[4]} for row in rows]


def _demand_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    result = {(row["session_id"], row["action"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate action demand")
    return result


def _plans(pack: dict, rows: list[dict]):
    sessions = [row["session_id"] for row in pack["sessions"]]
    demand = _demand_map(rows)
    if set(demand) != {(session, action) for session in sessions
                       for action in ACTIONS}:
        raise ValueError("pack action demands are incomplete")
    for choices in itertools.product(range(3), repeat=len(sessions)):
        mask, usage, duration = 0, dict.fromkeys(RESOURCES, 0.0), 0.0
        for index, choice in enumerate(choices):
            if not choice:
                continue
            mask |= 1 << index
            row = demand[sessions[index], ACTIONS[choice - 1]]
            duration = max(duration, _number(row["duration_s"], "duration"))
            for resource in RESOURCES:
                usage[resource] += _number(row[f"d_{resource}"], resource)
        yield float(pack["power_gains"][mask]), usage, choices, duration


def _moves(pack: dict, choices) -> list[dict]:
    return [{"session_id": session["session_id"],
             "action": ACTIONS[choice - 1], "order": order}
            for order, (session, choice) in enumerate(zip(pack["sessions"], choices))
            if choice]


def _feasible(plan, capacity: dict[str, float], pack: dict, timing) -> bool:
    if plan[3] > D_S or any(plan[1][name] > capacity[name] + 1e-9
                            for name in RESOURCES):
        return False
    if timing is None:
        return True
    result = timing(pack, _moves(pack, plan[2]), capacity)
    return _number(result["makespan_s"], "makespan") <= D_S


def oracle_case(pack: dict, demand_rows: list[dict], capacity: dict[str, float],
                target: float, timing=None) -> dict:
    """Return only the four requested discrete feasibility annotations."""
    if set(capacity) != set(RESOURCES):
        raise ValueError("oracle capacity must contain four resources")
    capacity = {name: _number(value, name) for name, value in capacity.items()}
    feasible = [row for row in _plans(pack, demand_rows)
                if row[0] + 1e-9 >= target
                and _feasible(row, capacity, pack, timing)]
    pure = lambda choice: any(all(value in (0, choice) for value in row[2])
                              for row in feasible)
    kv, replay = pure(1), pure(2)
    mixed = any(1 in row[2] and 2 in row[2] for row in feasible)
    return {"any_full": bool(feasible), "kv_full": kv,
            "replay_full": replay,
            "mixed_full": bool(mixed and not kv and not replay)}


def oracle_annotations(states: list[dict], packs: list[dict],
                       demands: list[dict], target: float, timing=None) -> list[dict]:
    rows = []
    for state in states:
        capacity = state["capacity_inputs"]["resources"]
        for pack in packs:
            selected = [row for row in demands if row["pack_id"] == pack["pack_id"]]
            rows.append({"state_id": state["state_id"],
                         "pack_id": pack["pack_id"],
                         **oracle_case(pack, selected, capacity, target, timing)})
    return rows


def per_session_greedy(pack: dict, demand_rows: list[dict],
                       capacity: dict[str, float], timing=None) -> list[dict]:
    """Choose each session's fastest current-state action and dispatch all."""
    demand = _demand_map(demand_rows)
    output = []
    for session in pack["sessions"]:
        session_id = session["session_id"]
        def duration(action):
            if timing is None:
                return _number(demand[session_id, action]["duration_s"], "duration")
            return _number(timing(
                pack, [{"session_id": session_id, "action": action, "order": 0}],
                capacity)["makespan_s"], "makespan")
        action = min(ACTIONS, key=lambda value: (duration(value), value))
        output.append({"session_id": session_id, "action": action,
                       "dispatch": True})
    return output


def execution_schedule(states: list[dict], packs: list[dict], seed: int,
                       repeats: int = REPEATS) -> list[dict]:
    """Schedule every state-pack-policy case with randomized policy order."""
    if repeats < 1:
        raise ValueError("at least one repeat is required")
    rows = []
    for state in states:
        for pack in packs:
            for repeat in range(repeats):
                block = digest([state["state_id"], pack["pack_id"], repeat])[:16]
                policies = list(POLICIES)
                random.Random(int(digest([seed, block])[:16], 16)).shuffle(policies)
                rows.extend({"episode_id": digest([block, policy])[:16],
                             "block_id": block, "state_id": state["state_id"],
                             "pack_id": pack["pack_id"], "repeat": repeat,
                             "policy": policy, "policy_order": order,
                             "deadline_s": D_S,
                             **{name: state[name] for name in
                                ("family", "wan_mbps", "n_prefill",
                                 "n_hbm", "n_serving")}}
                            for order, policy in enumerate(policies))
    return rows


def _gain(pack: dict, selected: set[str]) -> float:
    mask = sum(1 << index for index, row in enumerate(pack["sessions"])
               if row["session_id"] in selected)
    return float(pack["power_gains"][mask])


def window_relief(pack: dict, completions: list[tuple[float, str]], t: float,
                  window_s: float = W_S) -> float:
    start, selected, total, cursor = t - window_s, set(), 0.0, t - window_s
    events = sorted((float(at), session) for at, session in completions)
    selected.update(session for at, session in events if at <= start)
    for at, session in events:
        if start < at < t:
            total += _gain(pack, selected) * (at - cursor)
            selected.add(session)
            cursor = at
    return (total + _gain(pack, selected) * (t - cursor)) / window_s


def target_time(pack: dict, completions: list[tuple[float, str]], target: float,
                deadline_s: float = D_S) -> float | None:
    points = sorted({0.0, float(deadline_s),
                     *(min(deadline_s, max(0.0, at + W_S))
                       for at, _ in completions)})
    return next((at for at in points
                 if window_relief(pack, completions, at) + 1e-9 >= target), None)


def normalize_episodes(raw: list[dict], schedule: list[dict], packs: list[dict],
                       target: float) -> list[dict]:
    expected = {row["episode_id"]: row for row in schedule}
    if len(expected) != len(schedule) or {row.get("episode_id") for row in raw} \
            != set(expected) or len(raw) != len(expected):
        raise ValueError("raw episodes do not match the schedule")
    by_pack = {row["pack_id"]: row for row in packs}
    output = []
    for row in raw:
        job, pack = expected[row["episode_id"]], by_pack[expected[row["episode_id"]]["pack_id"]]
        sessions = {item["session_id"] for item in pack["sessions"]}
        decisions = row["decisions"]
        ids = [item["session_id"] for item in decisions]
        if len(ids) != len(set(ids)) or not set(ids) <= sessions \
                or any(item["action"] not in ACTIONS for item in decisions):
            raise ValueError("invalid policy decisions")
        if job["policy"] == "kv_only" and any(
                item["action"] != "kv_transfer" for item in decisions) \
                or job["policy"] == "replay_only" and any(
                    item["action"] != "replay" for item in decisions):
            raise ValueError("restricted policy chose the wrong action")
        completions = []
        for item in decisions:
            value = item.get("completion_s")
            if value is not None:
                value = _number(value, "completion")
                if value <= D_S:
                    completions.append((value, item["session_id"]))
        moved = {session for _, session in completions}
        result = {**job,
                  "capacity_inputs": json.dumps(row["capacity_inputs"],
                                                sort_keys=True),
                  "decisions": json.dumps(decisions, sort_keys=True),
                  "completion_times": json.dumps(
                      {session: at for at, session in completions},
                      sort_keys=True),
                  "kv_count": sum(item["action"] == "kv_transfer"
                                  for item in decisions),
                  "replay_count": sum(item["action"] == "replay"
                                      for item in decisions),
                  "not_moved_count": len(sessions - set(ids)),
                  "achieved_relief_w": _gain(pack, moved),
                  "target_time_s": target_time(pack, completions, target)}
        result["target_attained"] = result["target_time_s"] is not None
        output.append(result)
    return sorted(output, key=lambda row: tuple(
        row[name] for name in ("state_id", "pack_id", "repeat", "policy_order")))


def _csv_value(value):
    return json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows({key: _csv_value(value) for key, value in row.items()}
                         for row in rows)


def _save(fig, out: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=plot_style.SAVE_DPI)
    plt.close(fig)


def plot_results(episodes: list[dict], out: Path) -> None:
    """Write only target attainment and Queue-Haul action composition."""
    plot_style.apply()
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=plot_style.COMPACT_FIGSIZE)
    for policy in POLICIES:
        rows = [row for row in episodes if row["policy"] == policy]
        if not rows:
            continue
        times = sorted(float(row["target_time_s"]) for row in rows
                       if row["target_time_s"] not in (None, ""))
        x, y = [0.0, *times, D_S], [0.0, *(
            (index + 1) / len(rows) for index in range(len(times))),
            len(times) / len(rows)]
        ax.step(x, y, where="post", **plot_style.policy_style(policy))
    ax.set(xlim=(0, D_S), ylim=(0, 1.01), xlabel="Time (s)",
           ylabel="Fraction attaining full shed")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, out, "target_attainment")

    policies = ("queue_haul", "greedy")
    families = ("wan_prefill", "hbm", "serving")
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), squeeze=False)
    for row_index, policy in enumerate(policies):
        for column, family in enumerate(families):
            ax = axes[row_index][column]
            selected = [row for row in episodes
                        if row["policy"] == policy and row["family"] == family]
            keys = sorted({(row["state_id"], row["wan_mbps"], row["n_prefill"],
                            row["n_hbm"], row["n_serving"]) for row in selected},
                          key=lambda value: value[1:])
            bottoms = [0.0] * len(keys)
            for action, field in (("kv_transfer", "kv_count"),
                                  ("replay", "replay_count"),
                                  ("not_moved", "not_moved_count")):
                values = []
                for key in keys:
                    group = [row for row in selected if row["state_id"] == key[0]]
                    total = sum(row["kv_count"] + row["replay_count"]
                                + row["not_moved_count"] for row in group)
                    values.append(sum(row[field] for row in group) / total)
                ax.bar(range(len(keys)), values, bottom=bottoms,
                       color=plot_style.ACTION_COLORS[action],
                       label=plot_style.ACTION_NAMES[action])
                bottoms = [a + b for a, b in zip(bottoms, values)]
            labels = ([f"{key[1] / 1000:g}G/P{key[2]}" for key in keys]
                      if family == "wan_prefill" else
                      [str(key[3] if family == "hbm" else key[4]) for key in keys])
            ax.set_xticks(range(len(keys)), labels, rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(f"{plot_style.POLICY_NAMES[policy]} — "
                         f"{family.replace('_', ' ')}")
            if column == 0:
                ax.set_ylabel("Action share")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=.88)
    _save(fig, out, "action_composition")


def compile_campaign(frozen: dict, discovery: list[dict],
                     measured_states: list[dict], out: Path, timing=None) -> dict:
    """Compile every discovered state without selection or weighting."""
    verify_frozen(frozen)
    limits, retained = discovery_limits(discovery)
    expected = state_grid(limits, frozen["inputs"]["wan_mbps"])
    measured = {row["state_id"]: row for row in measured_states}
    if len(measured) != len(measured_states) or set(measured) != {
            row["state_id"] for row in expected}:
        raise ValueError("measured states do not match the generated grid")
    states = []
    for identity in expected:
        row = measured[identity["state_id"]]
        if not row.get("operational") or any(row.get(name) != value
                for name, value in identity.items()):
            raise ValueError("generated state is not operational or changed")
        resources = row.get("capacity_inputs", {}).get("resources", {})
        if set(resources) != set(RESOURCES):
            raise ValueError("state lacks Queue-Haul capacity inputs")
        states.append(row)
    inputs = frozen["inputs"]
    annotations = oracle_annotations(
        states, inputs["packs"], inputs["action_demands"], inputs["target"], timing)
    schedule = execution_schedule(states, inputs["packs"], frozen["seed"])
    plan = {"frozen": frozen, "states": states, "annotations": annotations,
            "schedule": schedule}
    plan["sha256"] = digest(plan)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    write_csv(out / "background_discovery.csv", retained)
    write_csv(out / "background_states.csv", states)
    write_csv(out / "oracle_annotations.csv", annotations)
    write_csv(out / "execution_schedule.csv", schedule)
    return plan


def _verify_plan(plan: dict) -> None:
    verify_frozen(plan["frozen"])
    expected = digest({key: value for key, value in plan.items() if key != "sha256"})
    if plan.get("sha256") not in (None, expected):
        raise ValueError("compiled plan changed")


def reduce_campaign(plan: dict, raw: list[dict], out: Path) -> list[dict]:
    _verify_plan(plan)
    episodes = normalize_episodes(raw, plan["schedule"],
                                  plan["frozen"]["inputs"]["packs"],
                                  plan["frozen"]["inputs"]["target"])
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "episodes.csv", episodes)
    plot_results(episodes, out)
    return episodes


def run_live(plan: dict, run_root: Path, episode_runner=None) -> list[dict]:
    """Execute the frozen order with a fresh, fixed background per policy."""
    _verify_plan(plan)
    run_root.mkdir(parents=True, exist_ok=True)
    raw_path = run_root / "raw_episodes.jsonl"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    inputs = plan["frozen"]["inputs"]
    states = {row["state_id"]: row for row in plan["states"]}
    packs = {row["pack_id"]: row for row in inputs["packs"]}
    runner, raw = episode_runner or a100_episode, []
    with raw_path.open("x", buffering=1) as handle:
        for job in plan["schedule"]:
            root = run_root / "scenarios" / job["episode_id"]
            result = runner(inputs, states[job["state_id"]],
                            packs[job["pack_id"]], job, root)
            row = {"episode_id": job["episode_id"], **result}
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            raw.append(row)
    reduce_campaign(plan, raw, run_root)
    return raw


class BackgroundLimit(RuntimeError):
    """The configured background rung cannot be held."""


def _paths(inputs: dict) -> dict:
    live = inputs["background_manifest"].get("live")
    required = {"manifest_path", "model_profile_path", "bundle_path",
                "service_profile_path", "prefill_unit_rps", "serving_unit_rps",
                "hbm_sessions_per_unit", "max_units"}
    if not isinstance(live, dict) or not required <= live.keys() \
            or set(live["max_units"]) != set(AXES):
        raise ValueError("live background manifest is incomplete")
    return live


def _metrics(testbed, destination, cfg) -> dict:
    rows = [destination.parse_metrics(testbed.http_text(
        cfg.host, cfg.sink_port, "GET", "/metrics")) for _ in range(5)]
    return {name: statistics.median(row[name] for row in rows)
            for name in ("vllm:gpu_cache_usage_perc",
                         "vllm:num_requests_running",
                         "vllm:num_requests_waiting")}


def _resident_telemetry(testbed, destination, cfg, minimum=0.0,
                        tolerance=0.0) -> dict:
    metrics = _metrics(testbed, destination, cfg)
    usage = metrics["vllm:gpu_cache_usage_perc"]
    if usage <= 0:
        raise BackgroundLimit("resident sessions did not consume visible HBM")
    if usage + tolerance < minimum:
        raise BackgroundLimit("resident-session HBM use decreased")
    return metrics


def _validate_serving(load, destination) -> None:
    if load.failure or load.blocked_arrivals or not load.rows or any(
            not destination.service_completion(row) for row in load.rows):
        raise RuntimeError("serving background was not maintained")


def _wait_background(load, kind: str, warmup_s: float, destination) -> None:
    if kind == "prefill":
        load.wait_ready()
    else:
        time.sleep(warmup_s)
        _validate_serving(load, destination)


@contextmanager
def _a100_background(inputs: dict, state: dict, root: Path):
    """Reuse the existing stack, load generator, prefix warming, and telemetry."""
    import destination_runner as destination
    import migration_testbed as testbed
    from profiles import ModelProfile

    live, cfg = _paths(inputs), testbed.Config()
    profile = ModelProfile.load(Path(live["model_profile_path"]))
    if json.loads(Path(live["model_profile_path"]).read_text()) != inputs["profile"]:
        raise ValueError("live model profile differs from the frozen profile")
    manifest = json.loads(Path(live["manifest_path"]).read_text())
    bundle = json.loads(Path(live["bundle_path"]).read_text())
    service = json.loads(Path(live["service_profile_path"]).read_text())
    stack = testbed.start_stack(cfg, root / "testbed", state["wan_mbps"], [])
    testbed.start_sink(stack, cfg, [])
    load = None
    try:
        testbed.flush_lmcache(stack, cfg)
        testbed.reset_vllm_caches(
            cfg, (stack.run_root / "source.log", stack.run_root / "sink.log"))
        hbm, resident = int(state["n_hbm"]), []
        if hbm:
            pool = sum((destination.manifest_sessions(
                bundle, job, split, 201088, inputs["background_manifest"]["seed"])
                for job, splits in bundle["manifest"]["splits"].items()
                for split in splits), [])
            needed = hbm * int(live["hbm_sessions_per_unit"])
            if needed > len(pool):
                raise BackgroundLimit("resident-session groups exhausted")
            resident = pool[:needed]
            try:
                destination.prewarm(cfg.host, cfg.sink_port, cfg.model,
                                    resident, bypass_lmcache=True)
            except RuntimeError as error:
                raise BackgroundLimit(str(error)) from error
        kind = "prefill" if state["n_prefill"] else (
            "serving" if state["n_serving"] else None)
        rps = 0.0
        if kind:
            sessions = destination.manifest_sessions(
                bundle, "agentic_tool_loop", "validation", 201088,
                inputs["background_manifest"]["seed"])
            if kind == "prefill":
                sessions = [replace(row, prefix_tokens=1, append_tokens=2048,
                                    output_tokens=32) for row in sessions]
            context = round(statistics.mean(
                row.prefix_tokens + row.append_tokens for row in sessions))
            rates = (destination.profile_rate(service, "prefill", context),
                     destination.profile_rate(service, "decode", context))
            count = int(state[f"n_{kind}"])
            rps = count * float(live[f"{kind}_unit_rps"])
            rho = rps * statistics.mean(
                row.append_tokens / rates[0] + row.output_tokens / rates[1]
                for row in sessions)
            load = destination.DestinationLoad(
                cfg.host, cfg.sink_port, cfg.model, sessions, rho, *rates,
                root / f"{kind}_background", 1000 + count,
                rps=rps, max_inflight=256, bypass_lmcache=True)
            load.start()
            try:
                _wait_background(
                    load, kind, float(live.get("warmup_s", 30)), destination)
            except RuntimeError as error:
                raise BackgroundLimit(str(error)) from error
        else:
            time.sleep(float(live.get("warmup_s", 30)))
            rates, sessions = None, []
        metrics = (_resident_telemetry(testbed, destination, cfg)
                   if resident else _metrics(testbed, destination, cfg))
        if metrics["vllm:num_requests_waiting"] > 0:
            raise BackgroundLimit("destination background queue is not stable")
        baseline = [0.0, 0.0]
        if load:
            work = [statistics.mean(getattr(row, field) / rate
                                    for row in sessions)
                    for field, rate in zip(("append_tokens", "output_tokens"), rates)]
            achieved = load.achieved if load.achieved is not None else load.target
            baseline = [achieved * value / sum(work) for value in work]
        kv = round(metrics["vllm:gpu_cache_usage_perc"]
                   * profile.kv_capacity_tokens)
        capacity = {
            "wan_mbps": float(state["wan_mbps"]),
            "background_kind": kind or "none", "background_rps": rps,
            "baseline_work": baseline, "baseline_kv_tokens": kv,
            "telemetry": metrics,
            "resources": {
                "wan": float(state["wan_mbps"]) * 125_000 * D_S,
                "service": max(0.0, 1 - sum(baseline)) * D_S,
                "prefill": max(0.0, 1 - baseline[0]) * D_S,
                "hbm": max(0, profile.kv_capacity_tokens - kv),
            },
        }
        yield stack, cfg, profile, manifest, capacity
        if resident:
            post = _resident_telemetry(
                testbed, destination, cfg,
                metrics["vllm:gpu_cache_usage_perc"],
                16 / profile.kv_capacity_tokens)
            if post["vllm:num_requests_waiting"] > 0:
                raise BackgroundLimit("destination background queue is not stable")
            capacity["post_hbm_telemetry"] = post
        elif kind == "serving":
            try:
                _validate_serving(load, destination)
            except RuntimeError as error:
                raise BackgroundLimit(str(error)) from error
            post = _metrics(testbed, destination, cfg)
            if post["vllm:num_requests_waiting"] > 0:
                raise BackgroundLimit("destination background queue is not stable")
            capacity["post_serving_telemetry"] = post
    finally:
        try:
            if load:
                load.close()
        finally:
            testbed.stop_stack(stack)


def measure_a100_background(inputs: dict, state: dict, root: Path) -> dict:
    with _a100_background(inputs, state, root) as context:
        capacity = context[-1]
        return {**state, "operational": True, "capacity_inputs": capacity}


def discover_live(frozen: dict, out: Path, measurement=None) -> tuple[list[dict], list[dict]]:
    """Discover integer rungs, then measure the complete generated state grid."""
    verify_frozen(frozen)
    inputs, measure = frozen["inputs"], measurement or measure_a100_background
    live, reference = _paths(inputs), max(inputs["wan_mbps"])
    discovery = []
    for axis in AXES:
        cap = int(live["max_units"][axis])
        for count in range(cap + 1):
            state = {"state_id": f"discover-{axis}-{count}", "family": axis,
                     "wan_mbps": reference, "n_prefill": 0, "n_hbm": 0,
                     "n_serving": 0}
            state[f"n_{axis}"] = count
            try:
                row = measure(inputs, state, out / "discovery" / axis / str(count))
                discovery.append({"axis": axis, "count": count,
                                  "operational": True,
                                  "telemetry": row["capacity_inputs"]})
            except BackgroundLimit as error:
                discovery.append({"axis": axis, "count": count,
                                  "operational": False,
                                  "telemetry": {"error": str(error)}})
                break
        if discovery[-1]["operational"]:
            raise BackgroundLimit(
                f"{axis} discovery reached max_units={cap} while operational")
    limits, _ = discovery_limits(discovery)
    states = [measure(inputs, state, out / "states" / state["state_id"])
              for state in state_grid(limits, inputs["wan_mbps"])]
    return discovery, states


def _planner_decisions(inputs: dict, state: dict, pack: dict, policy: str,
                       profile, capacity: dict) -> list[dict]:
    from destination import dedicated_sink_architecture
    from planner import plan, source_power
    from simulate import (ExecutionScenario, NetworkLink, PowerNode,
                          ServingInstance, SimSession)

    case, n = profile.case(), len(pack["sessions"])
    sessions = tuple(SimSession(
        row["session_id"], "source", int(row["initial_tokens"]),
        float(row.get("expected_f", 512 / n)),
        float(row.get("expected_g", 8 / n)), 2 * int(row["initial_tokens"]))
        for row in pack["sessions"])
    problem = ExecutionScenario(
        D_S, D_S, 0, "awake", 0,
        (PowerNode("source-node", 1, True),
         PowerNode("destination-node", 1, False)),
        (ServingInstance("source", ("source-node",)),
         ServingInstance("destination", ("destination-node",))), sessions,
        (NetworkLink("link", float(state["wan_mbps"]) * 125_000),))
    initial = source_power(problem, profile)
    problem = replace(problem, power_limit_w=initial - float(inputs["target"]))
    architecture = dedicated_sink_architecture(profile, "destination", ("link",))
    pool = architecture.pools[0]
    replica = replace(pool.replicas[0],
                      baseline_work=tuple(capacity["baseline_work"]),
                      baseline_kv_tokens=int(capacity["baseline_kv_tokens"]))
    architecture = replace(architecture,
                           pools=(replace(pool, replicas=(replica,)),),
                           residency_horizon_s=D_S)
    routes = {("source", "destination"): ("link",)}
    solvers = {"queue_haul": "lp_work_first", "greedy": "greedy",
               "kv_only": "kv_only", "replay_only": "replay_only"}
    if policy != "per_session_greedy":
        moves = plan(problem, profile, routes, solvers[policy],
                     destination=architecture).moves
        return [{"session_id": row.session_id, "action": row.method,
                 "order": row.order} for row in moves]
    choices = []
    for session in sessions:
        durations = {}
        for action, solver in (("kv_transfer", "kv_only"),
                               ("replay", "replay_only")):
            single = replace(problem, sessions=(session,))
            result = plan(single, profile, routes, solver,
                          destination=architecture)
            durations[action] = (result.predicted_migration_makespan_s
                                 if result.moves and
                                 result.predicted_migration_makespan_s is not None
                                 else math.inf)
        choices.append((session.session_id,
                        min(ACTIONS, key=lambda action: (durations[action], action))))
    return [{"session_id": session, "action": action, "order": order}
            for order, (session, action) in enumerate(choices)]


def a100_episode(inputs: dict, state: dict, pack: dict, job: dict,
                 root: Path) -> dict:
    """Create one physical background, plan once, and run one policy."""
    import migration_profiler as profiler

    root.mkdir(parents=True, exist_ok=False)
    with _a100_background(inputs, state, root) as (
            stack, cfg, profile, manifest, capacity):
        decisions = _planner_decisions(
            inputs, state, pack, job["policy"], profile, capacity)
        if not decisions:
            time.sleep(D_S)
            return {"capacity_inputs": capacity, "decisions": []}
        sessions = pack["sessions"]
        move_rows = [{**next(row for row in sessions
                             if row["session_id"] == move["session_id"]),
                      "method": move["action"], "order": move["order"]}
                     for move in decisions]
        scenario = {
            "scenario_id": job["episode_id"], "kind": "migration",
            "method": move_rows[0]["method"] if len({row["method"]
                     for row in move_rows}) == 1 else "mixed",
            "activity": "none", "request_schedule": [],
            "repeat": job["repeat"], "deadline_s": 180,
            "required_deadline_s": D_S,
            "sessions": sessions, "moves": move_rows,
            "serving_concurrency": 1, "concurrency": len(sessions),
            "move_concurrency": len(sessions), "copy_policy": "initial_final",
            "final_state": "awake", "bandwidth_mbps": state["wan_mbps"],
            "allow_partial_moves": True, "reset_caches": False,
        }
        result = profiler.run_scenario(
            stack, cfg, manifest, scenario, root, job["episode_id"],
            configure_proxy=False)
        completed = {row["move"]["session_id"]:
                     (row["switch_end_ns"] - result["started_ns"]) / 1e9
                     for row in result["migrations"]}
        return {"capacity_inputs": capacity,
                "decisions": [{**row,
                               "completion_s": completed.get(row["session_id"])}
                              for row in decisions]}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inputs = commands.add_parser("inputs")
    inputs.add_argument("--out", type=Path, required=True)
    inputs.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    inputs.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--inputs", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=1)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "inputs":
        with args.out.open("x") as handle:
            json.dump(default_inputs(args.template, args.profile), handle,
                      indent=2, sort_keys=True)
            handle.write("\n")
    elif args.command == "prepare":
        frozen = freeze_inputs(json.loads(args.inputs.read_text()), args.seed)
        discovery, states = discover_live(frozen, args.out)
        compile_campaign(frozen, discovery, states, args.out)
    else:
        run_live(json.loads(args.plan.read_text()), args.run_root)


if __name__ == "__main__":
    main()
