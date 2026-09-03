"""Compile and reduce the frozen two-A100 constrained-state campaign."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path


D_S = 30
W_S = 5
T_MIG_S = 25
SOBOL_POINTS = 128
STATE_QUOTA = 6
REPEATS = 2
SCHEMA = "queue-haul-constrained-state-v1"
AXES = ("hbm", "serving", "prefill")
RESOURCES = ("wan", "service", "prefill", "hbm")
ACTIONS = ("kv_transfer", "replay")
WAN_SETTINGS = ("natural", "controlled_80", "controlled_40")
POLICIES = ("queue_haul", "greedy", "kv_only", "replay_only",
            "per_session_greedy")
ORACLE_CLASSES = ("Slack", "KV-only feasible", "Replay-only feasible",
                  "Mixed actions required", "Full drain infeasible")


def digest(value) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def _number(value, name: str, nonnegative: bool = True) -> float:
    value = float(value)
    if not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError(f"invalid {name}")
    return value


def freeze_inputs(inputs: dict, seeds: dict[str, int]) -> dict:
    """Validate and hash the complete immutable campaign contract."""
    required = {"packs", "profile", "action_demands", "target",
                "wan_settings", "background_manifest"}
    if not required <= inputs.keys():
        raise ValueError(f"missing frozen inputs: {sorted(required - inputs.keys())}")
    packs = inputs["packs"]
    if len(packs) != 10 or any(len(pack.get("sessions", ())) != 8
                               for pack in packs):
        raise ValueError("campaign requires ten eight-session packs")
    pack_ids = [pack["pack_id"] for pack in packs]
    if len(set(pack_ids)) != 10:
        raise ValueError("pack IDs must be unique")
    target = _number(inputs["target"], "target")
    expected = set()
    for pack in packs:
        sessions = [row["session_id"] for row in pack["sessions"]]
        gains = [_number(gain, "power gain")
                 for gain in pack.get("power_gains", ())]
        if len(set(sessions)) != 8 or len(gains) != 256 \
                or gains[0] != 0 or not math.isclose(gains[-1], target):
            raise ValueError("invalid frozen pack")
        expected |= {(pack["pack_id"], session, action)
                     for session in sessions for action in ACTIONS}
    demands = inputs["action_demands"]
    actual = {(row["pack_id"], row["session_id"], row["action"])
              for row in demands}
    if len(actual) != len(demands) or actual != expected:
        raise ValueError("action-demand table must contain both actions exactly once")
    for row in demands:
        for name in ("duration_s", *(f"d_{resource}"
                                      for resource in RESOURCES)):
            _number(row[name], name)
    if set(inputs["wan_settings"]) != set(WAN_SETTINGS):
        raise ValueError("WAN settings must be natural, controlled_80, controlled_40")
    if not isinstance(inputs["profile"], dict) or not inputs["profile"] \
            or not isinstance(inputs["background_manifest"], dict) \
            or "seed" not in inputs["background_manifest"]:
        raise ValueError("profile and seeded background manifest are required")
    if not seeds or any(not isinstance(value, int) for value in seeds.values()):
        raise ValueError("integer campaign seeds are required")
    artifacts = {**inputs, "policies": POLICIES}
    frozen = {
        "schema": SCHEMA,
        "constants": {"D_s": D_S, "W_s": W_S, "T_mig_s": T_MIG_S},
        "seeds": dict(sorted(seeds.items())),
        "hashes": {name: digest(value) for name, value in artifacts.items()},
        "inputs": inputs,
    }
    frozen["inputs_hash"] = digest({
        key: value for key, value in frozen.items() if key != "inputs"})
    return frozen


def verify_frozen(frozen: dict) -> None:
    if freeze_inputs(frozen["inputs"], frozen["seeds"]) != frozen:
        raise ValueError("frozen input hash mismatch")


def _pack_hash(pack: dict) -> str:
    return pack["sha256"] if "sha256" in pack else digest(pack)


def _valid_axis_row(axis: str, row: dict, previous_throughput: float) -> bool:
    if axis == "hbm":
        return not _bool(row["rejected"]) and not _bool(row["evicted"]) \
            and _bool(row["persistently_resident"])
    if axis == "serving":
        return _bool(row["slo_met"]) \
            and _number(row["queue_growth"], "queue growth", False) <= 0
    return _number(row["queue_growth"], "queue growth", False) <= 0 \
        and _number(row["completed_throughput"], "throughput") \
        > previous_throughput


def discovery_limits(rows: list[dict]) -> tuple[dict[str, int], list[dict]]:
    """Validate each one-unit axis through its first invalid measurement."""
    retained, limits = [], {}
    for axis in AXES:
        selected = sorted((row for row in rows if row["axis"] == axis),
                          key=lambda row: row["count"])
        if [row["count"] for row in selected] != list(range(1, len(selected) + 1)):
            raise ValueError(f"{axis} discovery counts must be consecutive")
        previous = 0.0
        for index, row in enumerate(selected):
            if _number(row["warmup_s"], "warmup") != 30 \
                    or _number(row["measurement_s"], "measurement") != 30:
                raise ValueError("discovery requires 30 s warmup and measurement")
            valid = _valid_axis_row(axis, row, previous)
            retained.append({**row, "valid": valid})
            if not valid:
                if index != len(selected) - 1:
                    raise ValueError(f"{axis} includes states after first invalid")
                limits[axis] = int(row["count"]) - 1
                break
            if axis == "prefill":
                previous = _number(row["completed_throughput"], "throughput")
        else:
            raise ValueError(f"{axis} discovery is missing its first invalid point")
    return limits, retained


def candidate_counts(limits: dict[str, int], seed: int) -> list[tuple[int, int, int]]:
    """Return all pure axes plus the rounded, de-duplicated Sobol sample."""
    if set(limits) != set(AXES) or min(limits.values()) < 0:
        raise ValueError("invalid background limits")
    from scipy.stats import qmc

    bounds = [limits[name] for name in AXES]
    points = qmc.Sobol(3, scramble=True, seed=seed).random_base2(
        int(math.log2(SOBOL_POINTS)))
    counts = {
        tuple(min(bound, int(value * bound + .5))
              for value, bound in zip(point, bounds))
        for point in points
    }
    counts |= {(value, 0, 0) for value in range(bounds[0] + 1)}
    counts |= {(0, value, 0) for value in range(bounds[1] + 1)}
    counts |= {(0, 0, value) for value in range(bounds[2] + 1)}
    return sorted(counts)


def candidate_states(limits: dict[str, int], seed: int,
                     wan_settings=WAN_SETTINGS) -> list[dict]:
    return [{
        "state_id": digest([counts, wan])[:16],
        **dict(zip(("n_hbm", "n_serving", "n_prefill"), counts)),
        "wan_setting": wan,
    } for counts in candidate_counts(limits, seed) for wan in wan_settings]


def _demand_map(rows: list[dict]) -> dict[tuple[str, str], dict]:
    table = {(row["session_id"], row["action"]): row for row in rows}
    if len(table) != len(rows):
        raise ValueError("duplicate action demand")
    for row in rows:
        if row["action"] not in ACTIONS:
            raise ValueError("unknown action demand")
        for name in ("duration_s", *(f"d_{resource}"
                                      for resource in RESOURCES)):
            _number(row[name], name)
    return table


def _plans(pack: dict, rows: list[dict]):
    sessions = [row["session_id"] for row in pack["sessions"]]
    gains = [_number(gain, "power gain") for gain in pack["power_gains"]]
    demand = _demand_map(rows)
    if len(set(sessions)) != len(sessions) \
            or set(demand) != {(session, action) for session in sessions
                               for action in ACTIONS} \
            or len(gains) != 1 << len(sessions) or gains[0] != 0:
        raise ValueError("pack needs one nonlinear power gain per session subset")
    plans = []
    for choices in itertools.product(range(3), repeat=len(sessions)):
        mask, usage, valid = 0, dict.fromkeys(RESOURCES, 0.0), True
        for index, choice in enumerate(choices):
            if not choice:
                continue
            mask |= 1 << index
            row = demand[sessions[index], ACTIONS[choice - 1]]
            valid &= _number(row["duration_s"], "duration") <= T_MIG_S
            for resource in RESOURCES:
                usage[resource] += _number(row[f"d_{resource}"], resource)
        if valid:
            plans.append((float(gains[mask]), usage, choices))
    return plans


def _best(plans, capacity, action=None):
    index = None if action is None else ACTIONS.index(action) + 1
    feasible = [plan for plan in plans
                if (index is None or all(choice in (0, index)
                                         for choice in plan[2]))
                and all(plan[1][resource] <= float(capacity[resource]) + 1e-9
                        for resource in RESOURCES)]
    return max(feasible, key=lambda plan: plan[0])


def _oracle_class(j: bool, k: bool, r: bool) -> str:
    if not j:
        return ORACLE_CLASSES[4]
    if k and r:
        return ORACLE_CLASSES[0]
    if k:
        return ORACLE_CLASSES[1]
    if r:
        return ORACLE_CLASSES[2]
    return ORACLE_CLASSES[3]


def oracle_case(pack: dict, demand_rows: list[dict], capacity: dict[str, float],
                target: float, empty_capacity: dict[str, float],
                _compiled_plans=None) -> dict:
    """Exhaustively census one eight-session pack in one measured state."""
    if set(capacity) != set(RESOURCES) or set(empty_capacity) != set(RESOURCES):
        raise ValueError("invalid oracle capacities or target")
    capacity = {name: _number(value, name) for name, value in capacity.items()}
    empty_capacity = {name: _number(value, name)
                      for name, value in empty_capacity.items()}
    target = _number(target, "target")
    plans = (_plans(pack, demand_rows) if _compiled_plans is None
             else _compiled_plans)
    best = _best(plans, capacity)
    restricted = {action: _best(plans, capacity, action)[0]
                  for action in ACTIONS}
    p_star = best[0]
    j, k, r = (p_star + 1e-9 >= target,
               restricted["kv_transfer"] + 1e-9 >= target,
               restricted["replay"] + 1e-9 >= target)
    result = {"p_star": p_star, "j": j, "k": k, "r": r,
              "oracle_class": _oracle_class(j, k, r)}
    if j:
        target_plans = [plan for plan in plans if plan[0] + 1e-9 >= target]
        for resource in RESOURCES:
            eligible = [plan for plan in target_plans if all(
                other == resource
                or plan[1][other] <= float(capacity[other]) + 1e-9
                for other in RESOURCES)]
            required = min(plan[1][resource] for plan in eligible)
            result[f"margin_{resource}"] = (
                float(capacity[resource]) - required if required > 1e-9 else None)
            result[f"delta_{resource}"] = None
    else:
        for resource in RESOURCES:
            relaxed = {**capacity, resource: empty_capacity[resource]}
            result[f"delta_{resource}"] = max(0.0,
                                                _best(plans, relaxed)[0] - p_star)
            result[f"margin_{resource}"] = None
    labels = ([resource for resource in RESOURCES
               if result[f"delta_{resource}"] > 1e-9] if not j else
              [resource for resource in RESOURCES
               if result[f"margin_{resource}"] is not None
               and result[f"margin_{resource}"] <= min(
                   value for value in (result[f"margin_{name}"]
                                       for name in RESOURCES)
                   if value is not None) + 1e-9])
    result["resource_labels"] = json.dumps(labels or ["joint"])
    result["oracle_actions"] = json.dumps([
        {"session_id": session["session_id"], "action": ACTIONS[choice - 1]}
        for session, choice in zip(pack["sessions"], best[2]) if choice
    ], separators=(",", ":"))
    return result


def per_session_greedy(pack: dict, demand_rows: list[dict]) -> list[dict]:
    """Choose every session's fastest action without aggregate admission."""
    demand = _demand_map(demand_rows)
    return [{"session_id": row["session_id"],
             "action": min(ACTIONS, key=lambda action: (
                 _number(demand[row["session_id"], action]["duration_s"],
                         "duration"), action)),
             "dispatch": True} for row in pack["sessions"]]


def select_states(census: list[dict], seed: int,
                  quota: int = STATE_QUOTA) -> dict:
    """Sample background IDs uniformly within each observed case class."""
    hashes = {}
    for row in census:
        previous = hashes.setdefault(row["state_id"], row["state_hash"])
        if previous != row["state_hash"]:
            raise ValueError("state hash changed across packs")
    rng, draws, shortages = random.Random(seed), {}, {}
    histogram = Counter(row["oracle_class"] for row in census)
    for name in ORACLE_CLASSES:
        population = sorted({row["state_id"] for row in census
                             if row["oracle_class"] == name})
        draws[name] = rng.sample(population, min(quota, len(population)))
        if len(population) < quota:
            shortages[name] = quota - len(population)
    chosen = sorted(set(itertools.chain.from_iterable(draws.values())))
    return {
        "seed": seed, "quota_per_class": quota, "draws": draws,
        "shortages": shortages, "census_sha256": digest(census),
        "candidate_class_histogram": {
            name: histogram[name] for name in ORACLE_CLASSES},
        "states": [{"state_id": state, "sha256": hashes[state]}
                   for state in chosen],
    }


def execution_schedule(selected: dict, packs: list[dict], seed: int,
                       inputs_hash: str) -> list[dict]:
    """Cross frozen states with ten packs and randomize each matched block."""
    rows = []
    for state in selected["states"]:
        for pack in packs:
            for repeat in range(REPEATS):
                block_id = digest([state["state_id"], pack["pack_id"], repeat])[:16]
                policies = list(POLICIES)
                random.Random(int(digest([seed, block_id])[:16], 16)).shuffle(policies)
                rows.extend({
                    "episode_id": digest([block_id, policy])[:16],
                    "block_id": block_id, "state_id": state["state_id"],
                    "state_hash": state["sha256"], "pack_id": pack["pack_id"],
                    "pack_hash": _pack_hash(pack), "repeat": repeat,
                    "policy": policy, "policy_order": order,
                    "deadline_s": D_S, "power_window_s": W_S,
                    "migration_window_s": T_MIG_S,
                    "recreate_background": True, "verify_residual": True,
                    "inputs_hash": inputs_hash,
                } for order, policy in enumerate(policies))
    return rows


def _gain(pack: dict, selected: set[str]) -> float:
    mask = sum(1 << index for index, row in enumerate(pack["sessions"])
               if row["session_id"] in selected)
    return float(pack["power_gains"][mask])


def window_relief(pack: dict, completions: list[tuple[float, str]], t: float,
                  window_s: float = W_S) -> float:
    """Time-average nonlinear subset relief over the trailing window."""
    start, selected, total = t - window_s, set(), 0.0
    events = sorted((float(at), session) for at, session in completions)
    selected.update(session for at, session in events if at <= start)
    cursor = start
    for at, session in events:
        if at <= start or at >= t:
            continue
        total += _gain(pack, selected) * (at - cursor)
        selected.add(session)
        cursor = at
    return (total + _gain(pack, selected) * (t - cursor)) / window_s


def target_time(pack: dict, completions: list[tuple[float, str]], target: float,
                horizon_s: float = D_S) -> float | None:
    """Return the first trailing-window target crossing within the deadline."""
    points = sorted({0.0, float(horizon_s),
                     *(max(0.0, float(at)) for at, _ in completions),
                     *(min(float(horizon_s), float(at) + W_S)
                       for at, _ in completions)})
    points = [point for point in points if 0 <= point <= horizon_s]
    for left, right in zip(points, points[1:]):
        before, after = (window_relief(pack, completions, value)
                         for value in (left, right))
        if before + 1e-9 >= target:
            return left
        if after + 1e-9 >= target and after > before:
            return left + (target - before) * (right - left) / (after - before)
    return horizon_s if window_relief(pack, completions, horizon_s) + 1e-9 \
        >= target else None


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid boolean {value!r}")
    return str(value).lower() in {"true", "1"}


def validate_background_states(rows: list[dict], expected: list[dict]) -> list[dict]:
    """Require one complete no-migration measurement per candidate."""
    required = {"state_id", "n_hbm", "n_serving", "n_prefill", "wan_setting",
                "warmup_s", "measurement_s", "telemetry_complete",
                "background_valid", "kv_ingest_limiting",
                *(f"b_{name}" for name in RESOURCES)}
    wanted = {row["state_id"]: row for row in expected}
    observed = {row.get("state_id"): row for row in rows}
    if len(observed) != len(rows) or observed.keys() != wanted.keys():
        raise ValueError("background measurements must match candidate states")
    normalized = []
    for state_id, row in observed.items():
        if not required <= row.keys() or not _bool(row["telemetry_complete"]):
            raise ValueError(f"missing telemetry for {state_id}")
        identity = wanted[state_id]
        if any(str(row[name]) != str(identity[name]) for name in
               ("n_hbm", "n_serving", "n_prefill", "wan_setting")):
            raise ValueError(f"background identity changed for {state_id}")
        if _number(row["warmup_s"], "warmup") != 30 \
                or _number(row["measurement_s"], "measurement") != 30:
            raise ValueError("candidate measurement requires 30 s warmup and window")
        capacities = {f"b_{name}": _number(row[f"b_{name}"], name)
                      for name in RESOURCES}
        value = {**identity, **capacities,
                 "background_valid": _bool(row["background_valid"]),
                 "telemetry_complete": True,
                 "kv_ingest_limiting": _bool(row["kv_ingest_limiting"]),
                 "warmup_s": 30, "measurement_s": 30}
        value["valid"] = (value["background_valid"]
                          and not value["kv_ingest_limiting"])
        value["state_hash"] = digest(value)
        normalized.append(value)
    return sorted(normalized, key=lambda row: row["state_id"])


def oracle_census(background: list[dict], packs: list[dict],
                  demands: list[dict], target: float) -> list[dict]:
    """Cross every background-valid measured state with every frozen pack."""
    valid = [row for row in background if row["valid"]]
    empty = {}
    for row in valid:
        zero = all(int(row.get(f"n_{name}", 0)) == 0 for name in AXES)
        if zero:
            if row["wan_setting"] in empty:
                raise ValueError("duplicate empty-destination measurement")
            empty[row["wan_setting"]] = {
                name: _number(row[f"b_{name}"], name) for name in RESOURCES}
    missing = {row["wan_setting"] for row in valid} - empty.keys()
    if missing:
        raise ValueError(f"missing empty-destination measurement: {sorted(missing)}")
    compiled = {}
    for pack in packs:
        selected = [row for row in demands if row["pack_id"] == pack["pack_id"]]
        compiled[pack["pack_id"]] = (selected, _plans(pack, selected))
    rows = []
    for state in valid:
        capacity = {name: _number(state[f"b_{name}"], name)
                    for name in RESOURCES}
        b0 = empty[state["wan_setting"]]
        for pack in packs:
            selected, plans = compiled[pack["pack_id"]]
            result = oracle_case(
                pack, selected, capacity, target, b0, plans)
            rows.append({
                "state_id": state["state_id"], "state_hash": state["state_hash"],
                "wan_setting": state["wan_setting"],
                "pack_id": pack["pack_id"], "pack_hash": _pack_hash(pack),
                **{name: state[name] for name in
                   ("n_hbm", "n_serving", "n_prefill") if name in state},
                **result,
            })
    return rows


def normalize_episodes(raw: list[dict], schedule: list[dict], packs: list[dict],
                       demands: list[dict], target: float) -> list[dict]:
    """Validate matched hardware rows and derive deadline outcomes."""
    planned = {row["episode_id"]: row for row in schedule}
    observed = {row.get("episode_id"): row for row in raw}
    if len(planned) != len(schedule) or len(observed) != len(raw) \
            or observed.keys() != planned.keys():
        raise ValueError("episodes must match the frozen execution schedule")
    by_pack = {pack["pack_id"]: pack for pack in packs}
    output = []
    identity = ("state_id", "pack_id", "policy", "repeat", "block_id",
                "state_hash", "pack_hash", "inputs_hash", "policy_order",
                "deadline_s", "power_window_s", "migration_window_s")
    for episode_id, row in observed.items():
        spec = planned[episode_id]
        pack = by_pack[spec["pack_id"]]
        if any(name in spec and row.get(name) != spec[name] for name in identity):
            raise ValueError(f"episode identity changed for {episode_id}")
        decisions = row["decisions"]
        decisions = json.loads(decisions) if isinstance(decisions, str) else decisions
        session_ids = {session["session_id"] for session in pack["sessions"]}
        if len(decisions) != len(session_ids) \
                or {decision["session_id"] for decision in decisions} != session_ids \
                or any(decision["action"] not in (*ACTIONS, "not_moved")
                       or decision["action"] == "not_moved"
                       and decision.get("completion_s") is not None
                       for decision in decisions):
            raise ValueError(f"invalid decisions for {episode_id}")
        fixed = {"kv_only": "kv_transfer", "replay_only": "replay"}
        if spec["policy"] in fixed and any(
                decision["action"] not in (fixed[spec["policy"]], "not_moved")
                for decision in decisions):
            raise ValueError(f"{spec['policy']} contains a forbidden action")
        if spec["policy"] == "per_session_greedy":
            pack_demands = [value for value in demands
                            if value["pack_id"] == pack["pack_id"]]
            expected = {move["session_id"]: move["action"]
                        for move in per_session_greedy(pack, pack_demands)}
            if any(decision["action"] != expected[decision["session_id"]]
                   for decision in decisions):
                raise ValueError("per-session greedy must dispatch every fastest action")
        completions = [
            (_number(decision["completion_s"], "completion"),
             decision["session_id"])
            for decision in decisions
            if decision["action"] != "not_moved"
            and decision.get("completion_s") is not None
        ]
        counts = Counter(decision["action"] for decision in decisions)
        attained = target_time(pack, completions, target)
        evidence = all(_bool(row[name]) for name in
                       ("background_recreated", "background_valid",
                        "residual_verified"))
        output.append({
            **spec, "evidence_valid": evidence,
            "decisions": json.dumps(decisions, separators=(",", ":")),
            "completion_times": json.dumps(
                {session: at for at, session in completions}),
            "not_moved_sessions": json.dumps(sorted(
                decision["session_id"] for decision in decisions
                if decision["action"] == "not_moved")),
            "replay_count": counts["replay"],
            "kv_count": counts["kv_transfer"],
            "not_moved_count": counts["not_moved"],
            "achieved_relief": window_relief(pack, completions, D_S),
            "target_time_s": attained, "target_attained": attained is not None,
        })
    return output


def summarize_results(episodes: list[dict], census: list[dict]) -> list[dict]:
    """Return equal-class and full-population-weighted policy estimates."""
    cases = {(row["state_id"], row["pack_id"]): row for row in census}
    valid = [{**row, **cases[row["state_id"], row["pack_id"]]}
             for row in episodes if row["evidence_valid"]]
    metrics = {
        "relative_relief": (lambda row: row["p_star"] > 0,
                            lambda row: row["achieved_relief"] / row["p_star"]),
        "target_attainment": (lambda row: row["j"],
                              lambda row: float(row["target_attained"])),
    }
    output = []
    for metric, (eligible, value) in metrics.items():
        population = Counter(row["oracle_class"] for row in census if eligible(row))
        for policy in POLICIES:
            policy_rows = [row for row in valid if row["policy"] == policy
                           and eligible(row)]
            grouped = {name: [value(row) for row in policy_rows
                              if row["oracle_class"] == name]
                       for name in population}
            complete = bool(population) and all(grouped.values())
            means = {name: sum(values) / len(values)
                     for name, values in grouped.items() if values}
            values = {
                "class_balanced": sum(means.values()) / len(means)
                if complete else None,
                "population_weighted": sum(means[name] * count
                                           for name, count in population.items())
                / sum(population.values()) if complete else None,
            }
            output.extend({"policy": policy, "metric": metric,
                           "aggregation": name, "value": result,
                           "episodes": len(policy_rows)}
                          for name, result in values.items())
    return output


def _save(fig, out: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=200, bbox_inches="tight")


def plot_results(episodes: list[dict], census: list[dict], out: Path) -> None:
    """Write the four declared figure families from valid evidence only."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    out.mkdir(parents=True, exist_ok=True)
    cases = {(row["state_id"], row["pack_id"]): row for row in census}
    joined = [{**row, **cases[row["state_id"], row["pack_id"]]}
              for row in episodes if row["evidence_valid"]]
    aliases = {"per_session_greedy": "isolated_fastest"}
    style = lambda policy: aliases.get(policy, policy)

    fig, axis = plt.subplots()
    for policy in POLICIES:
        values = sorted(row["achieved_relief"] / row["p_star"] for row in joined
                        if row["policy"] == policy and row["p_star"] > 0)
        if values:
            axis.step(values, [index / len(values)
                               for index in range(1, len(values) + 1)],
                      where="post", color=plot_style.POLICY_COLORS[style(policy)],
                      linestyle=plot_style.POLICY_LINESTYLES[style(policy)],
                      label=plot_style.POLICY_NAMES[style(policy)])
    axis.set(xlabel=r"$\Delta P(30\,\mathrm{s})/P^*$",
             ylabel="Cumulative fraction")
    axis.legend(frameon=False)
    _save(fig, out, "relative_relief")
    plt.close(fig)

    fig, axis = plt.subplots()
    for policy in POLICIES:
        denominator = sum(row["policy"] == policy and row["j"] for row in joined)
        values = sorted(row["target_time_s"] for row in joined
                        if row["policy"] == policy and row["j"]
                        and row["target_time_s"] is not None)
        if denominator:
            axis.step([0, *values, D_S],
                      [0, *[index / denominator
                            for index in range(1, len(values) + 1)],
                       len(values) / denominator],
                      where="post", color=plot_style.POLICY_COLORS[style(policy)],
                      linestyle=plot_style.POLICY_LINESTYLES[style(policy)],
                      label=plot_style.POLICY_NAMES[style(policy)])
    axis.set(xlim=(0, D_S), ylim=(0, 1.02), xlabel="Time (s)",
             ylabel="Cumulative target attainment")
    axis.legend(frameon=False)
    _save(fig, out, "target_attainment")
    plt.close(fig)

    resource_labels = sorted({label for row in joined
                              for label in json.loads(row["resource_labels"])})
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    for axis, field, labels in (
            (axes[0], "oracle_class", list(ORACLE_CLASSES)),
            (axes[1], "resource_labels", resource_labels)):
        groups = [(label, policy) for label in labels for policy in POLICIES]
        x = list(range(len(groups)))
        bottom = [0.0] * len(groups)
        for action, key in (("replay", "replay_count"),
                            ("kv_transfer", "kv_count"),
                            ("not_moved", "not_moved_count")):
            values = []
            for label, policy in groups:
                selected = [row for row in joined if row["policy"] == policy
                            and (row[field] == label
                                 if field == "oracle_class"
                                 else label in json.loads(row[field]))]
                total = sum(row["replay_count"] + row["kv_count"]
                            + row["not_moved_count"] for row in selected)
                values.append(sum(row[key] for row in selected) / total
                              if total else 0)
            axis.bar(x, values, bottom=bottom,
                     color=plot_style.ACTION_COLORS[action],
                     hatch=plot_style.ACTION_HATCHES[action],
                     label=plot_style.ACTION_NAMES[action])
            bottom = [left + value for left, value in zip(bottom, values)]
        axis.set_xticks(x, [
            f"{label}\n{plot_style.COMPACT_POLICY_NAMES[style(policy)]}"
            for label, policy in groups], rotation=60, ha="right")
        axis.set(ylabel="Action fraction")
    axes[0].legend(frameon=False, ncol=3)
    _save(fig, out, "action_composition")
    plt.close(fig)

    population = Counter(row["oracle_class"] for row in census)
    executed_keys = {(row["state_id"], row["pack_id"]) for row in episodes}
    executed = Counter(cases[key]["oracle_class"] for key in executed_keys)
    x = list(range(len(ORACLE_CLASSES)))
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar([value - .2 for value in x],
             [population[name] for name in ORACLE_CLASSES], .4,
             label="Candidate population", color="#0072B2")
    axis.bar([value + .2 for value in x],
             [executed[name] for name in ORACLE_CLASSES], .4,
             label="Executed sample", color="#E69F00")
    axis.set_xticks(x, ORACLE_CLASSES, rotation=20, ha="right")
    axis.set(ylabel="State-pack cases")
    axis.legend(frameon=False)
    _save(fig, out, "class_histograms")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(out: Path, background: list[dict], census: list[dict],
                  selected: dict, episodes: list[dict] | None = None,
                  *, frozen: dict | None = None,
                  schedule: list[dict] | None = None) -> None:
    """Write frozen pre-execution artifacts and available results."""
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "background_states.csv", background)
    write_csv(out / "oracle_census.csv", census)
    chosen = {row["state_id"] for row in selected["states"]}
    histogram = [{
        "sample": sample, "oracle_class": name,
        "cases": sum(row["oracle_class"] == name
                     and (sample == "candidate_population"
                          or row["state_id"] in chosen)
                     for row in census),
    } for sample in ("candidate_population", "executed_sample")
        for name in ORACLE_CLASSES]
    write_csv(out / "class_histogram.csv", histogram)
    (out / "selected_states.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n")
    if frozen is not None:
        verify_frozen(frozen)
        (out / "frozen_inputs.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    if schedule is not None:
        write_csv(out / "execution_schedule.csv", schedule)
    if episodes is not None:
        write_csv(out / "episodes.csv", episodes)
        write_csv(out / "policy_summary.csv", summarize_results(episodes, census))
        (out / "summary.json").write_text(json.dumps({
            "p_star_zero_cases": sum(row["p_star"] == 0 for row in census),
            "valid_episodes": sum(row["evidence_valid"] for row in episodes),
            "invalid_background_episodes": sum(not row["evidence_valid"]
                                               for row in episodes),
            "selection_shortages": selected["shortages"],
        }, indent=2, sort_keys=True) + "\n")
        plot_results(episodes, census, out)



def _frozen_selection(census: list[dict], frozen: dict) -> dict:
    selected = select_states(census, frozen["seeds"]["selection"])
    selected["inputs_hash"] = frozen["inputs_hash"]
    selected["selection_sha256"] = digest(selected)
    return selected


def compile_campaign(frozen: dict, discovery: list[dict],
                     measured_background: list[dict], out: Path) -> dict:
    """Compile measured discovery into the immutable execution schedule."""
    verify_frozen(frozen)
    out.mkdir(parents=True, exist_ok=True)
    limits, retained = discovery_limits(discovery)
    candidates = candidate_states(limits, frozen["seeds"]["sobol"])
    background = validate_background_states(measured_background, candidates)
    inputs = frozen["inputs"]
    census = oracle_census(background, inputs["packs"],
                           inputs["action_demands"], inputs["target"])
    selected = _frozen_selection(census, frozen)
    schedule = execution_schedule(
        selected, inputs["packs"],
        frozen["seeds"].get("policy_order", frozen["seeds"]["selection"]),
        frozen["inputs_hash"])
    write_csv(out / "background_discovery.csv", retained)
    write_outputs(out, background, census, selected, frozen=frozen,
                  schedule=schedule)
    return {"limits": limits, "background": background, "census": census,
            "selected": selected, "schedule": schedule}


def reduce_campaign(frozen: dict, background: list[dict], census: list[dict],
                    selected: dict, schedule: list[dict], raw: list[dict],
                    out: Path) -> list[dict]:
    """Verify the frozen schedule and reduce completed hardware episodes."""
    verify_frozen(frozen)
    inputs = frozen["inputs"]
    expected_selected = _frozen_selection(census, frozen)
    expected_schedule = execution_schedule(
        expected_selected, inputs["packs"],
        frozen["seeds"].get("policy_order", frozen["seeds"]["selection"]),
        frozen["inputs_hash"])
    if selected != expected_selected or schedule != expected_schedule:
        raise ValueError("selection or execution schedule changed")
    episodes = normalize_episodes(
        raw, schedule, inputs["packs"], inputs["action_demands"],
        inputs["target"])
    write_outputs(out, background, census, selected, episodes,
                  frozen=frozen, schedule=schedule)
    return episodes
