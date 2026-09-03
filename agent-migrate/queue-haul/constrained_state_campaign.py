"""Compile and reduce the frozen two-A100 constrained-state campaign."""

from __future__ import annotations

import csv
import hashlib
import itertools
import inspect
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
SCHEMA = "queue-haul-constrained-state-v2"
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


def callable_sha256(value) -> str:
    return hashlib.sha256(inspect.getsource(value).encode()).hexdigest()


def _number(value, name: str, nonnegative: bool = True) -> float:
    value = float(value)
    if not math.isfinite(value) or nonnegative and value < 0:
        raise ValueError(f"invalid {name}")
    return value


def _sha256(value, name: str) -> str:
    value = str(value)
    try:
        valid = len(value) == 64 and int(value, 16) >= 0
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(f"invalid {name} SHA-256")
    return value


def freeze_inputs(inputs: dict, seeds: dict[str, int]) -> dict:
    """Validate and hash the complete immutable campaign contract."""
    required = {"packs", "profile", "action_demands", "target",
                "wan_settings", "background_manifest", "timing_model",
                "measurement_contract"}
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
        if any(not {"request_sha256", "continuation_sha256"} <= session.keys()
               for session in pack["sessions"]):
            raise ValueError("session evidence hashes are required")
        for session in pack["sessions"]:
            _sha256(session["request_sha256"], "request")
            _sha256(session["continuation_sha256"], "continuation")
        gains = [_number(gain, "power gain")
                 for gain in pack.get("power_gains", ())]
        if len(set(sessions)) != 8 or len(gains) != 256 \
                or gains[0] != 0 or not math.isclose(gains[-1], target):
            raise ValueError("invalid frozen pack")
        if any(gains[mask] > gains[mask | 1 << bit] + 1e-9
               for mask in range(256) for bit in range(8)
               if not mask & 1 << bit):
            raise ValueError("power gains must be monotone")
        if any(gain + 1e-9 >= target for gain in gains[:-1]):
            raise ValueError("full-drain target must require all eight sessions")

        expected |= {(pack["pack_id"], session, action)
                     for session in sessions for action in ACTIONS}
    demands = inputs["action_demands"]
    actual = {(row["pack_id"], row["session_id"], row["action"])
              for row in demands}
    if len(actual) != len(demands) or actual != expected:
        raise ValueError("action-demand table must contain both actions exactly once")
    for row in demands:
        for name in (f"d_{resource}" for resource in RESOURCES):
            _number(row[name], name)
    if set(inputs["wan_settings"]) != set(WAN_SETTINGS):
        raise ValueError("WAN settings must be natural, controlled_80, controlled_40")
    if not isinstance(inputs["profile"], dict) or not inputs["profile"] \
            or not isinstance(inputs["background_manifest"], dict) \
            or "seed" not in inputs["background_manifest"]:
        raise ValueError("profile and seeded background manifest are required")
    contract = inputs["measurement_contract"]
    required_contract = {
        "residual_tolerances", "min_telemetry_coverage",
        "max_scheduled_load_error", "min_samples",
        "queue_growth_tolerance", "prefill_throughput_tolerance",
        "min_power_samples", "max_power_gap_s",
    }
    if not isinstance(inputs["timing_model"], dict) \
            or not isinstance(contract, dict) \
            or set(contract) != required_contract \
            or set(contract["residual_tolerances"]) != set(RESOURCES):
        raise ValueError("invalid timing or measurement contract")
    _sha256(inputs["timing_model"].get("source_sha256"), "timing source")
    for name, value in contract["residual_tolerances"].items():
        _number(value, f"{name} residual tolerance")
    coverage = _number(contract["min_telemetry_coverage"], "telemetry coverage")
    _number(contract["max_scheduled_load_error"], "scheduled load error")
    _number(contract["queue_growth_tolerance"], "queue growth tolerance")
    _number(contract["prefill_throughput_tolerance"],
            "prefill throughput tolerance")
    gap = _number(contract["max_power_gap_s"], "power sample gap")
    samples = (contract["min_samples"], contract["min_power_samples"])
    if coverage > 1 or gap == 0 or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 2
            for value in samples):
        raise ValueError("invalid measurement thresholds")
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


def verify_timing(timing, frozen: dict) -> None:
    identity = frozen["inputs"]["timing_model"]
    if getattr(timing, "identity", None) != identity:
        raise ValueError("timing evaluator does not match frozen identity")
    if callable_sha256(timing) != identity["source_sha256"]:
        raise ValueError("timing source hash mismatch")



def _pack_hash(pack: dict) -> str:
    return pack["sha256"] if "sha256" in pack else digest(pack)


def _valid_axis_row(axis: str, row: dict, contract: dict) -> bool:
    if axis == "hbm":
        return not _bool(row["rejected"]) and not _bool(row["evicted"]) \
            and _bool(row["persistently_resident"])
    queue_ok = _number(row["queue_growth"], "queue growth", False) \
        <= contract["queue_growth_tolerance"]
    if axis == "serving":
        return _bool(row["slo_met"]) and queue_ok
    return queue_ok and _number(row["completed_throughput"], "throughput") \
        + contract["prefill_throughput_tolerance"] \
        >= _number(row["scheduled_throughput"], "scheduled throughput")


def _discovery_evidence(row: dict, contract: dict) -> dict:
    names = {"telemetry_complete", "measurement_id", "telemetry_sha256",
             "measurement_started_ns", "measurement_ended_ns", "sample_count",
             "telemetry_coverage", "scheduled_load_error"}
    if not names <= row.keys() or not _bool(row["telemetry_complete"]):
        raise ValueError("missing discovery telemetry")
    start = int(_number(row["measurement_started_ns"], "measurement start"))
    end = int(_number(row["measurement_ended_ns"], "measurement end"))
    samples = int(_number(row["sample_count"], "sample count"))
    coverage = _number(row["telemetry_coverage"], "telemetry coverage")
    load_error = _number(row["scheduled_load_error"], "scheduled load error")
    evidence = {
        "measurement_id": str(row["measurement_id"]),
        "telemetry_sha256": _sha256(row["telemetry_sha256"], "telemetry"),
        "measurement_started_ns": start, "measurement_ended_ns": end,
        "sample_count": samples, "telemetry_coverage": coverage,
        "scheduled_load_error": load_error,
    }
    if not evidence["measurement_id"] or end - start != 30_000_000_000 \
            or float(row["sample_count"]) != samples or coverage > 1 \
            or samples < contract["min_samples"] \
            or coverage < contract["min_telemetry_coverage"] \
            or load_error > contract["max_scheduled_load_error"]:
        raise ValueError("invalid discovery telemetry")
    return evidence


def discovery_limits(rows: list[dict], contract: dict) -> tuple[dict[str, int], list[dict]]:
    """Validate each one-unit axis through its first invalid measurement."""
    retained, limits = [], {}
    for axis in AXES:
        selected = sorted((row for row in rows if row["axis"] == axis),
                          key=lambda row: row["count"])
        if [row["count"] for row in selected] != list(range(1, len(selected) + 1)):
            raise ValueError(f"{axis} discovery counts must be consecutive")
        for index, row in enumerate(selected):
            if _number(row["warmup_s"], "warmup") != 30 \
                    or _number(row["measurement_s"], "measurement") != 30:
                raise ValueError("discovery requires 30 s warmup and measurement")
            evidence = _discovery_evidence(row, contract)
            valid = _valid_axis_row(axis, row, contract)
            retained.append({**row, **evidence, "valid": valid})
            if not valid:
                if index != len(selected) - 1:
                    raise ValueError(f"{axis} includes states after first invalid")
                limits[axis] = int(row["count"]) - 1
                break
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
        for name in (f"d_{resource}" for resource in RESOURCES):
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
        mask, usage = 0, dict.fromkeys(RESOURCES, 0.0)
        for index, choice in enumerate(choices):
            if not choice:
                continue
            mask |= 1 << index
            row = demand[sessions[index], ACTIONS[choice - 1]]
            for resource in RESOURCES:
                usage[resource] += _number(row[f"d_{resource}"], resource)
        plans.append((float(gains[mask]), usage, choices))
    return plans


def _moves(pack, choices):
    return [{"session_id": session["session_id"],
             "action": ACTIONS[choice - 1], "order": order}
            for order, (session, choice) in enumerate(
                zip(pack["sessions"], choices)) if choice]


def _makespan(timing, pack, choices, capacity):
    moves = _moves(pack, choices)
    result = timing(pack, moves, capacity)
    if not isinstance(result, dict) or not {
            "makespan_s", "completion_s"} <= result.keys():
        raise ValueError("timing evaluator returned incomplete evidence")
    makespan = float(result["makespan_s"])
    if math.isnan(makespan) or makespan < 0:
        raise ValueError("timing evaluator returned an invalid makespan")
    completions = result["completion_s"]
    if makespan <= T_MIG_S and (set(completions) != {
            move["session_id"] for move in moves} or any(
                _number(value, "completion") > makespan + 1e-9
                for value in completions.values())):
        raise ValueError("timing evaluator returned invalid completions")
    return makespan


def _best(plans, capacity, action=None, *, pack=None, timing=None):
    index = None if action is None else ACTIONS.index(action) + 1
    feasible = [plan for plan in plans
                if (index is None or all(choice in (0, index)
                                         for choice in plan[2]))
                and all(plan[1][resource] <= float(capacity[resource]) + 1e-9
                        for resource in RESOURCES)]
    if timing is None:
        return (*max(feasible, key=lambda plan: plan[0]), None)
    for plan in sorted(feasible, key=lambda plan: plan[0], reverse=True):
        makespan = _makespan(timing, pack, plan[2], capacity)
        if makespan <= T_MIG_S + 1e-9:
            return (*plan, makespan)
    raise RuntimeError("empty plan must be executable")


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
                timing,
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
    static = _best(plans, capacity)
    best = _best(plans, capacity, pack=pack, timing=timing)
    restricted = {action: _best(
        plans, capacity, action, pack=pack, timing=timing)[0]
                  for action in ACTIONS}
    p_star = best[0]
    j, k, r = (p_star + 1e-9 >= target,
               restricted["kv_transfer"] + 1e-9 >= target,
               restricted["replay"] + 1e-9 >= target)
    result = {"p_star": p_star, "static_p_star": static[0],
              "oracle_makespan_s": best[3],
              "static_scheduled_discordant": static[0] > p_star + 1e-9,
              "j": j, "k": k, "r": r,
              "oracle_class": _oracle_class(j, k, r)}
    target_plans = [plan for plan in plans if plan[0] + 1e-9 >= target]

    def reaches(trial):
        return any(
            all(plan[1][resource] <= trial[resource] + 1e-9
                for resource in RESOURCES)
            and _makespan(timing, pack, plan[2], trial) <= T_MIG_S + 1e-9
            for plan in target_plans
        )

    if j:
        for resource in RESOURCES:
            zero = {**capacity, resource: 0.0}
            if reaches(zero):
                margin = None
            else:
                low, high = 0.0, capacity[resource]
                for _ in range(24):
                    middle = (low + high) / 2
                    trial = {**capacity, resource: middle}
                    if reaches(trial):
                        high = middle
                    else:
                        low = middle
                margin = capacity[resource] - high
            result[f"margin_{resource}"] = margin
            result[f"delta_{resource}"] = None
    else:
        for resource in RESOURCES:
            result[f"margin_{resource}"] = None
    subsets = [subset for size in range(len(RESOURCES) + 1)
               for subset in itertools.combinations(RESOURCES, size)]
    values = {}
    for subset in subsets:
        relaxed = {**capacity, **{name: empty_capacity[name] for name in subset}}
        values["+".join(subset) or "none"] = _best(
            plans, relaxed, pack=pack, timing=timing)[0]
    improving = [set(subset) for subset in subsets
                 if values["+".join(subset) or "none"] > p_star + 1e-9]
    minimal = [subset for subset in improving
               if not any(other < subset for other in improving)]
    for resource in RESOURCES:
        result[f"delta_{resource}"] = (None if j else max(
            0.0, values[resource] - p_star))
    result["relaxation_values"] = json.dumps(values, separators=(",", ":"))
    result["minimal_improving_subsets"] = json.dumps(
        [sorted(subset) for subset in minimal], separators=(",", ":"))
    labels = (["+".join(sorted(subset)) for subset in minimal] if not j else
              [resource for resource in RESOURCES
               if result[f"margin_{resource}"] is not None])
    result["resource_labels"] = json.dumps(labels or ["none"])
    result["oracle_actions"] = json.dumps([
        {"session_id": session["session_id"], "action": ACTIONS[choice - 1]}
        for session, choice in zip(pack["sessions"], best[2]) if choice
    ], separators=(",", ":"))
    return result


def per_session_greedy(pack: dict, demand_rows: list[dict], capacity: dict,
                       timing) -> list[dict]:
    """Choose every session's fastest action without aggregate admission."""
    _demand_map(demand_rows)

    def duration(row, action):
        choices = tuple(ACTIONS.index(action) + 1
                        if session["session_id"] == row["session_id"] else 0
                        for session in pack["sessions"])
        return _makespan(timing, pack, choices, capacity)

    return [{"session_id": row["session_id"],
             "action": min(ACTIONS, key=lambda action: (
                 duration(row, action), action)),
             "dispatch": True} for row in pack["sessions"]]


def select_states(census: list[dict], seed: int,
                  quota: int = STATE_QUOTA) -> dict:
    """Sample states by class and freeze their union-inclusion weights."""
    hashes, memberships = {}, {}
    for row in census:
        previous = hashes.setdefault(row["state_id"], row["state_hash"])
        if previous != row["state_hash"]:
            raise ValueError("state hash changed across packs")
        memberships.setdefault(row["state_id"], set()).add(row["oracle_class"])
    draws, shortages, strata = {}, {}, {}
    histogram = Counter(row["oracle_class"] for row in census)
    for name in ORACLE_CLASSES:
        population = sorted({row["state_id"] for row in census
                             if row["oracle_class"] == name})
        class_seed = int(digest([seed, name])[:16], 16)
        size = min(quota, len(population))
        draws[name] = random.Random(class_seed).sample(population, size)
        strata[name] = {"population_states": len(population),
                        "draw_states": size, "seed": class_seed}
        if len(population) < quota:
            shortages[name] = quota - len(population)
    chosen = sorted(set(itertools.chain.from_iterable(draws.values())))
    states = []
    for state in chosen:
        probability = 1 - math.prod(
            1 - strata[name]["draw_states"] / strata[name]["population_states"]
            for name in memberships[state])
        states.append({
            "state_id": state, "sha256": hashes[state],
            "class_memberships": sorted(memberships[state]),
            "inclusion_probability": probability,
            "analysis_weight": 1 / probability,
        })
    return {
        "seed": seed, "quota_per_class": quota, "draws": draws,
        "shortages": shortages, "strata": strata,
        "census_sha256": digest(census),
        "candidate_class_histogram": {
            name: histogram[name] for name in ORACLE_CLASSES},
        "states": states,
    }



def policy_input_hash(row: dict) -> str:
    """Hash the immutable planner inputs carried by one schedule row."""
    fields = ("state_hash", "pack_hash", "inputs_hash", "policy",
              "deadline_s", "migration_window_s")
    return digest({name: row[name] for name in fields})

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
                    "inclusion_probability": state["inclusion_probability"],
                    "analysis_weight": state["analysis_weight"],
                    "inputs_hash": inputs_hash,
                } for order, policy in enumerate(policies))
    for row in rows:
        row["policy_input_sha256"] = policy_input_hash(row)
    return rows


def _gain(pack: dict, selected: set[str]) -> float:
    mask = sum(1 << index for index, row in enumerate(pack["sessions"])
               if row["session_id"] in selected)
    return float(pack["power_gains"][mask])


def modeled_window_relief(pack: dict, completions: list[tuple[float, str]], t: float,
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


def modeled_target_time(pack: dict, completions: list[tuple[float, str]], target: float,
                horizon_s: float = D_S) -> float | None:
    """Return the first trailing-window target crossing within the deadline."""
    points = sorted({0.0, float(horizon_s),
                     *(max(0.0, float(at)) for at, _ in completions),
                     *(min(float(horizon_s), float(at) + W_S)
                       for at, _ in completions)})
    points = [point for point in points if 0 <= point <= horizon_s]
    for left, right in zip(points, points[1:]):
        before, after = (modeled_window_relief(pack, completions, value)
                         for value in (left, right))
        if before + 1e-9 >= target:
            return left
        if after + 1e-9 >= target and after > before:
            return left + (target - before) * (right - left) / (after - before)
    return horizon_s if modeled_window_relief(pack, completions, horizon_s) + 1e-9 \
        >= target else None


def _bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() not in {"true", "false", "1", "0"}:
        raise ValueError(f"invalid boolean {value!r}")
    return str(value).lower() in {"true", "1"}


def validate_background_states(rows: list[dict], expected: list[dict],
                               contract: dict) -> list[dict]:
    """Require one complete no-migration measurement per candidate."""
    required = {"state_id", "n_hbm", "n_serving", "n_prefill", "wan_setting",
                "warmup_s", "measurement_s", "telemetry_complete",
                "background_valid", "kv_ingest_limiting",
                "measurement_id", "telemetry_sha256",
                "measurement_started_ns", "measurement_ended_ns",
                "sample_count", "telemetry_coverage",
                "scheduled_load_error",
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
        capacities = {f"observed_b_{name}":
                      _number(row[f"b_{name}"], name)
                      for name in RESOURCES}
        start = int(_number(row["measurement_started_ns"], "measurement start"))
        end = int(_number(row["measurement_ended_ns"], "measurement end"))
        samples = int(_number(row["sample_count"], "sample count"))
        coverage = _number(row["telemetry_coverage"], "telemetry coverage")
        load_error = _number(row["scheduled_load_error"], "load error")
        provenance = str(row["measurement_id"])
        telemetry_hash = str(row["telemetry_sha256"])
        if end - start != 30_000_000_000 or coverage > 1 \
                or float(row["sample_count"]) != samples \
                or not provenance or len(telemetry_hash) != 64:
            raise ValueError(f"invalid telemetry provenance for {state_id}")
        try:
            int(telemetry_hash, 16)
        except ValueError as error:
            raise ValueError(f"invalid telemetry provenance for {state_id}") from error
        measurement_valid = (
            samples >= contract["min_samples"]
            and coverage >= contract["min_telemetry_coverage"]
            and load_error <= contract["max_scheduled_load_error"])
        value = {
            **identity, **capacities,
            "background_valid": _bool(row["background_valid"]),
            "telemetry_complete": True,
            "kv_ingest_limiting": _bool(row["kv_ingest_limiting"]),
            "measurement_id": provenance, "telemetry_sha256": telemetry_hash,
            "measurement_started_ns": start, "measurement_ended_ns": end,
            "sample_count": samples, "telemetry_coverage": coverage,
            "scheduled_load_error": load_error, "measurement_valid": measurement_valid,
            "warmup_s": 30, "measurement_s": 30,
        }
        value["valid"] = (value["background_valid"] and measurement_valid
                          and not value["kv_ingest_limiting"])
        normalized.append(value)
    empty_rows = [
        row for row in normalized
        if all(int(row[f"n_{axis}"]) == 0 for axis in AXES)]
    empty = {row["wan_setting"]: row for row in empty_rows}
    if len(empty) != len(empty_rows):
        raise ValueError("duplicate empty-destination measurement")
    missing = {row["wan_setting"] for row in normalized} - empty.keys()
    if missing or "natural" not in empty:
        raise ValueError(f"missing empty-destination measurement: {sorted(missing)}")
    if not all(row["valid"] for row in empty.values()):
        raise ValueError("empty-destination measurements must be valid")
    for value in normalized:
        for resource in RESOURCES:
            setting = "natural" if resource == "wan" else value["wan_setting"]
            reference = empty[setting][f"observed_b_{resource}"]
            observed_value = value[f"observed_b_{resource}"]
            tolerance = _number(
                contract["residual_tolerances"][resource], "residual tolerance")
            if observed_value > reference + tolerance:
                raise ValueError(f"{resource} residual exceeds empty reference")
            value[f"b0_{resource}"] = reference
            value[f"b_{resource}"] = min(observed_value, reference)
        value["state_hash"] = digest(value)
    return sorted(normalized, key=lambda row: row["state_id"])


def oracle_census(background: list[dict], packs: list[dict],
                  demands: list[dict], target: float, timing) -> list[dict]:
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
    if "natural" not in empty:
        raise ValueError("missing natural empty-destination measurement")
    compiled = {}
    for pack in packs:
        selected = [row for row in demands if row["pack_id"] == pack["pack_id"]]
        compiled[pack["pack_id"]] = (selected, _plans(pack, selected))
    rows = []
    for state in valid:
        capacity = {name: _number(state[f"b_{name}"], name)
                    for name in RESOURCES}
        b0 = {
            name: empty["natural"][name] if name == "wan"
            else empty[state["wan_setting"]][name]
            for name in RESOURCES}
        for pack in packs:
            selected, plans = compiled[pack["pack_id"]]
            result = oracle_case(
                pack, selected, capacity, target, b0, timing, plans)
            rows.append({
                "state_id": state["state_id"], "state_hash": state["state_hash"],
                "wan_setting": state["wan_setting"],
                "pack_id": pack["pack_id"], "pack_hash": _pack_hash(pack),
                **{name: state[name] for name in
                   ("n_hbm", "n_serving", "n_prefill") if name in state},
                **result,
            })
    return rows


def _power_outcomes(samples: list[dict], target: float,
                    contract: dict) -> tuple[float, float | None]:
    """Integrate matched power samples with zero-order hold."""
    points = [(_number(row["t_s"], "power timestamp"),
               _number(row["control_power_w"], "control power")
               - _number(row["source_power_w"], "source power"))
              for row in samples]
    if len(points) < contract["min_power_samples"] or points != sorted(points) \
            or len({at for at, _ in points}) != len(points) \
            or points[0][0] > 0 or points[-1][0] < D_S \
            or max(right[0] - left[0] for left, right in zip(points, points[1:])) \
            > contract["max_power_gap_s"]:
        raise ValueError("power samples must uniquely cover [0,D]")

    def average(at):
        start, total = max(0.0, at - W_S), 0.0
        for (left, value), (right, _) in zip(points, points[1:]):
            duration = max(0.0, min(at, right) - max(start, left))
            total += value * duration
        return total / W_S

    boundaries = sorted({0.0, float(D_S),
                         *(at for at, _ in points if 0 <= at <= D_S),
                         *(at + W_S for at, _ in points
                           if 0 <= at + W_S <= D_S)})
    crossing = None
    for left, right in zip(boundaries, boundaries[1:]):
        before, after = average(left), average(right)
        if before + 1e-9 >= target:
            crossing = left
            break
        if after + 1e-9 >= target and after > before:
            crossing = left + (target - before) * (right - left) / (after - before)
            break
    if crossing is None and average(D_S) + 1e-9 >= target:
        crossing = float(D_S)
    return average(D_S), crossing


def _selection_quality(pack: dict, decisions: list[dict]) -> float:
    selected = {row["session_id"] for row in decisions
                if row["action"] != "not_moved"}
    size = len(selected)
    best = max(gain for mask, gain in enumerate(pack["power_gains"])
               if mask.bit_count() == size)
    return _gain(pack, selected) / best if best else 1.0


def normalize_episodes(raw: list[dict], schedule: list[dict], packs: list[dict],
                       demands: list[dict], target: float,
                       background: list[dict], contract: dict, timing) -> list[dict]:
    """Validate matched hardware rows and derive measured and modeled outcomes."""
    planned = {row["episode_id"]: row for row in schedule}
    observed = {row.get("episode_id"): row for row in raw}
    states = {row["state_id"]: row for row in background}
    if len(planned) != len(schedule) or len(observed) != len(raw) \
            or observed.keys() != planned.keys() or len(states) != len(background):
        raise ValueError("episodes must match the frozen execution schedule")
    by_pack = {pack["pack_id"]: pack for pack in packs}
    required = {
        "planner", "background_recreated", "background_valid",
        "background_warmup_s", "residual_measurement_id",
        "residual_telemetry_sha256", "residual_started_ns",
        "residual_ended_ns", "residual_sample_count",
        "residual_telemetry_coverage", "scheduled_load_error",
        "policy_output_sha256", "power_samples", "power_samples_sha256",
        "power_pair_id", "source_power_sha256", "control_power_sha256",
        *(f"observed_b_{name}" for name in RESOURCES),
    }
    output = []
    for episode_id, row in observed.items():
        spec = planned[episode_id]
        pack, state = by_pack[spec["pack_id"]], states[spec["state_id"]]
        if any(row.get(name) != value for name, value in spec.items()) \
                or state["state_hash"] != spec["state_hash"] \
                or _pack_hash(pack) != spec["pack_hash"]:
            raise ValueError(f"episode identity changed for {episode_id}")
        if spec.get("policy_input_sha256") != policy_input_hash(spec) \
                or row.get("planner") != spec["policy"]:
            raise ValueError(f"planner input mismatch for {episode_id}")
        if not required <= row.keys():
            raise ValueError(f"missing episode telemetry for {episode_id}")

        decisions = row["decisions"]
        decisions = json.loads(decisions) if isinstance(decisions, str) else decisions
        if row["policy_output_sha256"] != digest(decisions):
            raise ValueError(f"policy output hash mismatch for {episode_id}")
        sessions = {session["session_id"]: session for session in pack["sessions"]}

        def selected_action(decision):
            return (decision["action"] if decision["action"] != "not_moved"
                    else decision.get("attempted_action"))

        if len(decisions) != len(sessions) \
                or {decision["session_id"] for decision in decisions} != set(sessions) \
                or any(
                    decision["action"] not in (*ACTIONS, "not_moved")
                    or decision["action"] == "not_moved"
                    and decision.get("completion_s") is not None
                    or decision["action"] in ACTIONS
                    and (decision.get("completion_s") is None
                         or _number(decision["completion_s"], "completion")
                         > spec["migration_window_s"])
                    or decision.get("attempted_action") not in (*ACTIONS, None)
                    for decision in decisions):
            raise ValueError(f"invalid decisions for {episode_id}")
        moved = [decision for decision in decisions
                 if decision["action"] != "not_moved"]
        evidence = {"request_sha256", "continuation_sha256",
                    "reconstruction_evidence_sha256",
                    "reconstruction_verified", "continuation_verified"}
        if any(not evidence <= decision.keys() for decision in moved):
            raise ValueError(f"missing reconstruction evidence for {episode_id}")
        for decision in moved:
            _sha256(decision["reconstruction_evidence_sha256"], "reconstruction")
        fixed = {"kv_only": "kv_transfer", "replay_only": "replay"}
        if spec["policy"] in fixed and any(
                selected_action(decision) not in (fixed[spec["policy"]], None)
                for decision in decisions):
            raise ValueError(f"{spec['policy']} contains a forbidden action")
        capacity = {name: state[f"b_{name}"] for name in RESOURCES}
        if spec["policy"] == "per_session_greedy":
            pack_demands = [value for value in demands
                            if value["pack_id"] == pack["pack_id"]]
            expected = {move["session_id"]: move["action"] for move in
                        per_session_greedy(pack, pack_demands, capacity, timing)}
            if any(selected_action(decision) != expected[decision["session_id"]]
                   for decision in decisions):
                raise ValueError("per-session greedy must dispatch every fastest action")

        residual_match = True
        for resource in RESOURCES:
            value = _number(row[f"observed_b_{resource}"], resource)
            tolerance = contract["residual_tolerances"][resource]
            if value > state[f"b0_{resource}"] + tolerance:
                raise ValueError(f"{resource} residual exceeds empty reference")
            residual_match &= abs(value - capacity[resource]) <= tolerance
        samples = int(_number(row["residual_sample_count"], "sample count"))
        coverage = _number(row["residual_telemetry_coverage"], "coverage")
        load_error = _number(row["scheduled_load_error"], "load error")
        start = int(_number(row["residual_started_ns"], "residual start"))
        end = int(_number(row["residual_ended_ns"], "residual end"))
        telemetry_hash = str(row["residual_telemetry_sha256"])
        try:
            hash_valid = len(telemetry_hash) == 64 and int(telemetry_hash, 16) >= 0
        except ValueError:
            hash_valid = False
        if not row["residual_measurement_id"] or not hash_valid or end <= start \
                or coverage > 1 or float(row["residual_sample_count"]) != samples:
            raise ValueError(f"invalid residual provenance for {episode_id}")
        instrumentation_ok = (
            residual_match and samples >= contract["min_samples"]
            and coverage >= contract["min_telemetry_coverage"]
            and load_error <= contract["max_scheduled_load_error"])
        background_ok = (
            _bool(row["background_recreated"]) and _bool(row["background_valid"])
            and _number(row["background_warmup_s"], "background warmup") == 30)
        reconstruction_ok = all(
            _bool(decision["reconstruction_verified"])
            and _bool(decision["continuation_verified"])
            and decision["request_sha256"]
            == sessions[decision["session_id"]]["request_sha256"]
            and decision["continuation_sha256"]
            == sessions[decision["session_id"]]["continuation_sha256"]
            for decision in moved)

        power = row["power_samples"]
        power = json.loads(power) if isinstance(power, str) else power
        if row["power_samples_sha256"] != digest(power):
            raise ValueError(f"power sample hash mismatch for {episode_id}")
        pair_id = str(row["power_pair_id"])
        if not pair_id:
            raise ValueError(f"missing power pair for {episode_id}")
        source_hash = _sha256(row["source_power_sha256"], "source power")
        control_hash = _sha256(row["control_power_sha256"], "control power")
        measured_relief, measured_time = _power_outcomes(power, target, contract)
        completions = [
            (_number(decision["completion_s"], "completion"),
             decision["session_id"])
            for decision in decisions
            if decision["action"] != "not_moved"
            and decision.get("completion_s") is not None
        ]
        modeled_time = modeled_target_time(pack, completions, target)
        status = ("background_invalid" if not background_ok else
                  "retryable_instrumentation_failure"
                  if not instrumentation_ok else
                  "reconstruction_failure" if not reconstruction_ok else "valid")
        counts = Counter(decision["action"] for decision in decisions)
        output.append({
            **spec, "evidence_status": status, "evidence_valid": status == "valid",
            "policy_output_sha256": row["policy_output_sha256"],
            "decisions": json.dumps(decisions, separators=(",", ":")),
            "completion_times": json.dumps(
                {session: at for at, session in completions}),
            "not_moved_sessions": json.dumps(sorted(
                decision["session_id"] for decision in decisions
                if decision["action"] == "not_moved")),
            "replay_count": counts["replay"], "kv_count": counts["kv_transfer"],
            "not_moved_count": counts["not_moved"],
            "selection_quality": _selection_quality(pack, decisions),
            "modeled_relief_at_deadline": modeled_window_relief(pack, completions, D_S),
            "modeled_target_time_s": modeled_time,
            "modeled_target_attained": modeled_time is not None,
            "measured_relief_at_deadline": measured_relief,
            "measured_target_time_s": measured_time,
            "measured_target_attained": measured_time is not None,
            "residual_measurement_id": row["residual_measurement_id"],
            "residual_telemetry_sha256": telemetry_hash,
            "residual_started_ns": start, "residual_ended_ns": end,
            "residual_sample_count": samples,
            "residual_telemetry_coverage": coverage,
            "scheduled_load_error": load_error,
            **{f"observed_b_{name}": row[f"observed_b_{name}"]
               for name in RESOURCES},
            "power_samples": json.dumps(power, separators=(",", ":")),
            "power_samples_sha256": row["power_samples_sha256"],
            "power_pair_id": pair_id,
            "source_power_sha256": source_hash,
            "control_power_sha256": control_hash,
        })
    return output


def _repeat_cases(episodes: list[dict], census: list[dict]) -> list[dict]:
    cases = {(row["state_id"], row["pack_id"]): row for row in census}
    if len(cases) != len(census):
        raise ValueError("duplicate census case")
    grouped = {}
    for row in episodes:
        if not row["evidence_valid"]:
            continue
        key = row["state_id"], row["pack_id"]
        if key not in cases:
            raise ValueError("episode is absent from oracle census")
        grouped.setdefault((*key, row["policy"]), []).append({**row, **cases[key]})
    output = []
    for rows in grouped.values():
        weights = {row["analysis_weight"] for row in rows}
        repeats = [row["repeat"] for row in rows]
        if len(weights) != 1 or len(repeats) != len(set(repeats)):
            raise ValueError("invalid repeated case")
        output.append({**rows[0], "analysis_weight": weights.pop(),
                       "repeats": rows})
    return output


def _analysis_weights(rows: list[dict], census: list[dict], eligible,
                      aggregation: str) -> list[tuple[dict, float]]:
    if aggregation not in {"class_balanced", "population_weighted"}:
        raise ValueError("unknown aggregation")
    population = Counter(row["oracle_class"] for row in census if eligible(row))
    sampled = {name: [row for row in rows
                      if eligible(row) and row["oracle_class"] == name]
               for name in population}
    if not population or not all(sampled.values()):
        return []
    totals = {name: sum(_number(row["analysis_weight"], "analysis weight")
                        for row in values)
              for name, values in sampled.items()}
    masses = {name: (1 / len(population) if aggregation == "class_balanced"
                     else count / sum(population.values()))
              for name, count in population.items()}
    return [(row, row["analysis_weight"] / totals[name] * masses[name])
            for name, values in sampled.items() for row in values]


def summarize_results(episodes: list[dict], census: list[dict]) -> list[dict]:
    """Collapse repeats, then apply state-inclusion Hájek weights."""
    cases = _repeat_cases(episodes, census)
    metrics = {
        "modeled_relative_relief": (
            lambda row: row["p_star"] > 0,
            lambda row: row["modeled_relief_at_deadline"] / row["p_star"]),
        "measured_relative_relief": (
            lambda row: row["p_star"] > 0,
            lambda row: row["measured_relief_at_deadline"] / row["p_star"]),
        "modeled_target_attainment": (
            lambda row: row["j"], lambda row: float(row["modeled_target_attained"])),
        "measured_target_attainment": (
            lambda row: row["j"], lambda row: float(row["measured_target_attained"])),
        "selection_quality": (lambda row: True, lambda row: row["selection_quality"]),
    }
    output = []
    for metric, (eligible, value) in metrics.items():
        for policy in POLICIES:
            policy_rows = [row for row in cases
                           if row["policy"] == policy and eligible(row)]
            for aggregation in ("class_balanced", "population_weighted"):
                weighted = _analysis_weights(
                    policy_rows, census, eligible, aggregation)
                result = sum(
                    weight * sum(value(repeat) for repeat in row["repeats"])
                    / len(row["repeats"]) for row, weight in weighted
                ) if weighted else None
                output.append({
                    "policy": policy, "metric": metric,
                    "aggregation": aggregation, "value": result,
                    "episodes": sum(len(row["repeats"]) for row in policy_rows),
                    "cases": len(policy_rows), "weighting": "state_union_hajek",
                })
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
    joined = _repeat_cases(episodes, census)
    aggregations = ("class_balanced", "population_weighted")
    aliases = {"per_session_greedy": "isolated_fastest"}
    style = lambda policy: aliases.get(policy, policy)
    mean = lambda row, field: sum(
        repeat[field] for repeat in row["repeats"]) / len(row["repeats"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    eligible = lambda row: row["p_star"] > 0
    for row_index, signal in enumerate(("measured", "modeled")):
        field = f"{signal}_relief_at_deadline"
        for column, aggregation in enumerate(aggregations):
            axis = axes[row_index][column]
            for policy in POLICIES:
                policy_rows = [row for row in joined
                               if row["policy"] == policy and eligible(row)]
                values = sorted(
                    (mean(row, field) / row["p_star"], weight)
                    for row, weight in _analysis_weights(
                        policy_rows, census, eligible, aggregation))
                if values:
                    axis.step(
                        [value for value, _ in values],
                        list(itertools.accumulate(weight for _, weight in values)),
                        where="post", color=plot_style.POLICY_COLORS[style(policy)],
                        linestyle=plot_style.POLICY_LINESTYLES[style(policy)],
                        label=plot_style.POLICY_NAMES[style(policy)])
            axis.set(xlabel=r"$\Delta P(30\,\mathrm{s})/P^*$",
                     ylabel="Cumulative fraction", title=(
                         f"{plot_style.RELIEF_SIGNAL_NAMES[signal]} — "
                         f"{plot_style.AGGREGATION_NAMES[aggregation]}"))
    axes[0][0].legend(frameon=False)
    _save(fig, out, "relative_relief")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    eligible = lambda row: row["j"]
    for row_index, signal in enumerate(("measured", "modeled")):
        field = f"{signal}_target_time_s"
        for column, aggregation in enumerate(aggregations):
            axis = axes[row_index][column]
            for policy in POLICIES:
                policy_rows = [row for row in joined
                               if row["policy"] == policy and eligible(row)]
                weighted = _analysis_weights(
                    policy_rows, census, eligible, aggregation)
                points = sorted({0, D_S, *(repeat[field]
                                for row, _ in weighted for repeat in row["repeats"]
                                if repeat[field] is not None)})
                if weighted:
                    values = [sum(
                        weight * sum(repeat[field] is not None
                                     and repeat[field] <= at
                                     for repeat in row["repeats"])
                        / len(row["repeats"]) for row, weight in weighted)
                              for at in points]
                    axis.step(
                        points, values, where="post",
                        color=plot_style.POLICY_COLORS[style(policy)],
                        linestyle=plot_style.POLICY_LINESTYLES[style(policy)],
                        label=plot_style.POLICY_NAMES[style(policy)])
            axis.set(xlim=(0, D_S), ylim=(0, 1.02), xlabel="Time (s)",
                     ylabel="Cumulative target attainment", title=(
                         f"{plot_style.RELIEF_SIGNAL_NAMES[signal]} — "
                         f"{plot_style.AGGREGATION_NAMES[aggregation]}"))
    axes[0][0].legend(frameon=False)
    _save(fig, out, "target_attainment")
    plt.close(fig)

    resource_labels = sorted({label for row in joined
                              for label in json.loads(row["resource_labels"])})
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    for row_index, (field, labels, group_name) in enumerate((
            ("oracle_class", list(ORACLE_CLASSES), "Oracle class"),
            ("resource_labels", resource_labels, "Resource label"))):
        groups = [(label, policy) for label in labels for policy in POLICIES]
        x = list(range(len(groups)))
        for column, aggregation in enumerate(aggregations):
            axis = axes[row_index][column]
            weighted = {
                policy: _analysis_weights(
                    [row for row in joined if row["policy"] == policy],
                    census, lambda row: True, aggregation)
                for policy in POLICIES}
            bottom = [0.0] * len(groups)
            for action, key in (("replay", "replay_count"),
                                ("kv_transfer", "kv_count"),
                                ("not_moved", "not_moved_count")):
                values = []
                for label, policy in groups:
                    selected = [(row, weight) for row, weight in weighted[policy]
                                if (row[field] == label
                                    if field == "oracle_class"
                                    else label in json.loads(row[field]))]
                    total = sum(weight * sum(mean(row, value) for value in
                                ("replay_count", "kv_count", "not_moved_count"))
                                for row, weight in selected)
                    values.append(sum(weight * mean(row, key)
                                      for row, weight in selected) / total
                                  if total else 0)
                axis.bar(x, values, bottom=bottom,
                         color=plot_style.ACTION_COLORS[action],
                         hatch=plot_style.ACTION_HATCHES[action],
                         label=plot_style.ACTION_NAMES[action])
                bottom = [left + value for left, value in zip(bottom, values)]
            axis.set_xticks(x, [
                f"{label}\n{plot_style.COMPACT_POLICY_NAMES[style(policy)]}"
                for label, policy in groups], rotation=60, ha="right")
            axis.set(ylabel="Action fraction", title=(
                f"{group_name} — {plot_style.AGGREGATION_NAMES[aggregation]}"))
    axes[0][0].legend(frameon=False, ncol=3)
    _save(fig, out, "action_composition")
    plt.close(fig)

    keys = {
        "candidate_population": set(cases),
        "scheduled_sample": {(row["state_id"], row["pack_id"])
                             for row in episodes},
        "analyzed_sample": {(row["state_id"], row["pack_id"])
                            for row in episodes if row["evidence_valid"]},
    }
    histograms = {sample: Counter(cases[key]["oracle_class"] for key in selected)
                  for sample, selected in keys.items()}
    x = list(range(len(ORACLE_CLASSES)))
    fig, axis = plt.subplots(figsize=(10, 5))
    for offset, sample in zip((-.25, 0, .25), keys):
        axis.bar([value + offset for value in x],
                 [histograms[sample][name] for name in ORACLE_CLASSES], .25,
                 label=plot_style.SAMPLE_STAGE_NAMES[sample],
                 color=plot_style.SAMPLE_STAGE_COLORS[sample])
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
    stages = {
        "candidate_population": {(row["state_id"], row["pack_id"])
                                 for row in census},
        "scheduled_sample": {(row["state_id"], row["pack_id"])
                             for row in census if row["state_id"] in chosen},
    }
    if episodes is not None:
        stages["analyzed_sample"] = {
            (row["state_id"], row["pack_id"])
            for row in episodes if row["evidence_valid"]}
    classes = {(row["state_id"], row["pack_id"]): row["oracle_class"]
               for row in census}
    histogram = [{"sample": sample, "oracle_class": name,
                  "cases": sum(classes[key] == name for key in keys)}
                 for sample, keys in stages.items() for name in ORACLE_CLASSES]
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
        statuses = Counter(row["evidence_status"] for row in episodes)
        discordant = sum(row["static_scheduled_discordant"] for row in census)
        (out / "summary.json").write_text(json.dumps({
            "p_star_zero_cases": sum(row["p_star"] == 0 for row in census),
            "valid_episodes": statuses["valid"],
            "invalid_evidence_episodes": len(episodes) - statuses["valid"],
            "evidence_status_histogram": statuses,
            "static_schedule_discordance_cases": discordant,
            "static_schedule_discordance_rate": discordant / len(census),
            "kv_ingest_limited_by_pure_axis": {
                axis: sum(row["kv_ingest_limiting"]
                          and int(row[f"n_{axis}"]) > 0
                          and all(int(row[f"n_{other}"]) == 0
                                  for other in AXES if other != axis)
                          for row in background)
                for axis in AXES},
            "selection_shortages": selected["shortages"],
        }, indent=2, sort_keys=True) + "\n")
        plot_results(episodes, census, out)



def _frozen_selection(census: list[dict], frozen: dict) -> dict:
    selected = select_states(census, frozen["seeds"]["selection"])
    selected["inputs_hash"] = frozen["inputs_hash"]
    selected["selection_sha256"] = digest(selected)
    return selected


def compile_campaign(frozen: dict, discovery: list[dict],
                     measured_background: list[dict], out: Path, *,
                     timing) -> dict:
    """Compile measured discovery into the immutable execution schedule."""
    verify_frozen(frozen)
    verify_timing(timing, frozen)
    out.mkdir(parents=True, exist_ok=True)
    inputs = frozen["inputs"]
    limits, retained = discovery_limits(discovery, inputs["measurement_contract"])
    candidates = candidate_states(limits, frozen["seeds"]["sobol"])
    background = validate_background_states(
        measured_background, candidates, inputs["measurement_contract"])
    census = oracle_census(background, inputs["packs"],
                           inputs["action_demands"], inputs["target"], timing)
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
                    out: Path, *, timing) -> list[dict]:
    """Verify the frozen schedule and reduce completed hardware episodes."""
    verify_frozen(frozen)
    verify_timing(timing, frozen)
    inputs = frozen["inputs"]
    state_hashes = {row["state_id"]: row["state_hash"] for row in background}
    if len(state_hashes) != len(background) or any(
            state_hashes.get(row["state_id"]) != row["state_hash"]
            for row in census):
        raise ValueError("background or census state hash changed")
    if any(state_hashes.get(row["state_id"]) != row["sha256"]
           for row in selected["states"]):
        raise ValueError("selected background state hash changed")
    expected_selected = _frozen_selection(census, frozen)
    expected_schedule = execution_schedule(
        expected_selected, inputs["packs"],
        frozen["seeds"].get("policy_order", frozen["seeds"]["selection"]),
        frozen["inputs_hash"])
    if selected != expected_selected or schedule != expected_schedule:
        raise ValueError("selection or execution schedule changed")
    episodes = normalize_episodes(
        raw, schedule, inputs["packs"], inputs["action_demands"], inputs["target"],
        background, inputs["measurement_contract"], timing)
    write_outputs(out, background, census, selected, episodes,
                  frozen=frozen, schedule=schedule)
    return episodes
