"""Paired model-architecture migration campaign with hard evidence gates."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import migration_profiler as profiler
import migration_testbed as testbed
from planner import plan
from power_model import ExpectedPower
from profiles import HYBRID_PROFILE_SCHEMA, KVGeometry, ModelProfile, WorkloadProfile
from simulate import (ExecutionScenario, NetworkLink, PowerNode, ServingInstance,
                      SimSession, predict)


ROOT = Path(__file__).resolve().parent
SCHEMA = "queue-haul-model-architecture-campaign-v1"
GATE_SCHEMA = "queue-haul-model-architecture-gate-v1"
MODELS = tuple(testbed.MODEL_SPECS)
HARDWARE = ("A100", "H100")
CONTEXTS = (4096, 8192, 16384, 24576, 32256)
CONCURRENCIES = (1, 2, 4, 8)
BANDWIDTH_MBPS = (1000, 5000, 10000)
DEADLINES_S = (19, 30)
SHED_FRACTIONS = (2 / 3, 1.0)
METHODS = ("replay", "kv_transfer")
REPEATS = 3
TARGET_LOAD = .4
WORKLOADS = (
    ROOT / "profiles/coding.json",
    ROOT / "profiles/interactive_coding.json",
    ROOT / "profiles/agentic_tool_loop.json",
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def make_collection_plan(manifest_path: Path, model: str, hardware: str,
                         seed: int = 1) -> dict:
    if model not in MODELS or hardware not in HARDWARE:
        raise ValueError("unsupported model or hardware")
    result = profiler.make_plan(
        manifest_path, list(CONTEXTS), list(CONCURRENCIES),
        list(BANDWIDTH_MBPS), list(METHODS), ["none"], REPEATS, seed,
        exact_tokens=True,
    )
    for scenario in result["scenarios"]:
        sessions = [
            {**row, "initial_tokens": scenario["context_size"]}
            for row in scenario["sessions"]
        ]
        scenario.update(
            campaign="model_architecture",
            split="validation" if scenario["repeat"] == 2 else "train",
            sessions=sessions,
        )
        if scenario["kind"] == "migration":
            scenario["moves"] = [
                {**row, "method": scenario["method"]} for row in sessions
            ]
    smoke = next(
        row for row in result["scenarios"]
        if row["kind"] == "migration" and row["context_size"] == max(CONTEXTS)
        and row["move_concurrency"] == max(CONCURRENCIES)
        and row["method"] == "kv_transfer"
        and row["bandwidth_mbps"] == min(BANDWIDTH_MBPS)
        and row["repeat"] == 0
    )
    result["scenarios"].remove(smoke)
    smoke["smoke"] = True
    rest = result["scenarios"]
    random.Random(seed).shuffle(rest)
    rest.sort(key=lambda row: row["bandwidth_mbps"])
    result["scenarios"] = [smoke, *rest]
    result.update(
        campaign_schema=SCHEMA, model=model,
        revision=testbed.model_spec(model).revision, hardware=hardware,
        runtime={"precision": "bf16", "tensor_parallel": 1,
                 "max_model_len": 32768, "sessions": 8,
                 "gpu_memory_utilization": .9},
    )
    result["scenario_matrix_sha256"] = profiler.object_hash(
        result["scenarios"])
    profiler.validate_plan(
        result, json.loads(manifest_path.read_text()))
    validate_collection_plan(result)
    return result


def validate_collection_plan(value: dict) -> None:
    if value.get("campaign_schema") != SCHEMA \
            or value.get("model") not in MODELS \
            or value.get("revision") != testbed.model_spec(value["model"]).revision \
            or value.get("hardware") not in HARDWARE:
        raise ValueError("invalid campaign identity")
    scenarios = value.get("scenarios", [])
    migrations = [row for row in scenarios if row["kind"] == "migration"]
    expected = {
        (context, width, method, bandwidth, repeat)
        for context in CONTEXTS for width in CONCURRENCIES for method in METHODS
        for bandwidth in BANDWIDTH_MBPS for repeat in range(REPEATS)
    }
    actual = {
        (row["context_size"], row["move_concurrency"], row["method"],
         row["bandwidth_mbps"], row["repeat"])
        for row in migrations
    }
    if len(scenarios) != 375 or len(migrations) != 360 or actual != expected:
        raise ValueError("incomplete model-architecture collection matrix")
    if not scenarios[0].get("smoke") \
            or any(row.get("smoke") for row in scenarios[1:]) \
            or any(item.get("initial_tokens") != row["context_size"]
                   for row in scenarios for item in row["sessions"]) \
            or any(row["split"] != ("validation" if row["repeat"] == 2
                                    else "train") for row in scenarios):
        raise ValueError("campaign smoke, token, or split contract changed")


def validate_arm_plan(value: dict) -> None:
    if value.get("campaign") != "model_architecture_live":
        validate_collection_plan(value)
        return
    if value.get("campaign_schema") != SCHEMA \
            or value.get("model") not in MODELS \
            or value.get("revision") != testbed.model_spec(value["model"]).revision \
            or value.get("hardware") not in HARDWARE \
            or len(value.get("scenarios", ())) != 6 \
            or not value["scenarios"][0].get("smoke"):
        raise ValueError("invalid model-architecture live plan")
    manifest = json.loads(Path(value["manifest"]["path"]).read_text())
    profiler.validate_plan(value, manifest)


def prepare(manifest_path: Path, out_dir: Path, seed: int = 1) -> list[Path]:
    paths = []
    matrices = set()
    for hardware in HARDWARE:
        for model in MODELS:
            value = make_collection_plan(manifest_path, model, hardware, seed)
            matrices.add(value["scenario_matrix_sha256"])
            path = out_dir / f"{hardware.lower()}-{_slug(model)}.json"
            _write_json(path, value)
            paths.append(path)
    if len(matrices) != 1:
        raise RuntimeError("model/hardware arms do not share one scenario matrix")
    return paths


def validate_hardware(expected: str) -> tuple[str, ...]:
    names = tuple(line.strip() for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True).splitlines() if line.strip())
    if not names or any(expected.upper() not in name.upper() for name in names):
        raise RuntimeError(f"expected only {expected} GPUs, found {names}")
    return names


def run_profile(plan_path: Path, run_root: Path, cfg: testbed.Config,
                allow_dirty: bool, extra: list[str], smoke_only: bool = False,
                resume_from: str | None = None) -> None:
    value = json.loads(plan_path.read_text())
    validate_arm_plan(value)
    validate_hardware(value["hardware"])
    if cfg.model != value["model"]:
        raise ValueError("runtime model does not match campaign arm")
    testbed.validate_model_runtime(cfg)
    active = plan_path
    if smoke_only:
        if value.get("campaign") == "model_architecture_live":
            raise ValueError("live plans do not have a separate launch pilot")
        active = run_root / "smoke-plan.json"
        _write_json(active, {**value, "scenarios": value["scenarios"][:1]})
    profiler.run_plan(
        active, run_root, cfg, allow_dirty, extra, resume_from,
        fail_fast=True,
    )


def launch_gate(run_root: Path) -> dict:
    plan_value = json.loads((run_root / "plan.json").read_text())
    if plan_value.get("campaign_schema") != SCHEMA \
            or plan_value.get("model") not in MODELS \
            or plan_value.get("hardware") not in HARDWARE \
            or len(plan_value.get("scenarios", [])) not in {1, 375}:
        raise ValueError("invalid launch-gate plan")
    if len(plan_value["scenarios"]) == 375:
        validate_collection_plan(plan_value)
    scenario = plan_value["scenarios"][0]
    if not scenario.get("smoke") or scenario["context_size"] != max(CONTEXTS) \
            or scenario["move_concurrency"] != 8 \
            or len(scenario["sessions"]) != 8:
        raise RuntimeError("launch gate is not the 8x32K smoke cell")
    result = json.loads((run_root / "scenarios" / scenario["scenario_id"]
                         / "result.json").read_text())
    moves = result.get("migrations", [])
    chunk = testbed.model_spec(plan_value["model"]).chunk_tokens
    if result.get("status") != "complete" or len(moves) != 8 \
            or not profiler.valid_continuations(result, 8) \
            or any(row.get("error") or not row.get("initial")
                   or row["initial"]["logical_kv_bytes"] <= 0
                   or row["initial"]["processed_tokens"] >= chunk for row in moves):
        raise RuntimeError("8x32K launch/cache round-trip gate failed")
    logs = sorted((run_root / "debug").glob("testbed_*/[so][oi][un][rk]*.log"))
    text = "\n".join(path.read_text(errors="replace") for path in logs)
    cfg = testbed.model_campaign_config(plan_value["model"])
    testbed.validate_model_runtime_log(cfg, text)
    if re.search(r"out of memory|\bOOM\b", text, re.IGNORECASE):
        raise RuntimeError("launch gate contains an out-of-memory report")
    return {
        "passed": True, "model": plan_value["model"],
        "hardware": plan_value["hardware"], "sessions": 8,
        "context_tokens": max(CONTEXTS), "chunk_tokens": chunk,
    }


def _integer(value, label: str) -> int:
    number = float(value)
    if not number.is_integer() or number < 0:
        raise ValueError(f"{label} must contain nonnegative integer bytes")
    return int(number)


def geometry_evidence(rows: list[dict], migrations: list[dict], *,
                      contexts=CONTEXTS, repeats: int = REPEATS,
                      heterogeneous: bool = False) -> tuple[dict, list[list[int]]]:
    required = {
        "context_tokens", "repeat", "group", "resident_bytes",
        "capacity_bytes", "transfer_bytes",
    }
    if not rows or any(not required <= set(row) for row in rows):
        raise ValueError(f"geometry rows require {sorted(required)}")
    clean = [{
        "context": int(row["context_tokens"]), "repeat": int(row["repeat"]),
        "group": str(row["group"]),
        "resident": _integer(row["resident_bytes"], "resident_bytes"),
        "capacity": _integer(row["capacity_bytes"], "capacity_bytes"),
        "transfer": _integer(row["transfer_bytes"], "transfer_bytes"),
    } for row in rows]
    groups = sorted({row["group"] for row in clean})
    expected = {
        (context, repeat, group) for context in contexts
        for repeat in range(repeats) for group in groups
    }
    if {(row["context"], row["repeat"], row["group"])
        for row in clean} != expected or len(clean) != len(expected):
        raise ValueError("geometry evidence must contain one complete group matrix")
    if heterogeneous and len(groups) < 2:
        raise ValueError("hybrid model needs at least two measured cache groups")
    capacities = []
    for group in groups:
        values = {row["capacity"] for row in clean if row["group"] == group}
        if len(values) != 1:
            raise ValueError("group capacity changed across measurements")
        capacities.append(values.pop())
    resident = [[context, *[
        _integer(statistics.median(
            row["resident"] for row in clean
            if row["context"] == context and row["group"] == group
        ), "resident_bytes") for group in groups
    ]] for context in contexts]
    transferred = {
        (context, repeat): sum(
            row["transfer"] for row in clean
            if row["context"] == context and row["repeat"] == repeat
        ) for context in contexts for repeat in range(repeats)
    }
    measured: dict[tuple[int, int], list[int]] = {}
    for row in migrations:
        success = str(row.get("success", "")).lower() in {"true", "1"}
        if row.get("method") == "kv_transfer" \
                and row.get("activity") == "none" \
                and int(row.get("concurrency", 0)) == 1 and success:
            key = int(row["measured_prompt_tokens"]), int(row["repeat"])
            measured.setdefault(key, []).append(int(row["measured_kv_bytes"]))
    if set(measured) != set(transferred) or any(
            _integer(statistics.median(measured[key]), "measured_kv_bytes")
            != value for key, value in transferred.items()):
        raise ValueError("per-group transfer bytes do not match migration evidence")
    curve = [[context, _integer(statistics.median(
        transferred[context, repeat] for repeat in range(repeats)
    ), "transfer_bytes")] for context in contexts]
    raw = {"groups": groups, "capacity_bytes": capacities,
           "resident_bytes": resident}
    KVGeometry.parse(raw)
    return raw, curve


def _migration_frame(rows: list[dict]) -> pd.DataFrame:
    data = pd.DataFrame(rows)
    needed = {
        "method", "activity", "concurrency", "success", "repeat",
        "bandwidth_mbps", "measured_prompt_tokens", "measured_processed_tokens",
        "measured_kv_bytes", "initial_time_to_first_response_s",
    }
    if data.empty or not needed <= set(data):
        raise ValueError(f"migration evidence requires {sorted(needed)}")
    for key in needed - {"method", "activity", "success"}:
        data[key] = pd.to_numeric(data[key], errors="raise")
    data["success"] = data.success.astype(str).str.lower().isin(("true", "1"))
    return data[(data.activity == "none") & (data.concurrency == 1)
                & data.success].copy()


def _fit_kv(rows: pd.DataFrame) -> tuple[float, float]:
    size = rows.measured_kv_bytes.to_numpy(float)
    elapsed = rows.initial_time_to_first_response_s.to_numpy(float)
    network = size / (rows.bandwidth_mbps.to_numpy(float) * 1e6 / 8)
    lower = max(float(np.max(size / elapsed)), 1.0)
    upper = max(lower * 20, max(BANDWIDTH_MBPS) * 1e6 / 8)
    best = None
    for rate in np.geomspace(lower, upper, 300):
        work = np.maximum(network, size / rate)
        completion = max(0.0, float(np.median(elapsed - work)))
        error = float(np.mean((elapsed - work - completion) ** 2))
        if best is None or error < best[0]:
            best = error, rate, completion
    return float(best[1]), float(best[2])


def fit_profile_raw(base: dict, migrations: list[dict], scenarios: list[dict],
                    geometry: dict, transfer_curve: list[list[int]]) -> dict:
    raw = json.loads(json.dumps(base))
    frame = _migration_frame(migrations)
    train = frame[frame.repeat.isin((0, 1))]
    replay = train[train.method == "replay"].copy()
    replay["rate"] = (replay.measured_processed_tokens
                      / replay.initial_time_to_first_response_s)
    replay_curve = [[int(context), float(rate)] for context, rate in
                    replay.groupby("measured_prompt_tokens").rate.median().items()]
    if [row[0] for row in replay_curve] != list(CONTEXTS):
        raise ValueError("replay fit lacks the five training contexts")
    kv_rate, kv_completion = _fit_kv(train[train.method == "kv_transfer"])
    chunk = testbed.model_spec(raw["model"]).chunk_tokens
    x, y = np.asarray(transfer_curve, float).T
    block_bytes = chunk * max(1, round(float(x @ y / (x @ x))))
    scenario_frame = pd.DataFrame(scenarios)
    for key in ("repeat", "concurrency", "source_added_power_w",
                "destination_added_power_w"):
        scenario_frame[key] = pd.to_numeric(scenario_frame[key], errors="raise")
    action_power = {}
    for method in METHODS:
        selected = scenario_frame[
            (scenario_frame.kind == "migration")
            & (scenario_frame.method == method)
            & (scenario_frame.activity == "none")
            & scenario_frame.repeat.isin((0, 1))
        ]
        values = selected.groupby("concurrency")[[
            "source_added_power_w", "destination_added_power_w"
        ]].median().clip(lower=0).cummax()
        if list(map(int, values.index)) != list(CONCURRENCIES):
            raise ValueError(f"{method} action power lacks concurrency 1/2/4/8")
        action_power[method] = {
            str(int(width)): [float(source), float(destination)]
            for width, (source, destination) in values.iterrows()
        }
    for case in raw["cases"].values():
        case["replay_tps"] = {"1": replay_curve}
        case["replay_completion_s"] = 0
        old = case["kv_transfer"]
        case["kv_transfer"] = {
            **old, "block_tokens": chunk, "block_bytes": block_bytes,
            "setup_s": 0, "destination_bytes_per_s": kv_rate,
            "initial_completion_s": kv_completion,
            "bytes_by_context": transfer_curve,
        }
        case["action_power_w"].update(
            replay=action_power["replay"],
            kv_transfer=action_power["kv_transfer"],
            replay_on_request={"1": action_power["replay"]["1"]},
            catch_up=action_power["kv_transfer"],
        )
    parsed_geometry = KVGeometry.parse(geometry)
    nominal = round(max(CONTEXTS) / parsed_geometry.pressure(max(CONTEXTS)))
    profile_hash = profiler.object_hash([
        geometry, transfer_curve, replay_curve, kv_rate, kv_completion,
        action_power,
    ])[:8]
    raw.update(
        schema=HYBRID_PROFILE_SCHEMA, kv_geometry=geometry,
        kv_capacity_tokens=nominal, max_destination_replays=8,
        max_destination_kv_streams=8,
        profile_id=f"{raw['profile_id']}-architecture-{profile_hash}",
    )
    raw["sources"]["capacity"] = {
        "kind": "measured", "reference": "model-architecture kv_geometry.csv",
        "valid_range": [0, nominal], "relative_error": 0,
    }
    raw["sources"]["replay"] = {
        "kind": "measured", "reference": "model-architecture migrations.csv repeats 0-1",
        "valid_range": [CONTEXTS[0], CONTEXTS[-1]], "relative_error": 0,
    }
    raw["sources"]["kv_transfer"] = {
        "kind": "measured", "reference": "model-architecture per-group and migration evidence",
        "valid_range": [transfer_curve[0][1], transfer_curve[-1][1]],
        "relative_error": 0,
    }
    return raw


def timing_gate(profile: ModelProfile, migrations: list[dict]) -> dict:
    held = _migration_frame(migrations)
    held = held[held.repeat == 2]
    case, evaluated = profile.case(), []
    for row in held.itertuples():
        measured = float(row.initial_time_to_first_response_s)
        if row.method == "replay":
            predicted = (float(row.measured_processed_tokens)
                         / case.replay.rate(float(row.measured_prompt_tokens), 1)
                         + case.replay_completion_s)
        else:
            size = float(row.measured_kv_bytes)
            predicted = (case.kv_transfer.setup_s
                         + max(size / (float(row.bandwidth_mbps) * 1e6 / 8),
                               size / case.kv_transfer.destination_bytes_per_s)
                         + case.kv_transfer.initial_completion_s)
        evaluated.append({
            "method": row.method, "context_tokens": int(row.measured_prompt_tokens),
            "bandwidth_mbps": float(row.bandwidth_mbps),
            "measured_s": measured, "predicted_s": predicted,
            "absolute_error_s": abs(predicted - measured),
            "relative_error": abs(predicted / measured - 1),
        })
    if not evaluated:
        raise ValueError("timing gate has no held-out repeat")
    relative = [row["relative_error"] for row in evaluated]
    absolute = [row["absolute_error_s"] for row in evaluated]
    false_feasible = sum(
        row["predicted_s"] <= deadline < row["measured_s"]
        for row in evaluated for deadline in DEADLINES_S
    )
    result = {
        "median_relative_error": float(np.median(relative)),
        "p90_relative_error": float(np.quantile(relative, .9)),
        "p90_absolute_error_s": float(np.quantile(absolute, .9)),
        "false_feasible_deadlines": false_feasible,
        "rows": evaluated,
    }
    result["passed"] = (
        result["median_relative_error"] <= .10
        and result["p90_relative_error"] <= .15
        and result["p90_absolute_error_s"] <= 1
        and not false_feasible
    )
    return result


def freeze_profile(base_path: Path, run_root: Path, smoke_root: Path,
                   geometry_path: Path, out_path: Path) -> dict:
    if not (run_root / "migrations.csv").exists():
        profiler.reduce_run(run_root, None)
    base = json.loads(base_path.read_text())
    base_profile = ModelProfile.load(base_path)
    plan_value = json.loads((run_root / "plan.json").read_text())
    if base_profile.model != plan_value["model"] \
            or plan_value["hardware"].lower() not in base_profile.hardware.lower() \
            or base_profile.precision.lower() not in {"bf16", "bfloat16"} \
            or base_profile.tensor_parallel != 1:
        raise ValueError("base profile does not match the model/hardware arm")
    launch = launch_gate(smoke_root)
    migrations = _rows(run_root / "migrations.csv")
    geometry, curve = geometry_evidence(
        _rows(geometry_path), migrations,
        heterogeneous=plan_value["model"] != testbed.MODEL,
    )
    raw = fit_profile_raw(
        base, migrations, _rows(run_root / "scenarios.csv"), geometry, curve)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    _write_json(temporary, raw)
    candidate = ModelProfile.load(temporary)
    timing = timing_gate(candidate, migrations)
    for method in METHODS:
        errors = [row["relative_error"] for row in timing["rows"]
                  if row["method"] == method]
        raw["sources"][method]["relative_error"] = float(
            np.quantile(errors, .9))
    _write_json(temporary, raw)
    candidate = ModelProfile.load(temporary)
    gate = {
        "schema": GATE_SCHEMA, "passed": timing["passed"],
        "model": candidate.model, "hardware": plan_value["hardware"],
        "launch": launch, "timing": timing,
        "scalar_kv_residual": candidate.case().kv_transfer.scalar_residual(),
        "kv_representation": "measured_curve",
    }
    gate_path = out_path.with_suffix(".gate.json")
    if not timing["passed"]:
        _write_json(gate_path, gate)
        temporary.unlink()
        raise RuntimeError("held-out migration timing gate failed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(out_path)
    gate["profile_sha256"] = profiler.file_hash(out_path)
    _write_json(gate_path, gate)
    return gate


def session_shapes(workload: WorkloadProfile, repeat: int,
                   count: int = 8) -> tuple[dict, ...]:
    if repeat < 0 or count < 1:
        raise ValueError("repeat and session count must be nonnegative/positive")
    rng = np.random.default_rng(profiler.stable_seed(workload.profile_id, repeat))
    low = min(row.context_tokens for row in workload.records)
    high = max(row.context_tokens for row in workload.records)
    result = []
    for i in range(count):
        record = workload.records[int(rng.integers(len(workload.records)))]
        context = int(rng.integers(low, high + 1))
        cycle = record.request_gap_s + record.tool_delay_s
        if cycle <= 0:
            raise ValueError("workload request cycle must be positive")
        result.append({
            "session_id": f"s{i}", "context_tokens": context,
            "raw_f": record.prompt_tokens / cycle,
            "raw_g": record.output_tokens / cycle,
            "log_bytes": max(1, round(record.log_bytes * context
                                      / record.context_tokens)),
        })
    return tuple(result)


def arrival_scale(profile: ModelProfile, shapes: tuple[dict, ...],
                  target: float = TARGET_LOAD) -> float:
    case = profile.case()
    load = sum(case.service_load(row["raw_f"], row["raw_g"])
               for row in shapes)
    if load <= 0 or not 0 < target <= profile.max_ell:
        raise ValueError("invalid workload or utilization target")
    return target / load


def sim_sessions(shapes: tuple[dict, ...], scale: float) -> tuple[SimSession, ...]:
    if scale <= 0:
        raise ValueError("arrival scale must be positive")
    return tuple(SimSession(
        row["session_id"], "source", row["context_tokens"],
        row["raw_f"] * scale, row["raw_g"] * scale, row["log_bytes"],
    ) for row in shapes)


def _serviceable(profile: ModelProfile, sessions: tuple[SimSession, ...]) -> bool:
    case = profile.case()
    f, g = (sum(getattr(row, key) for row in sessions)
            for key in ("expected_f", "expected_g"))
    load = case.power_load(f, g)
    return bool(
        load <= profile.max_power_load + 1e-9
        and (case.phase_power is None or case.phase_power.contains(f, g))
        and (profile.kv_geometry is None
             or profile.kv_geometry.fits(row.context_tokens for row in sessions))
        and sum(profile.kv_admission_tokens(row.context_tokens)
                for row in sessions) <= profile.kv_capacity_tokens
    )


def _scenario(profile: ModelProfile, sessions: tuple[SimSession, ...],
              bandwidth_mbps: float, deadline_s: float,
              shed_fraction: float) -> ExecutionScenario:
    nodes = (PowerNode("source-node", 1, True),
             PowerNode("sink-node", 1, False))
    instances = (ServingInstance("source", ("source-node",)),
                 ServingInstance("sink", ("sink-node",)))
    shell = ExecutionScenario(
        deadline_s, deadline_s, 0, "awake", 0, nodes, instances, sessions,
        (NetworkLink("wan", bandwidth_mbps * 1e6 / 8),),
    )
    state = ExpectedPower(shell, profile)
    initial, idle = state.power(True), profile.case().power(0)
    limit = idle + (1 - shed_fraction) * (initial - idle)
    return replace(shell, power_limit_w=limit)


def _cell(profile: ModelProfile, sessions: tuple[SimSession, ...], *,
          hardware: str, model: str, variant: str, workload: str,
          load_mode: str, repeat: int, bandwidth_mbps: float,
          deadline_s: float, shed_fraction: float) -> dict:
    base = {
        "hardware": hardware, "model": model, "variant": variant,
        "profile_id": profile.profile_id, "workload": workload,
        "load_mode": load_mode, "repeat": repeat,
        "bandwidth_mbps": bandwidth_mbps, "deadline_s": deadline_s,
        "shed_fraction": shed_fraction,
        "sessions": [{
            "session_id": row.session_id, "context_tokens": row.context_tokens,
            "expected_f": row.expected_f, "expected_g": row.expected_g,
            "log_bytes": row.log_bytes,
        } for row in sessions],
    }
    if not _serviceable(profile, sessions):
        return {**base, "initial_serviceable": False, "feasible": False,
                "kv_share": 0.0, "replay_share": 0.0,
                "predicted_makespan_s": None, "moves": []}
    scenario = _scenario(
        profile, sessions, bandwidth_mbps, deadline_s, shed_fraction)
    result = plan(scenario, profile, {("source", "sink"): ("wan",)},
                  "lp_work_first")
    execution = predict(scenario, profile, result.moves)
    methods = [row.method for row in result.moves]
    total = len(methods)
    moved_contexts = {
        row.session_id: row.context_tokens for row in sessions
    }
    utilization = profile.kv_geometry.utilization(
        moved_contexts[row.session_id] for row in result.moves
    ).tolist() if profile.kv_geometry else []
    return {
        **base, "initial_serviceable": True, "feasible": result.feasible,
        "initial_power_w": result.initial_source_power_w,
        "power_limit_w": scenario.power_limit_w,
        "kv_share": methods.count("kv_transfer") / total if total else 0.0,
        "replay_share": methods.count("replay") / total if total else 0.0,
        "predicted_makespan_s": execution.migration_makespan_s,
        "destination_group_utilization": utilization,
        "moves": [{"session_id": row.session_id, "method": row.method}
                  for row in result.moves],
    }


def _representation(base, donor):
    return replace(
        base, block_tokens=donor.block_tokens, block_bytes=donor.block_bytes,
        bytes_by_context=donor.bytes_by_context,
    )


def counterfactual(reference: ModelProfile, donor: ModelProfile,
                   component: str) -> ModelProfile:
    if component == "cache_only":
        case = replace(
            reference.case(), kv_transfer=_representation(
                reference.case().kv_transfer, donor.case().kv_transfer))
        return replace(
            reference, profile_id=f"{donor.profile_id}-cache-only",
            cases={"central": case}, kv_geometry=donor.kv_geometry,
            kv_capacity_tokens=donor.kv_capacity_tokens,
        )
    if component == "compute_only":
        case = replace(
            donor.case(), kv_transfer=_representation(
                donor.case().kv_transfer, reference.case().kv_transfer))
        return replace(
            donor, profile_id=f"{donor.profile_id}-compute-only",
            cases={"central": case}, kv_geometry=reference.kv_geometry,
            kv_capacity_tokens=reference.kv_capacity_tokens,
        )
    raise ValueError("counterfactual component must be cache_only or compute_only")


def screen(profiles: dict[tuple[str, str], ModelProfile],
           workloads: tuple[WorkloadProfile, ...], *,
           bandwidths=BANDWIDTH_MBPS, deadlines=DEADLINES_S,
           shed_fractions=SHED_FRACTIONS, repeats: int = REPEATS,
           include_counterfactuals: bool = True) -> list[dict]:
    if set(profiles) != {(hardware, model) for hardware in HARDWARE
                         for model in MODELS}:
        raise ValueError("screening requires all six model/hardware profiles")
    rows = []
    for hardware in HARDWARE:
        reference = profiles[hardware, testbed.MODEL]
        variants = [(model, "both", profiles[hardware, model])
                    for model in MODELS]
        if include_counterfactuals:
            variants += [
                (model, component, counterfactual(
                    reference, profiles[hardware, model], component))
                for model in MODELS[1:]
                for component in ("cache_only", "compute_only")
            ]
        for workload in workloads:
            for repeat in range(repeats):
                shapes = session_shapes(workload, repeat)
                fixed = arrival_scale(reference, shapes)
                for model, variant, profile in variants:
                    for load_mode in ("fixed", "matched"):
                        scale = fixed if load_mode == "fixed" \
                            else arrival_scale(profile, shapes)
                        sessions = sim_sessions(shapes, scale)
                        for bandwidth in bandwidths:
                            for deadline in deadlines:
                                for fraction in shed_fractions:
                                    rows.append(_cell(
                                        profile, sessions, hardware=hardware,
                                        model=model, variant=variant,
                                        workload=workload.profile_id,
                                        load_mode=load_mode, repeat=repeat,
                                        bandwidth_mbps=bandwidth,
                                        deadline_s=deadline,
                                        shed_fraction=fraction,
                                    ))
    return rows


def _pair_key(row: dict) -> tuple:
    return tuple(row[key] for key in (
        "hardware", "workload", "load_mode", "repeat", "bandwidth_mbps",
        "deadline_s", "shed_fraction"))


def _interval(values: list[float], samples: int) -> tuple[float, float]:
    rng = np.random.default_rng(1)
    data = np.asarray(values, float)
    means = np.mean(rng.choice(data, (samples, len(data))), axis=1)
    return tuple(map(float, np.quantile(means, (.025, .975))))


def campaign_gate(rows: list[dict], bootstrap_samples: int = 2000) -> dict:
    main = [row for row in rows if row["variant"] == "both"
            and row["initial_serviceable"]]
    indexed = {(row["model"], _pair_key(row)): row for row in main}
    comparisons = []
    for hardware in HARDWARE:
        for model in MODELS[1:]:
            pairs = [
                (row, indexed.get((testbed.MODEL, _pair_key(row))))
                for row in main
                if row["hardware"] == hardware and row["model"] == model
            ]
            pairs = [(left, right) for left, right in pairs if right]
            if not pairs:
                raise ValueError(f"no paired rows for {hardware}/{model}")
            cells: dict[tuple, list[float]] = {}
            for left, right in pairs:
                key = tuple(left[name] for name in (
                    "workload", "load_mode", "bandwidth_mbps", "deadline_s",
                    "shed_fraction"))
                cells.setdefault(key, []).append(
                    left["kv_share"] - right["kv_share"])
            effects = []
            for key, differences in cells.items():
                lo, hi = _interval(differences, bootstrap_samples)
                shift = float(np.mean(differences))
                effects.append({
                    "cell": list(key), "mean_kv_share_change": shift,
                    "ci95": [lo, hi],
                    "material": abs(shift) >= .10 and lo * hi > 0,
                })
            strongest = max(effects, key=lambda row: abs(
                row["mean_kv_share_change"]))
            comparisons.append({
                "hardware": hardware, "model": model,
                "mean_kv_share_change": strongest["mean_kv_share_change"],
                "ci95": strongest["ci95"],
                "material_action_shift": any(row["material"] for row in effects),
                "material_cells": sum(row["material"] for row in effects),
                "feasibility_flips": sum(left["feasible"] != right["feasible"]
                                         for left, right in pairs),
            })
    groups: dict[tuple, list[dict]] = {}
    for row in main:
        key = tuple(row[name] for name in (
            "hardware", "model", "workload", "load_mode", "repeat",
            "shed_fraction"))
        groups.setdefault(key, []).append(row)
    crossovers = [{
        "hardware": key[0], "model": key[1], "workload": key[2],
        "load_mode": key[3], "repeat": key[4], "shed_fraction": key[5],
    } for key, group in groups.items()
        if min(row["kv_share"] for row in group) < .5
        and max(row["kv_share"] for row in group) >= .5
        and {row["bandwidth_mbps"] for row in group} <= set(BANDWIDTH_MBPS)
        and {row["deadline_s"] for row in group} <= set(DEADLINES_S)]
    interesting = bool(crossovers) and any(
        row["material_action_shift"] or row["feasibility_flips"]
        for row in comparisons)
    return {
        "schema": GATE_SCHEMA, "passed": interesting,
        "crossover_cells": crossovers, "comparisons": comparisons,
        "claim": "architecture/deployment behavior; not a causal sparsity effect",
    }


def _selection_key(row: dict) -> tuple:
    return tuple(row[name] for name in (
        "hardware", "workload", "load_mode", "bandwidth_mbps",
        "deadline_s", "shed_fraction"))


def select_live_rows(rows: list[dict]) -> list[dict]:
    main = [row for row in rows if row["variant"] == "both"]
    groups: dict[tuple, list[dict]] = {}
    for row in main:
        groups.setdefault(_selection_key(row), []).append(row)
    selected = []
    for hardware in HARDWARE:
        candidates = []
        for key, group in groups.items():
            if key[0] != hardware or len(group) != len(MODELS) * REPEATS \
                    or not all(row["initial_serviceable"] for row in group):
                continue
            shares = [statistics.median(
                row["kv_share"] for row in group if row["model"] == model)
                for model in MODELS]
            flips = sum(len({row["feasible"] for row in group
                             if row["repeat"] == repeat}) > 1
                        for repeat in range(REPEATS))
            margins = [abs((row["predicted_makespan_s"] or 0)
                           - row["deadline_s"]) for row in group]
            candidates.append((max(shares) - min(shares), flips,
                               -min(margins), key, group))
        if len(candidates) < 2:
            raise RuntimeError(f"{hardware} lacks two serviceable live cells")
        first = max(candidates, key=lambda item: (item[0], item[1], item[2]))
        remaining = [item for item in candidates if item[3] != first[3]]
        flipping = [item for item in remaining if item[1]]
        second = max(flipping or remaining,
                     key=lambda item: (item[1], item[2], item[0]))
        selected.extend(first[4] + second[4])
    if len(selected) != 36:
        raise RuntimeError("live subset must contain 36 paired executions")
    return sorted(selected, key=lambda row: (
        row["hardware"], row["model"], row["repeat"], _selection_key(row)))


def make_live_plans(rows: list[dict], manifest_path: Path,
                    out_dir: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text())
    profiler.validate_manifest(manifest)
    identities = sorted(manifest["sessions"], key=lambda row: row["id"])
    if len(identities) != 8:
        raise ValueError("live campaign requires the same eight-session manifest")
    paths = []
    for hardware in HARDWARE:
        for model in MODELS:
            arm = [row for row in rows
                   if row["hardware"] == hardware and row["model"] == model]
            scenarios = []
            for row in arm:
                mapping = {f"s{i}": item for i, item in enumerate(identities)}
                sessions = [{
                    "session_id": mapping[item["session_id"]]["id"],
                    "job_class": mapping[item["session_id"]]["job_class"],
                    "turn_index": 0, "initial_tokens": item["context_tokens"],
                    "order": i,
                } for i, item in enumerate(row["sessions"])]
                match = profiler.object_hash([
                    row[key] for key in (
                        "hardware", "workload", "load_mode", "repeat",
                        "bandwidth_mbps", "deadline_s", "shed_fraction")
                ])[:16]
                methods = {item["session_id"]: item["method"]
                           for item in row["moves"]}
                moves = [{**item, "method": methods[f"s{i}"]}
                         for i, item in enumerate(sessions) if f"s{i}" in methods]
                scenarios.append({
                    "scenario_id": f"a-{profiler.object_hash([match, model])[:16]}",
                    "match_id": match, "campaign": "model_architecture_live",
                    "split": "validation", "context_size": max(
                        item["initial_tokens"] for item in sessions),
                    "activity": "none", "activity_tokens": 0,
                    "request_schedule": [], "repeat": row["repeat"],
                    "deadline_s": row["deadline_s"], "sessions": sessions,
                    "serving_concurrency": 8, "concurrency": 8,
                    "move_concurrency": 8, "copy_policy": "initial_final",
                    "final_state": "awake", "bandwidth_mbps": row["bandwidth_mbps"],
                    "kind": "migration", "method": "mixed", "moves": moves,
                    "allow_partial_moves": True,
                    "predicted_feasible": row["feasible"],
                    "predicted_makespan_s": row["predicted_makespan_s"],
                })
            if len(scenarios) != 6:
                raise RuntimeError("each live arm requires six scenarios")
            scenarios.sort(key=lambda row: (-row["context_size"], row["scenario_id"]))
            scenarios[0]["smoke"] = True
            value = {
                "schema": profiler.PLAN_SCHEMA, "campaign_schema": SCHEMA,
                "campaign": "model_architecture_live", "model": model,
                "revision": testbed.model_spec(model).revision,
                "hardware": hardware, "seed": 1,
                "manifest": {"path": str(manifest_path),
                             "sha256": profiler.file_hash(manifest_path)},
                "scenarios": scenarios,
            }
            profiler.validate_plan(value, manifest)
            path = out_dir / f"live-{hardware.lower()}-{_slug(model)}.json"
            _write_json(path, value)
            paths.append(path)
    return paths


def plot_screen(rows: list[dict], stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plot_style

    plot_style.apply()
    main = [row for row in rows if row["variant"] == "both"
            and row["initial_serviceable"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, hardware in zip(axes, HARDWARE):
        for model in MODELS:
            points = []
            for bandwidth in BANDWIDTH_MBPS:
                values = [row["kv_share"] for row in main
                          if row["hardware"] == hardware and row["model"] == model
                          and row["bandwidth_mbps"] == bandwidth]
                if values:
                    points.append((bandwidth / 1000, statistics.median(values)))
            if points:
                axis.plot(*zip(*points), marker=plot_style.MODEL_MARKERS[model],
                          color=plot_style.MODEL_COLORS[model],
                          linestyle=plot_style.MODEL_LINESTYLES[model],
                          label=plot_style.MODEL_NAMES[model])
        axis.set(title=hardware, xlabel="Site link (Gbit/s)", ylim=(-.03, 1.03))
    axes[0].set_ylabel("KV-transfer share of planned moves")
    axes[1].legend()
    fig.tight_layout()
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=plot_style.SAVE_DPI)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)


def _load_arms(values: list[list[str]]) -> dict[tuple[str, str], ModelProfile]:
    profiles = {}
    for model, hardware, profile_path, gate_path in values:
        path, evidence = Path(profile_path), json.loads(Path(gate_path).read_text())
        profile = ModelProfile.load(path)
        raw = json.loads(path.read_text())
        if model not in MODELS or hardware not in HARDWARE \
                or profile.model != model or hardware.lower() not in profile.hardware.lower() \
                or raw.get("schema") != HYBRID_PROFILE_SCHEMA \
                or profile.precision.lower() not in {"bf16", "bfloat16"} \
                or profile.tensor_parallel != 1 \
                or evidence.get("schema") != GATE_SCHEMA or not evidence.get("passed") \
                or evidence.get("profile_sha256") != profiler.file_hash(path) \
                or tuple(map(int, profile.kv_geometry.contexts)) != CONTEXTS \
                or tuple(point[0] for point in
                         profile.case().kv_transfer.bytes_by_context) != CONTEXTS \
                or model != testbed.MODEL and len(profile.kv_geometry.groups) < 2:
            raise ValueError(f"profile arm failed its evidence contract: {model}/{hardware}")
        profiles[hardware, model] = profile
    return profiles


def screen_to_dir(arms: list[list[str]], workload_paths: list[Path],
                  manifest_path: Path, out_dir: Path) -> dict:
    profiles = _load_arms(arms)
    workloads = tuple(WorkloadProfile.load(path) for path in workload_paths)
    rows = screen(profiles, workloads)
    out_dir.mkdir(parents=True, exist_ok=True)
    profiler.write_csv(out_dir / "screen.csv", rows)
    gate = campaign_gate(rows)
    _write_json(out_dir / "gate.json", gate)
    plot_screen(rows, out_dir / "action_mix")
    if not gate["passed"]:
        raise RuntimeError("model campaign lacks an in-grid material effect")
    live = select_live_rows(rows)
    profiler.write_csv(out_dir / "live_selection.csv", live)
    make_live_plans(live, manifest_path, out_dir / "live_plans")
    return gate


def validate_live(runs: list[list[str]], out_path: Path) -> dict:
    if {(hardware, model) for model, hardware, _ in runs} != {
            (hardware, model) for hardware in HARDWARE for model in MODELS}:
        raise ValueError("live validation requires all six run roots")
    rows = []
    for model, hardware, root_text in runs:
        root = Path(root_text)
        value = json.loads((root / "plan.json").read_text())
        if value.get("model") != model or value.get("hardware") != hardware \
                or len(value["scenarios"]) != 6:
            raise ValueError("live run identity or cardinality changed")
        for scenario in value["scenarios"]:
            result = json.loads((root / "scenarios" / scenario["scenario_id"]
                                 / "result.json").read_text())
            expected = len(scenario["moves"])
            passed = result.get("status") == "complete" \
                and len(result.get("migrations", [])) == expected \
                and all(not row.get("error")
                        for row in result.get("migrations", [])) \
                and profiler.valid_continuations(result, expected)
            rows.append({
                "model": model, "hardware": hardware,
                "scenario_id": scenario["scenario_id"], "passed": passed,
                "predicted_feasible": scenario["predicted_feasible"],
                "deadline_met": result.get("deadline_met", False),
                "false_feasible": (scenario["predicted_feasible"]
                                   and not result.get("deadline_met", False)),
            })
    gate = {"schema": GATE_SCHEMA, "passed": all(
        row["passed"] and not row["false_feasible"] for row in rows),
        "rows": rows}
    _write_json(out_path, gate)
    if not gate["passed"]:
        raise RuntimeError("live model-architecture validation failed")
    return gate


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.add_argument("--manifest", type=Path,
                         default=ROOT / "outputs/coding-manifest.json")
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--seed", type=int, default=1)
    command = commands.add_parser("run-profile")
    command.add_argument("--plan", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--smoke-only", action="store_true")
    command.add_argument("--allow-dirty", action="store_true")
    command.add_argument("--resume-from-git-sha")
    testbed.add_common(command)
    command.set_defaults(model=None)
    command.add_argument("extra_vllm_args", nargs=argparse.REMAINDER)
    command = commands.add_parser("freeze-profile")
    command.add_argument("--base-profile", type=Path, required=True)
    command.add_argument("--run-root", type=Path, required=True)
    command.add_argument("--smoke-root", type=Path, required=True)
    command.add_argument("--geometry", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command = commands.add_parser("screen")
    command.add_argument("--arm", nargs=4, action="append", required=True,
                         metavar=("MODEL", "HARDWARE", "PROFILE", "GATE"))
    command.add_argument("--workload", type=Path, nargs="+", default=WORKLOADS)
    command.add_argument("--manifest", type=Path,
                         default=ROOT / "outputs/coding-manifest.json")
    command.add_argument("--out-dir", type=Path, required=True)
    command = commands.add_parser("validate-live")
    command.add_argument("--run", nargs=3, action="append", required=True,
                         metavar=("MODEL", "HARDWARE", "ROOT"))
    command.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        for path in prepare(args.manifest, args.out_dir, args.seed):
            print(path)
    elif args.command == "run-profile":
        value = json.loads(args.plan.read_text())
        if args.model is not None and args.model != value["model"]:
            raise ValueError("--model does not match the pinned plan")
        base = testbed.config_from_args(args)
        spec = testbed.model_spec(value["model"])
        cfg = replace(
            base, model=value["model"], max_model_len=32768,
            max_num_seqs=8, max_num_batched_tokens=spec.batched_tokens,
            architecture_campaign=True,
        )
        extra = args.extra_vllm_args[1:] \
            if args.extra_vllm_args[:1] == ["--"] else args.extra_vllm_args
        run_profile(
            args.plan, args.run_root, cfg, args.allow_dirty, extra,
            args.smoke_only, args.resume_from_git_sha,
        )
    elif args.command == "freeze-profile":
        freeze_profile(args.base_profile, args.run_root, args.smoke_root,
                       args.geometry, args.out)
    elif args.command == "screen":
        screen_to_dir(args.arm, args.workload, args.manifest, args.out_dir)
    else:
        validate_live(args.run, args.out)


if __name__ == "__main__":
    main()
