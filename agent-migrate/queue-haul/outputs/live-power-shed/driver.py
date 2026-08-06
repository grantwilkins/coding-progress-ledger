"""Full-shed migration power trace with both sites carrying live agentic load.

Source runs 4 rps of agentic-tool-loop traffic, destination runs 1 rps of its own
tenants throughout, then all eight sessions shed to the destination, the
destination takes over the source's traffic, and the source drains. Both engines
are kv_both with a >32 GB LMCache L1 so the handoff is visible in the cache.
"""
from __future__ import annotations

import collections
import csv
import datetime as dt
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import threading

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO = Path(__file__).resolve().parents[3]
QH = REPO / "queue-haul"
POWERTRACE = REPO.parents[1] / "powertrace-sim"
sys.path.insert(0, str(QH))

import destination_runner as destination
import migration_profiler as profiler
import migration_testbed as testbed


# Default is the queue_haul LP arm: 3 replay + 5 kv_transfer of 8 at 10 Gbps.
# p-bc461d9bb1fc33c6 is the replay_only baseline arm for the same context pack.
SCENARIO_ID = os.environ.get("QH_SCENARIO_ID", "p-09cfe127f6c1e59f")
PLAN = QH / "outputs/policy-hardware-width8-packing-plan/plan.json"
BUNDLE = QH / "outputs/destination-v7-20260722/content-free-manifest.json"
PROFILE = QH / "outputs/destination-v7-20260722/baseline-profile.json"
JOB_CLASS = "agentic_tool_loop"
# Twelve sessions, not all three splits: the foreground KV working set has to
# leave room for an uncapped-enough L2, which the kv_transfer moves ride on.
SPLITS = ("validation", "tune")
SOURCE_RPS, DEST_RPS = 4.0, 1.0
BUSY_APPEND_TOKENS, BUSY_OUTPUT_TOKENS = 2048, 32
SOURCE_INFLIGHT, DEST_INFLIGHT = 64, 48
STEADY_S, POST_S, MIGRATION_DEADLINE_S = 300, 300, 30
SOURCE, DESTINATION = "#E98300", "#007C92"
WINDOWS = (
    ("migration", "migration_start", "migration_complete", "#DAD7CB", .5,
     "Migration"),
    ("switch", "switch_start", "traffic_switched", "#007C92", .12, "Switch"),
    ("source_fall", "traffic_switched", "source_drained", "#6F4E7C", .12,
     "Power down"),
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class Markers:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", buffering=1)
        self.rows = []

    def add(self, event: str, **fields) -> dict:
        row = {
            "event": event,
            "monotonic_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
            **fields,
        }
        self.rows.append(row)
        self.handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(json.dumps(row), flush=True)
        return row

    def close(self) -> None:
        self.handle.close()


def gpu_uuids(devices: list[str]) -> list[str]:
    if not devices:
        rows = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            text=True,
        ).splitlines()
        if len(rows) != 2:
            raise RuntimeError(f"expected two GPUs, saw {len(rows)}")
        return [row.strip() for row in rows]
    return [subprocess.check_output(
        ["nvidia-smi", "-i", device, "--query-gpu=uuid",
         "--format=csv,noheader,nounits"], text=True,
    ).strip() for device in devices]


def start_power(local_root: Path, devices: list[str]):
    local_root.mkdir(parents=True, exist_ok=True)
    power = (local_root / "power.csv").open("w")
    stderr = (local_root / "power.stderr").open("w")
    command = [
        str(REPO / ".venv/bin/python"), "-u",
        str(POWERTRACE / "profiling/client/power_logger.py"),
        "--interval-ms", "100", "--profile", "core_timed_state",
    ]
    if devices:
        command += ["--gpu-ids", ",".join(devices)]
    proc = subprocess.Popen(command, stdout=power, stderr=stderr)
    return proc, power, stderr


def stop_power(proc, power, stderr) -> int:
    if proc.poll() is None:
        proc.terminate()
    try:
        status = proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        status = proc.wait()
    power.close()
    stderr.close()
    return status


def check(load: destination.DestinationLoad, label: str) -> None:
    if load.failure or load.sampler.error:
        raise RuntimeError(f"{label} failed") from (load.failure or load.sampler.error)


def wait_serving(load: destination.DestinationLoad, completions: int = 5,
                 seconds: float = 300, label: str = "destination") -> None:
    """A below-capacity background load is ready once it is completing requests."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check(load, label)
        if len(load.rows) >= completions:
            return
        time.sleep(.25)
    raise RuntimeError(f"{label} load completed fewer than {completions} requests")


def wait_drained(load: destination.DestinationLoad, seconds: float = 300,
                 label: str = "source") -> int:
    """Drained means the GPU is doing no work and holds no KV blocks.

    num_requests_waiting is deliberately not part of the predicate: an LMCache
    L2 eviction can strand a request in the connector, where it is counted as
    waiting but is never scheduled and consumes no GPU. Those stragglers are
    reported, not waited on.
    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check(load, label)
        rows = load.sampler.rows[-8:]
        if len(rows) == 8 and all(
            row["vllm:num_requests_running"] == 0
            and row["vllm:gpu_cache_usage_perc"] == 0 for row in rows
        ):
            return int(rows[-1]["vllm:num_requests_waiting"])
        time.sleep(.25)
    raise RuntimeError(f"{label} engine did not drain")


def hold(loads: list[destination.DestinationLoad], seconds: int, label: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for load in loads:
            check(load, label)
        remaining = deadline - time.monotonic()
        print(f"{label}: {max(0, int(remaining))} s remaining", flush=True)
        time.sleep(min(30, max(0, remaining)))


def busy_sessions(sessions):
    return [replace(session, append_tokens=BUSY_APPEND_TOKENS,
                   output_tokens=BUSY_OUTPUT_TOKENS)
            for session in sessions]


def load_inputs():
    plan = json.loads(PLAN.read_text())
    scenario = next(
        row for row in plan["scenarios"] if row["scenario_id"] == SCENARIO_ID
    )
    if scenario["method"] not in ("replay", "mixed", "kv_transfer") \
            or len(scenario["moves"]) != 8 or len(scenario["sessions"]) != 8 \
            or scenario["bandwidth_mbps"] != 10_000:
        raise RuntimeError("selected full-shed scenario changed")
    manifest_path = REPO / plan["manifest"]["path"]
    if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("migration manifest changed")
    manifest = json.loads(manifest_path.read_text())
    bundle = json.loads(BUNDLE.read_text())
    profile = json.loads(PROFILE.read_text())
    sessions = sum((
        destination.manifest_sessions(bundle, JOB_CLASS, split, 201088, 7)
        for split in SPLITS
    ), [])
    sessions = busy_sessions(sessions)
    context = round(statistics.mean(row.prefix_tokens for row in sessions))
    rates = (
        destination.profile_rate(profile, "prefill", context),
        destination.profile_rate(profile, "decode", context),
    )
    work = statistics.mean(
        row.append_tokens / rates[0] + row.output_tokens / rates[1]
        for row in sessions
    )
    return scenario, manifest, sessions, context, rates, work


def make_load(cfg, port, sessions, rates, work, root, seed, rps, inflight):
    return destination.DestinationLoad(
        cfg.host, port, cfg.model, sessions, rps * work, *rates, root, seed,
        rps=rps, max_inflight=inflight, bypass_lmcache=True, chat=True,
    )


def live_scenario(scenario):
    return {**scenario, "reset_caches": False,
            "verify_continuations": False, "wait_cache_idle": False,
            "warm_on_move": False, "prestage_all": True,
            "warm_concurrency": 8, "sample_power": False, "final_state": "awake",
            "deadline_s": MIGRATION_DEADLINE_S}


def switch_traffic(source, takeover, markers):
    markers.add("switch_start")
    takeover.resume()
    source.pause()
    markers.add("traffic_switched")


def parse_wall(value: str) -> float:
    return dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f").timestamp()


def parseable(row: dict) -> bool:
    try:
        parse_wall(row["timestamp"]), parse_wall(row["query.start"]), \
            parse_wall(row["query.end"]), float(row["power.draw [W]"])
    except (ValueError, TypeError, KeyError):
        return False
    return True


def load_power(run_root: Path, role_uuids: list[str]):
    # A logger killed mid-write can leave one very long torn line; read it and
    # discard it in `parseable` rather than letting csv refuse the whole file.
    csv.field_size_limit(1 << 24)
    raw = list(csv.DictReader((run_root / "power_100ms.csv").open()))
    rows = [row for row in raw if parseable(row)]
    # A logger killed mid-write leaves one torn line; anything more is corruption.
    if len(raw) - len(rows) > 2:
        raise RuntimeError(f"power log has {len(raw) - len(rows)} unparseable rows")
    if not rows:
        raise RuntimeError("100 ms power logger wrote no rows")
    samples = collections.defaultdict(list)
    for row in rows:
        if row["uuid"] not in role_uuids:
            raise RuntimeError(f"power log contains unallocated GPU {row['uuid']}")
        samples[row["query.start"], row["query.end"]].append(row)
    incomplete = [key for key, values in samples.items()
                  if sorted(row["uuid"] for row in values) != sorted(role_uuids)]
    if len(incomplete) > max(1, len(raw) - len(rows)) or any(
        len(samples[key]) != 1 for key in incomplete
    ):
        raise RuntimeError(f"power log has {len(incomplete)} incomplete ticks")
    ticks = sorted(((parse_wall(values[0]["timestamp"]), values)
                    for key, values in samples.items() if key not in incomplete),
                   key=lambda item: item[0])
    selected = []
    for tick in ticks:
        if not selected or tick[0] - selected[-1][0] >= .08:
            selected.append(tick)
    rows = [row for _, values in selected for row in values]
    by_uuid = {uuid: [row for row in rows if row["uuid"] == uuid]
               for uuid in role_uuids}
    if any(not values for values in by_uuid.values()):
        raise RuntimeError("100 ms power log is missing an allocated GPU")
    return rows, by_uuid


def series(by_uuid, uuid, origin):
    return ([parse_wall(row["timestamp"]) - origin for row in by_uuid[uuid]],
            [float(row["power.draw [W]"]) for row in by_uuid[uuid]])


def bin_power(rows: list[dict], origin: float) -> tuple[list[float], list[float]]:
    bins = collections.defaultdict(list)
    for row in rows:
        elapsed = parse_wall(row["timestamp"]) - origin
        if elapsed >= 0:
            bins[int(elapsed)].append(float(row["power.draw [W]"]))
    return ([second + .5 for second in sorted(bins)],
            [statistics.mean(bins[second]) for second in sorted(bins)])


def settle_time(times, watts, start, floor, band=15.0, dwell=5.0):
    """First time after `start` that power stays within `band` W of `floor`."""
    since = None
    for moment, value in zip(times, watts):
        if moment < start:
            continue
        if value <= floor + band:
            since = moment if since is None else since
            if moment - since >= dwell:
                return since
        else:
            since = None
    return None


def phase_mean(samples, start, end, event=False):
    if event:
        center = (start + end) / 2
        values = [watts for _, watts in
                  sorted(samples, key=lambda row: abs(row[0] - center))[:10]]
    else:
        values = [watts for moment, watts in samples if start <= moment <= end]
    if len(values) < 10:
        raise RuntimeError(f"power phase has only {len(values)} samples")
    return statistics.mean(values)


def reduce_power(run_root: Path, markers, role_uuids: list[str]) -> dict:
    rows, by_uuid = load_power(run_root, role_uuids)
    times = sorted({parse_wall(row["timestamp"]) for row in rows})
    gaps = [b - a for a, b in zip(times, times[1:])]
    query = [parse_wall(row["query.end"]) - parse_wall(row["query.start"])
             for row in rows]
    summary = {
        "rows": len(rows),
        "samples_per_gpu": {uuid: len(by_uuid[uuid]) for uuid in role_uuids},
        "sampling_coverage": len(times) / ((times[-1] - times[0]) * 10 + 1),
        "dropped_queries": sum(1 for _ in (run_root / "power_100ms.stderr").open()),
        "median_cadence_ms": 1000 * statistics.median(gaps),
        "max_cadence_ms": 1000 * max(gaps),
        "median_query_ms": 1000 * statistics.median(query),
        "max_query_ms": 1000 * max(query),
        "source_uuid": role_uuids[0],
        "destination_uuid": role_uuids[1],
    }
    if not 90 <= summary["median_cadence_ms"] <= 110 \
            or summary["sampling_coverage"] < .9 \
            or summary["max_cadence_ms"] > 3000:
        raise RuntimeError(f"100 ms cadence gate failed: {summary}")

    event = {row["event"]: row["wall_ns"] / 1e9 for row in markers.rows}
    monotonic = {row["event"]: row["monotonic_ns"] / 1e9 for row in markers.rows}

    def mean_power(uuid: str, start: float, end: float, event=False) -> float:
        samples = [(parse_wall(row["timestamp"]),
                    float(row["power.draw [W]"])) for row in by_uuid[uuid]]
        return phase_mean(samples, start, end, event)

    origin = markers.rows[0]["wall_ns"] / 1e9
    source_times, source_watts = series(by_uuid, role_uuids[0], origin)
    idle_floor = mean_power(role_uuids[0], event["idle_start"] + 5,
                            event["loads_prewarm_start"])
    settled = settle_time(source_times, source_watts,
                          event["traffic_switched"] - origin, idle_floor)
    summary["source_fall_settle_s"] = None if settled is None else \
        settled - (event["traffic_switched"] - origin)

    phases = {
        "loaded_idle": (event["idle_start"] + 5, event["loads_prewarm_start"]),
        "both_busy": (event["both_steady"], event["steady_hold_complete"]),
        "migration": (event["migration_start"], event["migration_complete"]),
        "switch": (event["switch_start"], event["traffic_switched"]),
        "source_fall": (event["traffic_switched"], event["source_drained"]),
        "post_switch": (event["source_drained"] + 5, event["measurement_complete"]),
    }
    summary["phase_seconds"] = {name: end - start for name, (start, end) in phases.items()}
    summary["mean_power_w"] = {
        name: {role: mean_power(uuid, *window, event=name == "switch")
               for role, uuid in zip(("source", "destination"), role_uuids)}
        for name, window in phases.items()
    }
    end = event["measurement_complete"]
    summary["post_switch_tail_drift_w"] = {
        role: mean_power(uuid, end - 60, end)
        - mean_power(uuid, end - 120, end - 60)
        for role, uuid in zip(("source", "destination"), role_uuids)
    }
    queue_phases = {
        "both_busy": (monotonic["both_steady"], monotonic["steady_hold_complete"]),
        "migration": (monotonic["migration_start"], monotonic["migration_complete"]),
        "post_switch": (monotonic["source_drained"], monotonic["measurement_complete"]),
    }
    summary["engine_queue"] = {
        name: {
            role: engine_queue(run_root / directory / "engine.csv", *window)
            for role, directory in (("source", "source_foreground"),
                                    ("destination", "destination_background"))
        }
        for name, window in queue_phases.items()
    }
    watts = summary["mean_power_w"]
    failures = []
    if watts["both_busy"]["source"] - watts["post_switch"]["source"] < 50:
        failures.append("source did not shed at least 50 W")
    if watts["both_busy"]["destination"] - watts["loaded_idle"]["destination"] < 10:
        failures.append("destination was not carrying background load")
    if watts["post_switch"]["destination"] - watts["both_busy"]["destination"] < 10:
        failures.append("destination did not take on the shed load")
    if abs(summary["post_switch_tail_drift_w"]["destination"]) > 25:
        failures.append("destination power drifted across the final two minutes")
    if failures:
        raise RuntimeError(f"power gate failed: {failures}; {watts}")
    write_json(run_root / "power_summary.json", summary)
    plot_power(run_root, markers, by_uuid, role_uuids, summary)
    return summary


def plot_power(run_root: Path, markers, by_uuid, role_uuids, summary) -> None:
    origin = markers.rows[0]["wall_ns"] / 1e9
    marker = {row["event"]: row["wall_ns"] / 1e9 - origin for row in markers.rows}
    fig, ax = plt.subplots(figsize=(9, 4))

    for label, uuid, color in (
        ("Source", role_uuids[0], SOURCE),
        ("Destination", role_uuids[1], DESTINATION),
    ):
        ax.plot(*bin_power(by_uuid[uuid], origin), lw=1.2, label=label,
                color=color, zorder=2)
    for name, start, end, color, alpha, label in WINDOWS:
        left, right = marker[start], marker[end]
        if name == "switch" and right - left < 2:
            left, right = (left + right) / 2 - 1, (left + right) / 2 + 1
        ax.axvspan(left, right, color=color, alpha=alpha, label=label, zorder=0)
    for event in dict.fromkeys(name for window in WINDOWS for name in window[1:3]):
        ax.axvline(marker[event], color="black", lw=.7, ls=":", zorder=1)

    ax.set(xlabel="Time (s)", ylabel="Power per GPU (W)")
    ax.set_xlim(0, marker["measurement_complete"])
    ax.grid(alpha=.25)
    ax.tick_params(labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    for spine in ax.spines.values():
        spine.set_color("black")
    ax.legend(frameon=False, ncol=5, fontsize=11, loc="upper center",
              bbox_to_anchor=(.5, -.2), columnspacing=1.4, handlelength=2.3)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(run_root / f"power_curve.{suffix}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    run_root = Path(sys.argv[1]).resolve()
    local_root = Path(os.environ["L_SCRATCH"]) / \
        f"qh-shed-{os.environ['SLURM_JOB_ID']}-{os.getpid()}"
    run_root.mkdir(parents=True, exist_ok=True)
    markers = Markers(run_root / "events.jsonl")
    scenario, manifest, sessions, context, rates, work = load_inputs()
    scenario = live_scenario(scenario)
    cfg = testbed.Config()
    devices = testbed.allocated_gpu_ids()
    role_uuids = gpu_uuids(devices)
    if len(role_uuids) != 2 or len(set(role_uuids)) != 2:
        raise RuntimeError(f"expected two unique allocated GPUs: {role_uuids}")
    write_json(run_root / "configuration.json", {
        "scenario_id": SCENARIO_ID,
        "policy": scenario["policy"],
        "scenario_method": scenario["method"],
        "shed_contexts": [row["initial_tokens"] for row in scenario["sessions"]],
        "shed_moves": len(scenario["moves"]),
        "move_methods": collections.Counter(
            row["method"] for row in scenario["moves"]),
        "move_method_by_context": sorted(
            (next(s["initial_tokens"] for s in scenario["sessions"]
                  if s["session_id"] == row["session_id"]), row["method"])
            for row in scenario["moves"]),
        "job_class": JOB_CLASS,
        "foreground_sessions": len(sessions),
        "foreground_mean_context": context,
        "source_rps": SOURCE_RPS, "destination_rps": DEST_RPS,
        "source_max_inflight": SOURCE_INFLIGHT,
        "destination_max_inflight": DEST_INFLIGHT,
        "work_per_request_s": work,
        "foreground_cache_policy": "unique_skip_save",
        "foreground_append_tokens": BUSY_APPEND_TOKENS,
        "foreground_output_tokens": BUSY_OUTPUT_TOKENS,
        "kv_roles": {role: testbed.kv_role_for(role) for role in ("source", "sink")},
        "lmcache_l1_gb": testbed.lmcache_l1_gb(),
        "steady_s": STEADY_S, "post_switch_s": POST_S,
        "migration_deadline_s": MIGRATION_DEADLINE_S,
        "reset_caches_during_migration": scenario["reset_caches"],
        "inline_continuation_verification": scenario["verify_continuations"],
        "gpu_uuids": {"source": role_uuids[0], "destination": role_uuids[1]},
        "plan_sha256": profiler.file_hash(PLAN),
    })

    stack = source = dest = takeover = None
    power_proc = power_handle = power_stderr = None
    migration_pool = future = move_gate = None
    try:
        testbed.preflight(cfg, 2)
        stack = testbed.start_stack(cfg, run_root / "testbed", 10_000, [])
        testbed.start_sink(stack, cfg, [])
        testbed.run_smoke2_probe(cfg, stack.run_root, 10_000)
        move_gate, contexts_ready = threading.Event(), threading.Event()

        def release_moves():
            contexts_ready.set()
            move_gate.wait()
            markers.add("migration_start")

        migration_pool = ThreadPoolExecutor(max_workers=1)
        future = migration_pool.submit(
            profiler.run_scenario, stack, cfg, manifest, scenario,
            run_root / "migration", "live-power-shed", None, False,
            release_moves,
        )
        if not contexts_ready.wait(180):
            move_gate.set()
            future.result()
            raise RuntimeError("context setup exceeded 180 s")

        markers.add("stack_ready")
        power_proc, power_handle, power_stderr = start_power(local_root, devices)
        time.sleep(2)
        markers.add("idle_start")
        time.sleep(30)

        dest = make_load(cfg, cfg.sink_port, sessions, rates, work,
                         run_root / "destination_background", 11, DEST_RPS,
                         DEST_INFLIGHT)
        source = make_load(cfg, cfg.src_port, sessions, rates, work,
                           run_root / "source_foreground", 1, SOURCE_RPS,
                           SOURCE_INFLIGHT)
        takeover = make_load(cfg, cfg.sink_port, sessions, rates, work,
                             run_root / "destination_takeover", 21, SOURCE_RPS,
                             SOURCE_INFLIGHT)
        markers.add("loads_prewarm_start", source_rps=SOURCE_RPS,
                    destination_rps=DEST_RPS)
        dest.start()
        wait_serving(dest, label="destination background")
        source.start()
        wait_serving(source, label="source foreground")
        takeover.pause()
        takeover.start()
        wait_serving(dest, len(dest.rows) + 3, label="destination background")
        markers.add("both_steady")

        hold([source, dest, takeover], STEADY_S,
             "both-busy steady load")
        markers.add("steady_hold_complete")
        move_gate.set()
        result = future.result()
        if result["status"] != "complete" or len(result["migrations"]) != 8 \
                or not all(row["error"] is None for row in result["migrations"]):
            raise RuntimeError("full-shed migration did not complete")
        markers.add("migration_complete")
        migration_s = (markers.rows[-1]["monotonic_ns"]
                       - markers.rows[-2]["monotonic_ns"]) / 1e9
        if migration_s > MIGRATION_DEADLINE_S:
            raise RuntimeError(
                f"migration missed {MIGRATION_DEADLINE_S} s deadline: "
                f"{migration_s:.3f} s"
            )

        switch_traffic(source, takeover, markers)
        markers.add("source_drained",
                    stranded=wait_drained(source, label="source foreground"))

        hold([dest, takeover], POST_S, "post-switch destination hold")
        markers.add("measurement_complete")
    finally:
        try:
            for name, load in (("source_foreground", source),
                               ("destination_background", dest),
                               ("destination_takeover", takeover)):
                if load:
                    load.close()
                    write_json(run_root / f"{name}_summary.json", load.summary())
        finally:
            if move_gate:
                move_gate.set()
            if migration_pool:
                migration_pool.shutdown(wait=True)
            if power_proc:
                markers.add("power_logger_stopped",
                            returncode=stop_power(power_proc, power_handle, power_stderr))
            if stack:
                testbed.stop_stack(stack)
            markers.add("stack_stopped")
            markers.close()
            for name in ("power.csv", "power.stderr"):
                if (local_root / name).exists():
                    shutil.copy2(local_root / name,
                                 run_root / name.replace("power", "power_100ms"))
    reduce_power(run_root, markers, role_uuids)
    print(f"completed {run_root}", flush=True)


def reduce_existing(run_root: Path) -> None:
    markers = type("Markers", (), {"rows": [
        json.loads(line) for line in (run_root / "events.jsonl").open()
    ]})()
    uuids = json.loads((run_root / "configuration.json").read_text())["gpu_uuids"]
    reduce_power(run_root, markers, [uuids[role] for role in ("source", "destination")])


def engine_queue(path: Path, start: float, end: float) -> dict:
    rows = [row for row in csv.DictReader(path.open())
            if start <= int(row["monotonic_ns"]) / 1e9 <= end]
    if not rows:
        raise RuntimeError(f"no engine samples in {start}..{end}: {path}")
    return {
        label: {
            "mean": statistics.mean(float(row[field]) for row in rows),
            "max": max(float(row[field]) for row in rows),
        }
        for label, field in (
            ("running", "vllm:num_requests_running"),
            ("waiting", "vllm:num_requests_waiting"),
        )
    }


if __name__ == "__main__":
    if sys.argv[1:2] == ["--reduce"]:
        reduce_existing(Path(sys.argv[2]).resolve())
    else:
        main()
