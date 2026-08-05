"""Full-shed migration power trace with both sites carrying live agentic load.

Source runs 4 rps of agentic-tool-loop traffic, destination runs 1 rps of its own
tenants throughout, then all eight sessions shed to the destination, the
destination takes over the source's traffic, and the source drains. Both engines
are kv_both with a >32 GB LMCache L1 so the handoff is visible in the cache.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

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


SCENARIO_ID = "p-bc461d9bb1fc33c6"  # replay, 8 of 8 sessions -> full shed
PLAN = QH / "outputs/policy-hardware-width8-packing-plan/plan.json"
BUNDLE = QH / "outputs/destination-v7-20260722/content-free-manifest.json"
PROFILE = QH / "outputs/destination-v7-20260722/baseline-profile.json"
JOB_CLASS = "agentic_tool_loop"
SPLITS = ("validation", "fit", "tune")
SOURCE_RPS, DEST_RPS = 4.0, 1.0
SOURCE_INFLIGHT, DEST_INFLIGHT = 128, 48
DEST_ONLY_S, STEADY_S, POST_S = 60, 300, 300
SOURCE, DESTINATION = "#B22E2E", "#00799B"
WINDOWS = (
    ("migration", "migration_start", "migration_complete", "#E98300", None,
     "Migration"),
    ("switch", "migration_complete", "source_admission_stopped", "#7B5EA7", None,
     "Switch"),
    ("source_fall", "source_admission_stopped", "source_drained", "#4F5B66", "///",
     "Source drain"),
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


def wait_queue(load: destination.DestinationLoad, seconds: float = 180,
               label: str = "source") -> None:
    """A saturating load must hold a standing queue before we call it steady."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check(load, label)
        rows = load.sampler.rows[-10:]
        if len(rows) == 10 and all(
            row["vllm:num_requests_waiting"] > 0 for row in rows
        ):
            return
        time.sleep(.25)
    raise RuntimeError(f"{label} load did not establish a persistent queue")


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
                 label: str = "source") -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        check(load, label)
        rows = load.sampler.rows[-4:]
        if len(rows) == 4 and all(
            row["vllm:num_requests_running"] == 0
            and row["vllm:num_requests_waiting"] == 0 for row in rows
        ):
            return
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


def load_inputs():
    plan = json.loads(PLAN.read_text())
    scenario = next(
        row for row in plan["scenarios"] if row["scenario_id"] == SCENARIO_ID
    )
    if scenario["method"] != "replay" or len(scenario["moves"]) != 8 \
            or len(scenario["sessions"]) != 8 \
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
        rps=rps, max_inflight=inflight,
    )


def parse_wall(value: str) -> float:
    return dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f").timestamp()


def load_power(run_root: Path, role_uuids: list[str]):
    rows = list(csv.DictReader((run_root / "power_100ms.csv").open()))
    if not rows:
        raise RuntimeError("100 ms power logger wrote no rows")
    by_uuid = {uuid: [] for uuid in role_uuids}
    sample_uuids = {}
    for row in rows:
        if row["uuid"] not in by_uuid:
            raise RuntimeError(f"power log contains unallocated GPU {row['uuid']}")
        by_uuid[row["uuid"]].append(row)
        sample_uuids.setdefault(row["timestamp"], []).append(row["uuid"])
    if any(not values for values in by_uuid.values()):
        raise RuntimeError("100 ms power log is missing an allocated GPU")
    if any(sorted(values) != sorted(role_uuids) for values in sample_uuids.values()):
        raise RuntimeError("100 ms power log lacks one row per GPU per tick")
    return rows, by_uuid


def series(by_uuid, uuid, origin):
    return ([parse_wall(row["timestamp"]) - origin for row in by_uuid[uuid]],
            [float(row["power.draw [W]"]) for row in by_uuid[uuid]])


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
            or summary["max_cadence_ms"] > 1000:
        raise RuntimeError(f"100 ms cadence gate failed: {summary}")

    event = {row["event"]: row["wall_ns"] / 1e9 for row in markers.rows}

    def mean_power(uuid: str, start: float, end: float) -> float:
        values = [float(row["power.draw [W]"]) for row in by_uuid[uuid]
                  if start <= parse_wall(row["timestamp"]) <= end]
        if len(values) < 10:
            raise RuntimeError(f"power phase has only {len(values)} samples")
        return statistics.mean(values)

    origin = markers.rows[0]["wall_ns"] / 1e9
    source_times, source_watts = series(by_uuid, role_uuids[0], origin)
    idle_floor = mean_power(role_uuids[0], event["idle_start"] + 5,
                            event["destination_prewarm_start"])
    settled = settle_time(source_times, source_watts,
                          event["source_admission_stopped"] - origin, idle_floor)
    summary["source_fall_settle_s"] = None if settled is None else \
        settled - (event["source_admission_stopped"] - origin)

    phases = {
        "loaded_idle": (event["idle_start"] + 5, event["destination_prewarm_start"]),
        "destination_only": (event["destination_steady"], event["source_prewarm_start"]),
        "both_busy": (event["source_steady"], event["steady_hold_complete"]),
        "migration": (event["migration_start"], event["migration_complete"]),
        "switch": (event["migration_complete"], event["source_admission_stopped"]),
        "source_fall": (event["source_admission_stopped"], event["source_drained"]),
        "post_switch": (event["source_drained"] + 5, event["measurement_complete"]),
    }
    summary["phase_seconds"] = {name: end - start for name, (start, end) in phases.items()}
    summary["mean_power_w"] = {
        name: {role: mean_power(uuid, *window)
               for role, uuid in zip(("source", "destination"), role_uuids)}
        for name, window in phases.items()
    }
    half = (event["source_drained"] + 5 + event["measurement_complete"]) / 2
    summary["post_switch_drift_w"] = {
        role: mean_power(uuid, half, event["measurement_complete"])
        - mean_power(uuid, event["source_drained"] + 5, half)
        for role, uuid in zip(("source", "destination"), role_uuids)
    }
    watts = summary["mean_power_w"]
    failures = []
    if watts["both_busy"]["source"] - watts["post_switch"]["source"] < 50:
        failures.append("source did not shed at least 50 W")
    if watts["both_busy"]["destination"] - watts["loaded_idle"]["destination"] < 10:
        failures.append("destination was not carrying background load")
    if watts["post_switch"]["destination"] - watts["destination_only"]["destination"] < 10:
        failures.append("destination did not take on the shed load")
    if abs(summary["post_switch_drift_w"]["destination"]) > 25:
        failures.append("destination power drifted across the post-switch hold")
    if failures:
        raise RuntimeError(f"power gate failed: {failures}; {watts}")
    write_json(run_root / "power_summary.json", summary)
    plot_power(run_root, markers, by_uuid, role_uuids, summary)
    return summary


def plot_power(run_root: Path, markers, by_uuid, role_uuids, summary) -> None:
    origin = markers.rows[0]["wall_ns"] / 1e9
    marker = {row["event"]: row["wall_ns"] / 1e9 - origin for row in markers.rows}
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_facecolor("#ffffff")
    top = max(float(row["power.draw [W]"])
              for uuid in role_uuids for row in by_uuid[uuid])

    # Rotated in-band labels sit inside their own span, so narrow windows cannot
    # collide with their neighbours the way centred horizontal labels do.
    for _, start, end, color, hatch, label in WINDOWS:
        ax.axvspan(marker[start], marker[end], facecolor=color,
                   alpha=.14 if hatch else .20, lw=0, zorder=0, hatch=hatch,
                   edgecolor=color)
        ax.text(marker[start] + (marker[end] - marker[start]) * .5, top * .06,
                label, rotation=90, ha="center", va="bottom", fontsize=8,
                color="#3f3f42", zorder=3)

    for label, uuid, color in (
        ("Source GPU", role_uuids[0], SOURCE),
        ("Destination GPU", role_uuids[1], DESTINATION),
    ):
        ax.plot(*series(by_uuid, uuid, origin), lw=.7, label=label, color=color,
                zorder=2, solid_joinstyle="round")

    ax.grid(axis="y", color="#e6e6e4", lw=.6, zorder=1)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c9c9c6")
    ax.tick_params(colors="#5f5f63", labelsize=9)
    ax.set_xlabel("Elapsed time (s)", color="#3f3f42")
    ax.set_ylabel("GPU power (W)", color="#3f3f42")
    ax.set_title(
        f"Full-shed migration under live load: source {SOURCE_RPS:g} rps, "
        f"destination {DEST_RPS:g} rps",
        color="#1f1f22", fontsize=12, pad=26, loc="left",
    )
    ax.set_ylim(0, top * 1.06)
    ax.set_xlim(0, marker["measurement_complete"])
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color,
                              alpha=.14 if hatch else .20, hatch=hatch)
                for _, _, _, color, hatch, _ in WINDOWS]
    labels += [f"{label} window" for *_, label in WINDOWS]
    ax.legend(handles, labels, frameon=False, ncol=5, fontsize=9,
              loc="upper center", bbox_to_anchor=(.5, -.14), labelcolor="#3f3f42")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(run_root / f"power_curve.{suffix}", dpi=220,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    run_root = Path(sys.argv[1]).resolve()
    local_root = Path(os.environ["L_SCRATCH"]) / f"qh-shed-{os.environ['SLURM_JOB_ID']}"
    run_root.mkdir(parents=True, exist_ok=True)
    markers = Markers(run_root / "events.jsonl")
    scenario, manifest, sessions, context, rates, work = load_inputs()
    cfg = testbed.Config()
    devices = testbed.allocated_gpu_ids()
    role_uuids = gpu_uuids(devices)
    if len(role_uuids) != 2 or len(set(role_uuids)) != 2:
        raise RuntimeError(f"expected two unique allocated GPUs: {role_uuids}")
    write_json(run_root / "configuration.json", {
        "scenario_id": SCENARIO_ID,
        "shed_contexts": [row["initial_tokens"] for row in scenario["sessions"]],
        "shed_moves": len(scenario["moves"]),
        "job_class": JOB_CLASS,
        "foreground_sessions": len(sessions),
        "foreground_mean_context": context,
        "source_rps": SOURCE_RPS, "destination_rps": DEST_RPS,
        "source_max_inflight": SOURCE_INFLIGHT,
        "destination_max_inflight": DEST_INFLIGHT,
        "work_per_request_s": work,
        "kv_roles": {role: testbed.kv_role_for(role) for role in ("source", "sink")},
        "lmcache_l1_gb": testbed.lmcache_l1_gb(),
        "steady_s": STEADY_S, "post_switch_s": POST_S,
        "gpu_uuids": {"source": role_uuids[0], "destination": role_uuids[1]},
        "plan_sha256": profiler.file_hash(PLAN),
    })

    stack = source = dest = takeover = None
    power_proc = power_handle = power_stderr = None
    original_idle = profiler.b.mp_wait_idle
    try:
        testbed.preflight(cfg, 2)
        stack = testbed.start_stack(cfg, run_root / "testbed", 10_000, [])
        testbed.start_sink(stack, cfg, [])
        testbed.run_smoke2_probe(cfg, stack.run_root, 10_000)
        markers.add("stack_ready")
        power_proc, power_handle, power_stderr = start_power(local_root, devices)
        time.sleep(2)
        markers.add("idle_start")
        time.sleep(30)

        testbed.flush_lmcache(stack, cfg)
        testbed.reset_vllm_caches(
            cfg, (stack.run_root / "source.log", stack.run_root / "sink.log")
        )

        dest = make_load(cfg, cfg.sink_port, sessions, rates, work,
                         run_root / "destination_background", 11, DEST_RPS,
                         DEST_INFLIGHT)
        markers.add("destination_prewarm_start", offered_rps=DEST_RPS)
        dest.start()
        wait_serving(dest, label="destination background")
        markers.add("destination_steady")
        hold([dest], DEST_ONLY_S, "destination-only background load")

        source = make_load(cfg, cfg.src_port, sessions, rates, work,
                           run_root / "source_foreground", 1, SOURCE_RPS,
                           SOURCE_INFLIGHT)
        markers.add("source_prewarm_start", offered_rps=SOURCE_RPS)
        source.start()
        wait_queue(source, label="source foreground")
        markers.add("source_steady")

        hold([source, dest], STEADY_S, "both-busy steady load")
        markers.add("steady_hold_complete")

        markers.add("source_paused_for_reset")
        source.pause()
        wait_drained(source, label="source foreground")
        markers.add("source_drained_for_reset")

        original_flush = profiler.b.flush_lmcache
        original_reset = profiler.b.reset_vllm_caches
        flush_count = 0

        def coordinated_flush(*args, **kwargs):
            nonlocal flush_count
            flush_count += 1
            result = original_flush(*args, **kwargs)
            if flush_count == 2:
                markers.add("source_load_resumed")
                source.resume()
                wait_queue(source, label="source foreground")
                markers.add("migration_start")
            elif flush_count != 1:
                raise RuntimeError(f"unexpected scenario flush {flush_count}")
            return result

        def source_only_reset(reset_cfg, logs):
            """The destination keeps serving its tenants across the shed."""
            original_reset(reset_cfg, logs[:1], (reset_cfg.src_port,))

        profiler.b.flush_lmcache = coordinated_flush
        profiler.b.reset_vllm_caches = source_only_reset
        profiler.b.mp_wait_idle = lambda *_args, **_kwargs: None
        markers.add("shed_start")
        try:
            result = profiler.run_scenario(
                stack, cfg, manifest, scenario, run_root / "migration",
                "live-power-shed", configure_proxy=False,
            )
        finally:
            profiler.b.flush_lmcache = original_flush
            profiler.b.reset_vllm_caches = original_reset
        if result["status"] != "complete" or len(result["migrations"]) != 8 \
                or not all(row["error"] is None for row in result["migrations"]):
            raise RuntimeError("full-shed replay migration did not complete")
        markers.add("migration_complete")

        takeover = make_load(cfg, cfg.sink_port, sessions, rates, work,
                             run_root / "destination_takeover", 21, SOURCE_RPS,
                             SOURCE_INFLIGHT)
        markers.add("takeover_prewarm_start", offered_rps=SOURCE_RPS)
        takeover.start()
        wait_queue(takeover, label="destination takeover")
        markers.add("takeover_steady")

        markers.add("source_admission_stopped")
        source.pause()
        wait_drained(source, label="source foreground")
        markers.add("source_drained")

        hold([dest, takeover], POST_S, "post-switch destination hold")
        markers.add("measurement_complete")
    finally:
        profiler.b.mp_wait_idle = original_idle
        try:
            for name, load in (("source_foreground", source),
                               ("destination_background", dest),
                               ("destination_takeover", takeover)):
                if load:
                    load.close()
                    write_json(run_root / f"{name}_summary.json", load.summary())
        finally:
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


if __name__ == "__main__":
    if sys.argv[1:2] == ["--reduce"]:
        reduce_existing(Path(sys.argv[2]).resolve())
    else:
        main()
