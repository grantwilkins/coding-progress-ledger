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
import seaborn as sns


REPO = Path(__file__).resolve().parents[3]
QH = REPO / "queue-haul"
POWERTRACE = REPO.parents[1] / "powertrace-sim"
sys.path.insert(0, str(QH))

import destination_runner as destination
import migration_profiler as profiler
import migration_testbed as testbed


SCENARIO_ID = "p-bc461d9bb1fc33c6"
PLAN = QH / "outputs/policy-hardware-width8-packing-plan/plan.json"
BUNDLE = QH / "outputs/destination-v7-20260722/content-free-manifest.json"
PROFILE = QH / "outputs/destination-v7-20260722/baseline-profile.json"
TARGET_RHO = 16.5


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


def wait_queue(load: destination.DestinationLoad, seconds: float = 120) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if load.failure or load.sampler.error:
            raise RuntimeError("foreground load failed") from (
                load.failure or load.sampler.error
            )
        rows = load.sampler.rows[-10:]
        if len(rows) == 10 and all(
            row["vllm:num_requests_waiting"] > 0 for row in rows
        ):
            return
        time.sleep(.25)
    raise RuntimeError("foreground load did not establish a persistent queue")


def hold(load: destination.DestinationLoad, seconds: int, label: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if load.failure or load.sampler.error:
            raise RuntimeError(f"{label} failed") from (
                load.failure or load.sampler.error
            )
        remaining = deadline - time.monotonic()
        print(f"{label}: {max(0, int(remaining))} s remaining", flush=True)
        time.sleep(min(30, max(0, remaining)))


def load_inputs():
    plan = json.loads(PLAN.read_text())
    scenario = next(
        row for row in plan["scenarios"] if row["scenario_id"] == SCENARIO_ID
    )
    if scenario["method"] != "replay" or len(scenario["moves"]) != 8 \
            or scenario["bandwidth_mbps"] != 10_000:
        raise RuntimeError("selected scenario changed")
    manifest_path = REPO / plan["manifest"]["path"]
    if profiler.file_hash(manifest_path) != plan["manifest"]["sha256"]:
        raise RuntimeError("migration manifest changed")
    manifest = json.loads(manifest_path.read_text())
    bundle = json.loads(BUNDLE.read_text())
    profile = json.loads(PROFILE.read_text())
    sessions = sum((
        destination.manifest_sessions(
            bundle, job, "validation", 201088, 7
        )
        for job in sorted(bundle["manifest"]["splits"])
    ), [])
    context = round(statistics.mean(row.prefix_tokens for row in sessions))
    rates = (
        destination.profile_rate(profile, "prefill", context),
        destination.profile_rate(profile, "decode", context),
    )
    return scenario, manifest, sessions, profile, context, rates


def parse_wall(value: str) -> float:
    return dt.datetime.strptime(value, "%Y/%m/%d %H:%M:%S.%f").timestamp()


def bin_power(rows: list[dict], origin: float) -> tuple[list[float], list[float]]:
    bins = {}
    for row in rows:
        index = int((parse_wall(row["timestamp"]) - origin) // 5)
        bins.setdefault(index, []).append(float(row["power.draw [W]"]))
    return ([index * 5 + 2.5 for index in sorted(bins)],
            [statistics.mean(bins[index]) for index in sorted(bins)])


def reduce_power(run_root: Path, local_power: Path, markers: Markers,
                 role_uuids: list[str]) -> None:
    destination_path = run_root / "power_100ms.csv"
    if local_power.resolve() != destination_path.resolve():
        shutil.copy2(local_power, destination_path)
    rows = list(csv.DictReader(destination_path.open()))
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
        raise RuntimeError("100 ms power log does not contain one row per GPU per tick")
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
        "logger_returncode": next(
            row["returncode"] for row in markers.rows
            if row["event"] == "power_logger_stopped"
        ),
    }
    if not 90 <= summary["median_cadence_ms"] <= 110 \
            or summary["sampling_coverage"] < .9 \
            or summary["max_cadence_ms"] > 1000 \
            or summary["max_query_ms"] > 205:
        raise RuntimeError(f"100 ms cadence gate failed: {summary}")

    event = {row["event"]: row["wall_ns"] / 1e9 for row in markers.rows}
    def mean_power(uuid: str, start: float, end: float) -> float:
        values = [float(row["power.draw [W]"]) for row in by_uuid[uuid]
                  if start <= parse_wall(row["timestamp"]) <= end]
        if len(values) < 10:
            raise RuntimeError(f"power phase has only {len(values)} samples")
        return statistics.mean(values)

    phases = {
        "loaded_idle": (event["idle_start"] + 5, event["source_prewarm_start"]),
        "source_steady": (event["migration_start"] - 30, event["migration_start"]),
        "overlap": (event["migration_start"], event["migration_complete"]),
        "post_switch": (event["source_stopped"] + 5, event["measurement_complete"]),
    }
    summary["mean_power_w"] = {
        name: {role: mean_power(uuid, *window)
               for role, uuid in zip(("source", "destination"), role_uuids)}
        for name, window in phases.items()
    }
    watts = summary["mean_power_w"]
    if watts["source_steady"]["source"] - watts["post_switch"]["source"] < 50 \
            or any(watts["overlap"][role] - watts["loaded_idle"][role] < 10
                   for role in ("source", "destination")) \
            or watts["post_switch"]["destination"] \
            - watts["loaded_idle"]["destination"] < 10:
        raise RuntimeError(f"overlap/drop power gate failed: {watts}")
    write_json(run_root / "power_summary.json", summary)

    origin = markers.rows[0]["wall_ns"] / 1e9
    sns.set_context("talk", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, uuid, color in (
        ("Source", role_uuids[0], "#8C1515"),
        ("Dest", role_uuids[1], "#007C92"),
    ):
        ax.plot(*bin_power(by_uuid[uuid], origin), lw=1.5, label=label,
                color=color)
    marker = {row["event"]: row["wall_ns"] / 1e9 - origin
              for row in markers.rows}
    if "migration_start" in marker and "source_stopped" in marker:
        ax.axvspan(marker["migration_start"], marker["source_stopped"],
                   color="#E98300", alpha=.12)
    for name in ("source_steady", "migration_start", "migration_complete",
                 "source_stopped"):
        if name in marker:
            ax.axvline(marker[name], color="black", lw=.7, ls=":")
    ax.set(xlabel="Time (s)", ylabel="Power per GPU (W)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(run_root / f"power_curve.{suffix}", dpi=220)
    plt.close(fig)


def main() -> None:
    run_root = Path(sys.argv[1]).resolve()
    local_root = Path(os.environ["L_SCRATCH"]) / f"qh-overlap-{os.environ['SLURM_JOB_ID']}"
    run_root.mkdir(parents=True, exist_ok=True)
    markers = Markers(run_root / "events.jsonl")
    scenario, manifest, sessions, profile, context, rates = load_inputs()
    cfg = testbed.Config()
    devices = testbed.allocated_gpu_ids()
    role_uuids = gpu_uuids(devices)
    if len(role_uuids) != 2 or len(set(role_uuids)) != 2:
        raise RuntimeError(f"expected two unique allocated GPUs: {role_uuids}")
    write_json(run_root / "configuration.json", {
        "scenario_id": SCENARIO_ID,
        "contexts": [row["initial_tokens"] for row in scenario["sessions"]],
        "foreground_sessions": len(sessions),
        "foreground_mean_context": context,
        "foreground_offered_rho": TARGET_RHO,
        "gpu_uuids": {"source": role_uuids[0], "destination": role_uuids[1]},
        "plan_sha256": profiler.file_hash(PLAN),
    })

    stack = source = post_destination = None
    power_proc = power_handle = power_stderr = None
    power_status = None
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
        target = TARGET_RHO
        markers.add("source_prewarm_start", offered_rho=target)
        source = destination.DestinationLoad(
            cfg.host, cfg.src_port, cfg.model, sessions, target, *rates,
            run_root / "source_foreground", 1, chunk_s=15, normal_bound=1,
        )
        source.start()
        wait_queue(source)
        markers.add("source_steady")
        hold(source, 300, "source pre-migration load")
        wait_queue(source)

        markers.add("source_setup_drain_requested")
        active, source = source, None
        active.close()
        markers.add("source_setup_drained")

        original_flush = profiler.b.flush_lmcache
        original_idle = profiler.b.mp_wait_idle
        flush_count = 0
        def coordinated_flush(*args, **kwargs):
            nonlocal flush_count, source
            flush_count += 1
            result = original_flush(*args, **kwargs)
            if flush_count == 2:
                markers.add("source_migration_load_start")
                source = destination.DestinationLoad(
                    cfg.host, cfg.src_port, cfg.model, sessions, target, *rates,
                    run_root / "source_foreground_migration", 3,
                    chunk_s=15, normal_bound=1,
                )
                source.start()
                wait_queue(source)
                markers.add("migration_start")
            elif flush_count != 1:
                raise RuntimeError(f"unexpected scenario flush {flush_count}")
            return result
        profiler.b.flush_lmcache = coordinated_flush
        profiler.b.mp_wait_idle = lambda *_args, **_kwargs: None
        try:
            result = profiler.run_scenario(
                stack, cfg, manifest, scenario, run_root / "migration",
                "live-power-oneoff", configure_proxy=False,
            )
        finally:
            profiler.b.flush_lmcache = original_flush
            profiler.b.mp_wait_idle = original_idle
        if result["status"] != "complete" or len(result["migrations"]) != 8 \
                or not all(row["error"] is None for row in result["migrations"]):
            raise RuntimeError("width-eight replay migration did not complete")
        markers.add("migration_complete")

        post_destination = destination.DestinationLoad(
            cfg.host, cfg.sink_port, cfg.model, sessions, target, *rates,
            run_root / "destination_foreground", 2, chunk_s=15, normal_bound=1,
        )
        markers.add("destination_prewarm_start")
        post_destination.start()
        wait_queue(post_destination)
        markers.add("destination_steady")
        markers.add("source_stop_requested")
        active, source = source, None
        active.close()
        markers.add("source_stopped")
        hold(post_destination, 60, "destination post-switch load")
        markers.add("measurement_complete")
    finally:
        try:
            if source:
                source.close()
            if post_destination:
                post_destination.close()
        finally:
            if power_proc:
                power_status = stop_power(power_proc, power_handle, power_stderr)
                markers.add("power_logger_stopped", returncode=power_status)
            if stack:
                testbed.stop_stack(stack)
            markers.add("stack_stopped")
            markers.close()
            if (local_root / "power.csv").exists():
                shutil.copy2(local_root / "power.csv", run_root / "power_100ms.csv")
            if (local_root / "power.stderr").exists():
                shutil.copy2(local_root / "power.stderr", run_root / "power_100ms.stderr")
    reduce_power(run_root, local_root / "power.csv", markers, role_uuids)
    print(f"completed {run_root}", flush=True)


def reduce_existing(run_root: Path) -> None:
    markers = type("Markers", (), {"rows": [
        json.loads(line) for line in (run_root / "events.jsonl").open()
    ]})()
    uuids = json.loads((run_root / "configuration.json").read_text())["gpu_uuids"]
    reduce_power(run_root, run_root / "power_100ms.csv", markers,
                 [uuids[role] for role in ("source", "destination")])


if __name__ == "__main__":
    if sys.argv[1:2] == ["--reduce"]:
        reduce_existing(Path(sys.argv[2]).resolve())
    else:
        main()
