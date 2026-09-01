"""Run and plot isolated production greedy versus ordinary LP scaling cells."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
import time
from time import perf_counter

import numpy as np

import fleet_shed_frontier_campaign as campaign
import pool_planner
from planner import source_power
from power_model import ExpectedPower
from profiles import ModelProfile, WorkloadProfile


ROOT = Path(__file__).parent
DEFAULT_OUT = ROOT / "outputs/planner-scaling-greedy-vs-lp"
SESSIONS = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000,
            200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000)
SOLVERS = ("greedy", "lp")
SEED, DEADLINE_S, RHO, TARGET_FRACTION = 1001, 600.0, .38, .25
OOM = re.compile(r"out of memory|memoryerror|cannot allocate|bad_alloc|memory allocation",
                 re.IGNORECASE)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 * 1024 if sys.platform == "darwin" else 1024)


def _limit_memory(gib: float) -> None:
    if gib and sys.platform != "darwin":
        limit = int(gib * 1024**3)
        resource.setrlimit(
            resource.RLIMIT_AS,
            (limit, resource.getrlimit(resource.RLIMIT_AS)[1]),
        )


def _process_rss_mib(pid: int) -> float | None:
    try:
        result = subprocess.run(
            ("ps", "-o", "rss=", "-p", str(pid)), capture_output=True, text=True,
        )
    except OSError as error:
        raise RuntimeError(f"resident-memory monitor failed: {error}") from error
    if result.stderr:
        raise RuntimeError(f"resident-memory monitor failed: {result.stderr.strip()}")
    return int(result.stdout) / 1024 if result.stdout.strip() else None


def _terminate(process) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def build_problem(sessions: int):
    profile = ModelProfile.load(campaign.MODEL)
    workload = WorkloadProfile.load(campaign.WORKLOADS[campaign.HEADLINE_WORKLOAD])
    bound = float(campaign.request_work(profile.case()).sum() * 5)
    bounds = {mode: bound for mode in ("normal", "emergency", "stable")}
    scenario, replicas, demand, fits = campaign.build_fleet(
        profile, workload, sessions, SEED, DEADLINE_S, bound, "natural",
    )
    architecture = campaign.build_architecture(
        profile, replicas, bounds, fits, RHO,
        campaign.migration_headroom(RHO, demand, replicas, bound), None,
    )
    power = ExpectedPower(scenario, profile)
    initial = power.power(True)
    target = TARGET_FRACTION * (initial - source_power(
        scenario, profile, (session.session_id for session in scenario.sessions),
    ))
    return profile, replace(scenario, power_limit_w=initial - target), \
        architecture, power, target, replicas


def measure_cell(sessions: int, solver: str, memory_gib=0, stage=None) -> dict:
    if solver not in SOLVERS:
        raise ValueError(f"solver must be one of {SOLVERS}")
    _limit_memory(memory_gib)
    cell_started = perf_counter()
    profile, scenario, architecture, power, target, replicas = build_problem(sessions)
    build_s = perf_counter() - cell_started
    if stage is not None:
        _write(stage, {"state": "solve", "started_local": datetime.now().isoformat()})
    started = perf_counter()
    stats = {}
    if solver == "greedy":
        oracle = pool_planner._candidate_oracle(
            scenario, profile, architecture, "normal", power,
        )
        oracle_s = perf_counter() - started
        table, selected, generation_s, selection_s = \
            pool_planner._compact_greedy(oracle, target)
        candidate_generation_s = oracle_s + generation_s
        candidate_metric = {
            "candidate_universe_slots": len(oracle.sessions) * len(oracle.options)}
    else:
        table = pool_planner.candidate_table(
            scenario, profile, architecture, "normal", power,
        )
        candidate_generation_s = perf_counter() - started
        selected_started = perf_counter()
        selected = pool_planner._lp(table, target, stats, integral_recovery=False)
        selection_s = perf_counter() - selected_started
        candidate_metric = {"materialized_candidates": len(table.candidates)}
        if stats.get("milp_recovery_s", 0):
            raise RuntimeError("scaling LP invoked MILP recovery")
    solve_s = perf_counter() - started
    credit = sum(table.candidates[i].credit for i in selected)
    usage = np.asarray(table.resources[:, list(selected)].sum(1)).ravel() \
        if selected else np.zeros(table.resources.shape[0])
    if credit < target - 1e-6:
        raise RuntimeError(f"{solver} reached {credit:g} of target {target:g}")
    if usage.max(initial=0) > 1 + 1e-8:
        raise RuntimeError(f"{solver} returned an aggregate-infeasible selection")
    return {
        "status": "ok", "solver": solver, "sessions": sessions,
        **candidate_metric, "selected_moves": len(selected),
        "source_replicas": replicas, "target_w": target,
        "selected_credit_w": credit, "build_s": build_s,
        "candidate_generation_s": candidate_generation_s,
        "selection_s": selection_s,
        "solve_s": solve_s,
        "cell_wall_s": perf_counter() - cell_started,
        "milp_recovery_s": stats.get("milp_recovery_s", 0),
        "peak_rss_mib": _rss_mib(), "memory_limit_gib": memory_gib,
    }


def cell(args) -> None:
    try:
        row = measure_cell(args.sessions, args.solver, args.memory_gib, args.stage)
    except MemoryError as error:
        phase = "solve" if args.stage.exists() else "setup"
        row = {
            "status": "oom" if phase == "solve" else "setup_oom",
            "failure_phase": phase, "solver": args.solver, "sessions": args.sessions,
            "memory_limit_gib": args.memory_gib, "error": repr(error),
        }
    _write(args.output, row)


def _failed_row(solver, sessions, repeat, status, phase, error, args, process, wall_s):
    return {
        "status": status, "solver": solver, "sessions": sessions,
        "failure_phase": phase,
        "repeat": repeat, "time_limit_s": (args.setup_timeout_s
                                              if status == "setup_timeout"
                                              else args.timeout_s),
        "memory_limit_gib": args.memory_gib, "error": error[-2000:],
        "returncode": process.returncode,
        "signal": -process.returncode if process.returncode < 0 else None,
        "parent_wall_s": wall_s,
    }


def _run_cell(solver, sessions, repeat, args, path):
    stage = path.with_suffix(".stage.json")
    stage.unlink(missing_ok=True)
    command = (
        sys.executable, str(Path(__file__).resolve()), "cell", "--solver", solver,
        "--sessions", str(sessions), "--memory-gib", str(args.memory_gib),
        "--output", str(path), "--stage", str(stage),
    )
    env = {**os.environ, **{name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    )}}
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, env=env,
                               start_new_session=True)
    parent_started, solve_started, status = perf_counter(), None, None
    while process.poll() is None:
        try:
            rss = (_process_rss_mib(process.pid)
                   if sys.platform == "darwin" and args.memory_gib else None)
        except RuntimeError:
            _terminate(process)
            raise
        if rss is None and process.poll() is not None:
            break
        if rss is None and sys.platform == "darwin" and args.memory_gib:
            _terminate(process)
            raise RuntimeError("resident-memory monitor lost a live child")
        memory_exceeded = rss is not None and rss > args.memory_gib * 1024
        if solve_started is None and stage.exists() and not memory_exceeded:
            solve_started = perf_counter()
        elapsed = perf_counter() - (solve_started or parent_started)
        limit = args.timeout_s if solve_started is not None else args.setup_timeout_s
        if elapsed > limit or memory_exceeded:
            status = (("oom" if solve_started is not None else "setup_oom")
                      if memory_exceeded else
                      ("timeout" if solve_started is not None else "setup_timeout"))
            _terminate(process)
            break
        time.sleep(.5)
    stdout, stderr = process.communicate()
    phase = "solve" if solve_started is not None else "setup"
    stage.unlink(missing_ok=True)
    wall_s = perf_counter() - parent_started
    if status is not None:
        return _failed_row(
            solver, sessions, repeat, status, phase, stdout + stderr, args, process,
            wall_s,
        )
    if path.exists():
        row = json.loads(path.read_text())
        row.update(repeat=repeat, returncode=process.returncode,
                   signal=None, parent_wall_s=wall_s)
        return row
    text = stdout + stderr
    status = (("oom" if phase == "solve" else "setup_oom")
              if OOM.search(text) else "failed")
    return _failed_row(
        solver, sessions, repeat, status, phase, text, args, process, wall_s)


def _rows(out: Path):
    return [json.loads(path.read_text()) for path in sorted((out / "cells").glob("*.json"))
            if not path.name.endswith(".stage.json")]


def _validate(rows) -> None:
    rejected = [row for row in rows if row["status"] not in {"ok", "timeout", "oom"}]
    if rejected:
        raise RuntimeError(rejected[0]["error"])


def _write_csv(path: Path, rows) -> None:
    fields = sorted(set().union(*(row for row in rows)))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(paths) -> str:
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def _identity(args) -> dict:
    sources = tuple(ROOT / name for name in (
        "planner_scaling_campaign.py", "pool_planner.py", "planner.py",
        "power_model.py", "fleet_shed_frontier_campaign.py", "native/src/lib.rs",
    ))
    inputs = (campaign.MODEL, campaign.WORKLOADS[campaign.HEADLINE_WORKLOAD],
              campaign.TIMING, campaign.LOADED)
    native_spec = importlib.util.find_spec("_queue_haul_native._queue_haul_native")
    if native_spec is None or native_spec.origin is None:
        raise ModuleNotFoundError("compiled queue-haul native extension not found")
    native = Path(native_spec.origin)
    return {
        "schema": "queue-haul-planner-scaling-v1",
        "code_sha256": _sha256(sources), "native_sha256": _sha256((native,)),
        "input_sha256": _sha256(inputs),
        "sessions": list(args.sessions), "solvers": list(SOLVERS),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("clarabel", "cvxpy", "highspy", "numpy", "scipy")},
        "repeats": args.repeats,
        "repeat_until": args.repeat_until, "timeout_s": args.timeout_s,
        "setup_timeout_s": args.setup_timeout_s,
        "memory_limit_gib": args.memory_gib, "scenario_seed": SEED,
        "deadline_s": DEADLINE_S, "rho": RHO,
        "target_fraction_of_removable_power": TARGET_FRACTION,
        "lp_integral_recovery": False,
        "timed_region": "candidate generation plus selection; excludes packing and DES",
        "plotted_metric": "selection_s for completed cells only",
    }


def run(args) -> list[dict]:
    args.out.mkdir(parents=True, exist_ok=True)
    identity, manifest = _identity(args), args.out / "run_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError("scaling output belongs to a different run identity")
    if (args.out / "cells").exists() and not manifest.exists():
        raise RuntimeError("scaling cells exist without a run manifest")
    _write(manifest, identity)
    _validate(_rows(args.out))
    for index, sessions in enumerate(args.sessions):
        repeats = args.repeats if sessions <= args.repeat_until else 1
        for repeat in range(repeats):
            order = SOLVERS if (index + repeat) % 2 else tuple(reversed(SOLVERS))
            for solver in order:
                path = args.out / "cells" / f"{sessions}-{solver}-{repeat}.json"
                if path.exists():
                    continue
                row = _run_cell(solver, sessions, repeat, args, path)
                _write(path, row)
                print(f"n={sessions:,} solver={solver} status={row['status']} "
                      f"solve_s={row.get('solve_s', float('nan')):.3f}", flush=True)
                if row["status"] not in {"ok", "timeout", "oom"}:
                    raise RuntimeError(row["error"])
    rows = _rows(args.out)
    _validate(rows)
    _write_csv(args.out / "results.csv", rows)
    metadata = {**identity,
        "created_local": datetime.now().astimezone().isoformat(),
        "git_sha": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "platform": platform.platform(), "python": platform.python_version(),
    }
    _write(args.out / "run_metadata.json", metadata)
    plot(rows, args.out / "planner_scaling")
    return rows


def plot(rows, output: Path) -> None:
    import matplotlib
    import plot_style

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("scaling rows must not be empty")
    plot_style.apply()
    fig, axis = plt.subplots(figsize=plot_style.FIGSIZE)
    for solver, policy in (("lp", "queue_haul"), ("greedy", "greedy")):
        ok = [row for row in rows if row["solver"] == solver
              and row["status"] == "ok"]
        grouped = {sessions: [row["selection_s"] for row in ok
                              if row["sessions"] == sessions]
                   for sessions in sorted({row["sessions"] for row in ok})}
        style = plot_style.policy_style(policy)
        x = list(grouped)
        medians = [float(np.median(value)) for value in grouped.values()]
        axis.plot(x, medians, marker="o", **style)
        low, high = [min(value) for value in grouped.values()], \
            [max(value) for value in grouped.values()]
        if low != high:
            axis.fill_between(x, low, high, color=plot_style.POLICY_COLORS[policy],
                              alpha=.14)
    axis.set(xscale="log", yscale="log", xlabel="Sessions",
             ylabel="Selection time (s)")
    axis.grid(True, which="both", alpha=.25)
    axis.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"))
    plt.close(fig)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    commands = value.add_subparsers(dest="command", required=True)
    cell_ = commands.add_parser("cell")
    cell_.add_argument("--solver", required=True, choices=SOLVERS)
    cell_.add_argument("--sessions", required=True, type=int)
    cell_.add_argument("--memory-gib", type=float, default=0)
    cell_.add_argument("--output", required=True, type=Path)
    cell_.add_argument("--stage", required=True, type=Path)
    run_ = commands.add_parser("run")
    run_.add_argument("--sessions", type=int, nargs="+", default=SESSIONS)
    run_.add_argument("--repeats", type=int, default=3)
    run_.add_argument("--repeat-until", type=int, default=100_000)
    run_.add_argument("--timeout-s", type=float, default=1_800)
    run_.add_argument("--setup-timeout-s", type=float, default=1_800)
    run_.add_argument("--memory-gib", type=float, default=24)
    run_.add_argument("--out", type=Path, default=DEFAULT_OUT)
    plot_ = commands.add_parser("plot")
    plot_.add_argument("--input", type=Path, required=True)
    plot_.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "cell":
        cell(args)
    elif args.command == "run":
        run(args)
    else:
        with args.input.open() as stream:
            rows = list(csv.DictReader(stream))
        numeric = {"sessions", "selection_s"}
        plot([{key: float(value) if key in numeric and value else value
               for key, value in row.items()} for row in rows], args.output)


if __name__ == "__main__":
    main()
