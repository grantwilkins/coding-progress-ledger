"""Attach-only, bounded controlled-source diagnostics; not agentic service episodes."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import destination_runner as serving
import migration_profiler as p
import migration_testbed as b


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def validate_inventory(inventory):
    source, destination = (inventory[key] for key in ("source", "destination"))
    if source["gpu_uuid"] == destination["gpu_uuid"]:
        raise ValueError("source and destination must be separate physical GPUs")
    for endpoint in (source, destination):
        if "A100" not in endpoint["gpu_name"] or not endpoint["reference_runtime_verified"]:
            raise ValueError("A100 reference runtime must be verified explicitly")
        for key in ("identity_evidence", "runtime_evidence"):
            if not Path(endpoint[key]).is_file():
                raise ValueError(f"missing {key}")
    if inventory["proxy_clock"] != "coordinator_monotonic":
        raise ValueError("phase attribution requires a coordinator-local transfer proxy")
    root = Path(inventory["stack_root"])
    for name in (*p.MP_SCENARIO_CSVS, "lmcache-source.log", "lmcache-sink.log"):
        if not (root/name).is_file():
            raise ValueError(f"missing attached stack telemetry: {name}")
    if inventory["bandwidth_mbps"] <= 0:
        raise ValueError("explicit positive observed proxy bandwidth required")
    cfg = b.Config(**inventory["config"])
    if cfg.src_port == cfg.sink_port or cfg.max_model_len != 32768:
        raise ValueError("distinct serving endpoints and 32768-token context required")
    return cfg


def scenarios(selected_method=None, selected_context=None):
    rows = []
    for seed in (7101, 7102):
        for context, appended in ((8192, 32), (30000, 2048)):
            for method in (("replay", "kv_transfer") if seed == 7101 else ("kv_transfer", "replay")):
                label = f"paired-{seed}-{context}-{method}"
                sessions = [{"session_id": f"{label}-{i}", "turn_index": 0,
                             "initial_tokens": context, "order": i} for i in range(8)]
                rows.append({"scenario_id": label, "seed": seed, "kind": "migration",
                    "method": method, "context_size": context, "activity": "one_turn",
                    "activity_tokens": appended, "request_schedule": [{"at_s": 0, "append_tokens": appended}],
                    "sessions": sessions, "moves": [{**row, "method": method} for row in sessions],
                    "concurrency": 8, "move_concurrency": 8, "serving_concurrency": 8,
                    "warm_concurrency": 8, "prestage_all": True, "copy_policy": "initial_final",
                    "reset_caches": False, "wait_cache_idle": False, "sample_power": False,
                    "final_state": "awake", "deadline_s": 180})
    return [row for row in rows if (selected_method is None or row["method"] == selected_method) and (selected_context is None or row["context_size"] == selected_context)]


class CheckedSession(p.LiveSession):
    def request(self, port, messages, label, prompt=None, bypass_lmcache=False):
        tokens = b.mp_chat_tokens(self.cfg, self.probe(messages, prompt), p.PROBE_MAX_TOKENS)
        self.event_log.write("rendered_request", session_id=self.session_id, request_label=label,
                             token_ids=tokens, prompt_tokens=len(tokens), max_tokens=p.PROBE_MAX_TOKENS)
        if len(tokens) + p.PROBE_MAX_TOKENS > self.cfg.max_model_len:
            raise ValueError("full request plus original 512-token generation exceeds context limit")
        result, text = super().request(port, messages, label, prompt, bypass_lmcache)
        self.event_log.write("state_validation", session_id=self.session_id, request_id=result.request_id,
                             response_text=text, expected_state_code=self.state_code,
                             valid=self.state_code in text)
        if self.state_code not in text:
            raise RuntimeError("migration probe omitted expected state code")
        if result.prompt_tokens != len(tokens):
            raise RuntimeError("actual migration prompt differs from rendered token count")
        return result, text


def worker(inventory, scenario, out):
    scenario = {**scenario, **{key: [{**row, "session_id": f"{out.parent.name}-{row['session_id']}"} for row in scenario[key]] for key in ("sessions", "moves")}}
    cfg = replace(validate_inventory(inventory), architecture_campaign=True)
    stack = b.Stack(None, None, None, None, Path(inventory["stack_root"]),
                    bandwidth_mbps=inventory["bandwidth_mbps"])
    manifest = {"sessions": [{"id": row["session_id"], "job_class": "coding",
                "state_code": hashlib.sha256(row["session_id"].encode()).hexdigest()[:12].upper()}
                for row in scenario["sessions"]]}
    scenario = {**scenario, "bandwidth_mbps": inventory["bandwidth_mbps"]}
    p.LiveSession = CheckedSession
    p.run_scenario(stack, cfg, manifest, scenario, out, out.parent.name, configure_proxy=False)


def remaining_seconds(started, deadline_wall_ns=None):
    return max(0., min(1200 - (time.monotonic() - started), (deadline_wall_ns - time.time_ns()) / 1e9 if deadline_wall_ns is not None else 1200.))


def run_worker(command, timeout, log):
    with log.open("w") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return {"status": "complete" if child.wait(timeout=timeout) == 0 else "failed",
                    "returncode": child.returncode}
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            return {"status": "timeout", "returncode": child.returncode,
                    "interpretation": "failed acquisition; serving endpoints not stopped"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run", "worker"))
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--method", choices=("replay", "kv_transfer"))
    parser.add_argument("--context", type=int, choices=(8192, 30000))
    args = parser.parse_args()
    os.environ.update(QH_LMCACHE_MODE="mp", QH_RUNTIME="native")
    inventory = json.loads(args.inventory.read_text())
    cfg = validate_inventory(inventory)
    plan_path = args.out/"paired-plan.json"
    if args.stage == "freeze":
        args.out.mkdir(parents=True, exist_ok=True)
        if plan_path.exists():
            raise FileExistsError(plan_path)
        evidence = [args.inventory, Path(p.__file__), Path(b.__file__), *[Path(endpoint[key]) for endpoint in
                    (inventory["source"], inventory["destination"])
                    for key in ("identity_evidence", "runtime_evidence")]]
        write(plan_path, {"scope": "controlled source append and paired KV diagnostics; not recorded agentic service",
            "total_limit_s": 1200, "per_scenario_limit_s": 180, "method": args.method, "context": args.context, "deadline_wall_ns": inventory.get("deadline_wall_ns"), "scenarios": scenarios(args.method, args.context),
            "input_sha256": {str(path): p.file_hash(path) for path in evidence},
            "driver_sha256": p.file_hash(Path(__file__)), "argv": sys.argv,
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dirty": subprocess.check_output(["git", "status", "--porcelain"], text=True),
            "arrival_schedule": "assumed one controlled source append at migration epoch; no production-arrival claim",
            "pairing": "same target context/append/width, independent unique histories between methods; counterbalanced order",
            "source_gpu": inventory["source"], "destination_gpu": inventory["destination"],
            "runtime_config": asdict(validate_inventory(inventory))})
        return
    plan = json.loads(plan_path.read_text())
    if plan["driver_sha256"] != p.file_hash(Path(__file__)):
        raise ValueError("driver changed after freezing plan")
    for path, expected in plan["input_sha256"].items():
        if p.file_hash(Path(path)) != expected:
            raise ValueError(f"frozen input changed: {path}")
    if plan.get("method") != args.method or plan.get("context") != args.context or plan.get("deadline_wall_ns") != inventory.get("deadline_wall_ns") or plan["scenarios"] != scenarios(args.method, args.context) or plan["total_limit_s"] != 1200 or plan["per_scenario_limit_s"] != 180:
        raise ValueError("frozen bounded plan differs from driver")
    if args.stage == "worker":
        worker(inventory, plan["scenarios"][args.index], args.out/plan["scenarios"][args.index]["scenario_id"])
        return
    launch = args.out/"paired-launch.json"
    if launch.exists():
        raise FileExistsError("paired acquisition already attempted; no retries")
    started = time.monotonic()
    write(launch, {"argv": sys.argv, "start_wall_ns": time.time_ns(), "start_monotonic": started,
                   "plan_sha256": p.file_hash(plan_path), "driver_sha256": p.file_hash(Path(__file__))})
    results = []
    for index, row in enumerate(plan["scenarios"]):
        remaining = remaining_seconds(started, plan.get("deadline_wall_ns"))
        if remaining <= 0:
            results.extend({"scenario": item["scenario_id"], "status": "unmeasured_budget_limit"}
                           for item in plan["scenarios"][index:])
            break
        root = args.out/row["scenario_id"]
        root.mkdir(exist_ok=False)
        offsets = {name: (Path(inventory["stack_root"])/name).stat().st_size
                   for name in (*p.MP_SCENARIO_CSVS, "lmcache-source.log", "lmcache-sink.log")}
        command = [sys.executable, str(Path(__file__).resolve()), "worker", "--inventory",
                   str(args.inventory.resolve()), "--out", str(args.out.resolve()), "--index", str(index)]
        if args.method:
            command += ["--method", args.method]
        if args.context:
            command += ["--context", str(args.context)]
        samplers = [serving.MetricsSampler(cfg.host, port, root/f"engine-{role}.csv", .5)
                    for role, port in (("source", cfg.src_port), ("destination", cfg.sink_port))]
        samplers.append(p.PowerSampler(root/"power-source.csv", .5))
        for sampler in samplers:
            sampler.start()
        try:
            remaining = remaining_seconds(started, plan.get("deadline_wall_ns"))
            outcome = run_worker(command, min(180, remaining), root/"worker.log") if remaining > 0 else {"status": "unmeasured_global_deadline"}
        finally:
            for sampler in samplers:
                sampler.close()
        for name, offset in offsets.items():
            source = Path(inventory["stack_root"])/name
            with source.open("rb") as handle:
                handle.seek(offset)
                (root/f"attached-{name}.raw").write_bytes(handle.read())
        results.append({"scenario": row["scenario_id"], "command": command, "log_offsets": offsets, **outcome})
        write(root/"attempt.json", results[-1])
        write(args.out/"paired-attempts.json", results)
        if outcome["status"] == "timeout":
            results.extend({"scenario": item["scenario_id"], "status": "unmeasured_after_timeout"}
                           for item in plan["scenarios"][index+1:])
            break
    write(args.out/"paired-attempts.json", results)
    write(args.out/"paired-stop.json", {"elapsed_s": time.monotonic()-started, "end_wall_ns": time.time_ns(), "deadline_wall_ns": plan.get("deadline_wall_ns")})


if __name__ == "__main__":
    main()
