"""Single-H100 serving-to-awake-idle controls; no migration or fleet claim."""
import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import destination_runner as serving
import migration_testbed as testbed
import network_campaign as network
from power_trace import NvmlPowerSampler


def schedules(manifest, phase, seconds, repeat):
    """Preserve recorded inter-turn gaps; token contents are deterministic surrogates."""
    rows = []
    if phase == "coding_trace":
        for record in manifest["sessions"]:
            turns = record["turns"]
            origin = turns[0]["time_s"]
            gaps = [b["time_s"] - a["time_s"] for a, b in zip(turns, turns[1:])]
            period = max(1., turns[-1]["time_s"] - origin + (min(gaps) if gaps else 10.))
            cycle = 0
            while cycle * period < seconds:
                for index, turn in enumerate(turns):
                    at = cycle * period + turn["time_s"] - origin
                    if at < seconds:
                        rows.append((at, record["id"], turn["input_tokens"],
                                     turn["output_tokens"], cycle * len(turns) + index))
                cycle += 1
    else:
        raise ValueError("only recorded timing schedules are supported")
    if any(p < 2 or g < 1 or p + g > 32768 for _, _, p, g, _ in rows):
        raise ValueError("recorded request does not fit the pinned 32K runtime")
    return sorted(rows)


def run(model, manifest_path, root, seconds=60., idle_s=30., repeats=1):
    manifest = json.loads(manifest_path.read_text())
    schedules(manifest, "coding_trace", seconds, 0)
    network.configure_handoff_environment(model)
    cfg = testbed.model_campaign_config(model, serving=True)
    testbed.preflight(cfg, 1)
    root.mkdir(parents=True, exist_ok=False)
    (root / "metadata.json").write_text(json.dumps({
        "model": model, "revision": testbed.model_spec(model).revision,
        "runtime": network.expected_runtime(), "config": asdict(cfg),
        "manifest_path": str(manifest_path), "manifest_sha256": network.profiler.file_hash(manifest_path),
        "scope": "Recorded coding shapes/inter-turn timing with synthetic tokens; sessions start together and loop independently, with overlapping turns allowed. Forced-length synthetic tokens; separate eight-concurrency prefill/decode/mixed controls. No migration or fleet claim.",
        "seconds": seconds, "idle_s": idle_s, "repeats": repeats,
    }, default=str, indent=2) + "\n")
    processes = []
    sampler = NvmlPowerSampler(root / "power.jsonl")
    metrics = None
    sampler_started = False
    events = (root / "phases.jsonl").open("x", buffering=1)
    def mark(phase, event, repeat):
        events.write(json.dumps({"phase": phase, "event": event, "repeat": repeat,
            "monotonic_ns": time.monotonic_ns(), "wall_ns": time.time_ns()}) + "\n")
    try:
        for command, log, port in [
            (testbed.redis_cmd(cfg), "redis.log", cfg.lmc_port),
            (testbed.mp_server_cmd(cfg, "source", l2_host="127.0.0.1", l2_port=cfg.lmc_port), "cache.log", cfg.src_lmc_port),
        ]:
            process = testbed.start_logged(command, root / log); processes.append(process)
            testbed.wait_tcp_process("127.0.0.1", port, 300, process, root / log)
        command = testbed.vllm_cmd(cfg, "source", gpu_index=0, sleep_mode=False)
        (root / "server_command.json").write_text(json.dumps(command, indent=2) + "\n")
        process = testbed.start_logged(command, root / "source.log"); processes.append(process)
        testbed.wait_health_process("127.0.0.1", cfg.src_port, testbed.health_timeout(), process, root / "source.log")
        testbed.validate_model_runtime_log(cfg, testbed.read_text(root / "source.log"))
        sampler.start()
        sampler_started = True
        metrics = serving.MetricsSampler("127.0.0.1", cfg.src_port, root / "engine.csv")
        metrics.start()
        from transformers import AutoTokenizer
        vocabulary = AutoTokenizer.from_pretrained(str(testbed.model_path(cfg))).vocab_size
        for repeat in range(repeats):
            for phase in ("coding_trace", "prefill", "decode", "mixed"):
                mark(phase, "idle_before", repeat); time.sleep(idle_s)
                output = root / f"{phase}-r{repeat}.jsonl"
                if output.exists(): raise FileExistsError(output)
                mark(phase, "active_start", repeat)
                epoch = time.monotonic_ns()
                events.write(json.dumps({"phase": phase, "event": "admissions_deadline", "repeat": repeat, "monotonic_ns": epoch + int(seconds * 1e9), "wall_ns": time.time_ns() + int(seconds * 1e9)}) + "\n")
                deadline = epoch + int(seconds * 1e9)
                lock = threading.Lock()
                def request(index, sid, p, g, at):
                    session = serving.Session(sid, p - 1, 1, g, vocabulary, repeat, force_output=False)
                    with lock:
                        stream.write(json.dumps({"event": "arrival", "session_id": sid, "request_index": index, "scheduled_ns": at, "dispatch_ns": time.monotonic_ns()}) + "\n")
                    row = serving.issue("127.0.0.1", cfg.src_port, model, session, index, at, 900, True)
                    row.update(phase=phase, repeat=repeat, event="completion")
                    with lock: stream.write(json.dumps(row) + "\n")
                    return row
                with output.open("x", buffering=1) as stream, ThreadPoolExecutor(max_workers=256) as pool:
                    if phase == "coding_trace":
                        pending = []
                        for offset, sid, p, g, index in schedules(manifest, phase, seconds, repeat):
                            at = epoch + int(offset * 1e9)
                            time.sleep(max(0, (at - time.monotonic_ns()) / 1e9))
                            pending.append(pool.submit(request, index, sid, p, g, at))
                        rows = [future.result() for future in pending]
                    else:
                        def worker(index):
                            rows = []
                            while time.monotonic_ns() < deadline:
                                prefill = phase == "prefill" or phase == "mixed" and index % 2 == 0
                                p, g = (16384, 1) if prefill else (4096, 512)
                                rows.append(request(len(rows), f"{phase}-{index}", p, g, time.monotonic_ns()))
                            return rows
                        rows = []
                        for batch in pool.map(worker, range(8)):
                            rows.extend(batch)
                    if not rows or any(not serving.service_completion(row) for row in rows):
                        raise RuntimeError("serving control has incomplete or failed requests")
                time.sleep(max(0, (deadline - time.monotonic_ns()) / 1e9))
                mark(phase, "active_drained", repeat)
                mark(phase, "idle_after", repeat); time.sleep(idle_s)
                mark(phase, "idle_after_end", repeat)
    finally:
        try:
            if metrics: metrics.close()
        finally:
            try:
                if sampler_started: sampler.close()
            finally:
                for process in reversed(processes): testbed.stop_proc(process)
                events.close()
    (root / "complete.json").write_text(json.dumps({"status": "complete"}) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=testbed.MODEL_SPECS)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--idle-s", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    if min(args.seconds, args.idle_s, args.repeats) <= 0: parser.error("windows and repeats must be positive")
    run(args.model, args.manifest, args.run_root, args.seconds, args.idle_s, args.repeats)
