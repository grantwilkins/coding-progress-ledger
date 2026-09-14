"""Matched destination interference controls on two H100s; not source-drain relief."""
import argparse
import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import destination_runner as serving
import migration_testbed as testbed
import network_campaign as network
from power_trace import NvmlPowerSampler


def actions(arm, count=8):
    if count < 2 or count % 2:
        raise ValueError("matched mixed controls require an even migration count")
    return {"none": [], "replay": ["replay"] * count,
            "kv": ["kv_transfer"] * count,
            "mixed": ["replay", "kv_transfer"] * (count // 2)}[arm]


def run(model, cluster, calibration, root, key, resident_rps, context=16384,
        seconds=120., warmup=30., reference=None, arms=("none", "replay", "kv", "mixed")):
    if len(cluster.destinations) != 1 or not 0 < warmup < seconds or resident_rps <= 0:
        raise ValueError("need one destination and positive resident rate/observation windows")
    if not arms or len(set(arms)) != len(arms) or set(arms) - {"none", "replay", "kv", "mixed"}:
        raise ValueError("invalid matched control arms")
    network.configure_handoff_environment(model)
    contract = network.freeze_contract(calibration)
    node = cluster.destinations[0]
    contract["paths"] = {node.id: contract["paths"][node.id]}
    contract["aggregate"] = dict(contract["paths"][node.id])
    hosts = network.host_check(cluster, key)
    stack = network.start_cluster(cluster, key, contract, "natural", root,
                                  model=model, literal_token_timing=True)
    power = NvmlPowerSampler(root / "source-power.jsonl")
    started = False
    try:
        network.write_checkpoint(root / "metadata.json", {
            "model": model, "revision": testbed.model_spec(model).revision,
            "config": json.loads(json.dumps(asdict(stack.cfg), default=str)), "hosts": hosts, "cluster": asdict(cluster),
            "runtime": network.expected_runtime(), "resident_rps": resident_rps,
            "context_tokens": context, "migration_count": 8, "repeats": 1,
            "seconds": seconds, "warmup_s": warmup, "arms": list(arms),
            "scope": "Destination interference at an explicit offered rate, not a claim of 50% utilization or source-drain power relief. Synthetic resident inputs. Source histories are idle after export.",
            "network": "Natural South Central route; prior two-route envelope subset, not a new independent route calibration.",
            "calibration_sha256": network.profiler.object_hash(calibration),
            "destination_power_sensor": "NVML standard power usage: one-second average",
        })
        power.start(); started = True
        gate = network._network_state_equivalence(stack, 32256 if reference else context, reference)
        from transformers import AutoTokenizer
        vocabulary = AutoTokenizer.from_pretrained(str(testbed.model_path(stack.cfg))).vocab_size
        sessions = [serving.Session(f"resident-{i}", context - 1, 1, 128,
                                    vocabulary, 0, force_output=False) for i in range(8)]
        for arm in arms:
            arm_root = root / arm; arm_root.mkdir()
            network._clear_cluster(stack)
            prepared = []
            for index in range(8):
                session = {"id": f"matched-{index}", "state_code": f"QHM{index:03d}"}
                messages = network.profiler.exact_calibration_messages(stack.cfg, session, context, max_tokens=128)
                tokens = testbed.mp_chat_tokens(stack.cfg, network._probe(stack.cfg, messages, session["state_code"]), max_tokens=128)
                warm = network._warm(stack, messages, session["state_code"], 900, tokens)
                prepared.append((session, messages, tokens, warm))
            network.write_checkpoint(arm_root / "exports.json", [dict(session=s, warm=w) for s, _, _, w in prepared])
            metrics = serving.MetricsSampler(node.host, stack.cfg.sink_port, arm_root / "engine.csv")
            metrics.start()
            epoch, wall = time.monotonic_ns(), time.time_ns()
            network.write_checkpoint(arm_root / "schedule.json", {
                "epoch_monotonic_ns": epoch, "epoch_wall_ns": wall,
                "arrival_offsets_s": [i / resident_rps for i in range(math.ceil(seconds * resident_rps))],
                "migration_offset_s": warmup, "arm": arm})
            lock = threading.Lock()
            before = testbed.proxy_counts(root / "proxy_bytes.csv")
            try:
                with (arm_root / "events.jsonl").open("x", buffering=1) as output, ThreadPoolExecutor(max_workers=256) as pool:
                    def write(row):
                        with lock: output.write(json.dumps(row) + "\n")
                    resident_rows, migration_rows = [], []
                    def resident(index, at):
                        write({"event": "arrival", "kind": "resident", "index": index, "scheduled_ns": at, "dispatch_ns": time.monotonic_ns()})
                        row = serving.issue(node.host, stack.cfg.sink_port, model, sessions[index % 8], index, at, 900, True)
                        write({"event": "completion", "kind": "resident", "index": index, **row})
                        with lock: resident_rows.append(row)
                        if not serving.service_completion(row): raise RuntimeError("resident request failed")
                    methods = actions(arm)
                    barrier = threading.Barrier(len(methods)) if methods else None
                    def migrate(index, method):
                        session, messages, tokens, warm = prepared[index]
                        barrier.wait()
                        scheduled = epoch + int(warmup * 1e9)
                        write({"event": "arrival", "kind": "migration", "session_id": session["id"], "method": method, "scheduled_ns": scheduled, "dispatch_ns": time.monotonic_ns()})
                        row = network._chat(stack.cfg, stack.ports[node.id]["api"], messages, session["state_code"], 900, method == "replay", prompt_ids=tokens)
                        write({"event": "completion", "kind": "migration", "session_id": session["id"], "method": method, **row})
                        with lock: migration_rows.append(row)
                        expected = context // testbed.model_chunk_tokens(stack.cfg) * testbed.model_chunk_tokens(stack.cfg) if method == "kv_transfer" else 0
                        if row["cached_tokens"] != expected: raise RuntimeError("migration method/cache evidence mismatch")
                    jobs = [(i / resident_rps, "resident", i) for i in range(math.ceil(seconds * resident_rps))]
                    jobs += [(warmup, "migration", i) for i in range(len(methods))]
                    pending = []
                    for offset, kind, index in sorted(jobs):
                        at = epoch + int(offset * 1e9)
                        time.sleep(max(0, (at - time.monotonic_ns()) / 1e9))
                        pending.append(pool.submit(resident, index, at) if kind == "resident" else pool.submit(migrate, index, methods[index]))
                    time.sleep(max(0, (epoch + int(seconds * 1e9) - time.monotonic_ns()) / 1e9))
                    write({"event": "admissions_end", "monotonic_ns": time.monotonic_ns()})
                    for future in pending: future.result()
                    write({"event": "all_completed", "monotonic_ns": time.monotonic_ns()})
            finally:
                metrics.close()
            time.sleep(1)
            wire = testbed.count_delta(before, testbed.proxy_counts(root / "proxy_bytes.csv"))
            kv_bytes = wire.get(f"kv/{node.id}/target_to_client", 0)
            if (arm in ("none", "replay") and kv_bytes != 0) or (arm in ("kv", "mixed") and kv_bytes <= 0):
                raise RuntimeError("arm wire evidence does not match assigned actions")
            geometry = gate["geometry"]
            expected_bytes = sum(g["chunk_bytes"] * (context // geometry["chunk_tokens"] if g["sw_size_chunks"] < 0 else min(context // geometry["chunk_tokens"], g["sw_size_chunks"])) for g in geometry["object_groups"]) * methods.count("kv_transfer")
            resident_lateness = max(row["send_lateness_s"] for row in resident_rows)
            migration_lateness = max(((row["start_ns"] - epoch) / 1e9 - warmup for row in migration_rows), default=0)
            valid = max(resident_lateness, migration_lateness) <= .25 and expected_bytes <= kv_bytes <= expected_bytes + methods.count("kv_transfer") * 1_000_000
            network.write_checkpoint(arm_root / "complete.json", {"wire_bytes": wire, "expected_compact_bytes": expected_bytes, "max_resident_lateness_s": resident_lateness, "max_migration_lateness_s": migration_lateness, "status": "complete" if valid else "invalid_measurement"})
            if not valid: raise RuntimeError("matched schedule or compact payload bound failed; raw evidence retained")
    finally:
        try:
            if started: power.close()
        finally:
            network.stop_cluster(stack)
    network.write_checkpoint(root / "complete.json", {"status": "complete"})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=testbed.MODEL_SPECS)
    p.add_argument("--cluster", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--ssh-key", type=Path, default=Path.home() / ".ssh/azrs")
    p.add_argument("--resident-rps", type=float, required=True)
    p.add_argument("--state-reference", type=Path)
    p.add_argument("--arms", nargs="+", choices=("none", "replay", "kv", "mixed"), default=("none", "replay", "kv", "mixed"))
    p.add_argument("--seconds", type=float, default=120)
    p.add_argument("--warmup-s", type=float, default=30)
    a = p.parse_args()
    run(a.model, network.Cluster.load(a.cluster), json.loads(a.calibration.read_text()),
        a.run_root, a.ssh_key, a.resident_rps, seconds=a.seconds, warmup=a.warmup_s,
        reference=a.state_reference, arms=tuple(a.arms))
