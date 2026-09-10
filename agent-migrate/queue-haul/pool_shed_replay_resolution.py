"""Reproduce the replay discrepancy from frozen A100 evidence and bounded CPU comparisons."""

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pool_shed_campaign as q
from pool_shed_execution import initial_work, kv_transfer_bytes, source_snapshot
from pool_shed_replay_audit import FIELDS, fleet_summary

HARDWARE = "outputs/a100-replay-final-20260910T0352"
EXTRA_FIELDS = ("resident_displaced_work_s", "resident_pool_compensation_work_s", "peak_migration_replicas",
                "service_recovery_scope", "pending_source_buffer_work_s", "pending_backlog_reference_work_s")


def hardware_rows(analysis, calibration):
    rows, requests = [], []
    for episode in analysis["episodes"]:
        spec = episode["spec"]
        if not spec["episode"].startswith("episodes-"):
            continue
        events = episode["migration_events"]
        starts = [e for e in events if e["kind"] == "initial_start"]
        switches = [e for e in events if e["kind"] == "route_switch"]
        resident = next(w for w in episode["windows"] if (w["cohort"], w["start_s"], w["end_s"]) == ("resident", 60, 300))
        row = {"episode": spec["episode"], "workload": spec["workload"], "arm": spec["arm"], "seed": spec["seed"],
               "resident_rps": spec["rate"], "resident_arrivals": resident["offered_requests"],
               "resident_completions": resident["completed_arrival_cohort_requests"],
               "resident_p90_ttft_s": resident["p90_original_arrival_ttft_s"],
               "resident_p90_mean_tpot_s": resident["p90_request_mean_tpot_s"],
               "resident_known_ttft_over_1s": resident["known_original_arrival_ttft_over_1s"]}
        if spec["arm"] != "control":
            if len(starts) != 8 or len(switches) != 8:
                raise ValueError("hardware migration requires eight captured and switched histories")
            contexts = np.array([e["context_tokens"] for e in starts], float)
            row.update(initial_context_tokens=contexts.astype(int).tolist(),
                       initial_contexts_below_24k=int(sum(contexts < 24576)),
                       handoff_elapsed_s=(max(e["monotonic_ns"] for e in switches) - min(e["monotonic_ns"] for e in starts)) / 1e9)
            if spec["arm"] == "replay":
                fleet = SimpleNamespace(context=contexts, t1=q.replay_seconds(contexts, calibration), log=2 * contexts,
                                        metadata={k: calibration[k] for k in ("batch_context_limit", "packing_context_tokens")})
                initial = [r for r in episode["migration_requests"] if r["phase"] == "initial"]
                row.update(model_initial_idle_s=initial_work(fleet, np.ones(8), 0, 1, contexts, calibration["timing"][0], calibration)[1],
                           initial_last_first_token_s=(max(r["first_ns"] for r in initial) - min(r["start_ns"] for r in initial)) / 1e9,
                           initial_last_completion_s=(max(r["end_ns"] for r in initial) - min(r["start_ns"] for r in initial)) / 1e9,
                           initial_output_tokens=[r["output_tokens"] for r in initial])
        rows.append(row)
        for request in episode["migration_requests"]:
            if request["status"] != 200 or not request["done"] or not request["exact_token_timestamps"]:
                raise ValueError("migration timing requires a complete exact token stream")
            requests.append({**{k: row[k] for k in ("episode", "workload", "arm", "seed")},
                **{k: request[k] for k in ("session", "phase", "serving_role", "prompt_tokens", "cached_tokens", "output_tokens", "ttft_s")},
                "elapsed_s": (request["end_ns"] - request["start_ns"]) / 1e9,
                "after_first_token_s": (request["end_ns"] - request["first_ns"]) / 1e9})
    return rows, requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=q.ROOT / "outputs/a100-replay-resolution-20260910")
    args = parser.parse_args()
    if (args.out / "report.json").exists():
        raise ValueError("use a fresh output directory; do not overwrite a frozen audit")
    calibration = q.calibration(0)
    timing = calibration["timing"][0]
    sources = {**q.provenance(calibration), **{name: hashlib.sha256((q.ROOT / name).read_bytes()).hexdigest()
               for name in (Path(__file__).name, "pool_shed_replay_audit.py")}}

    def read(name):
        file = q.ROOT / name
        sources[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        return json.loads(file.read_text())

    acquisition = read(f"{HARDWARE}/report.json")
    analysis = read(f"{HARDWARE}/service-analysis.json")
    cold = read("outputs/a100-replay-completion-20260910T0116/model-audit.json")
    previous = read("outputs/a100-replay-live-20260909T1920/policy-verification.json")
    hardware, requests = hardware_rows(analysis, calibration)
    report = {"scope": "Diagnosis, unchanged timing coefficients, 2 MW source and each destination, full-context replay; no GPU or full campaign run.",
              "sources": sources, "resident_latency_validated": False, "campaign_ready": False,
              "wire_bytes_per_32768_tokens": 800_000_000, "shared_wan_gbps": 1000,
              "selected_hardware_resident_rates": acquisition["selected_resident_rates"],
              "unloaded_holdout": cold["heldout_by_phase_and_width"], "hardware": hardware,
              "hardware_scope": "60–300s arrival windows, two seeds, eight physical resident histories; TTFT includes queueing. Initial cold-model comparison uses actual snapshots but omits loaded scheduler interference and varying output generation; it is not a matched fit.",
              "request_timing_scope": "After-first-token wall time includes decode, interleaved work and scheduling; not a pure GPU decode duration. Full request completion must not be fitted as cold prefill.",
              "fleets": {}, "comparisons": []}
    endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    for workload in ("coding", "coding_long"):
        original = q.sample_fleet(workload, gpus=6666, gpus_per_node=8)
        fleet = replace(original, metadata={**original.metadata, "planning_reference_s": 4.,
            "kv_wire_scale": 800_000_000 / (32768 * 49152),
            "replay_cached_tokens": np.zeros(len(original.count)).tolist(), "kv_shared_tokens": np.zeros(len(original.count)).tolist()})
        fleet = replace(fleet, kv=kv_transfer_bytes(fleet, fleet.context, calibration))
        budgets = q.bandwidth(endpoint, fleet.nodes, 1000)
        replay, kv = q.include_isolated(*q.library(fleet), q.isolated_methods(fleet, .5, endpoint, budgets, timing))
        candidate_hash = hashlib.sha256(replay.tobytes() + kv.tobytes()).hexdigest()
        frozen = fleet_summary(fleet, calibration)
        frozen.update(standing_service_ceiling=min(1., 2 * 6666 * .5 / (6666 * q.SOURCE_LOAD)),
                      initial_wire_time_at_shared_cap_s=frozen["initial_kv_wire_bytes"] / budgets[2],
                      initial_replay_text_time_at_shared_cap_s=frozen["initial_replay_log_bytes"] / budgets[2],
                      width8_batches_per_destination_gpu=float(fleet.count.sum() / (8 * 2 * 6666)),
                      source_duration_proxy_mean_s=float(fleet.count @ np.array([np.mean(s) for s in fleet.metadata["turn_duration_s"]]) / fleet.count.sum()),
                      contexts_at_time_s={str(t): source_snapshot(fleet, t)[0].astype(int).tolist() for t in (0, 30, 120)})
        report["fleets"][workload] = frozen
        variants = [("primary", fleet, timing, 30., q.POLICIES), ("primary", fleet, timing, 120., q.POLICIES),
                    ("no_regional_replay_discount", fleet, {**timing, "regional_replay_factor": [1., 1.]}, 30., ("queue_haul", "replay_only")),
                    ("half_destination_inventory", replace(fleet, metadata={**fleet.metadata, "destination_gpus": 3333}), timing, 30., ("queue_haul", "replay_only"))]
        for variant, candidate, coefficients, deadline, policies in variants:
            table = q.schedule_table(candidate, replay, kv, .5, deadline, endpoint, budgets, coefficients)
            cell = {"workload": workload, "variant": variant, "deadline_s": deadline,
                    "candidate_sha256": candidate_hash, "results": {}}
            for policy in policies:
                result = q.execute_feedback(table, table, policy, coefficients, calibration)
                displaced, compensated, generated, recovered, pending = [np.array(result[k]) for k in
                    ("resident_displaced_work_s", "resident_pool_compensation_work_s", "resident_debt_generated_work_s", "resident_debt_recovered_work_s", "pending_resident_debt_work_s")]
                if (result["max_relative_residual"] > 1e-8 or result["last_completion_s"] > deadline + 1e-8
                        or result["resident_latency_validated"] or np.any(np.array(result["transferred_bytes"]) > budgets * deadline * (1 + 1e-8))
                        or not np.allclose(displaced - compensated, generated, rtol=1e-8, atol=1e-6)
                        or not np.allclose(generated - recovered, pending, rtol=1e-8, atol=1e-6)):
                    raise ValueError("capacity, resident conservation or scope check failed")
                cell["results"][policy] = {k: result[k] for k in (*FIELDS, *EXTRA_FIELDS)}
                if variant == "primary":
                    old = next(c for c in previous["cells"] if c["workload"] == workload and c["deadline_s"] == deadline)
                    if old["candidate_sha256"] != candidate_hash or not np.isclose(result["shed_fraction"], old["results"][policy]["shed_fraction"], atol=1e-8, rtol=0):
                        raise ValueError("diagnostics changed archived primary candidates or handoff fraction")
                print(workload, variant, deadline, policy, round(result["shed_fraction"], 6),
                      "displaced", round(displaced.sum(), 2), "pool-compensated", round(compensated.sum(), 2), flush=True)
            report["comparisons"].append(cell)
    report["policy_evaluations"] = sum(len(c["results"]) for c in report["comparisons"])
    report["interpretation"] = [
        "Cold width-eight replay is close to unloaded measurements. Warm full-message catch-up naturally hits retained prefixes; the default full-rebuild assumption can overcharge it, not explain fast replay.",
        "Pooled spare GPUs cancel the local resident throughput-loss proxy before debt is reported. Aggregate service recovery therefore does not establish that pinned resident histories avoided stalls.",
        "Two equal-size destinations at half nominal load provide more standing spare service than the source uses. Width-eight migrations initially need only half the combined destination GPU inventory; shrinking all sites equally preserves this ratio.",
        "The 1000 Gbit/s fleet WAN and the 1000 Mbit/s hardware GET cap differ by 1000. KV moves tens of TB for this fleet; replay moves a few GB of token text and rebuilds in parallel.",
        "Nominal half-load is 0.2213 coding or 0.1882 coding_long RPS/GPU. The inherited short-request timing conversion yields less than 1% replay slowdown. New coding hardware ran at twice that coding RPS; neither label is validated GPU utilization.",
        "Historical regional factors make replay about 22% faster. The no-discount ablation isolates that assumption without refitting from a policy outcome.",
        "Agentic traces have no arrival timestamps. Source durations are throughput proxies and cyclic wraps are synthetic lifecycle resets; initially long cohorts can migrate after wrapping to short contexts.",
        "Shed fraction credits ownership switches; buffered requests and resident latency are separate. QH optimizes a receding temporal LP approximation, so executed outcomes need not dominate replay even when initial LP baseline containment holds."]
    report["remaining_model_work"] = [
        "Use a replica-affine service/queue model validated against the existing paired request traces before claiming SLO-feasible shedding; do not turn displaced work into an arbitrary global pause.",
        "Keep full-message replay requests and separately account for observed prefix reuse, rebuild, generation and source quiescence when comparing a specific hardware episode.",
        "Keep actual RPS, per-replica history count, arrival assumptions, wire units and raw versus recovered service explicit in any campaign contract; no extra broad GPU acquisition is needed to establish this diagnosis."]
    if report["policy_evaluations"] != 28 or len(hardware) != 12:
        raise ValueError("bounded audit coverage changed")
    if any(hashlib.sha256((q.ROOT / path).read_bytes()).hexdigest() != checksum for path, checksum in sources.items()):
        raise ValueError("audit inputs changed during execution")
    q.write_json(args.out / "report.json", report)
    with (args.out / "migration-requests.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(requests[0]))
        writer.writeheader()
        writer.writerows(requests)
    print(args.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
