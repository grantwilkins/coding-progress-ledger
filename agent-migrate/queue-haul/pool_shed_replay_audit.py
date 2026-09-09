"""Bounded replay/resident audit using existing measurements; no campaign or refit."""

import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import pool_shed_campaign as q
from loaded_service_model import historical_execution_check
from pool_shed_cache_sensitivity import build
from pool_shed_calibration import loaded_execution_check, resident_execution_check
from pool_shed_execution import catchup, destination_gpus, kv_transfer_bytes, regional_execution_check

OUT = q.ROOT / "outputs/a100-replay-realism/audit.json"
FIELDS = ("shed_fraction", "admitted_shed_fraction", "last_completion_s", "service_ready_s", "action_fractions",
          "migration_idle_work_s", "batch_replica_seconds", "transferred_bytes", "buffered_requests",
          "pending_buffered_work_s", "resident_debt_generated_work_s", "resident_debt_recovered_work_s",
          "pending_resident_debt_work_s", "final_destination_load", "service_recovered_by_deadline",
          "resident_latency_validated", "max_relative_residual", "planning_s")


def fleet_summary(fleet, calibration):
    metadata = fleet.metadata
    shapes = np.array([[np.mean([r[k] for r in rows]) for k in ("prompt", "output")]
                       for rows in metadata["turn_sequences"]])
    means = fleet.count @ shapes / fleet.count.sum()
    return {"source_gpus": fleet.gpus, "destination_gpus_per_site": destination_gpus(fleet),
            "source_sessions": int(fleet.count.sum()), "cycle_mean_prompt_output_tokens": means.tolist(),
            "source_rps_per_gpu": float(fleet.count.sum() * metadata["source_session_rps"] / fleet.gpus),
            "resident_rps_per_gpu_at_load_half": .5 * metadata["reference_rps"],
            "resident_prompt_output_tps_at_load_half": (.5 * metadata["reference_rps"] * means).tolist(),
            "source_period_s": 1 / metadata["source_session_rps"], "source_phase_s": metadata["source_phase_s"],
            "timing_load_factor": metadata["timing_load_factor"],
            "replay_slowdown_at_load_half": float(np.exp(.5 * metadata["timing_load_factor"] * calibration["timing"][0]["beta"])),
            "initial_kv_wire_bytes": float(fleet.count @ fleet.kv),
            "initial_replay_log_bytes": float(fleet.count @ fleet.log),
            "reserved_resident_kv_capacity_fraction": fleet.baseline_kv / fleet.kv_capacity,
            "service_context_extrapolated": metadata["service_context_extrapolated"],
            "excluded_states": metadata["excluded_states"],
            "full_handoff_source_power_mw": fleet.gpus * metadata["source_power"]["delta_w"] / 1e6}


def main():
    calibration, timing = q.calibration(0), q.calibration(0)["timing"][0]
    sources = {**q.provenance(calibration), **{name: hashlib.sha256((q.ROOT / name).read_bytes()).hexdigest()
               for name in (Path(__file__).name, "pool_shed_cache_sensitivity.py")}}

    def read(path):
        file = q.ROOT / path
        sources[path] = hashlib.sha256(file.read_bytes()).hexdigest()
        return json.loads(file.read_text()) if file.suffix == ".json" else list(csv.DictReader(file.read_text().splitlines()))

    packing = read("outputs/policy-hardware-width8-packing-20260730/policy_episodes.csv")
    normalize = read("outputs/service-headroom-a100-20260815/normalization.json")
    confirmation = read("outputs/service-headroom-a100-20260815/confirmed.json")
    transition = read("outputs/service-admission-transition-a100-20260816/summary.json")
    pressure = read("outputs/prefill-pressure-a100-20260906/run/prefill_validation.json")
    capacity = read("outputs/single-gpu-capacity-a100-20260815/summary.json")
    archive = read("outputs/a100-replay-audit/audit.json")
    if confirmation["planner_usable"] or transition["planner_usable"] or not transition["campaign_pass"]:
        raise ValueError("frozen SLO evidence scope changed")
    base = build(3, calibration)
    fleet = base.fleet
    if np.any(fleet.metadata["replay_cached_tokens"]) or np.any(fleet.metadata["kv_shared_tokens"]):
        raise ValueError("audit requires full-context replay and no extra private-KV discount")
    report = {"scope": "Full-context replay, shared resident contention and explicit recovery; central calibration, no refits or campaign restart.",
        "sources": sources, "resident_latency_validated": False, "campaign_ready": False,
        "solver_version": q.highspy.Highs().version(),
        "fleets": {w: fleet_summary(q.sample_fleet(w), calibration) for w in q.WORKLOADS},
        "historical_reference": {"path": "outputs/a100-replay-audit/audit.json",
            "scope": "Archived v8 isolation/synchronized-arrival results; not a matched ablation of the current source dynamics.",
            "results": archive["ablations"][0]["results"]},
        "measurement_review": {
            "width8_replay": {name: {"episodes": len(values), "median_commit_s": float(np.median(values))}
                for name in sorted({r["context_profile"] for r in packing})
                for values in [[float(r["commit_100_s"]) for r in packing if r["context_profile"] == name and r["policy"] == "replay_only"]]},
            "width8_scope": "Existing measured packing calibration and held-out reproduction; no fit to datacenter policy rankings.",
            "slo_targets": transition["targets"], "slo_headroom_checks": confirmation["checks"],
            "slo_transition_checks": transition["mix_checks"], "slo_transition_recipes": transition["recipes"],
            "slo_scope": "Three discrete 4K recipes pass sustained incumbent/added-cohort service; no universal scalar service envelope was confirmed.",
            "cold_burst_32k": [{k: r[k] for k in ("context_tokens", "first_saturated_width", "max_peak_running_requests",
                "first_saturated_p90_ttft_s", "first_saturated_p90_mean_tpot_s", "right_censored")}
                for r in capacity["rows"] if r["model"] == "openai/gpt-oss-20b" and r["context_tokens"] == 32256],
            "cold_burst_scope": "Independent current-stack evidence of queueing at long contexts; 32 output tokens per cold request, not a direct replay-completion or warm resident-latency calibration.",
            "prefill_pressure_campaign_robust": pressure["robust"],
            "prefill_pressure_scope": "The control already separates policies; the artifact rejects a genuine prefill-pressure transition. Do not fit a contention multiplier from selected wins."}}
    prompt, output = report["fleets"]["coding"]["resident_prompt_output_tps_at_load_half"]
    report["load_mismatch"] = {
        "coding_half_rps": .5 * fleet.metadata["reference_rps"],
        "loaded_2048_32_half_rps": .5 * calibration["reference_rps"],
        "regional_604_64_half_rps": .5 / calibration["resident_interference"]["reference_work_s"],
        "coding_half_in_slo_normalizer": [prompt / normalize["prefill_tps"], output / normalize["decode_tps"]],
        "scope": "RPS, context, shape and normalization differ. The 4K normalizer illustrates the scale mismatch, not a coding occupancy or SLO estimate. Existing contextual coding cadence remains unchanged."}
    counts, probes = np.eye(len(fleet.count))[0], []
    for origin in (2048., 30000.):
        context = np.full(len(counts), origin)
        network, work = catchup(fleet, counts, 0, 0, context + 256, np.zeros(len(counts), bool), timing, calibration, origin_context=context)
        probes.append({"origin_context": origin, "appended_tokens": 256, "wire_bytes": network, "idle_work_s": work})
    report["catchup"] = {"probes": probes, "singleton_completion_overhead_s": calibration["replay_completion_s"],
        "scope": "Changed state resubmits full context and uses the existing full-context rate, packing and request overhead. No retained-prefix hit is assumed. This is a full-rebuild transfer assumption, not a fitted current-stack warm-prefix catch-up curve."}
    report["timing_checks"] = {"loaded_replay": loaded_execution_check(calibration),
        "regional": regional_execution_check(calibration), "historical": historical_execution_check(calibration)}
    report["resident_execution_check"] = resident_execution_check(calibration, report["timing_checks"]["regional"])
    if not all(r["gate_pass"] for r in report["timing_checks"].values()):
        raise ValueError("existing hardware timing reproduction failed")
    longer = q.sample_fleet("coding_long")
    longer = replace(longer, metadata={**longer.metadata, "planning_reference_s": 4., "kv_wire_scale": fleet.metadata["kv_wire_scale"]})
    longer = replace(longer, kv=kv_transfer_bytes(longer, longer.context, calibration))
    variants = [("coding_20_to_20_20", fleet, q.POLICIES),
        ("coding_20_to_10_10", replace(fleet, metadata={**fleet.metadata, "destination_gpus": 33333}), q.POLICIES),
        ("coding_2_to_2_2", build(21, calibration).fleet, ("replay_only", "queue_haul")),
        ("coding_long_20_to_20_20", longer, ("replay_only", "queue_haul"))]
    report["comparisons"] = []
    for name, candidate, policies in variants:
        replay, kv = (q.library(candidate) if name.endswith("long_20_to_20_20") else
                      (base.replay[base.route == 0], base.kv[base.route == 0]))
        table = q.schedule_table(candidate, replay, kv, .5, 30., base.endpoint, base.budgets, timing)
        ceiling = min(1., 2 * destination_gpus(candidate) * .5 / (candidate.gpus * q.SOURCE_LOAD))
        results = {}
        for policy in policies:
            result = q.execute_feedback(table, table, policy, timing, calibration)
            if (result["max_relative_residual"] > 1e-8 or not 0 <= result["shed_fraction"] <= ceiling + 1e-8
                    or result["last_completion_s"] > 30 + 1e-8 or result["resident_latency_validated"]
                    or np.any(np.asarray(result["transferred_bytes"]) > table.budgets * 30 * (1 + 1e-8))
                    or np.max(result["batch_replica_seconds"]) > destination_gpus(candidate) * 30 * (1 + 1e-8)
                    or not np.allclose(np.asarray(result["resident_debt_generated_work_s"]) - result["resident_debt_recovered_work_s"],
                                       result["pending_resident_debt_work_s"], rtol=1e-8, atol=1e-6)):
                raise ValueError("comparison violated conservation, capacity or scope")
            results[policy] = {k: result[k] for k in FIELDS}
            print(name, policy, round(result["shed_fraction"], 6), "recovered", result["service_recovered_by_deadline"], flush=True)
        report["comparisons"].append({"name": name, "fleet": fleet_summary(candidate, calibration),
            "serving_ceiling": ceiling, "results": results})
    report["comparison_scope"] = "Fourteen policy evaluations, 30s, shared 1000Gbps, 0.80GB/32K effective wire without another private-KV discount. All paired policies share candidates and a fixed planning clock; the long-context scenario uses its own library. Source population stays fixed when destinations halve; destination compute, memory and endpoint inventories shrink together. Raw handoff fraction does not establish recovered service or resident SLOs."
    report["required_before_campaign"] = [
        "Match resident RPS, context, prompt/output mix and scheduler configuration to a measured SLO recipe; no equivalence between the existing 50% labels.",
        "Validate shared replay and resident latency with continuing arrivals; the throughput-loss proxy and aggregate recovery constraints do not predict TTFT/TPOT.",
        "Validate current-stack full-message catch-up at retained contexts and append sizes, preserving native cache-hit and exact token-timing evidence.",
        "Arrival phases are synthetic and request durations remain throughput proxies; per-GPU placement, kernel contention, preemption and warm-prefix residency are unvalidated."]
    report["primary_references"] = ["https://docs.vllm.ai/en/stable/configuration/optimization/",
        "https://docs.vllm.ai/en/v0.10.1/features/automatic_prefix_caching.html",
        "https://www.usenix.org/conference/osdi24/presentation/agrawal"]
    if any(hashlib.sha256((q.ROOT / path).read_bytes()).hexdigest() != checksum for path, checksum in sources.items()):
        raise ValueError("audit sources changed during execution")
    q.write_json(OUT, report)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
