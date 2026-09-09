"""Bounded replay/resident audit using frozen measurements; no campaign or refit."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import pool_shed_campaign as q
from loaded_service_model import historical_execution_check
from pool_shed_cache_sensitivity import build, fleet_for
from pool_shed_calibration import loaded_execution_check, resident_execution_check
from pool_shed_execution import catchup, kv_transfer_bytes, regional_execution_check

OUT = q.ROOT / "outputs/a100-replay-audit/audit.json"
FIELDS = ("shed_fraction", "admitted_shed_fraction", "last_completion_s", "action_fractions",
          "migration_idle_work_s", "batch_replica_seconds", "transferred_bytes", "buffered_requests",
          "pending_buffered_work_s", "resident_debt_generated_work_s", "final_destination_load",
          "max_relative_residual", "planning_s")


def fleet_summary(fleet, calibration):
    metadata = fleet.metadata
    shapes = np.array([[np.mean([r[k] for r in rows]) for k in ("prompt", "output")]
                       for rows in metadata["turn_sequences"]])
    means = fleet.count @ shapes / fleet.count.sum()
    first_durations = [rows[offset] for rows, offset in zip(metadata["turn_duration_s"], metadata["turn_offset"])]
    return {"gpus_per_site": fleet.gpus, "source_sessions": int(fleet.count.sum()),
            "mean_initial_context_tokens": float(fleet.count @ fleet.context / fleet.count.sum()),
            "initial_context_range": [float(fleet.context.min()), float(fleet.context.max())],
            "cycle_mean_prompt_output_tokens": means.tolist(),
            "source_rps_per_gpu": float(fleet.count.sum() * metadata["source_session_rps"] / fleet.gpus),
            "resident_rps_per_gpu_at_load_half": .5 * metadata["reference_rps"],
            "source_period_s": 1 / metadata["source_session_rps"],
            "first_request_duration_proxy_range_s": [min(first_durations), max(first_durations)],
            "timing_load_factor": metadata["timing_load_factor"],
            "timing_rho_at_load_half": .5 * metadata["timing_load_factor"],
            "replay_slowdown_at_load_half": float(np.exp(.5 * metadata["timing_load_factor"] * calibration["timing"][0]["beta"])),
            "initial_kv_wire_bytes": float(fleet.count @ fleet.kv),
            "initial_replay_log_bytes": float(fleet.count @ fleet.log),
            "reserved_resident_kv_capacity_fraction": fleet.baseline_kv / fleet.kv_capacity,
            "service_context_extrapolated": metadata["service_context_extrapolated"],
            "excluded_states": metadata["excluded_states"],
            "full_handoff_source_power_mw": fleet.gpus * metadata["source_power"]["delta_w"] / 1e6}


def main():
    calibration = q.calibration(0)
    timing = calibration["timing"][0]
    sources = {**q.provenance(calibration), Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report = {"scope": "Central, full-context replay audit; no new measurements, refits, synthetic cache hits, or campaign restart. Ablations test assumptions, not alternative hardware calibrations.",
              "sources": sources, "solver_version": q.highspy.Highs().version(),
              "resident_latency_validated": False, "campaign_ready": False,
              "fleets": {w: fleet_summary(q.sample_fleet(w), calibration) for w in q.WORKLOADS},
              "replay_calibration": {"contexts": calibration["replay_context_tokens"],
                  "tps": calibration["replay_tps"], "completion_s": calibration["replay_completion_s"],
                  "timing": timing}, "preserved_controls": []}
    for index in (0, 3, 18, 21):
        path = q.ROOT / f"outputs/a100-cache-sensitivity/case-{index:02d}.json"
        value = json.loads(path.read_text())
        if any(value["sources"].get(k) != v for k, v in q.provenance(calibration).items()):
            raise ValueError("preserved control does not match current simulation sources")
        sources[str(path.relative_to(q.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        result = value["results"]["replay_only"]
        report["preserved_controls"].append({"case": value["case"],
            "results": {p: {k: r[k] for k in FIELDS} for p, r in value["results"].items()},
            "next_source_arrival_s": 1 / value["metadata"]["source_session_rps"],
            "replay_finishes_before_next_arrival": result["last_completion_s"] < 1 / value["metadata"]["source_session_rps"]})
    fleet = fleet_for(q.GPUS, "reported800MB", calibration)
    original = q.sample_fleet("coding")
    np.testing.assert_array_equal(fleet.t1, original.t1)
    np.testing.assert_array_equal(fleet.demand, original.demand)
    np.testing.assert_array_equal(fleet.memory_tokens, original.memory_tokens)
    if np.any(fleet.metadata["replay_cached_tokens"]) or np.any(fleet.metadata["kv_shared_tokens"]):
        raise ValueError("audit requires full-context replay and no extra private-KV discount")
    report["resource_arithmetic"] = {
        "installed_source_mw": fleet.gpus * 300 / 1e6,
        "installed_destination_mw_total": 2 * fleet.gpus * 300 / 1e6,
        "initial_spare_destination_gpu_equivalents": float(2 * fleet.gpus * .5),
        "imported_source_service_gpu_equivalents": float(fleet.gpus * q.SOURCE_LOAD),
        "standing_service_ceiling_by_destination_load": {str(u): min(1., 2 * (1 - u) / q.SOURCE_LOAD) for u in q.LOADS},
        "reported_anchor_initial_kv_bytes": float(fleet.count @ fleet.kv),
        "initial_snapshot_only_kv_wan_lower_bound_s_at_1000gbps": float(fleet.count @ fleet.kv / 125e9),
        "initial_snapshot_only_replay_wan_lower_bound_s_at_1000gbps": float(fleet.count @ fleet.log / 125e9),
        "scope": "Initial snapshot byte lower bounds exclude growth, resets, endpoint limits, compute and policy selection; not completion-time predictions. Fixed sessions/GPU and proportional spare compute preserve replay work/capacity under fleet scaling. Fixed aggregate WAN does not scale with KV population."}
    report["resident_contracts"] = {
        "coding_fleet_at_half": report["fleets"]["coding"]["resident_rps_per_gpu_at_load_half"],
        "loaded_2048_32_reference_at_half_rps": .5 * calibration["reference_rps"],
        "regional_604_64_reference_at_half_rps": .5 / calibration["resident_interference"]["reference_work_s"],
        "loaded_reference_replay_slowdown_at_rho_half": float(np.exp(.5 * timing["beta"])),
        "regional_observed_replay_resident_loss": calibration["resident_interference"]["validation"]["fixed_replay"]["median_observed_loss"],
        "regional_observed_kv_resident_loss": calibration["resident_interference"]["validation"]["fixed_kv_transfer"]["median_observed_loss"],
        "service_bound": calibration["resident_service"]["bound"],
        "resident_slo_targets": calibration["resident_service"]["targets"],
        "full_service_profile_accepted": calibration["resident_service"]["full_profile_accepted"],
        "scope": "Different request shapes and normalization; RPS ratios are not occupancy ratios. Protected execution enforces an aggregate work budget and generates zero resident debt by construction. Historical resident-loss reproduction uses unprotected execution, so it does not validate protected TTFT/TPOT, migration time slicing, placement, or continuing-arrival recovery."}
    counts = np.eye(len(fleet.count))[0]
    probes = []
    for origin in (2048., 30000.):
        context = np.full(len(counts), origin)
        final = context + 256
        network, work = catchup(fleet, counts, 0, 0, final, np.zeros(len(counts), bool), timing, calibration, origin_context=context)
        probes.append({"origin_context": origin, "appended_tokens": 256, "wire_bytes": network, "idle_work_s": work})
    report["catchup"] = {"probes": probes,
        "singleton_completion_overhead_s": calibration["replay_completion_s"],
        "scope": "Current incremental replay uses appended-token count for rate and packing, independent of retained prefix length. Below the smallest measured context it interpolates from zero, scaling down completion overhead too. No measured context-by-append replay calibration validates this rule. Initial full-context replay is unchanged. Hardware prepare submits full messages on catch-up; simulator delta logs additionally assume an incremental log transport."}
    report["timing_checks"] = {"loaded_replay": loaded_execution_check(calibration),
                               "regional": regional_execution_check(calibration),
                               "historical": historical_execution_check(calibration)}
    report["resident_execution_check"] = resident_execution_check(calibration, report["timing_checks"]["regional"])
    if not all(r["gate_pass"] for r in report["timing_checks"].values()):
        raise ValueError("existing hardware timing reproduction failed")
    report["limitations"] = [
        "Two equally large destination sites and freely pooled fractional replicas allow work to spread over the whole fleet; no per-GPU resident placement, HBM bandwidth, kernel scheduling, preemption, or tail latency is modeled.",
        "Measured batch elapsed time is used as divisible replica work; preservation under partial compute shares has not been measured. Load conversion also assumes one scalar captures a different prompt/decode mixture.",
        "Source durations use phase-throughput work proxies, not measured per-request latencies. Equal cadence starts all cohorts together; no source request queue is executed. First request shapes are often much cheaper than their cycle average.",
        "Source traces cycle and reset, so contexts cannot grow indefinitely. Trajectories beyond the approximately 32K replay support are excluded. Forecasts know future shapes and resets.",
        "Packing is calibrated at width eight; intermediate widths and mixed execution are transfers. A single context over 16384 serializes a whole batch, producing a model discontinuity.",
        "Timing uses a measured regional factor near 0.78. It is not a cache-hit fraction or permission for an additional replay discount. Native prefix hits are not fully observable.",
        "KV ingest is omitted, which favors KV. Peak-cycle memory reservations are conservative. Neither explains a replay advantage.",
        "Shed is workload-weighted active-minus-awake-idle power, not 20 MW of removable power or a GPU shutdown schedule.",
        "The planner is a fractional, finite-library temporal LP with iterative nonlinear load forecasts and no runtime charge. Static LP containment cannot guarantee executed QH beats replay."]
    base = build(3, calibration)
    variants = [
        ("effective_wire_anchor", base.fleet, .5, timing),
        ("offered_reference_factor_one", replace(base.fleet, metadata={**base.fleet.metadata, "timing_load_factor": 1.}), .5, timing),
        ("without_regional_speedup", base.fleet, .5, {**timing, "regional_replay_factor": [1., 1.]}),
        ("serial_all_replay_batches", replace(base.fleet, metadata={**base.fleet.metadata, "batch_context_limit": 0}), .5, timing),
        ("destination_load_075", base.fleet, .75, timing),
        ("coding_long", replace(q.sample_fleet("coding_long"), metadata={**q.sample_fleet("coding_long").metadata,
              "planning_reference_s": 4., "kv_wire_scale": base.fleet.metadata["kv_wire_scale"]}), .5, timing)]
    # The long-context scenario needs its own bytes and library; paired policies still share both.
    report["ablations"] = []
    for name, candidate, load, fitted in variants:
        if name == "coding_long":
            candidate = replace(candidate, kv=kv_transfer_bytes(candidate, candidate.context, calibration))
            replay, kv = q.library(candidate)
        else:
            replay, kv = base.replay[base.route == 0], base.kv[base.route == 0]
        table = q.schedule_table(candidate, replay, kv, load, 30., base.endpoint, base.budgets, fitted)
        results = {p: q.execute_feedback(table, table, p, fitted, calibration) for p in ("replay_only", "queue_haul")}
        for result in results.values():
            if (result["max_relative_residual"] > 1e-8 or not 0 <= result["shed_fraction"] <= 1 + 1e-8
                    or np.any(result["resident_debt_generated_work_s"]) or result["last_completion_s"] > 30 + 1e-8
                    or result["pending_backlog_reference_work_s"] > 1e-8
                    or np.any(np.asarray(result["transferred_bytes"]) > table.budgets * 30 * (1 + 1e-8))
                    or np.max(result["batch_replica_seconds"]) > candidate.gpus * (1 - load) * 30 * (1 + 1e-8)):
                raise ValueError("ablation violated resource, resident or deadline accounting")
        report["ablations"].append({"name": name, "results": {p: {k: r[k] for k in FIELDS} for p, r in results.items()}})
        print(name, {p: round(r["shed_fraction"], 6) for p, r in results.items()}, flush=True)
    report["ablation_scope"] = "Twelve policy executions, 20 MW/site, 30 s, 1000 Gbps, effective 0.80 GB/32K wire with no further discount. Identical candidate library and fixed planning clock for the first five scenarios; a separate long-context library is shared by its two policies. Source cadence, resident shape and capacity stay fixed in the timing-factor stresses. Setting the factor to one is not a hardware-load match. Load 0.75 and coding_long change scenario conditions. No asynchronous-arrival claim is made. No ablation establishes a deployment-ready replacement model."
    report["required_before_campaign"] = [
        "Specify available destination GPUs and actual resident request rates, shapes, contexts and arrival phases; do not equate the existing 50% labels.",
        "Check protected replay against measured concurrent resident TTFT/TPOT and throughput on that contract; aggregate zero debt is insufficient.",
        "Validate source quiescence and replay catch-up with ongoing arrivals, retained context and buffer recovery. Existing frozen-source timing holdouts do not cover this.",
        "Keep full-context replay and the effective wire anchor; do not add a private-KV or synthetic cache-hit discount without distinct evidence."]
    if any(hashlib.sha256((q.ROOT / path).read_bytes()).hexdigest() != checksum for path, checksum in sources.items()):
        raise ValueError("audit sources changed during execution")
    q.write_json(OUT, report)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
