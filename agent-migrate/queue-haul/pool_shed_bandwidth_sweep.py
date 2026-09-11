"""Bounded bandwidth comparison with a separate, conditional request-queue audit."""

import argparse
import gzip
import hashlib
import json
import platform
import time
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np

import pool_shed_campaign as q
from pool_shed_execution import destination_gpus, kv_transfer_bytes
from pool_shed_network import transport_workers
from pool_shed_ttft import check_requests, resident_templates

OUT = q.ROOT / "outputs/a100-bandwidth-ttft-20260911"
BANDWIDTHS = (.1, .4, 1., 2., 5., 10., 25.6, 100.)
DEADLINES = (30, 60, 120)
SEEDS = (7101, 7102)
BASELINE, FOLLOWUP, LOAD = 60., 120., .5
SCOPE = {
    "readiness": "Conditional CPU sensitivity; no fleet SLO certificate or globally optimal execution frontier.",
    "sites": "6666 A100s (1.9998 MW nameplate) at source and EACH of two destinations; eight GPUs per host.",
    "network": "Shared source-site migration allocation across both routes. Each route and destination ingress can carry the whole allocation; no additional rack bottleneck. Fixed nonborrowable 80 Gb/s host reservations: 1.25 Gb/s per source history, 10 Gb/s per destination GPU. Measured per-wave endpoint goodput remains a cap. Pooled per-GPU transport-worker equivalents are unmeasured parallel scaling, not measured local worker concurrency. 80 Gb/s is a scenario assumption, not the Azure A100 SKU's advertised IP rate. Other IP traffic is separately provisioned.",
    "replay": "Full context on initial replay and on any changed/reset catch-up. No shared/private cache discount. Request audit uses one output token for first-token readiness; 512-output acquisition probes are not this contract. KV initial import has no GPU request; catch-up validates the uncached sealed-block tail and one token. The request checker adds no synthetic delay or initial debt; admission retains the existing fluid local-recovery criterion, which is not a TTFT constraint.",
    "queue_audit": "Frozen per-wave network releases and source pause/context snapshots; request scheduling is checked after admission, not optimized jointly. If initial replay exceeds planned pause, catch-up GPU work exceeds switch entry, or delivery exceeds handoff, downstream frozen timing is inconsistent. Passing-wave power is descriptive: removing failed waves would require rescheduling the whole plan. Two equally weighted resident calendars per replica are a sensitivity, not a fleet-state distribution. Frozen German prefill/endpoint coefficients plus East A100 server decode cadence transfer to both destinations.",
    "service": "Original offered arrivals retain pause and predecessor waiting. Eight prewarmed resident histories, 60 s startup baseline, block16 self-prefix reuse without eviction; recorded resets/wraps cold. Report deadline-wide offers and each trigger-aligned 30 s window through the deadline, observing for another 120 s. Primary resident attainment is the worst such window, avoiding dilution by a longer deadline. Uncommitted waves have no handoff before the deadline; later network progress is not predicted. Audit statistics are conditional on this frozen continuation.",
    "targets": "Diagnostic target: at least 90% of output requests TTFT <=1 s and at least 90% of multi-output requests mean TPOT <=0.1 s. Missing tokens/completions count conservatively; migration commands are excluded from resident/incoming service targets.",
    "power": "Linear allocation of measured source active-minus-awake-idle power by handoff work fraction, not nameplate shed or a nonlinear residual-load prediction.",
}


def fleet_for(workload, measured):
    fleet = q.replica_fleet(q.sample_fleet(workload, gpus=6666, gpus_per_node=8))
    fleet = replace(fleet, metadata={**fleet.metadata, "planning_reference_s": 4.,
        "require_local_recovery": True, "record_wave_schedules": True,
        "fixed_host_shares": True, "host_migration_gbps": 80.,
        "replay_cached_tokens": [0.] * len(fleet.count), "kv_shared_tokens": [0.] * len(fleet.count),
        "kv_wire_scale": 800_000_000 / (32768 * 49152)})
    return replace(fleet, kv=kv_transfer_bytes(fleet, fleet.context, measured))


def wave_requests(fleet, wave, until, block):
    """Commands queue at trigger; release times preserve network and source waits."""
    phases, rows, gates = wave["phase_enter_s"], [], {}
    counts = np.asarray(wave["counts"])
    if np.any(counts < 0) or np.any(counts != counts.astype(int)) or len(counts) != len(fleet.count):
        raise ValueError("wave requires aligned integer history counts")
    for i in np.flatnonzero(counts):
        for copy in range(int(counts[i])):
            history = f"incoming-{i}-{copy}"
            def command(stage, context, cached, release, limit, server_limit=None):
                if context <= 0:
                    return
                request_id = f"{history}-{stage}"
                rows.append(dict(request_id=request_id, gpu="destination", history=history,
                    cohort="migration", stage=stage, arrival_s=BASELINE, release_s=BASELINE + release,
                    prompt_tokens=int(context), cached_tokens=int(cached), output_tokens=1))
                if limit is not None:
                    gates[request_id] = {"end_s": BASELINE + limit}
                    if server_limit is not None:
                        gates[request_id]["server_end_s"] = BASELINE + server_limit
            if wave["action"] == "replay" and phases[1] is not None:
                command("initial", wave["origin_context"][i], 0, phases[1], wave["pause_requested_s"])
            if phases[4] is not None:
                context = int(wave["quiesced_context"][i])
                if wave["action"] == "kv_transfer":
                    cached = min(context // block * block, (context - 1) // 16 * 16)
                    command("catchup", context, cached, phases[4], phases[6], phases[5])
                elif wave["reset"][i] or context != wave["origin_context"][i]:
                    command("catchup", context, 0, phases[4], phases[6], phases[5])
            if wave["paused_turns"] is None:
                continue
            sequence = fleet.metadata["turn_sequences"][i]
            offset, n = fleet.metadata["turn_offset"][i], int(wave["paused_turns"][i])
            retained = int(wave["quiesced_context"][i])
            cadence, phase = fleet.metadata["source_session_rps"], fleet.metadata["source_phase_s"][i]
            while fleet.metadata["sequence_cycle"] or n < len(sequence):
                arrival = BASELINE + n / cadence - phase
                if arrival >= until:
                    break
                if arrival < BASELINE:
                    raise ValueError("unserved source turn precedes migration trigger")
                index = (offset + n) % len(sequence)
                turn = sequence[index]
                prompt = int(turn["context"] + turn["prompt"])
                reset = index == 0 or turn["reset"]
                prefix = 0 if reset else retained
                if prefix > prompt:
                    raise ValueError("incoming retained context exceeds next prompt without reset")
                rows.append(dict(request_id=f"{history}-turn-{n}", gpu="destination", history=history,
                    cohort="incoming", arrival_s=float(arrival),
                    release_s=max(float(arrival), BASELINE + phases[6] if phases[6] is not None else until + 1),
                    prompt_tokens=prompt, cached_tokens=int(min(prefix, prompt - 1) // 16 * 16),
                    output_tokens=int(turn["output"]), source_cohort=int(i), turn=n, recorded_turn=index))
                retained, n = prompt + int(turn["output"]), n + 1
    return rows, gates


def target_pass(stats):
    ttft = stats["ttft_violation_fraction_upper"]
    multi = stats["multi_output_requests"]
    return ((ttft is None or ttft <= .1 + 1e-12)
            and (not multi or (stats["known_tpot_violations"] + stats["tpot_right_censored"]) / multi <= .1 + 1e-12))


def observation_windows(deadline):
    until = BASELINE + deadline + FOLLOWUP
    return {"baseline": (0., BASELINE, until), "deadline": (BASELINE, BASELINE + deadline, until),
            **{f"burst-{start:g}": (BASELINE + start, BASELINE + min(start + 30, deadline), until)
               for start in range(0, deadline, 30)}}


def audit_plan(fleet, result, deadline, measured, templates, controls, cache):
    until = BASELINE + deadline + FOLLOWUP
    windows = observation_windows(deadline)
    coefficients = fleet.metadata["source_timing_coefficients"]
    weighted, audits = {}, []
    consistent, passed = 0., 0.
    def account(summary, weight):
        for name, window in summary["windows"].items():
            if name == "baseline":
                continue
            for cohort, stats in window["cohorts"].items():
                if cohort in ("resident", "incoming"):
                    accumulate(stats, weight, weighted.setdefault(name, {}).setdefault(cohort, {}))
    def accumulate(stats, weight, dest):
        for key in ("output_requests", "known_ttft_violations", "ttft_unresolved", "multi_output_requests",
                    "known_tpot_violations", "tpot_right_censored", "arrivals", "unfinished"):
            dest[key] = dest.get(key, 0.) + weight * stats[key]
    reserved = sum(w["mass"] for w in result["wave_schedules"])
    unused = 2 * destination_gpus(fleet) - reserved
    if unused < -1e-6:
        raise ValueError("destination replica inventory exceeded")
    for control in controls:
        account(control, max(unused, 0) / len(SEEDS))
    for wave in result["wave_schedules"]:
        extra, gates = wave_requests(fleet, wave, until, measured["kv_block_tokens"])
        key = q.digest(dict(requests=extra, gates=gates, deadline=deadline,
                            templates=[t["metadata"]["requests_sha256"] for t in templates]))
        if key not in cache:
            summaries = []
            for template in templates:
                checked = check_requests([*template["requests"], *extra], coefficients, windows, until)
                outcomes = {r["request_id"]: r for r in checked["requests"]}
                late = {f"{rid}/{metric}": None if outcomes[rid][metric] is None else outcomes[rid][metric] - limit
                        for rid, limits in gates.items() for metric, limit in limits.items()
                        if outcomes[rid][metric] is None or outcomes[rid][metric] > limit + 1e-8}
                summaries.append(dict(windows=checked["windows"], timing_overruns_s=late,
                    requests_sha256=checked["metadata"]["requests_sha256"]))
            cache[key] = summaries
        summaries = cache[key]
        for summary in summaries:
            account(summary, wave["mass"] / len(SEEDS))
        on_time = wave["state"] == 6 and all(not s["timing_overruns_s"] for s in summaries)
        local_pass = on_time and all(target_pass(stats) for s in summaries
            for name, window in s["windows"].items() if name.startswith("burst-")
            for cohort, stats in window["cohorts"].items() if cohort in ("resident", "incoming"))
        gain = float(wave["mass"] * (np.asarray(wave["counts"]) @ fleet.gain))
        consistent += gain * on_time
        passed += gain * local_pass
        audits.append(dict(wave_id=wave["wave_id"], gain= gain, timing_consistent=on_time,
                           local_target_pass=local_pass, audit_key=key))
    for window in weighted.values():
        for stats in window.values():
            count = stats["output_requests"]
            stats["ttft_attainment_lower"] = 1 - (stats["known_ttft_violations"] + stats["ttft_unresolved"]) / count if count else None
            stats["ttft_attainment_upper"] = 1 - stats["known_ttft_violations"] / count if count else None
    return dict(timing_consistent_handoff_fraction=consistent, locally_passing_handoff_fraction=passed,
        service=weighted["deadline"], service_windows=weighted, waves=audits, controls=controls,
        resident_templates=[t["metadata"] for t in templates],
        unique_queue_checks=len({a["audit_key"] for a in audits}) * len(SEEDS),
        checks={key: cache[key] for key in {a["audit_key"] for a in audits}},
        resident_latency_validated=False)


def validate_result(fleet, result, deadline, budgets, policy):
    if (not 0 <= result["shed_fraction"] <= result["admitted_shed_fraction"] + 1e-8 <= 1 + 2e-8
            or result["max_relative_residual"] > 1e-8 or result["last_completion_s"] > deadline + 1e-8):
        raise ValueError("handoff, admission or planner certificate failed")
    fractions = np.asarray(result["action_fractions"])
    if not np.isclose(fractions.sum(), result["shed_fraction"]):
        raise ValueError("action work fractions do not conserve handoff")
    if policy == "kv_only" and np.any(fractions[::2]) or policy == "replay_only" and np.any(fractions[1::2]):
        raise ValueError("pure-policy action contamination")
    if np.any(np.asarray(result["transferred_bytes"]) > budgets * deadline * (1 + 1e-8)):
        raise ValueError("shared network budget exceeded")
    if not np.allclose(np.asarray(result["resident_debt_generated_work_s"]) - result["resident_debt_recovered_work_s"],
                       result["pending_resident_debt_work_s"], rtol=1e-8, atol=1e-6):
        raise ValueError("resident work is not conserved")
    if np.any(np.asarray(result["batch_replica_seconds"]) > destination_gpus(fleet) * deadline * (1 + 1e-8)):
        raise ValueError("destination compute capacity exceeded")
    used = np.zeros(len(fleet.count))
    completed = 0.
    for wave in result["wave_schedules"]:
        used += wave["mass"] * np.asarray(wave["counts"])
        if wave["state"] == 6:
            completed += float(wave["mass"] * (np.asarray(wave["counts"]) @ fleet.gain))
    if np.any(used > fleet.count + 1e-6) or not np.isclose(completed, result["shed_fraction"], atol=1e-8):
        raise ValueError("wave history conservation failed")


def run(out, workloads, bandwidths, deadlines, policies):
    out.mkdir(parents=True, exist_ok=True)
    measured = q.calibration(0)
    fleets = {workload: fleet_for(workload, measured) for workload in workloads}
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in q.ROOT.glob("pool_shed*.py")}
    config = dict(workloads=workloads, bandwidths_tbps=bandwidths, deadlines_s=deadlines, policies=policies,
        scope=SCOPE, source_sha256=sources, measurement_sha256=q.provenance(measured),
        environment=dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__),
        fleet_contract_sha256={name: q.digest(dict(metadata=f.metadata, count=f.count.tolist(), context=f.context.tolist()))
                               for name, f in fleets.items()})
    identity = q.digest(config)
    if (out / "config.json").exists() and json.loads((out / "config.json").read_text())["identity"] != identity:
        raise ValueError("output configuration changed; choose a fresh output directory")
    q.write_json(out / "config.json", {**config, "identity": identity})
    endpoint = np.r_[np.median(q.network_samples()[:, :2], axis=0), 0.]
    endpoint[2] = endpoint[:2].sum()
    rows = []
    for workload in workloads:
        fleet = fleets[workload]
        r, k = q.include_isolated(*q.library(fleet), q.isolated_methods(fleet, LOAD, endpoint, np.full(3, 1e12 / 8), measured["timing"][0]))
        candidates = q.digest(dict(replay=r.tolist(), kv=k.tolist()))
        for deadline in deadlines:
            until = BASELINE + deadline + FOLLOWUP
            templates = [resident_templates(fleet, LOAD, seed, until, BASELINE) for seed in SEEDS]
            windows = observation_windows(deadline)
            controls = [check_requests(t["requests"], fleet.metadata["source_timing_coefficients"], windows, until) for t in templates]
            controls = [{k: v for k, v in c.items() if k != "requests"} for c in controls]
            cache = {}
            for bw, policy in product(bandwidths, policies):
                name = f"{workload}-d{deadline}-b{bw:g}-{policy}"
                path = out / f"{name}.json.gz"
                if path.exists():
                    with gzip.open(path, "rt") as handle:
                        record = json.load(handle)
                    if record["identity"] != identity or record["candidate_sha256"] != candidates:
                        raise ValueError("checkpoint inputs changed")
                else:
                    started = time.perf_counter()
                    budgets = np.full(3, bw * 1e12 / 8)
                    table = q.schedule_table(fleet, r, k, LOAD, deadline, endpoint, budgets, measured["timing"][0])
                    result = q.execute_feedback(table, table, policy, measured["timing"][0], measured)
                    validate_result(fleet, result, deadline, table.budgets, policy)
                    simulate_s = time.perf_counter() - started
                    audit = audit_plan(fleet, result, deadline, measured, templates, controls, cache)
                    record = dict(identity=identity, workload=workload, bandwidth_tbps=bw, deadline_s=deadline,
                        policy=policy, candidate_sha256=candidates, result=result, audit=audit,
                        simulate_s=simulate_s, audit_s=time.perf_counter() - started - simulate_s,
                        requested_budgets_tbps=(budgets * 8e-12).tolist(),
                        effective_budgets_tbps=(table.budgets * 8e-12).tolist(),
                        transport_worker_equivalents=transport_workers(fleet).tolist(),
                        source_power=fleet.metadata["source_power"], fleet_metadata=fleet.metadata)
                    q.write_json(path, record)
                result, audit = record["result"], record["audit"]
                action = np.asarray(result["action_counts"])
                moved = action.sum()
                power = record["source_power"]["delta_w"] * fleet.gpus / 1e6
                row = dict(workload=workload, deadline_s=deadline, bandwidth_tbps=bw, policy=policy,
                    handoff_fraction=result["shed_fraction"], recovered_handoff_fraction=result["recovered_handoff_fraction"],
                    power_shed_mw=power * result["shed_fraction"], full_handoff_power_mw=power,
                    kv_source_session_fraction=float(action[1::2].sum() / fleet.count.sum()),
                    replay_source_session_fraction=float(action[::2].sum() / fleet.count.sum()),
                    kv_share_of_moved=float(action[1::2].sum() / moved) if moved else None,
                    timing_consistent_handoff_fraction=audit["timing_consistent_handoff_fraction"],
                    locally_passing_handoff_fraction=audit["locally_passing_handoff_fraction"],
                    simulate_s=record["simulate_s"], audit_s=record["audit_s"],
                    **{f"{c}_ttft_attainment": min((window[c]["ttft_attainment_lower"]
                        for name, window in audit["service_windows"].items()
                        if name.startswith("burst-") and c in window and window[c]["ttft_attainment_lower"] is not None), default=None)
                       for c in audit["service"]})
                rows.append(row)
                q.write_csv(out / "summary.csv", rows)
                print(json.dumps(row, sort_keys=True), flush=True)
    q.write_json(out / "summary.json", dict(identity=identity, scope=SCOPE, rows=rows, resident_latency_validated=False))
    plot(rows, out)


def plot(rows, out):
    import matplotlib.pyplot as plt
    import plot_style as style

    style.apply()
    workloads = list(dict.fromkeys(r["workload"] for r in rows))
    deadlines = sorted({r["deadline_s"] for r in rows})
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    for metric, label, limit in (("handoff_fraction", "Frozen model: handoff work by deadline (%)", (0, 102)),
        ("power_shed_mw", "Source power shed proxy (MW)", None),
        ("kv_share_of_moved", "KV share of completed sessions (%)", (0, 102)),
        ("resident_ttft_attainment", "Worst 30 s resident TTFT attainment (%)", (0, 102)),
        ("timing_consistent_handoff_fraction", "Request timing compatible subset (%)", (0, 102))):
        fig, axes = plt.subplots(len(workloads), len(deadlines), squeeze=False,
                                 figsize=(5 * len(deadlines), 3.8 * len(workloads)), sharex=True, sharey=True)
        for i, workload in enumerate(workloads):
            for j, deadline in enumerate(deadlines):
                ax = axes[i, j]
                for policy in policies:
                    selected = sorted((r for r in rows if (r["workload"], r["deadline_s"], r["policy"]) == (workload, deadline, policy)), key=lambda r: r["bandwidth_tbps"])
                    ax.plot([r["bandwidth_tbps"] for r in selected],
                        [np.nan if r.get(metric) is None else r[metric] * (1 if metric == "power_shed_mw" else 100) for r in selected],
                        marker=style.POLICY_MARKERS[policy], markersize=5, **style.policy_style(policy, style.PAPER_POLICY_NAMES))
                ax.set(xscale="log", title=f"{workload} · {deadline} s", xlabel="Shared site migration allocation (Tb/s)")
                if j == 0:
                    ax.set_ylabel(label)
                if limit:
                    ax.set_ylim(*limit)
                ax.grid(True, alpha=.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(policies), frameon=False)
        fig.suptitle("2 MW source / 2 MW per destination · conditional simulation", fontsize=15)
        fig.tight_layout(rect=(0, .08, 1, .95))
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"{metric}.{suffix}")
        plt.close(fig)
    for deadline in deadlines:
        fig, axes = plt.subplots(len(workloads), len(policies), squeeze=False,
                                 figsize=(3.4 * len(policies), 3.4 * len(workloads)), sharey=True)
        for i, workload in enumerate(workloads):
            for j, policy in enumerate(policies):
                ax = axes[i, j]
                selected = sorted((r for r in rows if (r["workload"], r["deadline_s"], r["policy"]) == (workload, deadline, policy)), key=lambda r: r["bandwidth_tbps"])
                bottom = np.zeros(len(selected))
                for action, field in (("kv_transfer", "kv_source_session_fraction"), ("replay", "replay_source_session_fraction"), ("not_moved", None)):
                    values = np.array([r[field] * 100 for r in selected]) if field else 100 - bottom
                    ax.bar(range(len(selected)), values, bottom=bottom, color=style.ACTION_COLORS[action], label=style.ACTION_NAMES[action])
                    bottom += values
                ax.set(ylim=(0, 100), title=f"{workload}\n{style.PAPER_POLICY_NAMES[policy]}", xlabel="Site allocation (Tb/s)")
                ax.set_xticks(range(len(selected)), [f'{r["bandwidth_tbps"]:g}' for r in selected], rotation=45)
                if j == 0:
                    ax.set_ylabel("Source sessions (%)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.suptitle(f"{deadline} s · completed action mix in the frozen fleet model", fontsize=15)
        fig.tight_layout(rect=(0, .08, 1, .94))
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"action_mix_d{deadline}.{suffix}")
        plt.close(fig)
    bandwidths = sorted({r["bandwidth_tbps"] for r in rows})
    shown = sorted({bandwidths[0], bandwidths[-1], *({1., 10.} & set(bandwidths))})
    fig, axes = plt.subplots(len(workloads), len(shown), squeeze=False,
                             figsize=(4.2 * len(shown), 3.8 * len(workloads)), sharex=True, sharey=True)
    for i, workload in enumerate(workloads):
        for j, bw in enumerate(shown):
            ax = axes[i, j]
            for policy in policies:
                selected = sorted((r for r in rows if (r["workload"], r["bandwidth_tbps"], r["policy"]) == (workload, bw, policy)), key=lambda r: r["deadline_s"])
                ax.plot([r["deadline_s"] for r in selected], [r["power_shed_mw"] for r in selected],
                        marker=style.POLICY_MARKERS[policy], **style.policy_style(policy, style.PAPER_POLICY_NAMES))
            ax.set(title=f"{workload} · {bw:g} Tb/s", xlabel="Deadline (s)", ylim=(0, .47))
            if j == 0:
                ax.set_ylabel("Source power shed proxy (MW)")
            ax.grid(True, alpha=.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(policies), frameon=False)
    fig.suptitle("Power–deadline frontier estimate · frozen fleet model, conditional TTFT audit", fontsize=15)
    fig.tight_layout(rect=(0, .08, 1, .94))
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"power_deadline_frontier.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--workloads", nargs="+", choices=("coding", "coding_long"), default=["coding", "coding_long"])
    parser.add_argument("--bandwidths", nargs="+", type=float, default=list(BANDWIDTHS))
    parser.add_argument("--deadlines", nargs="+", type=int, default=list(DEADLINES))
    parser.add_argument("--policies", nargs="+", choices=q.POLICIES, default=list(q.POLICIES))
    args = parser.parse_args()
    if any(not np.isfinite(b) or b <= 0 for b in args.bandwidths) or any(d <= 0 for d in args.deadlines):
        parser.error("positive finite bandwidths and deadlines required")
    run(args.out, args.workloads, args.bandwidths, args.deadlines, args.policies)
