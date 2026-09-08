"""Independent event execution of divisible migration batches and finite source traces."""

from __future__ import annotations

import numpy as np

DISPATCH_CHUNKS = 32


def flow_rates(mass, route, endpoint, budgets):
    """Weighted max-min per-batch rates under endpoint, route and shared caps."""
    return _flow_with_caps(mass, route, np.asarray(endpoint)[route], budgets)


def _flow_with_caps(mass, route, caps, budgets, kv=None, application=None, nodes=1):
    mass, route = np.asarray(mass), np.asarray(route)
    rates, active = np.zeros(len(mass)), np.ones(len(mass), bool)
    masks = np.array([route == 0, route == 1, np.ones(len(route), bool)])
    if application is not None:
        masks = np.vstack((masks, [(route == r) & kv for r in (0, 1)]))
        budgets = np.r_[budgets, np.asarray(application) * nodes]
    for _ in range(len(mass) + len(budgets)):
        if not active.any():
            return rates
        used = masks @ (mass * rates)
        weights = masks @ (mass * active)
        remaining = np.maximum(np.asarray(budgets) - used, 0)
        step = min(float(np.min(caps[active] - rates[active])),
                   float(np.min(np.divide(remaining, weights, out=np.full(len(budgets), np.inf), where=weights > 0))))
        rates[active] += max(step, 0)
        full = (weights > 0) & (remaining - step * weights <= np.maximum(budgets, 1) * 1e-12)
        active &= (rates < caps * (1 - 1e-12)) & ~np.any(masks[full], axis=0)
    raise RuntimeError("network sharing failed to exhaust a resource")


def _quiesce(fleet, counts, now):
    sequences = fleet.metadata.get("turn_sequences")
    if sequences is None:
        raise ValueError("pooled execution requires explicit finite source turn sequences")
    cadence = fleet.metadata.get("source_session_rps", 0.)
    if len(sequences) != len(fleet.count) or cadence < 0 or (not cadence and any(sequences)):
        raise ValueError("invalid source trace pacing")
    active = np.flatnonzero(counts)
    lengths = np.array([len(sequences[i]) for i in active])
    step = int(np.ceil(max(now * cadence - 1e-10, 0)))
    cycle = fleet.metadata.get("sequence_cycle", False)
    if cycle and np.any(lengths == 0):
        raise ValueError("cannot cycle an empty source trace")
    completed = np.full_like(lengths, step) if cycle else np.minimum(lengths, step)
    end = max(now, float(completed.max(initial=0) / cadence)) if cadence else now
    context, reset = fleet.context.copy(), np.zeros(len(fleet.count), bool)
    for i, n in zip(active, completed):
        if n:
            offset = fleet.metadata.get("turn_offset", [0] * len(sequences))[i] if cycle else 0
            row = sequences[i][(offset + n - 1) % len(sequences[i])]
            context[i] = row["context"] + row["prompt"] + row["output"]
            reset[i] = (cycle and offset + n > len(sequences[i])) or any(
                sequences[i][(offset + j) % len(sequences[i])].get("reset", False) for j in range(min(n, len(sequences[i]))))
    return end, context, reset, bool(not cycle and np.all(completed == lengths))


def _buffered(fleet, counts, start, end, calibration):
    cadence = fleet.metadata.get("source_session_rps", 0.)
    first, last = int(np.ceil(start * cadence - 1e-9)), int(np.ceil(end * cadence - 1e-9))
    number, work = 0., 0.
    for i in np.flatnonzero(counts):
        sequence = fleet.metadata["turn_sequences"][i]
        if not sequence:
            continue
        values = np.array([r["prompt"] / calibration["F"] + r["output"] / calibration["G"] for r in sequence])
        prefix = np.r_[0., np.cumsum(values)]
        if fleet.metadata.get("sequence_cycle"):
            offset = fleet.metadata.get("turn_offset", [0] * len(counts))[i]
            a, b = offset + first, offset + last
            work += counts[i] * ((b // len(values) - a // len(values)) * prefix[-1] + prefix[b % len(values)] - prefix[a % len(values)])
            number += counts[i] * (last - first)
        else:
            a, b = min(first, len(values)), min(last, len(values))
            work += counts[i] * (prefix[b] - prefix[a])
            number += counts[i] * (b - a)
    return number, work


def catchup(fleet, counts, action, route, context, reset, timing, calibration):
    """Primitive bytes and idle work for a newly captured source state."""
    if action == 1:
        block = calibration["kv_block_tokens"]
        sealed = np.floor(context / block) * block
        old = np.where(reset, 0, np.floor(fleet.context / block) * block)
        return (float(counts @ np.maximum(sealed - old, 0) / block * calibration["kv_block_bytes"]),
                float(counts @ (context - sealed) / calibration["kv_tail_replay_tps"] + np.interp(counts.sum(), [0, 1, 8],
                    [0, timing["kv_completion_s"], timing["kv_batch_completion_s"]])))
    if action != 0:
        raise ValueError("unknown catch-up action")
    changed = np.where(reset, context, np.maximum(context - fleet.context, 0))
    rebuilding = (counts > 0) & (changed > 0)
    idle = 0.
    if rebuilding.any():
        from pool_shed_calibration import replay_seconds
        support = np.asarray(calibration["replay_context_tokens"])
        if np.any(changed[rebuilding] > support.max()):
            raise ValueError("source replay catch-up exceeds measured context support")
        work = np.interp(changed[rebuilding], np.r_[0., support],
                         np.r_[0., np.maximum.accumulate(replay_seconds(support, calibration))])
        knots = fleet.metadata.get("packing_context_tokens")
        packing = np.interp(changed[rebuilding], knots, timing["packing_kappa"]) if knots else np.full(len(work), timing["kappa"])
        if np.any(changed[rebuilding] > fleet.metadata.get("batch_context_limit", np.inf)):
            packing[:] = 1.
        regional = timing.get("regional_replay_factor", calibration.get("regional_components", {}).get("replay_factor", [1., 1.]))
        idle = (counts[rebuilding] @ (packing * work) + np.max((1 - packing) * work)) * regional[route]
    return float(2 * (counts @ changed)), float(idle)


def execute_pooled(table, chosen, timing, calibration=None, chunks=1):
    """Time-share batch mass; imported serving starts at each committed handoff."""
    if calibration is None:
        from pool_shed_calibration import calibration as load_calibration
        calibration = load_calibration(0)
    fleet, chosen = table.fleet, np.asarray(chosen, float)
    if not isinstance(chunks, (int, np.integer)) or isinstance(chunks, bool) or chunks < 1:
        raise ValueError("dispatch chunks must be a positive integer")
    if (fleet.gpus <= 0 or np.any(~np.isin(table.route, [0, 1]))
            or not np.isfinite(np.r_[np.ravel(table.load), table.deadline, table.endpoint, table.budgets, timing["beta"]]).all()
            or np.any(np.asarray(table.load) < 0) or np.any(np.asarray(table.load) > 1)
            or np.any(np.asarray(table.endpoint) <= 0) or np.any(np.asarray(table.budgets) <= 0)
            or np.any(table.replay < 0) or np.any(table.kv < 0)
            or np.any(table.replay != np.floor(table.replay)) or np.any(table.kv != np.floor(table.kv))):
        raise ValueError("invalid independent execution primitives")
    if (chosen.shape != (len(table.route),) or not np.isfinite(chosen).all() or np.any(chosen < 0)
            or np.any((table.replay + table.kv).T @ chosen > fleet.count * (1 + 1e-8) + 1e-8)):
        raise ValueError("invalid or duplicated source migration mass")
    selected, mass = [], []
    for j in np.flatnonzero(chosen > 1e-10):
        for action, counts in enumerate((table.replay[j], table.kv[j])):
            if counts.any():
                selected.extend([(j, action, counts)] * chunks)
                mass.extend([chosen[j] / chunks] * chunks)
    mass = np.array(mass)
    route = np.array([table.route[j] for j, _, _ in selected], int)
    action = np.array([a for _, a, _ in selected], int)
    counts = np.array([c for _, _, c in selected]).reshape(-1, len(fleet.count))
    memory_tokens, free_memory = fleet.memory_tokens, fleet.kv_capacity - fleet.baseline_kv
    demand, memory = counts @ fleet.demand, counts @ memory_tokens
    initial_load = np.broadcast_to(np.asarray(table.load), (2,)).astype(float)
    for r in (0, 1):
        ids = route == r
        if (mass[ids] @ demand[ids] > fleet.gpus * (1 - initial_load[r]) + 1e-7 * fleet.gpus
                or mass[ids] @ memory[ids] > free_memory + 1e-7 * fleet.kv_capacity):
            raise ValueError("selected migrations exceed destination serving or memory capacity")
    n = len(selected)
    group, initial_ready = np.arange(n) // chunks, np.zeros(n, bool)
    state, remaining, release = np.zeros(n, int), np.zeros(n), np.zeros(n)
    tail, delta, compute_used, idle_work = np.zeros(n), np.zeros(n), np.zeros(2), np.zeros(2)
    quiesced, buffered, backlog, backlog_total = (np.zeros(n) for _ in range(4))
    resident_debt, resident_generated, resident_recovered = (np.zeros(2) for _ in range(3))
    resident_loss = np.broadcast_to(np.asarray(timing.get("resident_replay_loss", 0.)), (2,))
    if not np.isfinite(resident_loss).all() or np.any((resident_loss < 0) | (resident_loss > 1)):
        raise ValueError("resident replay throughput loss must be in [0, 1]")
    committed, network_used, loads = np.zeros((4, len(fleet.count))), np.zeros(3), initial_load.copy()
    reserved = mass * memory
    peak_memory = np.array([reserved[route == r].sum() for r in (0, 1)])
    peak_load, events, exhausted = loads.copy(), {}, 0
    tail_rate = calibration["kv_tail_replay_tps"]
    if tail_rate <= 0 or table.deadline <= 0:
        raise ValueError("positive tail service and deadline required")
    knots = fleet.metadata.get("packing_context_tokens")
    kappa = np.interp(fleet.context, knots, timing["packing_kappa"]) if knots else np.full(len(fleet.count), timing["kappa"])
    regional = timing.get("regional_replay_factor", calibration.get("regional_components", {}).get("replay_factor", [1., 1.]))
    for i, (_, a, c) in enumerate(selected):
        remaining[i] = c @ (fleet.log if a == 0 else fleet.kv)
        if a == 0:
            packing = np.ones_like(kappa) if np.any((c > 0) & (fleet.context > fleet.metadata.get("batch_context_limit", np.inf))) else kappa
            tail[i] = (c @ (packing * fleet.t1) + np.max(np.where(c > 0, (1 - packing) * fleet.t1, 0))) * regional[route[i]]
    now, iterations = 0., 0
    network_key, network_rates = None, np.zeros(n)
    while (np.any(state < 6) or np.any(backlog > 1e-9) or np.any(resident_debt > 1e-9)) and now <= table.deadline:
        iterations += 1
        if iterations > 20 * n + 20:
            raise RuntimeError("pooled phase execution failed to advance")
        ready = np.flatnonzero((state < 6) & (remaining <= 1e-9) & (release <= now + 1e-10))
        if len(ready):
            for i in ready:
                a, c, r = action[i], counts[i], route[i]
                if state[i] == 0 and a == 0:
                    state[i], remaining[i] = 1, tail[i]
                elif state[i] in (0, 1):
                    release[i], context, reset, terminal = _quiesce(fleet, c, now)
                    quiesced[i] = release[i]
                    exhausted += int(terminal)
                    required = mass[i] * (c @ np.maximum(memory_tokens, np.ceil(context / 16) * 16))
                    occupied = reserved[route == r].sum() - reserved[i] + required
                    if occupied > free_memory + 1e-7:
                        state[i], reserved[i] = 7, 0.
                        continue
                    reserved[i] = required
                    peak_memory[r] = max(peak_memory[r], occupied)
                    delta[i], tail[i] = catchup(fleet, c, a, r, context, reset, timing, calibration)
                    state[i], remaining[i] = 2, 0.
                elif state[i] == 2:
                    state[i], remaining[i] = 3, delta[i]
                elif state[i] == 3:
                    state[i], remaining[i] = 4, tail[i]
                elif state[i] == 4:
                    state[i], release[i] = 5, now + calibration.get("switch_s", 0.)
                else:
                    state[i] = 6
                    committed[2 * r + a] += mass[i] * c
                    loads[r] = min(1., loads[r] + mass[i] * demand[i] / fleet.gpus)
                    buffered[i], backlog[i] = _buffered(fleet, c, quiesced[i], now, calibration)
                    backlog_total[i] = backlog[i]
                    peak_load[r] = max(peak_load[r], loads[r])
                    event = events.setdefault((int(selected[i][0]), int(a)), {
                        "column": int(selected[i][0]), "action": "replay" if a == 0 else "kv_transfer", "route": int(r),
                        "multiplicity": 0., "first_completion_s": float(now)})
                    event.update(multiplicity=event["multiplicity"] + float(mass[i]), completion_s=float(now),
                                 resident_debt_work_s=float(resident_debt[r]), source_buffer_work_s=float(mass[route == r] @ backlog[route == r]))
            continue
        rates = np.zeros(n)
        transfers = np.flatnonzero((state == 0) | (state == 3))
        endpoint = np.asarray(table.endpoint)
        application = timing.get("regional_kv_bytes_per_s", calibration.get("regional_components", {}).get("endpoint_bytes_per_s", endpoint[:2]))
        # A KV application cap is additional to the measured bulk endpoint cap.
        key = (transfers.tobytes(), state[transfers].tobytes())
        if len(transfers) and key != network_key:
            nodes = getattr(fleet, "nodes", fleet.gpus)
            residual, app_residual = np.array(table.budgets, float), np.asarray(application) * nodes
            for phase in (3, 0):
                ids = transfers[state[transfers] == phase]
                if not len(ids):
                    continue
                if phase == 0:
                    routes, actions = route[::chunks], action[::chunks]
                    caps = np.where(actions == 1, np.minimum(endpoint[routes], np.asarray(application)[routes]), endpoint[routes])
                    pending = np.bincount(group[ids], weights=mass[ids], minlength=n // chunks)
                    virtual = _flow_with_caps(pending, routes, caps, residual, actions == 1, app_residual / nodes, nodes)
                    active = ids[initial_ready[ids]]
                    admitted = np.bincount(group[active], weights=mass[active], minlength=n // chunks)
                    needed = np.ceil(np.maximum(pending * virtual / caps - admitted, 0.) / mass[::chunks] - 1e-12).astype(int)
                    for g in np.flatnonzero(needed > 0):
                        start = g * chunks
                        waiting = np.flatnonzero((state[start:start + chunks] == 0) & ~initial_ready[start:start + chunks]) + start
                        initial_ready[waiting[:needed[g]]] = True
                    ids = ids[initial_ready[ids]]
                    if not len(ids):
                        continue
                caps = np.where(action[ids] == 1, np.minimum(endpoint[route[ids]], np.asarray(application)[route[ids]]), endpoint[route[ids]])
                rates[ids] = _flow_with_caps(mass[ids], route[ids], caps, residual, action[ids] == 1, app_residual / nodes, nodes)
                used = mass[ids] * rates[ids]
                residual = np.maximum(residual - np.r_[[used[route[ids] == r].sum() for r in (0, 1)], used.sum()], 0.)
                app_residual = np.maximum(app_residual - [used[(route[ids] == r) & (action[ids] == 1)].sum() for r in (0, 1)], 0.)
            network_key, network_rates = key, rates.copy()
        rates[transfers] = network_rates[transfers]
        computing = (state == 1) | (state == 4)
        backlog_rates = np.zeros(n)
        resident_growth, resident_recovery = np.zeros(2), np.zeros(2)
        for r in (0, 1):
            active = computing & (route == r)
            queued = (backlog > 1e-9) & (route == r)
            sharing = min(1., fleet.gpus / max(float(mass[active].sum()), 1e-30))
            capacity = fleet.gpus
            if "resident_replay_loss" in timing:
                replay_busy, kv_busy = [sharing * mass[active & (action == a)].sum() for a in (0, 1)]
                capacity -= replay_busy * (1 - loads[r] * (1 - resident_loss[r])) + kv_busy
            resident_growth[r] = max(loads[r] * fleet.gpus - capacity, 0.)
            spare = max(capacity - loads[r] * fleet.gpus, 0.)
            resident_recovery[r] = spare if resident_debt[r] > 1e-9 else 0.
            backlog_rates[queued] = min(1., (spare - resident_recovery[r]) / max(float(mass[queued].sum()), 1e-30))
            effective_load = loads[r] + (resident_recovery[r] + float(mass[queued] @ backlog_rates[queued])) / fleet.gpus
            peak_load[r] = max(peak_load[r], effective_load)
            rates[active] = sharing * np.exp(-timing["beta"] * effective_load)
        completions = np.divide(remaining, rates, out=np.full(n, np.inf), where=rates > 0)
        waiting = ((state == 2) | (state == 5)) & (release > now)
        step = min(table.deadline - now, float(completions.min(initial=np.inf)),
                   float(np.min(np.divide(resident_debt, resident_recovery, out=np.full(2, np.inf), where=resident_recovery > 0))),
                   float(np.min(np.divide(backlog, backlog_rates, out=np.full(n, np.inf), where=backlog_rates > 0), initial=np.inf)),
                   float(np.min(release[waiting] - now, initial=np.inf)))
        if not np.isfinite(step) or step <= 0:
            break
        sent = mass[transfers] * rates[transfers] * step
        network_used += [sent[route[transfers] == 0].sum(), sent[route[transfers] == 1].sum(), sent.sum()]
        compute_used += [min(float(mass[computing & (route == r)].sum()), fleet.gpus) * step for r in (0, 1)]
        idle_work += [float(mass[computing & (route == r)] @ rates[computing & (route == r)] * step) for r in (0, 1)]
        resident_generated += resident_growth * step
        resident_recovered += resident_recovery * step
        resident_debt = np.maximum(resident_debt + (resident_growth - resident_recovery) * step, 0.)
        backlog = np.maximum(backlog - backlog_rates * step, 0)
        remaining = np.maximum(remaining - rates * step, 0)
        now += step
    fractions, numbers = committed @ fleet.gain, committed.sum(1)
    events = list(events.values())
    pending = float(mass @ (buffered * np.divide(backlog, backlog_total, out=np.zeros(n), where=backlog_total > 0)))
    source_buffered, source_work = sum((mass[i] * np.array(_buffered(fleet, counts[i], quiesced[i], now, calibration))
                                       for i in np.flatnonzero((state >= 3) & (state < 6))), start=np.zeros(2))
    return {"shed_fraction": float(fractions.sum()), "action_counts": numbers.tolist(),
            "action_fractions": fractions.tolist(), "completed_sessions": float(numbers.sum()),
            "last_completion_s": max((e["completion_s"] for e in events), default=0.),
            "service_ready_s": float(now) if np.all(state == 6) and np.all(backlog <= 1e-9) and np.all(resident_debt <= 1e-9) else None,
            "resident_debt_generated_work_s": resident_generated.tolist(),
            "resident_debt_recovered_work_s": resident_recovered.tolist(),
            "pending_resident_debt_work_s": resident_debt.tolist(),
            "batch_replica_seconds": compute_used.tolist(), "migration_idle_work_s": idle_work.tolist(),
            "transferred_bytes": network_used.tolist(),
            "peak_destination_load": peak_load.tolist(), "final_destination_load": loads.tolist(), "completion_events": events,
            "peak_reserved_kv_tokens": peak_memory.tolist(),
            "buffered_requests": float(mass @ buffered) + float(source_buffered),
            "transferred_buffered_requests": float(mass @ buffered), "source_buffered_requests": float(source_buffered),
            "pending_buffered_requests": pending + float(source_buffered),
            "pending_destination_buffered_requests": pending,
            "completed_buffered_requests": float(mass @ buffered) - pending,
            "backlog_reference_work_s": float(mass @ backlog_total), "pending_backlog_reference_work_s": float(mass @ backlog),
            "pending_source_buffer_work_s": float(source_work),
            "pending_buffered_work_s": float(source_work + mass @ backlog),
            "unfinished_batch_mass": float(mass[state != 6].sum()), "memory_blocked_batch_mass": float(mass[state == 7].sum()),
            "trace_exhausted_waves": exhausted,
            "dispatch_chunks": int(chunks), "dispatch_wave_count": n,
            "dispatch_scope": "bounded waves fill a dynamic fair-share network window independent of wave count; final deltas have priority; frozen initial snapshot while source continues",
            "execution_model": "independent_event_fluid_batch_mass_finite_trace",
            "source_pacing": ("paced recorded trajectories; explicit reset on cyclic wrap" if fleet.metadata.get("sequence_cycle")
                              else "finite recorded turns at explicit equal cadence; terminal context retained"),
            "compute_scope": "fractional batch processor sharing; dynamic measured load factor; no ingestion",
            "backlog_scope": "migration reduces ordinary service; spare capacity repays resident debt before source buffers; recovery utilization enters the measured migration load factor",
            "resident_debt_scope": ("measured resident throughput loss during replay; conservative zero ordinary service during KV response compute; network waiting consumes no ordinary service"
                                    if "resident_replay_loss" in timing else "reverse interference disabled in legacy primitive fixture"),
            "memory_scope": "reserve declared cohort memory, including recorded-cycle peaks, before transfer; grow or reject before catch-up"}


def regional_execution_check(calibration):
    """Replay recorded regional actions through the engine with frozen source context."""
    import csv
    import json
    from pathlib import Path
    from types import SimpleNamespace
    from pool_shed_calibration import kv_state, replay_seconds

    root = Path(__file__).parent / "outputs/a100-parity-20260907/timing"
    plan, protocol = [json.loads((root / name).read_text()) for name in ("plan.json", "scale-protocol.json")]
    scenarios = {s["scenario_id"]: s for s in plan["scenarios"]}
    rows = []
    with (root / "results.csv").open() as handle:
        for row in csv.DictReader(handle):
            if row["scenario_id"] not in protocol["holdout_ids"]:
                continue
            if row["status"] != "complete":
                raise ValueError("regional validation requires complete episodes")
            scenario = scenarios[row["scenario_id"]]
            ids = {s["session_id"]: i for i, s in enumerate(scenario["sessions"])}
            context = np.array([s["initial_tokens"] for s in scenario["sessions"]], float)
            replay, kv = np.zeros((2, len(context))), np.zeros((2, len(context)))
            for move in scenario["moves"]:
                (replay if move["method"] == "replay" else kv)[int(move["destination_instance"] == "germany"), ids[move["session_id"]]] += 1
            fleet = SimpleNamespace(count=np.ones(len(context)), context=context, demand=np.zeros(len(context)),
                gain=np.ones(len(context)) / len(context), memory_tokens=np.ceil(context / 16) * 16,
                baseline_kv=0., kv_capacity=1e12, gpus=1, t1=replay_seconds(context, calibration),
                log=2 * context, kv=kv_state(context, calibration)[0], metadata={"turn_sequences": [[] for _ in context],
                "source_session_rps": 0., "batch_context_limit": calibration["batch_context_limit"],
                "packing_context_tokens": calibration["packing_context_tokens"]})
            endpoint = np.array([scenario["bandwidth_mbps"][r] * 125_000 for r in ("east", "germany")])
            table = SimpleNamespace(fleet=fleet, replay=replay, kv=kv, route=np.array([0, 1]),
                deadline=float(scenario["deadline_s"]), endpoint=endpoint,
                load=np.array([scenario["background"][r][0] for r in ("east", "germany")]),
                budgets=np.r_[endpoint, plan["network_contract"]["aggregate"]["natural_mbps"] * 125_000])
            result = execute_pooled(table, np.ones(2), calibration["timing"][0], calibration, chunks=1)
            if result["completed_sessions"] != len(context):
                raise RuntimeError("regional execution did not complete the recorded actions")
            rows.append({"scenario_id": row["scenario_id"], "policy": row["policy"],
                         "observed_s": float(row["migration_s"]), "predicted_s": result["last_completion_s"],
                         "completion_events": result["completion_events"]})
    if len(rows) != 24:
        raise ValueError("regional execution requires the frozen 24 episode partition")
    metrics = {}
    for policy in [*sorted({r["policy"] for r in rows}), "aggregate"]:
        selected = [r for r in rows if policy == "aggregate" or r["policy"] == policy]
        observed, predicted = np.array([[r["observed_s"], r["predicted_s"]] for r in selected]).T
        metrics[policy] = {"episodes": len(selected), "mae_s": float(np.mean(abs(predicted - observed))),
            "r2": float(1 - np.sum((predicted - observed) ** 2) / np.sum((observed - observed.mean()) ** 2)),
            "false_feasible_25s": int(np.sum((predicted <= 25) & (observed > 25)))}
    aggregate, gates = metrics["aggregate"], protocol["gates"]
    return {**metrics, "gate_pass": aggregate["mae_s"] <= gates["mae_s"] and aggregate["r2"] >= gates["r2"],
            "gates": gates, "predictions": rows,
            "scope": "Previously inspected regional partition; fixed actions, frozen context and resident load. Tests engine transport/compute, not source growth, dynamic serving admission or fleet-scale fidelity."}
