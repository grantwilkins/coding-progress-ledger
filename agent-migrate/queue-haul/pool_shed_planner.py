"""Time-indexed pooled admissions with measured queue feedback, not global optimality."""

from types import SimpleNamespace

import numpy as np

from pool_shed_execution import _buffered, _quiesce, catchup, flow_rates, compute_allocation, source_snapshot, initial_work

PLANNING_RESOLUTION = .5
PLANNING_ITERATIONS = 3


def planning_grid(now, deadline, duration, resolution=1.):
    step = max(.25, min(2., float(np.median(duration[duration > 0])) / 4))
    anchors = np.r_[0., step * 2. ** np.arange(max(1, int(np.ceil(np.log2(max(deadline / step, 1)))) + 1))]
    if resolution <= 0 or resolution > 1 or not np.isclose(1 / resolution, round(1 / resolution)):
        raise ValueError("temporal resolution must be the reciprocal of a positive integer")
    anchors = np.concatenate([a + (b - a) * np.arange(round(1 / resolution)) * resolution for a, b in zip(anchors[:-1], anchors[1:])])
    return np.unique(np.r_[now, anchors[(anchors > now + 1e-10) & (anchors < deadline)], deadline])


def _overlap(edges, start, end):
    return np.maximum(np.minimum(edges[1:], end) - np.maximum(edges[:-1], start), 0.)


def recovery_prefix(engine, edges, events=None):
    """Clear observed gates before other compute, retaining per-batch recovery caps."""
    profile, end = np.zeros((2, len(edges) - 1)), np.full(2, engine.now)
    for r in (0, 1):
        ids = np.flatnonzero((engine.route == r) & engine.gated & (engine.backlog > 1e-9))
        left, mass = engine.backlog[ids].copy(), engine.mass[ids]
        spare = max(engine.fleet.gpus * (1 - engine.serving_load()[r]), 0.)
        while np.any(left > 1e-9) and end[r] < edges[-1] and spare > 0:
            active = left > 1e-9
            rate = min(1., spare / mass[active].sum())
            step = min(left[active].min() / rate, edges[-1] - end[r])
            profile[r] += _overlap(edges, end[r], end[r] + step) * rate * mass[active].sum()
            left[active] = np.maximum(left[active] - step * rate, 0.)
            end[r] += step
            if events is not None:
                events.append(end[r])
        if np.any(left > 1e-9):
            end[r] = edges[-1] + 1.
    return profile, end


def phase_profile(table, counts, action, route, start, edges, loads, timing, calibration,
                  state=0, quiesced=0., transferred=0., completed_work=0., elapsed=0., rate=None, sharing=1., buffer_remaining=None,
                  compute_after=0., recovery_sharing=1., primitive_cache=None):
    """One central-calibration batch; phase dependencies and source resets remain causal."""
    fleet, bins = table.fleet, len(edges) - 1
    if fleet.metadata.get("protect_resident") and any((state, transferred, completed_work, elapsed)):
        raise ValueError("active protected migrations require their observed-origin engine continuation")
    profile = {name: np.zeros(bins) for name in ("replay", "kv", "network", "application", "serving", "buffers", "recovery", "occupancy", "service_peak")}
    endpoint = min(table.endpoint[route], table.budgets[route], table.budgets[2])
    if action:
        endpoint = min(endpoint, timing.get("regional_kv_bytes_per_s", table.endpoint[:2])[route])
    endpoint = endpoint if rate is None else max(min(rate, endpoint), 1e-30)
    now = start
    origin_context, origin_turn = source_snapshot(fleet, start, primitive_cache) if fleet.metadata.get("protect_resident") else (None, None)
    initial_bytes, initial_compute = initial_work(fleet, counts, action, route, origin_context, timing, calibration, primitive_cache)

    def phase(work, compute=False):
        nonlocal now
        if not compute:
            end = now + max(work, 0.) / endpoint
            sent = _overlap(edges, now, end) * endpoint
            profile["network"] += sent
            if action:
                profile["application"] += sent
            now = end
            return
        left = max(work, 0.)
        if left:
            now = max(now, compute_after)
        for k in range(max(0, np.searchsorted(edges, now, side="right") - 1), bins):
            speed = sharing * np.exp(-timing["beta"] * loads[route, k] * fleet.metadata.get("timing_load_factor", 1.))
            step = min(max(edges[k + 1] - now, 0.), left / speed)
            profile["kv" if action else "replay"][k] += step * sharing
            if step > 0:
                profile["occupancy"][k] = max(profile["occupancy"][k], sharing * (edges[k + 1] - edges[k]))
            now += step
            left = max(left - step * speed, 0.)
            if left <= 1e-12:
                break
        if left:
            now += left * np.exp(timing["beta"] * loads[route, -1] * fleet.metadata.get("timing_load_factor", 1.)) / sharing

    if state == 0:
        phase(initial_bytes - transferred)
    if state <= 1 and not action:
        phase(initial_compute - (completed_work if state == 1 else 0.), True)
    if now > edges[-1]:
        profile["finish"] = now
        return profile
    if state <= 1:
        quiesced, context, reset, _ = _quiesce(fleet, counts, now, primitive_cache, origin_turn=origin_turn)
        now = quiesced
    else:
        _, context, reset, _ = _quiesce(fleet, counts, quiesced, primitive_cache, origin_turn=origin_turn)
        if state == 2:
            now = max(now, quiesced)
    delta, tail = catchup(fleet, counts, action, route, context, reset, timing, calibration, primitive_cache, origin_context=origin_context)
    if state <= 3:
        phase(delta - (transferred if state == 3 else 0.))
    if state <= 4:
        phase(tail - (completed_work if state == 4 else 0.), True)
    if fleet.metadata.get("protect_resident"):
        now = max(now, compute_after)
    now += max(calibration.get("switch_s", 0.) - (elapsed if state == 5 else 0.), 0.)
    profile["serving"] = _overlap(edges, now, edges[-1]) * float(counts @ fleet.demand)
    profile["service_peak"] = (profile["serving"] > 0) * float(counts @ fleet.demand) * np.diff(edges)
    if now <= edges[-1] + 1e-10:
        _, buffer = _buffered(fleet, counts, quiesced, now, calibration, primitive_cache)
        if fleet.metadata.get("protect_resident"):
            buffer = buffer if buffer_remaining is None else buffer_remaining
            profile["buffers"][min(bins - 1, max(0, np.searchsorted(edges, now, side="right") - 1))] = buffer
            left = buffer
            for k in range(max(0, np.searchsorted(edges, now, side="right") - 1), bins):
                speed = min(recovery_sharing, max(fleet.gpus * (1 - table.load) - float(counts @ fleet.demand), 0.))
                step = min(max(edges[k + 1] - now, 0.), left / max(speed, 1e-30))
                profile["recovery"][k] += step * speed
                if step > 0:
                    profile["service_peak"][k] += speed * (edges[k + 1] - edges[k])
                now += step
                left = max(left - step * speed, 0.)
                if left <= 1e-12:
                    break
            if left > 1e-12:
                now = edges[-1] + left / max(speed, 1e-30)
        else:
            k = min(bins - 1, max(0, np.searchsorted(edges, now, side="right") - 1))
            profile["buffers"][k] = buffer
    profile["occupancy"] = np.maximum(profile["occupancy"], profile["service_peak"])
    profile["finish"] = now
    return profile


def project_queues(edges, initial_load, resident, buffered, replay, kv, serving, arrivals, gpus, loss, buffer_groups=None):
    """Resident-first recovery, including newly generated debt in subsequent rate forecasts."""
    dt = np.diff(edges)
    resident, buffered = np.array(resident, float), np.array(buffered, float)
    groups = np.asarray(buffer_groups, float).reshape(-1, 4) if buffer_groups is not None else None
    if groups is None and (buffered.any() or np.any(arrivals)):
        raise ValueError("buffer projection requires batch masses and remaining per-batch work")
    effective, history = np.zeros((2, len(dt))), []
    for k, step in enumerate(dt):
        load = np.minimum(1., np.asarray(initial_load) + serving[:, k] / (gpus * step))
        replay_busy, kv_busy = replay[:, k] / step, kv[:, k] / step
        sharing = np.minimum(1., gpus / np.maximum(replay_busy + kv_busy, 1e-30))
        capacity = gpus - sharing * (replay_busy * (1 - load * (1 - loss)) + kv_busy)
        generated = np.maximum(load * gpus - capacity, 0.) * step
        spare = np.maximum(capacity - load * gpus, 0.) * step
        recovered = np.minimum(resident, spare)
        resident = resident + generated - recovered
        buffers_recovered = np.minimum(buffered, spare - recovered)
        if groups is not None:
            buffers_recovered = np.zeros(2)
            for r in (0, 1):
                seconds = step * (1 - recovered[r] / spare[r]) if spare[r] > 0 else 0.
                ids = np.flatnonzero((groups[:, 0] == r) & (groups[:, 1] < k) & (groups[:, 2] > 1e-12))
                while seconds > 1e-12 and len(ids):
                    rate = min(1., spare[r] / step / groups[ids, 3].sum())
                    duration = min(seconds, groups[ids, 2].min() / max(rate, 1e-30))
                    groups[ids, 2] = np.maximum(groups[ids, 2] - duration * rate, 0.)
                    buffers_recovered[r] += duration * rate * groups[ids, 3].sum()
                    seconds -= duration
                    ids = ids[groups[ids, 2] > 1e-12]
        buffered = np.maximum(buffered - buffers_recovered, 0.) + arrivals[:, k]
        effective[:, k] = load + (recovered + buffers_recovered) / (gpus * step)
        history.append(np.r_[resident, buffered])
    return effective, np.asarray(history), resident + buffered


def _allowed(table, policy, fastest=None):
    if policy == "kv_only":
        return table.replay.sum(1) == 0
    if policy == "replay_only":
        return table.kv.sum(1) == 0
    if policy == "isolated_fastest":
        fastest = table.fastest if fastest is None else fastest
        return ~np.any((table.replay > 0) & ~fastest, axis=1) & ~np.any((table.kv > 0) & fastest, axis=1)
    if policy not in ("queue_haul", "greedy"):
        raise ValueError(policy)
    return np.ones(len(table.route), bool)


def _choose(matrix, capacity, gains, debt, fleet, greedy):
    from pool_shed_campaign import solve_lp

    table = SimpleNamespace(matrix=matrix, capacities=capacity, gains=gains, fleet=fleet)
    allowed = gains > 0
    if not greedy:
        chosen = solve_lp(table, allowed, -gains)
        return solve_lp(table, allowed, debt, float(gains @ chosen)) if debt.any() else chosen
    scale = np.maximum(capacity, 1.)
    normalized, remaining = matrix / scale[:, None], capacity / scale
    chosen = np.zeros(len(gains))
    for _ in range(len(capacity) + 1):
        feasible = allowed & ~np.any((normalized > 0) & (remaining[:, None] <= 1e-10), axis=0)
        if not feasible.any():
            return chosen
        costs = np.max(normalized / np.maximum(remaining[:, None], 1e-30), axis=0)
        scores = np.where(feasible, gains / np.maximum(costs, 1e-30), -np.inf)
        tied = feasible & np.isclose(scores, scores.max(), rtol=1e-12, atol=0.)
        j = int(np.argmin(np.where(tied, debt / np.maximum(gains, 1e-30), np.inf)))
        used = normalized[:, j] > 0
        take = np.min(remaining[used] / normalized[used, j])
        chosen[j] += take
        remaining = np.maximum(remaining - take * normalized[:, j], 0.)
    raise RuntimeError("temporal greedy failed to exhaust a constraint")


def mandatory_profile(engine, table, timing, calibration, edges):
    forecast = engine.nominal_continuation(table, timing, calibration)
    fixed = {name: np.zeros((2, len(edges) - 1)) for name in
             ("replay", "kv", "network", "application", "serving", "buffers", "recovery", "occupancy", "service_peak")}
    queued = np.array([forecast.mass[forecast.route == r] @ forecast.backlog[forecast.route == r] for r in (0, 1)])
    for k, (start, end) in enumerate(zip(edges[:-1], edges[1:])):
        usage = forecast.advance(end, collect=True)
        pending = np.array([forecast.mass[forecast.route == r] @ forecast.backlog[forecast.route == r] for r in (0, 1)])
        fixed["recovery"][:, k] = usage["recovery"]
        fixed["network"][:, k], fixed["application"][:, k] = usage["network"][:2], usage["application"]
        fixed["serving"][:, k] = usage["serving"] - engine.loads * engine.fleet.gpus * (end - start)
        fixed["service_peak"][:, k] = np.maximum(usage["service_peak"] - engine.loads * engine.fleet.gpus, 0.) * (end - start)
        fixed["occupancy"][:, k] = usage["peak"] * (end - start)
        fixed["buffers"][:, k] = np.maximum(pending - queued + usage["recovery"], 0.)
        queued = pending
    finish = np.array([table.deadline + 1. if np.any((forecast.route == r) & (forecast.state < 6)) else
                       max((e["completion_s"] for e in forecast.events.values() if e["route"] == r), default=engine.now)
                       for r in (0, 1)])
    return fixed, finish


def plan_admission(engine, nominal_table, policy, timing=None, calibration=None,
                   iterations=PLANNING_ITERATIONS, resolution=PLANNING_RESOLUTION):
    """Plan starts over a geometric horizon; commit only starts in its first interval."""
    table, fleet = nominal_table, nominal_table.fleet
    timing, calibration = table.timing if timing is None else timing, engine.calibration if calibration is None else calibration
    primitive_cache = {}  # Scoped to this fixed nominal fleet, timing and calibration.
    if iterations < 1 or engine.now >= table.deadline:
        raise ValueError("planning needs a positive iteration budget and remaining time")
    edges = planning_grid(engine.now, table.deadline, table.nominal_commit, resolution)
    if fleet.metadata.get("protect_resident"):
        events = []
        recovery_prefix(engine, edges, events)
        edges = np.unique(np.r_[edges, events])
    next_decision, start_times = float(edges[1]), edges[:-1][:3].copy()
    bins, columns = len(edges) - 1, len(table.route)
    total, route_masks = table.replay + table.kv, np.array([table.route == r for r in (0, 1)])
    available = np.maximum(fleet.count - total.T @ engine.selected_total, 0.)
    serving = route_masks * (total @ fleet.demand)
    memory = route_masks * (total @ fleet.memory_tokens)
    static = np.vstack((total.T, serving, memory))
    capacity = np.r_[available, fleet.gpus * (1 - engine.initial_load) - serving @ engine.selected_total,
                     np.full(2, engine.free_memory) - np.array([engine.reserved[engine.route == r].sum() for r in (0, 1)])]
    static_scale = np.maximum(np.r_[fleet.count, [fleet.gpus] * 2, [fleet.kv_capacity] * 2], 1.)
    if np.min(capacity / static_scale) < -1e-8:
        raise RuntimeError("admitted migrations exceed static capacity")
    capacity = np.maximum(capacity, 0.)
    loss = np.broadcast_to(np.asarray(timing.get("resident_replay_loss", 0.)), (2,))
    observed = engine.serving_load() if fleet.metadata.get("protect_resident") else engine.loads.copy()
    for r in (0, 1):
        computing = ((engine.state == 1) | (engine.state == 4)) & (engine.route == r)
        busy = np.array([engine.mass[computing & (engine.action == a)].sum() for a in (0, 1)])
        sharing, capacity_now = compute_allocation(observed[r], *busy, fleet.gpus, loss[r], fleet.metadata.get("protect_resident", False))
        spare = max(fleet.gpus * (1 - observed[r]) if fleet.metadata.get("protect_resident") else capacity_now - fleet.gpus * observed[r], 0.)
        buffer_mass = engine.mass[(engine.route == r) & (engine.backlog > 1e-9)].sum()
        observed[r] += (spare if engine.resident_debt[r] > 1e-9 else min(spare, buffer_mass)) / fleet.gpus
    fastest = table.fastest
    if policy == "isolated_fastest":
        isolated_load = np.broadcast_to(observed[:, None], (2, bins))
        finish = np.array([[[phase_profile(table, counts, action, route, engine.now, edges, isolated_load, timing, calibration, primitive_cache=primitive_cache)["finish"]
                            for counts in np.eye(len(fleet.count))] for route in (0, 1)] for action in (0, 1)])
        fastest = finish[0].min(0) < finish[1].min(0)
    allowed = _allowed(table, policy, fastest) & ~np.any((static > 0) & (capacity[:, None] <= 1e-10 * static_scale[:, None]), axis=0)
    ids = np.flatnonzero(allowed)
    # ponytail: three upcoming start times; replan later starts after observing actual progress.
    start_bins = np.searchsorted(edges, start_times)
    starts, original = np.repeat(start_bins, len(ids)), np.tile(ids, len(start_bins))
    if not len(ids):
        return np.zeros(columns), float(table.deadline), {"iterations": 0, "max_relative_residual": 0.,
            "projected_debt_work_s": (engine.resident_debt + np.array([engine.mass[engine.route == r] @ engine.backlog[engine.route == r] for r in (0, 1)])).tolist(),
            "fixed_point_residual": 0., "predicted_shed_fraction": 0., "variables": 0, "time_bins": bins,
            "planning_scope": "no remaining source population fits reserved serving and memory capacity"}
    loads = np.broadcast_to(engine.loads[:, None], (2, bins)).copy()
    queued = np.array([engine.mass[engine.route == r] @ engine.backlog[engine.route == r] for r in (0, 1)])
    completed_work = engine.phase_replica_seconds * np.exp(-timing["beta"] * observed[engine.route] * fleet.metadata.get("timing_load_factor", 1.))
    residual, overload, change, chosen = 0., 0., 0., np.zeros(len(original))
    predicted, debt_history = engine.resident_debt + queued, np.zeros((bins, 4))
    protected = fleet.metadata.get("protect_resident", False)
    prefix, compute_after = recovery_prefix(engine, edges) if protected else (np.zeros((2, bins)), np.full(2, engine.now))
    mandatory, mandatory_finish = mandatory_profile(engine, table, timing, calibration, edges) if protected else (None, None)
    active = np.flatnonzero((engine.state < 6) & (not protected))
    active_mass = np.array([engine.mass[active[engine.route[active] == r]].sum() for r in (0, 1)])
    reserved_load = engine.loads + np.array([engine.mass[(engine.route == r) & (engine.state < 6)] @ engine.demand[(engine.route == r) & (engine.state < 6)] for r in (0, 1)]) / fleet.gpus
    fixed_sharing = np.minimum(1., fleet.gpus * np.maximum(1 - reserved_load if protected else np.ones(2), 0.) / np.maximum(active_mass, 1e-30))
    for iteration in range(iterations):
        fixed = {name: np.zeros((2, bins)) for name in ("replay", "kv", "network", "application", "serving", "buffers", "recovery", "occupancy", "service_peak")}
        if protected:
            fixed = mandatory
        active_rate = flow_rates(active_mass, np.arange(2), table.endpoint[:2], table.budgets)
        active_kv = np.array([engine.mass[active[(engine.route[active] == r) & (engine.action[active] == 1)]].sum() for r in (0, 1)])
        app = np.asarray(timing.get("regional_kv_bytes_per_s", table.endpoint[:2])) * fleet.nodes
        active_rate = np.minimum(active_rate, app / np.maximum(active_kv, 1e-30))
        buffer_groups = [(int(engine.route[i]), -1, float(engine.backlog[i]), float(engine.mass[i]))
                         for i in np.flatnonzero(engine.backlog > 1e-12)]
        profiles, fixed_finish = {}, mandatory_finish.copy() if protected else compute_after.copy()
        for i in active:
            key = (int(engine.selected[i][0]), int(engine.action[i]), int(engine.state[i]), float(engine.quiesced[i]),
                   float(engine.phase_transferred_bytes[i]), float(engine.phase_replica_seconds[i]), float(engine.phase_started[i]),
                   bool(engine.gated[i]) if fleet.metadata.get("protect_resident") else False, float(engine.backlog[i]))
            if key not in profiles:
                # ponytail: past replica service uses fixed observed load; retain load history only if refinement requires it.
                profiles[key] = phase_profile(table, engine.counts[i], engine.action[i], engine.route[i], engine.now,
                    edges, loads, timing, calibration, state=engine.state[i], quiesced=engine.quiesced[i],
                    transferred=engine.phase_transferred_bytes[i], completed_work=completed_work[i],
                    elapsed=calibration.get("switch_s", 0.) if fleet.metadata.get("protect_resident") and engine.gated[i] else engine.now - engine.phase_started[i],
                    rate=active_rate[engine.route[i]],
                    sharing=max(1e-30, fixed_sharing[engine.route[i]]), compute_after=compute_after[engine.route[i]],
                    recovery_sharing=fixed_sharing[engine.route[i]],
                    buffer_remaining=float(engine.backlog[i]) if fleet.metadata.get("protect_resident") and engine.gated[i] else None,
                    primitive_cache=primitive_cache)
            fixed_finish[engine.route[i]] = max(fixed_finish[engine.route[i]], profiles[key]["finish"])
            for name in fixed:
                fixed[name][engine.route[i]] += engine.mass[i] * profiles[key][name]
            for k in np.flatnonzero(profiles[key]["buffers"] > 0):
                buffer_groups.append((int(engine.route[i]), int(k), float(profiles[key]["buffers"][k]), float(engine.mass[i])))
        data = {name: np.zeros((2, bins, len(original))) for name in fixed}
        finish = np.zeros(len(original))
        profiles, future_buffers = {}, []
        for v, (k, j) in enumerate(zip(starts, original)):
            for action, counts in enumerate((table.replay[j], table.kv[j])):
                if not counts.any():
                    continue
                key = (int(k), int(action), int(table.route[j]), counts.tobytes())
                if key not in profiles:
                    profiles[key] = phase_profile(table, counts, action, table.route[j], edges[k], edges, loads, timing, calibration,
                        compute_after=compute_after[table.route[j]],
                        sharing=max(1e-30, fixed_sharing[table.route[j]]) if protected else 1.,
                        recovery_sharing=fixed_sharing[table.route[j]] if protected else 1., primitive_cache=primitive_cache)
                profile = profiles[key]
                finish[v] = max(finish[v], profile["finish"])
                for name in data:
                    data[name][table.route[j], :, v] += profile[name]
                future_buffers.extend((int(table.route[j]), int(b), float(profile["buffers"][b]), int(v))
                                      for b in np.flatnonzero(profile["buffers"] > 0))
        dt = np.diff(edges)
        compute = data["replay"] + data["kv"]
        fixed_compute = fixed["replay"] + fixed["kv"]
        compute_limit = np.tile(fleet.gpus * dt, 2)
        if fleet.metadata.get("protect_resident"):
            compute, fixed_compute = data["occupancy"], fixed["occupancy"]
            compute_limit = np.tile(fleet.gpus * dt, 2)
        resource = np.vstack((compute.reshape(2 * bins, -1), data["network"].reshape(2 * bins, -1),
                              data["network"].sum(0), data["application"].reshape(2 * bins, -1)))
        fixed_resource = np.r_[fixed_compute.ravel(), fixed["network"].ravel(),
                               fixed["network"].sum(0), fixed["application"].ravel()]
        budgets = np.minimum(table.budgets, table.endpoint * fleet.nodes)
        limit = np.r_[compute_limit, (budgets[:2, None] * dt).ravel(), budgets[2] * dt, (app[:, None] * dt).ravel()]
        overload = max(overload, float(np.max((fixed_resource - limit) / np.maximum(limit, 1.), initial=0.)))
        matrix = np.vstack((static[:, original], resource))
        limits = np.r_[capacity, np.where(limit - fixed_resource > 1e-10 * np.maximum(limit, 1.), limit - fixed_resource, 0.)]
        gains = table.gains[original] * (finish <= table.deadline + 1e-10)
        debt = (data["replay"] * loads[:, :, None] * loss[:, None, None] + data["kv"] * loads[:, :, None] + data["buffers"]).sum((0, 1))
        if fleet.metadata.get("protect_resident"):
            debt = (data["replay"] + data["kv"] + data["recovery"]).sum((0, 1))
        debt += fleet.gpus * table.gains[original] * (finish - engine.now) / max(table.deadline - engine.now, 1e-30) * 1e-6
        chosen = _choose(matrix, limits, gains, debt, fleet, policy == "greedy") if len(original) else np.zeros(0)
        residual = max(residual, float(np.max((matrix @ chosen - limits) / np.maximum(limits, 1.), initial=0.)))
        aggregate = {name: fixed[name] + data[name] @ chosen for name in data}
        buffer_groups.extend((r, k, work, float(chosen[v])) for r, k, work, v in future_buffers if chosen[v] > 1e-12)
        if fleet.metadata.get("protect_resident"):
            updated = np.minimum(1., engine.loads[:, None] + aggregate["service_peak"] / (fleet.gpus * dt))
            pending = np.maximum(queued[:, None] + np.cumsum(aggregate["buffers"] - aggregate["recovery"], axis=1), 0.)
            predicted, debt_history = pending[:, -1], np.column_stack((np.zeros((bins, 2)), pending.T))
        else:
            updated, debt_history, predicted = project_queues(edges, engine.loads, engine.resident_debt, queued,
                aggregate["replay"], aggregate["kv"], aggregate["serving"], aggregate["buffers"], fleet.gpus, loss, buffer_groups)
        change = float(np.max(abs(updated - loads)))
        loads = updated
        if change < 1e-3:
            break
    if max(residual, overload) > 1e-8:
        raise RuntimeError("temporal plan violates resource constraints")
    admission = np.bincount(original[starts == 0], weights=chosen[starts == 0], minlength=columns)
    return admission, next_decision, {"iterations": iteration + 1, "max_relative_residual": residual,
        "projected_debt_work_s": predicted.tolist(), "projected_queue_history_work_s": debt_history.tolist(),
        "fixed_point_residual": change, "fixed_obligation_overload": overload,
        "mandatory_forecast_finish_s": fixed_finish.tolist(),
        "predicted_shed_fraction": float(gains @ chosen), "variables": len(original),
        "time_bins": bins, "planning_scope": "central mandatory continuation (deadline+1 denotes unfinished); candidate temporal LPs with bounded load iteration, compute peak envelopes and network volumes; handoff objective"}
