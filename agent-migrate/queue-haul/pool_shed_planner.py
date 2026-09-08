"""Time-indexed pooled admissions with measured queue feedback, not global optimality."""

from types import SimpleNamespace

import numpy as np

from pool_shed_execution import _buffered, _quiesce, catchup, flow_rates

PLANNING_RESOLUTION = 1.
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


def phase_profile(table, counts, action, route, start, edges, loads, timing, calibration,
                  state=0, quiesced=0., transferred=0., completed_work=0., elapsed=0., rate=None, sharing=1.):
    """One central-calibration batch; phase dependencies and source resets remain causal."""
    fleet, bins = table.fleet, len(edges) - 1
    profile = {name: np.zeros(bins) for name in ("replay", "kv", "network", "application", "serving", "buffers")}
    endpoint = min(table.endpoint[route], table.budgets[route], table.budgets[2])
    if action:
        endpoint = min(endpoint, timing.get("regional_kv_bytes_per_s", table.endpoint[:2])[route])
    endpoint = endpoint if rate is None else max(min(rate, endpoint), 1e-30)
    now = start

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
        for k in range(max(0, np.searchsorted(edges, now, side="right") - 1), bins):
            speed = sharing * np.exp(-timing["beta"] * loads[route, k])
            step = min(max(edges[k + 1] - now, 0.), left / speed)
            profile["kv" if action else "replay"][k] += step * sharing
            now += step
            left = max(left - step * speed, 0.)
            if left <= 1e-12:
                break
        if left:
            now += left * np.exp(timing["beta"] * loads[route, -1]) / sharing

    if state == 0:
        phase(float(counts @ (fleet.kv if action else fleet.log)) - transferred)
    if state <= 1 and not action:
        knots = fleet.metadata.get("packing_context_tokens")
        packing = np.interp(fleet.context, knots, timing["packing_kappa"]) if knots else np.full(len(counts), timing["kappa"])
        if np.any((counts > 0) & (fleet.context > fleet.metadata.get("batch_context_limit", np.inf))):
            packing = np.ones(len(counts))
        work = counts @ (packing * fleet.t1) + np.max(np.where(counts > 0, (1 - packing) * fleet.t1, 0.))
        work *= timing.get("regional_replay_factor", [1., 1.])[route]
        phase(float(work) - (completed_work if state == 1 else 0.), True)
    if state <= 1:
        quiesced, context, reset, _ = _quiesce(fleet, counts, now)
        now = quiesced
    else:
        _, context, reset, _ = _quiesce(fleet, counts, quiesced)
        if state == 2:
            now = max(now, quiesced)
    delta, tail = catchup(fleet, counts, action, route, context, reset, timing, calibration)
    if state <= 3:
        phase(delta - (transferred if state == 3 else 0.))
    if state <= 4:
        phase(tail - (completed_work if state == 4 else 0.), True)
    now += max(calibration.get("switch_s", 0.) - (elapsed if state == 5 else 0.), 0.)
    profile["serving"] = _overlap(edges, now, edges[-1]) * float(counts @ fleet.demand)
    if now <= edges[-1] + 1e-10:
        _, buffer = _buffered(fleet, counts, quiesced, now, calibration)
        k = min(bins - 1, max(0, np.searchsorted(edges, now, side="right") - 1))
        profile["buffers"][k] = buffer
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
        costs += debt / max(float(np.max(debt[feasible], initial=0.)), 1e-30) * np.max(costs[feasible], initial=0.)
        j = int(np.argmax(np.where(feasible, gains / np.maximum(costs, 1e-30), -np.inf)))
        used = normalized[:, j] > 0
        take = np.min(remaining[used] / normalized[used, j])
        chosen[j] += take
        remaining = np.maximum(remaining - take * normalized[:, j], 0.)
    raise RuntimeError("temporal greedy failed to exhaust a constraint")


def plan_admission(engine, nominal_table, policy, timing=None, calibration=None,
                   iterations=PLANNING_ITERATIONS, resolution=PLANNING_RESOLUTION):
    """Plan starts over a geometric horizon; commit only starts in its first interval."""
    table, fleet = nominal_table, nominal_table.fleet
    timing, calibration = table.timing if timing is None else timing, engine.calibration if calibration is None else calibration
    if iterations < 1 or engine.now >= table.deadline:
        raise ValueError("planning needs a positive iteration budget and remaining time")
    edges = planning_grid(engine.now, table.deadline, table.nominal_commit, resolution)
    bins, columns = len(edges) - 1, len(table.route)
    total, route_masks = table.replay + table.kv, np.array([table.route == r for r in (0, 1)])
    available = np.maximum(fleet.count - total.T @ engine.selected_total, 0.)
    serving = route_masks * (total @ fleet.demand)
    memory = route_masks * (total @ fleet.memory_tokens)
    static = np.vstack((total.T, serving, memory))
    capacity = np.r_[available, fleet.gpus * (1 - engine.initial_load) - serving @ engine.selected_total,
                     np.full(2, engine.free_memory) - np.array([engine.reserved[engine.route == r].sum() for r in (0, 1)])]
    if np.min(capacity / np.maximum(np.r_[fleet.count, [fleet.gpus] * 2, [fleet.kv_capacity] * 2], 1.)) < -1e-8:
        raise RuntimeError("admitted migrations exceed static capacity")
    capacity = np.maximum(capacity, 0.)
    loss = np.broadcast_to(np.asarray(timing.get("resident_replay_loss", 0.)), (2,))
    observed = engine.loads.copy()
    for r in (0, 1):
        computing = ((engine.state == 1) | (engine.state == 4)) & (engine.route == r)
        busy = np.array([engine.mass[computing & (engine.action == a)].sum() for a in (0, 1)])
        busy *= min(1., fleet.gpus / max(busy.sum(), 1e-30))
        spare = max(fleet.gpus * (1 - engine.loads[r]) - busy[0] * (1 - engine.loads[r] * (1 - loss[r])) - busy[1], 0.)
        buffer_mass = engine.mass[(engine.route == r) & (engine.backlog > 1e-9)].sum()
        observed[r] += (spare if engine.resident_debt[r] > 1e-9 else min(spare, buffer_mass)) / fleet.gpus
    fastest = table.fastest
    if policy == "isolated_fastest":
        isolated_load = np.broadcast_to(observed[:, None], (2, bins))
        finish = np.array([[[phase_profile(table, counts, action, route, engine.now, edges, isolated_load, timing, calibration)["finish"]
                            for counts in np.eye(len(fleet.count))] for route in (0, 1)] for action in (0, 1)])
        fastest = finish[0].min(0) < finish[1].min(0)
    allowed = _allowed(table, policy, fastest) & ~np.any((static > 0) & (capacity[:, None] <= 1e-10), axis=0)
    ids = np.flatnonzero(allowed)
    # ponytail: three upcoming start times; replan later starts after observing actual progress.
    start_bins = np.arange(min(3, bins))
    starts, original = np.repeat(start_bins, len(ids)), np.tile(ids, len(start_bins))
    if not len(ids):
        return np.zeros(columns), float(table.deadline), {"iterations": 0, "max_relative_residual": 0.,
            "projected_debt_work_s": (engine.resident_debt + np.array([engine.mass[engine.route == r] @ engine.backlog[engine.route == r] for r in (0, 1)])).tolist(),
            "fixed_point_residual": 0., "predicted_shed_fraction": 0., "variables": 0, "time_bins": bins,
            "planning_scope": "no remaining source population fits reserved serving and memory capacity"}
    loads = np.broadcast_to(engine.loads[:, None], (2, bins)).copy()
    queued = np.array([engine.mass[engine.route == r] @ engine.backlog[engine.route == r] for r in (0, 1)])
    completed_work = engine.phase_replica_seconds * np.exp(-timing["beta"] * observed[engine.route])
    residual, overload, change, chosen = 0., 0., 0., np.zeros(len(original))
    predicted, debt_history = engine.resident_debt + queued, np.zeros((bins, 4))
    for iteration in range(iterations):
        fixed = {name: np.zeros((2, bins)) for name in ("replay", "kv", "network", "application", "serving", "buffers")}
        active = np.flatnonzero(engine.state < 6)
        active_mass = np.array([engine.mass[active[engine.route[active] == r]].sum() for r in (0, 1)])
        active_rate = flow_rates(active_mass, np.arange(2), table.endpoint[:2], table.budgets)
        active_kv = np.array([engine.mass[active[(engine.route[active] == r) & (engine.action[active] == 1)]].sum() for r in (0, 1)])
        app = np.asarray(timing.get("regional_kv_bytes_per_s", table.endpoint[:2])) * fleet.nodes
        active_rate = np.minimum(active_rate, app / np.maximum(active_kv, 1e-30))
        buffer_groups = [(int(engine.route[i]), -1, float(engine.backlog[i]), float(engine.mass[i]))
                         for i in np.flatnonzero(engine.backlog > 1e-12)]
        profiles = {}
        for i in active:
            key = (int(engine.selected[i][0]), int(engine.action[i]), int(engine.state[i]), float(engine.quiesced[i]),
                   float(engine.phase_transferred_bytes[i]), float(engine.phase_replica_seconds[i]), float(engine.phase_started[i]))
            if key not in profiles:
                # ponytail: past replica service uses fixed observed load; retain load history only if refinement requires it.
                profiles[key] = phase_profile(table, engine.counts[i], engine.action[i], engine.route[i], engine.now,
                    edges, loads, timing, calibration, state=engine.state[i], quiesced=engine.quiesced[i],
                    transferred=engine.phase_transferred_bytes[i], completed_work=completed_work[i],
                    elapsed=engine.now - engine.phase_started[i], rate=active_rate[engine.route[i]],
                    sharing=min(1., fleet.gpus / max(active_mass[engine.route[i]], 1e-30)))
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
                    profiles[key] = phase_profile(table, counts, action, table.route[j], edges[k], edges, loads, timing, calibration)
                profile = profiles[key]
                finish[v] = max(finish[v], profile["finish"])
                for name in data:
                    data[name][table.route[j], :, v] += profile[name]
                future_buffers.extend((int(table.route[j]), int(b), float(profile["buffers"][b]), int(v))
                                      for b in np.flatnonzero(profile["buffers"] > 0))
        dt = np.diff(edges)
        resource = np.vstack(((data["replay"] + data["kv"]).reshape(2 * bins, -1), data["network"].reshape(2 * bins, -1),
                              data["network"].sum(0), data["application"].reshape(2 * bins, -1)))
        fixed_resource = np.r_[(fixed["replay"] + fixed["kv"]).ravel(), fixed["network"].ravel(),
                               fixed["network"].sum(0), fixed["application"].ravel()]
        budgets = np.minimum(table.budgets, table.endpoint * fleet.nodes)
        limit = np.r_[np.tile(fleet.gpus * dt, 2), (budgets[:2, None] * dt).ravel(), budgets[2] * dt, (app[:, None] * dt).ravel()]
        overload = max(overload, float(np.max((fixed_resource - limit) / np.maximum(limit, 1.), initial=0.)))
        matrix = np.vstack((static[:, original], resource))
        limits = np.r_[capacity, np.maximum(limit - fixed_resource, 0.)]
        gains = table.gains[original] * (finish <= table.deadline + 1e-10)
        debt = (data["replay"] * loads[:, :, None] * loss[:, None, None] + data["kv"] * loads[:, :, None] + data["buffers"]).sum((0, 1))
        debt += table.gains[original] * (finish - engine.now) / max(table.deadline - engine.now, 1e-30) * 1e-6
        chosen = _choose(matrix, limits, gains, debt, fleet, policy == "greedy") if len(original) else np.zeros(0)
        residual = max(residual, float(np.max((matrix @ chosen - limits) / np.maximum(limits, 1.), initial=0.)))
        aggregate = {name: fixed[name] + data[name] @ chosen for name in data}
        buffer_groups.extend((r, k, work, float(chosen[v])) for r, k, work, v in future_buffers if chosen[v] > 1e-12)
        updated, debt_history, predicted = project_queues(edges, engine.loads, engine.resident_debt, queued,
            aggregate["replay"], aggregate["kv"], aggregate["serving"], aggregate["buffers"], fleet.gpus, loss, buffer_groups)
        change = float(np.max(abs(updated - loads)))
        loads = updated
        if change < 1e-3:
            break
    if max(residual, overload) > 1e-8:
        raise RuntimeError("temporal plan violates resource constraints")
    admission = np.bincount(original[starts == 0], weights=chosen[starts == 0], minlength=columns)
    return admission, float(edges[1]), {"iterations": iteration + 1, "max_relative_residual": residual,
        "projected_debt_work_s": predicted.tolist(), "projected_queue_history_work_s": debt_history.tolist(),
        "fixed_point_residual": change, "fixed_obligation_overload": overload,
        "predicted_shed_fraction": float(gains @ chosen), "variables": len(original),
        "time_bins": bins, "planning_scope": "successive central-calibration temporal LPs; bounded queue/load iteration; handoff objective"}
