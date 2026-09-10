"""Independent event execution of divisible migration batches and finite source traces."""

from __future__ import annotations

from copy import copy

import numpy as np

DISPATCH_CHUNKS = 64


def destination_gpus(fleet):
    count = fleet.metadata.get("destination_gpus", fleet.gpus)
    if not isinstance(count, (int, np.integer)) or isinstance(count, bool) or count <= 0:
        raise ValueError("destination GPU count must be a positive integer per site")
    return count


def network_nodes(fleet):
    width = getattr(fleet, "gpus_per_node", 1)
    source, sink = [(count + width - 1) // width for count in (fleet.gpus, destination_gpus(fleet))]
    return np.array([min(source, sink), min(source, sink), source])


def paced_source(fleet):
    return fleet.metadata.get("paced_source", fleet.metadata.get("protect_resident", False))


def _started_turns(fleet, now):
    phases = np.asarray(fleet.metadata.get("source_phase_s", np.zeros(len(fleet.count))))
    cadence = fleet.metadata.get("source_session_rps", 0.)
    if phases.shape != fleet.count.shape or not np.isfinite(phases).all() or np.any(phases < 0) or np.any(phases * cadence >= 1):
        raise ValueError("source phases must have one finite offset per cohort within its period")
    return np.floor((now + phases) * cadence + 1e-10).astype(int) + 1


def compute_allocation(load, replay_mass, kv_mass, gpus, loss=None, protected=False):
    mass = replay_mass + kv_mass
    sharing = min(1., gpus * (1 - load if protected else 1.) / max(mass, 1e-30))
    cost = mass if protected else (replay_mass * (1 - load * (1 - loss)) + kv_mass if loss is not None else 0.)
    return sharing, max(0., gpus - sharing * cost)


def replica_feasible(fleet, replay, kv, load):
    """Each nonempty action pack remains on its own resident-bearing replica."""
    free = (fleet.kv_capacity - fleet.baseline_kv) / fleet.gpus
    return np.logical_and.reduce([(counts @ fleet.demand <= 1 - np.asarray(load) + 1e-8)
                                  & (counts @ fleet.memory_tokens <= free + 1e-8 * max(free, 1.))
                                  for counts in (replay, kv)])


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


def source_snapshot(fleet, now, cache=None):
    key = ("snapshot", float(now))
    if cache is not None and key in cache:
        return cache[key]
    context, completed = fleet.context.copy(), np.zeros(len(fleet.count), int)
    cadence, cycle = fleet.metadata.get("source_session_rps", 0.), fleet.metadata.get("sequence_cycle", False)
    if paced_source(fleet) and cadence:
        started = _started_turns(fleet, now)
        phases = fleet.metadata.get("source_phase_s", np.zeros(len(context)))
        for i, sequence in enumerate(fleet.metadata["turn_sequences"]):
            if not sequence:
                continue
            offset = fleet.metadata.get("turn_offset", [0] * len(context))[i] if cycle else 0
            n = started[i]
            if cycle or n <= len(sequence):
                duration = fleet.metadata["turn_duration_s"][i][(offset + n - 1) % len(sequence)]
                n -= (n - 1) / cadence - phases[i] + duration > now + 1e-10
            completed[i] = n if cycle else min(n, len(sequence))
            if completed[i]:
                row = sequence[(offset + completed[i] - 1) % len(sequence)]
                context[i] = row["context"] + row["prompt"] + row["output"]
    if cache is not None:
        cache[key] = context, completed
    return context, completed


def kv_transfer_bytes(fleet, contexts, calibration):
    block = calibration["kv_block_tokens"]
    shared = np.asarray(fleet.metadata.get("kv_shared_tokens", 0))
    scale = fleet.metadata.get("kv_wire_scale", 1.)
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(shared).all() or np.any(shared < 0):
        raise ValueError("invalid KV wire scale or shared prefix")
    return np.maximum(np.floor(np.asarray(contexts) / block) - np.floor(shared / block), 0) * calibration["kv_block_bytes"] * scale


def initial_work(fleet, counts, action, route, contexts, timing, calibration, cache=None):
    contexts = fleet.context if contexts is None else contexts
    key = ("initial", counts.astype(float, copy=False).tobytes(), int(action), int(route), contexts.astype(float, copy=False).tobytes())
    if cache is not None and key in cache:
        return cache[key]
    same = contexts == fleet.context
    if action:
        result = float(counts @ np.where(same, fleet.kv, kv_transfer_bytes(fleet, contexts, calibration))), 0.
    else:
        from pool_shed_calibration import replay_seconds
        work = np.array(fleet.t1, float)
        changed = ~same & (contexts > 0)
        work[~same & (contexts == 0)] = 0.
        if changed.any():
            cached = np.broadcast_to(fleet.metadata.get("replay_cached_tokens", 0), contexts.shape)
            work[changed] = replay_seconds(contexts[changed], calibration, cached[changed])
        knots = fleet.metadata.get("packing_context_tokens")
        packing = np.interp(contexts, knots, timing["packing_kappa"]) if knots else np.full(len(counts), timing["kappa"])
        if np.any((counts > 0) & (contexts > fleet.metadata.get("batch_context_limit", np.inf))):
            packing[:] = 1.
        regional = timing.get("regional_replay_factor", calibration.get("regional_components", {}).get("replay_factor", [1., 1.]))
        result = float(counts @ np.where(same, fleet.log, 2 * contexts)), float((counts @ (packing * work) + np.max(np.where(counts > 0, (1 - packing) * work, 0.))) * regional[route])
    if cache is not None:
        cache[key] = result
    return result


def _quiesce(fleet, counts, now, cache=None, origin_turn=None):
    sequences = fleet.metadata.get("turn_sequences")
    if sequences is None:
        raise ValueError("pooled execution requires explicit finite source turn sequences")
    cadence = fleet.metadata.get("source_session_rps", 0.)
    if len(sequences) != len(fleet.count) or cadence < 0 or (not cadence and any(sequences)):
        raise ValueError("invalid source trace pacing")
    active = np.flatnonzero(counts)
    lengths = np.array([len(sequences[i]) for i in active])
    steps = np.full(len(fleet.count), int(np.ceil(max(now * cadence - 1e-10, 0))))
    cycle = fleet.metadata.get("sequence_cycle", False)
    if cycle and np.any(lengths == 0):
        raise ValueError("cannot cycle an empty source trace")
    paced = paced_source(fleet)
    if paced and cadence:
        steps = _started_turns(fleet, now)
    origin_turn = np.zeros(len(fleet.count), int) if origin_turn is None else origin_turn
    key = ("quiesce", counts.astype(float, copy=False).tobytes(), steps.tobytes(), origin_turn.tobytes())
    if cache is not None and key in cache:
        end, context, reset, terminal = cache[key]
        return max(now, end), context, reset, terminal
    completed = steps[active] if cycle else np.minimum(lengths, steps[active])
    end = 0. if paced or not cadence else float(completed.max(initial=0) / cadence)
    if paced and cadence:
        durations = fleet.metadata["turn_duration_s"]
        phases = fleet.metadata.get("source_phase_s", np.zeros(len(fleet.count)))
        for i, n, length in zip(active, completed, lengths):
            if not length or not cycle and steps[i] > length:
                continue
            offset = fleet.metadata.get("turn_offset", [0] * len(sequences))[i] if cycle else 0
            duration = durations[i][(offset + n - 1) % length]
            if not np.isfinite(duration) or not 0 <= duration <= 1 / cadence:
                raise ValueError("source request duration exceeds the paced single-request contract")
            end = max(end, (steps[i] - 1) / cadence - phases[i] + duration)
    context, reset = fleet.context.copy(), np.zeros(len(fleet.count), bool)
    for i, n in zip(active, completed):
        if n:
            offset = fleet.metadata.get("turn_offset", [0] * len(sequences))[i] if cycle else 0
            row = sequences[i][(offset + n - 1) % len(sequences[i])]
            context[i] = row["context"] + row["prompt"] + row["output"]
            first, length = int(origin_turn[i]), len(sequences[i])
            reset[i] = (cycle and (offset + n - 1) // length > (offset + max(first, 1) - 1) // length) or any(
                sequences[i][(offset + j) % length].get("reset", False) for j in range(first, min(n, first + length)))
    terminal = bool(not cycle and np.all(completed == lengths))
    if cache is not None:
        cache[key] = end, context, reset, terminal
    return max(now, end), context, reset, terminal


def _buffered(fleet, counts, start, end, calibration, cache=None, quiescing=False):
    cadence = fleet.metadata.get("source_session_rps", 0.)
    phases = np.asarray(fleet.metadata.get("source_phase_s", np.zeros(len(fleet.count))))
    first = _started_turns(fleet, start) if quiescing else np.ceil((start + phases) * cadence - 1e-9).astype(int)
    last = np.maximum(first, np.ceil((end + phases) * cadence - 1e-9).astype(int))
    if cache is not None:
        key = ("buffer", counts.astype(float, copy=False).tobytes(), first.tobytes(), last.tobytes())
        if key not in cache:
            cache[key] = _buffered(fleet, counts, start, end, calibration, quiescing=quiescing)
        return cache[key]
    number, work = 0., 0.
    for i in np.flatnonzero(counts):
        sequence = fleet.metadata["turn_sequences"][i]
        if not sequence:
            continue
        if paced_source(fleet) and "turn_work_s" in fleet.metadata:
            values = np.asarray(fleet.metadata["turn_work_s"][i])
        elif paced_source(fleet):
            from pool_shed_calibration import service_work
            values = np.array([service_work(r["context"] + r["prompt"], r["prompt"], r["output"], calibration) for r in sequence])
        else:
            values = np.array([r["prompt"] / calibration["F"] + r["output"] / calibration["G"] for r in sequence])
        prefix = np.r_[0., np.cumsum(values)]
        if fleet.metadata.get("sequence_cycle"):
            offset = fleet.metadata.get("turn_offset", [0] * len(counts))[i]
            a, b = offset + first[i], offset + last[i]
            work += counts[i] * ((b // len(values) - a // len(values)) * prefix[-1] + prefix[b % len(values)] - prefix[a % len(values)])
            number += counts[i] * (last[i] - first[i])
        else:
            a, b = min(first[i], len(values)), min(last[i], len(values))
            work += counts[i] * (prefix[b] - prefix[a])
            number += counts[i] * (b - a)
    return number, work


def catchup(fleet, counts, action, route, context, reset, timing, calibration, cache=None, origin_context=None):
    """Primitive bytes and idle work for a newly captured source state."""
    origin_context = fleet.context if origin_context is None else origin_context
    if cache is not None:
        key = ("catchup", counts.astype(float, copy=False).tobytes(), int(action), int(route),
               context.astype(float, copy=False).tobytes(), reset.astype(bool, copy=False).tobytes(), origin_context.astype(float, copy=False).tobytes())
        if key not in cache:
            cache[key] = catchup(fleet, counts, action, route, context, reset, timing, calibration, origin_context=origin_context)
        return cache[key]
    if action == 1:
        block = calibration["kv_block_tokens"]
        sealed = np.floor(context / block) * block
        old = np.where(reset, 0, kv_transfer_bytes(fleet, origin_context, calibration))
        return (float(counts @ np.maximum(kv_transfer_bytes(fleet, context, calibration) - old, 0)),
                float(counts @ (context - sealed) / calibration["kv_tail_replay_tps"] + np.interp(counts.sum(), [0, 1, 8],
                    [0, timing["kv_completion_s"], timing["kv_batch_completion_s"]])))
    if action != 0:
        raise ValueError("unknown catch-up action")
    rebuilding = (counts > 0) & (context > 0) & (reset | (context != origin_context))
    if not rebuilding.any():
        return 0., 0.
    if np.any(context[rebuilding] > max(calibration["replay_context_tokens"])):
        raise ValueError("source replay catch-up exceeds measured context support")
    _, work = initial_work(fleet, counts * rebuilding, 0, route, context, timing, calibration)
    return float(2 * (counts * rebuilding @ context)), work


def execute_pooled(table, chosen, timing, calibration=None, chunks=1):
    """Time-share batch mass; imported serving starts at each committed handoff."""
    execution = PooledExecution(table, timing, calibration, chunks)
    execution.admit(chosen)
    execution.advance(table.deadline)
    return execution.result()


class PooledExecution:
    """Persistent pooled execution state; admissions reserve source mass permanently."""

    def __init__(self, table, timing, calibration=None, chunks=1):
        if calibration is None:
            from pool_shed_calibration import calibration as load_calibration
            calibration = load_calibration(0)
        self.table, self.timing, self.calibration, self.chunks = table, timing, calibration, chunks
        self.primitive_cache = {}  # One fixed fleet, timing and calibration for this execution.
        self.fleet = table.fleet
        self.gpus = destination_gpus(self.fleet)
        self.protected = self.fleet.metadata.get("protect_resident", False)
        self.affinity = self.fleet.metadata.get("resident_affinity", False)
        if self.affinity and (self.protected or "resident_replay_loss" not in self.timing):
            raise ValueError("resident affinity requires measured interference and unprotected migration")
        self.paced = paced_source(self.fleet)
        self.timing_load_factor = self.fleet.metadata.get("timing_load_factor", 1.)
        if not np.isfinite(self.timing_load_factor) or self.timing_load_factor <= 0:
            raise ValueError("invalid measured timing-load conversion")
        if not isinstance(self.chunks, (int, np.integer)) or isinstance(self.chunks, bool) or self.chunks < 1:
            raise ValueError("dispatch chunks must be a positive integer")
        if (self.fleet.gpus <= 0 or np.any(~np.isin(self.table.route, [0, 1]))
                or not np.isfinite(np.r_[np.ravel(self.table.load), self.table.deadline, self.table.endpoint, self.table.budgets, self.timing["beta"]]).all()
                or np.any(np.asarray(self.table.load) < 0) or np.any(np.asarray(self.table.load) > 1)
                or np.any(np.asarray(self.table.endpoint) <= 0) or np.any(np.asarray(self.table.budgets) <= 0)
                or np.any(self.table.replay < 0) or np.any(self.table.kv < 0)
                or np.any(self.table.replay != np.floor(self.table.replay)) or np.any(self.table.kv != np.floor(self.table.kv))):
            raise ValueError("invalid independent execution primitives")
        self.selected, self.mass = [], np.zeros(0)
        self.route, self.action = np.zeros(0, int), np.zeros(0, int)
        self.counts = np.zeros((0, len(self.fleet.count)))
        self.memory_tokens = self.fleet.memory_tokens
        self.free_memory = (self.fleet.kv_capacity - self.fleet.baseline_kv) * self.gpus / self.fleet.gpus
        self.demand, self.memory = self.counts @ self.fleet.demand, self.counts @ self.memory_tokens
        self.initial_load = np.broadcast_to(np.asarray(self.table.load), (2,)).astype(float)
        self.n = len(self.selected)
        self.group, self.initial_ready = np.arange(self.n) // self.chunks, np.zeros(self.n, bool)
        self.origin_time = np.zeros(self.n)
        self.state, self.remaining, self.release = np.zeros(self.n, int), np.zeros(self.n), np.zeros(self.n)
        self.tail, self.delta, self.compute_used, self.idle_work = np.zeros(self.n), np.zeros(self.n), np.zeros(2), np.zeros(2)
        self.quiesced, self.buffered, self.backlog, self.backlog_total = (np.zeros(self.n) for _ in range(4))
        self.pause_requested = np.zeros(self.n)
        self.gated = np.zeros(self.n, bool)
        self.resident_debt, self.resident_generated, self.resident_recovered = (np.zeros(2) for _ in range(3))
        self.replica_debt = np.zeros(self.n)
        self.resident_displaced, self.peak_migration_replicas = np.zeros(2), np.zeros(2)
        self.resident_loss = np.broadcast_to(np.asarray(self.timing.get("resident_replay_loss", 0.)), (2,))
        if not np.isfinite(self.resident_loss).all() or np.any((self.resident_loss < 0) | (self.resident_loss > 1)):
            raise ValueError("resident replay throughput loss must be in [0, 1]")
        self.committed, self.network_used, self.loads = np.zeros((4, len(self.fleet.count))), np.zeros(3), self.initial_load.copy()
        self.reserved = self.mass * self.memory
        self.peak_memory = np.array([self.reserved[self.route == r].sum() for r in (0, 1)])
        self.peak_load, self.events, self.exhausted = self.loads.copy(), {}, 0
        self.tail_rate = self.calibration["kv_tail_replay_tps"]
        if self.tail_rate <= 0 or self.table.deadline <= 0:
            raise ValueError("positive tail service and deadline required")
        self.nominal_beta = self.calibration.get("timing", [self.timing])[0]["beta"]
        self.phase_started, self.phase_replica_seconds, self.phase_transferred_bytes, self.phase_nominal_work = (np.zeros(self.n) for _ in range(4))
        self.now = 0.
        self.network_key, self.network_rates = None, np.zeros(self.n)
        self.selected_total = np.zeros(len(self.table.route))
        self.service_ready_s = 0.

    def admit(self, chosen):
        chosen = np.asarray(chosen, float)
        total = self.selected_total + chosen if chosen.shape == self.selected_total.shape else chosen
        if (chosen.shape != self.selected_total.shape or not np.isfinite(chosen).all() or np.any(chosen < 0)
                or np.any((self.table.replay + self.table.kv).T @ total > self.fleet.count * (1 + 1e-8) + 1e-8)):
            raise ValueError("invalid or duplicated source migration mass")
        selected = [(j, a, c) for j in np.flatnonzero(chosen > 1e-10)
                    for a, c in enumerate((self.table.replay[j], self.table.kv[j])) if c.any()
                    for _ in range(self.chunks)]
        if not selected:
            return
        mass = np.array([chosen[j] / self.chunks for j, _, _ in selected])
        counts = np.array([c for _, _, c in selected])
        route = np.array([self.table.route[j] for j, _, _ in selected], int)
        demand, memory = counts @ self.fleet.demand, counts @ self.memory_tokens
        if self.affinity and not replica_feasible(self.fleet, counts, np.zeros_like(counts), self.initial_load[route]).all():
            raise ValueError("action pack exceeds its destination replica serving or memory capacity")
        for r in (0, 1):
            ids, old = route == r, self.route == r
            if self.affinity and mass[ids].sum() + self.mass[old].sum() > self.gpus * (1 + 1e-8):
                raise ValueError("selected migrations exceed permanently reserved destination replicas")
            if (mass[ids] @ demand[ids] + self.mass[old] @ self.demand[old] > self.gpus * (1 - self.initial_load[r]) + 1e-7 * self.gpus
                    or mass[ids] @ memory[ids] + self.reserved[old].sum() > self.free_memory + 1e-7 * self.fleet.kv_capacity):
                raise ValueError("selected migrations exceed destination serving or memory capacity")
        n = len(selected)
        remaining, tail = np.zeros(n), np.zeros(n)
        for i, (_, a, c) in enumerate(selected):
            remaining[i], tail[i] = initial_work(self.fleet, c, a, route[i], None, self.timing, self.calibration, self.primitive_cache)
        values = dict(mass=mass, route=route, action=np.array([a for _, a, _ in selected], int),
                      counts=counts, demand=demand, memory=memory, reserved=mass * memory,
                      group=np.arange(self.n, self.n + n) // self.chunks, initial_ready=np.zeros(n, bool), gated=np.zeros(n, bool),
                      origin_time=np.full(n, np.nan if self.paced else 0.),
                      state=np.zeros(n, int), remaining=remaining, tail=tail, phase_started=np.full(n, self.now))
        for name in ("release", "delta", "quiesced", "pause_requested", "buffered", "backlog", "backlog_total", "phase_replica_seconds", "phase_transferred_bytes", "phase_nominal_work", "replica_debt"):
            values[name] = np.zeros(n)
        for name, value in values.items():
            setattr(self, name, np.concatenate((getattr(self, name), value)))
        self.selected.extend(selected)
        self.n += n
        self.selected_total = total.copy()
        self.peak_memory = np.maximum(self.peak_memory, [self.reserved[self.route == r].sum() for r in (0, 1)])
        self.network_key, self.network_rates = None, np.zeros(self.n)
        self.service_ready_s = None

    def serving_load(self):
        gate = (self.state == 5) & self.gated
        load = self.loads + np.array([self.mass[gate & (self.route == r)] @ self.demand[gate & (self.route == r)] for r in (0, 1)]) / self.gpus
        if self.protected and np.any(load > 1 + 1e-8):
            raise RuntimeError("handoff gate exceeded safe serving capacity")
        return np.minimum(1., load)

    def origin(self, i):
        return source_snapshot(self.fleet, self.origin_time[i], self.primitive_cache) if self.paced and np.isfinite(self.origin_time[i]) else (self.fleet.context, np.zeros(len(self.fleet.count), int))

    def buffered_work(self, i, until):
        quiescing = self.fleet.metadata.get("paced_source", False)
        return _buffered(self.fleet, self.counts[i], self.pause_requested[i] if quiescing else self.quiesced[i],
                         until, self.calibration, self.primitive_cache, quiescing=quiescing)

    def nominal_continuation(self, table, timing, calibration):
        central = PooledExecution(table, timing, calibration, self.chunks)
        clone = copy(self)
        clone.__dict__ = {name: value.copy() if isinstance(value, np.ndarray) else value for name, value in vars(self).items()}
        for name in ("table", "timing", "calibration", "fleet", "nominal_beta", "resident_loss", "tail_rate", "timing_load_factor", "primitive_cache"):
            setattr(clone, name, getattr(central, name))
        clone.events, clone.selected = {key: value.copy() for key, value in self.events.items()}, self.selected.copy()
        clone.network_key, clone.network_rates = None, np.zeros(self.n)
        clone.remaining[:], clone.tail[:], clone.delta[:] = 0., 0., 0.
        for i in np.flatnonzero(self.state < 6):
            c, a, r, phase = clone.counts[i], clone.action[i], clone.route[i], clone.state[i]
            origin_context, origin_turn = clone.origin(i)
            if phase <= 1:
                payload, clone.tail[i] = initial_work(clone.fleet, c, a, r, origin_context, timing, calibration, clone.primitive_cache)
                clone.remaining[i] = max(payload - clone.phase_transferred_bytes[i], 0.) if phase == 0 else max(clone.tail[i] - clone.phase_nominal_work[i], 0.)
            else:
                _, context, reset, _ = _quiesce(clone.fleet, c, clone.pause_requested[i], clone.primitive_cache, origin_turn=origin_turn)
                clone.delta[i], clone.tail[i] = catchup(clone.fleet, c, a, r, context, reset, timing, calibration, clone.primitive_cache, origin_context=origin_context)
                if phase == 3:
                    clone.remaining[i] = max(clone.delta[i] - clone.phase_transferred_bytes[i], 0.)
                elif phase == 4:
                    clone.remaining[i] = max(clone.tail[i] - clone.phase_nominal_work[i], 0.)
                elif phase == 5 and not clone.gated[i]:
                    clone.release[i] = clone.phase_started[i] + calibration.get("switch_s", 0.)
        return clone

    def advance(self, until, collect=False):
        if not np.isfinite(until) or until < self.now or until > self.table.deadline:
            raise ValueError("advance requires current time <= until <= deadline")
        usage = {name: np.zeros(3 if name == "network" else 2) for name in ("network", "application", "migration", "replay", "kv", "recovery", "buffer_recovery", "serving", "peak", "service_peak")} if collect else None
        iterations = 0
        while (np.any(self.state < 6) or np.any(self.backlog > 1e-9) or np.any(self.resident_debt > 1e-9)) and self.now <= until:
            iterations += 1
            if iterations > 20 * self.n + 20:
                raise RuntimeError("pooled phase execution failed to advance")
            ready = np.flatnonzero((self.state < 6) & (self.remaining <= 1e-9) & (self.release <= self.now + 1e-10)
                                  & ~(self.gated & (self.backlog > 1e-9)) & ~((self.state == 0) & ~np.isfinite(self.origin_time)))
            if len(ready):
                for i in ready:
                    self.phase_started[i], self.phase_replica_seconds[i], self.phase_transferred_bytes[i], self.phase_nominal_work[i] = self.now, 0., 0., 0.
                    a, c, r = self.action[i], self.counts[i], self.route[i]
                    if self.state[i] == 0 and a == 0:
                        self.state[i], self.remaining[i] = 1, self.tail[i]
                    elif self.state[i] in (0, 1):
                        origin_context, origin_turn = self.origin(i)
                        self.pause_requested[i] = self.now
                        self.release[i], context, reset, terminal = _quiesce(self.fleet, c, self.now, self.primitive_cache, origin_turn=origin_turn)
                        self.quiesced[i] = self.release[i]
                        self.exhausted += int(terminal)
                        required = self.mass[i] * (c @ np.maximum(self.memory_tokens, np.ceil(context / 16) * 16))
                        occupied = self.reserved[self.route == r].sum() - self.reserved[i] + required
                        if (occupied > self.free_memory + 1e-7 or self.affinity
                                and required / self.mass[i] > self.free_memory / self.gpus + 1e-7):
                            self.state[i], self.reserved[i] = 7, 0.
                            continue
                        self.reserved[i] = required
                        self.peak_memory[r] = max(self.peak_memory[r], occupied)
                        self.delta[i], self.tail[i] = catchup(self.fleet, c, a, r, context, reset, self.timing, self.calibration, self.primitive_cache, origin_context=origin_context)
                        self.state[i], self.remaining[i] = 2, 0.
                    elif self.state[i] == 2:
                        self.state[i], self.remaining[i] = 3, self.delta[i]
                    elif self.state[i] == 3:
                        self.state[i], self.remaining[i] = 4, self.tail[i]
                    elif self.state[i] == 4:
                        self.state[i], self.release[i] = 5, self.now + self.calibration.get("switch_s", 0.)
                    else:
                        if self.protected and not self.gated[i]:
                            self.gated[i] = True
                            self.buffered[i], self.backlog[i] = self.buffered_work(i, self.now)
                            self.backlog_total[i] = self.backlog[i]
                            if self.backlog[i] > 1e-9:
                                continue
                        self.state[i] = 6
                        self.committed[2 * r + a] += self.mass[i] * c
                        if self.protected and self.loads[r] + self.mass[i] * self.demand[i] / self.gpus > 1 + 1e-8:
                            raise RuntimeError("handoff exceeded safe serving capacity")
                        self.loads[r] = min(1., self.loads[r] + self.mass[i] * self.demand[i] / self.gpus)
                        if not self.protected:
                            self.buffered[i], self.backlog[i] = self.buffered_work(i, self.now)
                            self.backlog_total[i] = self.backlog[i]
                        self.peak_load[r] = max(self.peak_load[r], self.loads[r])
                        event = self.events.setdefault((int(self.selected[i][0]), int(a)), {
                            "column": int(self.selected[i][0]), "action": "replay" if a == 0 else "kv_transfer", "route": int(r),
                            "multiplicity": 0., "first_completion_s": float(self.now)})
                        event.update(multiplicity=event["multiplicity"] + float(self.mass[i]), completion_s=float(self.now),
                                     resident_debt_work_s=float(self.resident_debt[r]), source_buffer_work_s=float(self.mass[self.route == r] @ self.backlog[self.route == r]))
                continue
            rates = np.zeros(self.n)
            transfers = np.flatnonzero((self.state == 0) | (self.state == 3))
            endpoint = np.asarray(self.table.endpoint)
            application = self.timing.get("regional_kv_bytes_per_s", self.calibration.get("regional_components", {}).get("endpoint_bytes_per_s", endpoint[:2]))
            # A KV application cap is additional to the measured bulk endpoint cap.
            key = (transfers.tobytes(), self.state[transfers].tobytes())
            if len(transfers) and key != self.network_key:
                nodes = network_nodes(self.fleet)[:2]
                residual, app_residual = np.array(self.table.budgets, float), np.asarray(application) * nodes
                captured = False
                for phase in (3, 0):
                    ids = transfers[self.state[transfers] == phase]
                    if not len(ids):
                        continue
                    if phase == 0:
                        routes, actions = self.route[::self.chunks], self.action[::self.chunks]
                        caps = np.where(actions == 1, np.minimum(endpoint[routes], np.asarray(application)[routes]), endpoint[routes])
                        pending = np.bincount(self.group[ids], weights=self.mass[ids], minlength=self.n // self.chunks)
                        virtual = _flow_with_caps(pending, routes, caps, residual, actions == 1, app_residual / nodes, nodes)
                        active = ids[self.initial_ready[ids]]
                        admitted = np.bincount(self.group[active], weights=self.mass[active], minlength=self.n // self.chunks)
                        needed = np.ceil(np.maximum(pending * virtual / caps - admitted, 0.) / self.mass[::self.chunks] - 1e-12).astype(int)
                        for g in np.flatnonzero(needed > 0):
                            start = g * self.chunks
                            waiting = np.flatnonzero((self.state[start:start + self.chunks] == 0) & ~self.initial_ready[start:start + self.chunks]) + start
                            self.initial_ready[waiting[:needed[g]]] = True
                        ids = ids[self.initial_ready[ids]]
                        if not len(ids):
                            continue
                        for i in ids[~np.isfinite(self.origin_time[ids])]:
                            self.origin_time[i] = self.now
                            context, _ = self.origin(i)
                            self.remaining[i], self.tail[i] = initial_work(self.fleet, self.counts[i], self.action[i], self.route[i], context, self.timing, self.calibration, self.primitive_cache)
                            captured = True
                        if captured:
                            break
                    caps = np.where(self.action[ids] == 1, np.minimum(endpoint[self.route[ids]], np.asarray(application)[self.route[ids]]), endpoint[self.route[ids]])
                    rates[ids] = _flow_with_caps(self.mass[ids], self.route[ids], caps, residual, self.action[ids] == 1, app_residual / nodes, nodes)
                    used = self.mass[ids] * rates[ids]
                    residual = np.maximum(residual - np.r_[[used[self.route[ids] == r].sum() for r in (0, 1)], used.sum()], 0.)
                    app_residual = np.maximum(app_residual - [used[(self.route[ids] == r) & (self.action[ids] == 1)].sum() for r in (0, 1)], 0.)
                if captured:
                    self.network_key = None
                    continue
                self.network_key, self.network_rates = key, rates.copy()
            rates[transfers] = self.network_rates[transfers]
            computing = (self.state == 1) | (self.state == 4)
            backlog_rates = np.zeros(self.n)
            resident_growth, resident_recovery, resident_displacement, compute_shares = (np.zeros(2) for _ in range(4))
            local_growth, local_recovery = np.zeros(self.n), np.zeros(self.n)
            nominal_rates = np.zeros(2)
            serving_load = self.serving_load()
            for r in (0, 1):
                active = computing & (self.route == r)
                queued = (self.backlog > 1e-9) & (self.route == r)
                load = serving_load[r]
                if self.affinity:
                    local_load = self.initial_load[r] + self.demand * (self.state == 6)
                    headroom = np.maximum(1 - local_load, 0.)
                    recovering = (self.route == r) & ~computing & (self.replica_debt > 1e-9)
                    local_growth[active] = self.initial_load[r] * np.where(self.action[active] == 0, self.resident_loss[r], 1.)
                    local_recovery[recovering] = headroom[recovering]
                    backlog_rates[queued & ~computing] = (headroom - local_recovery)[queued & ~computing]
                    resident_growth[r] = resident_displacement[r] = float(self.mass[active] @ local_growth[active])
                    resident_recovery[r] = float(self.mass[recovering] @ local_recovery[recovering])
                    compute_shares[r] = 1.
                    self.peak_migration_replicas[r] = max(self.peak_migration_replicas[r], self.mass[active].sum())
                    self.peak_load[r] = max(self.peak_load[r], load + (resident_recovery[r] + self.mass[queued] @ backlog_rates[queued]) / self.gpus)
                    rates[active] = np.exp(-self.timing["beta"] * self.initial_load[r] * self.timing_load_factor)
                    nominal_rates[r] = np.exp(-self.nominal_beta * self.initial_load[r] * self.timing_load_factor)
                    continue
                if self.protected:
                    backlog_rates[queued] = min(1., self.gpus * (1 - load) / max(float(self.mass[queued].sum()), 1e-30))
                    load = min(1., load + float(self.mass[queued] @ backlog_rates[queued]) / self.gpus)
                replay_mass, kv_mass = [float(self.mass[active & (self.action == a)].sum()) for a in (0, 1)]
                sharing, capacity = compute_allocation(load, replay_mass, kv_mass,
                    self.gpus, self.resident_loss[r] if "resident_replay_loss" in self.timing else None, self.protected)
                compute_shares[r] = sharing
                self.peak_migration_replicas[r] = max(self.peak_migration_replicas[r], sharing * (replay_mass + kv_mass))
                if not self.protected and "resident_replay_loss" in self.timing:
                    resident_displacement[r] = sharing * load * (self.resident_loss[r] * replay_mass + kv_mass)
                if self.protected and capacity < load * self.gpus - 1e-9 * self.gpus:
                    raise RuntimeError("migration violated protected serving capacity")
                resident_growth[r] = 0. if self.protected else max(load * self.gpus - capacity, 0.)
                spare = max(capacity - load * self.gpus, 0.)
                resident_recovery[r] = spare if self.resident_debt[r] > 1e-9 else 0.
                if not self.protected:
                    backlog_rates[queued] = min(1., (spare - resident_recovery[r]) / max(float(self.mass[queued].sum()), 1e-30))
                effective_load = serving_load[r] + (resident_recovery[r] + float(self.mass[queued] @ backlog_rates[queued])) / self.gpus
                self.peak_load[r] = max(self.peak_load[r], effective_load)
                rates[active] = sharing * np.exp(-self.timing["beta"] * effective_load * self.timing_load_factor)
                nominal_rates[r] = sharing * np.exp(-self.nominal_beta * effective_load * self.timing_load_factor)
            completions = np.divide(self.remaining, rates, out=np.full(self.n, np.inf), where=rates > 0)
            waiting = ((self.state == 2) | (self.state == 5)) & (self.release > self.now)
            step = min(until - self.now, float(completions.min(initial=np.inf)),
                       float(np.min(np.divide(self.replica_debt, local_recovery, out=np.full(self.n, np.inf), where=local_recovery > 0), initial=np.inf)) if self.affinity else
                       float(np.min(np.divide(self.resident_debt, resident_recovery, out=np.full(2, np.inf), where=resident_recovery > 0))),
                       float(np.min(np.divide(self.backlog, backlog_rates, out=np.full(self.n, np.inf), where=backlog_rates > 0), initial=np.inf)),
                       float(np.min(self.release[waiting] - self.now, initial=np.inf)))
            step = min(until - self.now, max(step, float(completions[completions <= step + 1e-10].max(initial=0.))))
            if not np.isfinite(step) or step <= 0:
                break
            progress = np.where(completions <= step, self.remaining, np.minimum(self.remaining, rates * step))
            self.phase_replica_seconds[computing] += compute_shares[self.route[computing]] * step
            self.phase_nominal_work[computing] += nominal_rates[self.route[computing]] * step
            self.phase_transferred_bytes[transfers] += progress[transfers]
            sent = self.mass[transfers] * progress[transfers]
            self.network_used += [sent[self.route[transfers] == 0].sum(), sent[self.route[transfers] == 1].sum(), sent.sum()]
            self.compute_used += [compute_shares[r] * float(self.mass[computing & (self.route == r)].sum()) * step for r in (0, 1)]
            if collect:
                migration = np.array([compute_shares[r] * self.mass[computing & (self.route == r)].sum() for r in (0, 1)])
                recovery = resident_recovery + np.array([self.mass[self.route == r] @ backlog_rates[self.route == r] for r in (0, 1)])
                usage["buffer_recovery"] += (recovery - resident_recovery) * step
                ordinary = self.gpus * serving_load
                usage["network"] += [sent[self.route[transfers] == 0].sum(), sent[self.route[transfers] == 1].sum(), sent.sum()]
                usage["application"] += [sent[(self.route[transfers] == r) & (self.action[transfers] == 1)].sum() for r in (0, 1)]
                for name, value in (("migration", migration), ("recovery", recovery), ("serving", ordinary)):
                    usage[name] += value * step
                for action, name in enumerate(("replay", "kv")):
                    usage[name] += np.array([compute_shares[r] * self.mass[computing & (self.route == r) & (self.action == action)].sum() for r in (0, 1)]) * step
                usage["peak"] = np.maximum(usage["peak"], ordinary + recovery + migration)
                usage["service_peak"] = np.maximum(usage["service_peak"], ordinary + recovery)
            self.idle_work += [float(self.mass[computing & (self.route == r)] @ progress[computing & (self.route == r)]) for r in (0, 1)]
            self.resident_generated += resident_growth * step
            self.resident_displaced += resident_displacement * step
            self.resident_recovered += resident_recovery * step
            if self.affinity:
                self.replica_debt = np.maximum(self.replica_debt + (local_growth - local_recovery) * step, 0.)
                self.resident_debt = np.array([self.mass[self.route == r] @ self.replica_debt[self.route == r] for r in (0, 1)])
            else:
                self.resident_debt = np.maximum(self.resident_debt + (resident_growth - resident_recovery) * step, 0.)
            self.backlog = np.maximum(self.backlog - backlog_rates * step, 0)
            gate = (self.state == 5) & self.gated
            self.buffered[gate] += self.counts[gate].sum(1) * self.fleet.metadata.get("source_session_rps", 0.) * step
            self.backlog_total[gate] += self.demand[gate] * step
            self.remaining = np.maximum(self.remaining - progress, 0)
            self.now += step
        if np.all(self.state == 6) and np.all(self.backlog <= 1e-9) and np.all(self.resident_debt <= 1e-9) and self.service_ready_s is None:
            self.service_ready_s = float(self.now)
        if collect and until > self.now:
            ordinary = self.gpus * self.serving_load()
            usage["serving"] += ordinary * (until - self.now)
            usage["peak"], usage["service_peak"] = np.maximum(usage["peak"], ordinary), np.maximum(usage["service_peak"], ordinary)
        self.now = float(until)
        return usage

    def result(self):
        fractions, numbers = self.committed @ self.fleet.gain, self.committed.sum(1)
        events = [event.copy() for event in self.events.values()]
        gate = (self.state == 5) & self.gated
        pending_by_wave = self.mass * self.buffered * np.divide(self.backlog, self.backlog_total, out=np.zeros(self.n), where=self.backlog_total > 0)
        pending, gate_pending = float(pending_by_wave[~gate].sum()), float(pending_by_wave[gate].sum())
        gate_work = float(self.mass[gate] @ self.backlog[gate])
        source_buffered, source_work = sum((self.mass[i] * np.array(self.buffered_work(i, self.now))
                                           for i in np.flatnonzero((self.state >= 2) & (self.state < 6) & ~gate)), start=np.zeros(2))
        source_buffered, source_work = source_buffered + gate_pending, source_work + gate_work
        transferred_buffered = float(self.mass @ self.buffered) - gate_pending
        recovered = (self.state == 6) & (self.replica_debt <= 1e-9) & (self.backlog <= 1e-9)
        recovered_fractions = [float(self.mass[recovered & (2 * self.route + self.action == a)] @
                                    (self.counts[recovered & (2 * self.route + self.action == a)] @ self.fleet.gain)) for a in range(4)]
        return {"shed_fraction": float(fractions.sum()), "action_counts": numbers.tolist(),
                "resident_latency_validated": False,
                "service_recovered_by_deadline": self.service_ready_s is not None,
                "service_recovery_scope": ("resident debt and source buffers recover on their permanently assigned replica cohorts; work proxy, no TTFT/TPOT guarantee" if self.affinity else "aggregate pooled work only; no resident GPU affinity or TTFT/TPOT guarantee"),
                "action_fractions": fractions.tolist(), "completed_sessions": float(numbers.sum()),
                "last_completion_s": max((e["completion_s"] for e in events), default=0.),
                "service_ready_s": self.service_ready_s,
                "resident_debt_generated_work_s": self.resident_generated.tolist(),
                "resident_displaced_work_s": self.resident_displaced.tolist(),
                "resident_pool_compensation_work_s": np.maximum(self.resident_displaced - self.resident_generated, 0.).tolist(),
                "resident_affinity": bool(self.affinity),
                "recovered_handoff_fraction": float(sum(recovered_fractions)) if self.affinity else None,
                "recovered_action_fractions": recovered_fractions if self.affinity else None,
                "reserved_destination_replicas": [float(self.mass[self.route == r].sum()) for r in (0, 1)] if self.affinity else None,
                "resident_debt_recovered_work_s": self.resident_recovered.tolist(),
                "pending_resident_debt_work_s": self.resident_debt.tolist(),
                "batch_replica_seconds": self.compute_used.tolist(), "migration_idle_work_s": self.idle_work.tolist(),
                "peak_migration_replicas": self.peak_migration_replicas.tolist(),
                "transferred_bytes": self.network_used.tolist(),
                "peak_destination_load": self.peak_load.tolist(), "final_destination_load": self.loads.tolist(), "completion_events": events,
                "peak_reserved_kv_tokens": self.peak_memory.tolist(),
                "buffered_requests": transferred_buffered + float(source_buffered),
                "transferred_buffered_requests": transferred_buffered, "source_buffered_requests": float(source_buffered),
                "pending_buffered_requests": pending + float(source_buffered),
                "pending_destination_buffered_requests": pending,
                "completed_buffered_requests": transferred_buffered - pending,
                "backlog_reference_work_s": float(self.mass @ self.backlog_total) - gate_work, "pending_backlog_reference_work_s": float(self.mass @ self.backlog) - gate_work,
                "pending_source_buffer_work_s": float(source_work),
                "pending_buffered_work_s": float(source_work + self.mass @ self.backlog - gate_work),
                "protected_serving_load": self.serving_load().tolist(),
                "unfinished_batch_mass": float(self.mass[self.state != 6].sum()), "memory_blocked_batch_mass": float(self.mass[self.state == 7].sum()),
                "trace_exhausted_waves": self.exhausted,
                "dispatch_chunks": int(self.chunks), "dispatch_wave_count": self.n,
                "dispatch_scope": "bounded waves fill a dynamic fair-share network window independent of wave count; final deltas have priority; " + ("capture current source state at each wave's first dispatch" if self.paced else "frozen initial snapshot while source continues"),
                "execution_model": "independent_event_fluid_batch_mass_finite_trace",
                "source_pacing": ("paced recorded trajectories with contextual request-duration proxy and declared arrival phases" if self.paced else "paced recorded trajectories; explicit reset on cyclic wrap" if self.fleet.metadata.get("sequence_cycle")
                                  else "finite recorded turns at explicit equal cadence; terminal context retained"),
                "compute_scope": ("one action pack per permanently reserved replica cohort; measured slowdown uses that replica's initial resident load; no ingestion" if self.affinity else "shared safe occupancy limits migration; measured replay slowdown uses offered-reference load conversion; aggregate pool transfer, no ingestion" if self.protected else "fractional batch processor sharing; dynamic measured load factor; no ingestion"),
                "backlog_scope": ("incoming standing service remains on its action replica; local headroom repays resident debt before source buffers" if self.affinity else "ordinary safe occupancy reserved first; pre-handoff buffers use remaining capacity before migration; fresh gate arrivals use reserved demand; no handoff until buffer clears" if self.protected else "migration reduces ordinary service; spare capacity repays resident debt before source buffers; recovery utilization enters the measured migration load factor"),
                "resident_debt_scope": ("disjoint fluid replica cohorts; measured local throughput loss generates debt during compute, same-replica headroom recovers it outside compute; no cross-GPU compensation; conservative placement, not general bin packing" if self.affinity else "resident debt disallowed by the shared safe-occupancy constraint" if self.protected else "site-wide deficit after spare GPUs compensate displaced service; resident_displaced_work_s records the local throughput-loss proxy before compensation; zero ordinary service during KV response compute; network waiting consumes no ordinary service"
                                        if "resident_replay_loss" in self.timing else "reverse interference disabled in legacy primitive fixture"),
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
