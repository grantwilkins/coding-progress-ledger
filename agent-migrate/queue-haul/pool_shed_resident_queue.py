"""CPU pilot of GPU-local, cache-conditioned request scheduling; not a fleet SLO model."""

import heapq
import math
from bisect import bisect_right
from collections import defaultdict

import numpy as np


def source_turns(fleet, now, cache=None):
    """Return actual started/completed counts and latest-started finish per cohort.

    The first offered turn starts at -phase, without earlier queued history.
    Cyclic histories use turn_offset; finite histories retain their first row.
    A cache belongs to fixed fleet metadata, and permits arbitrary query order.
    """
    if not math.isfinite(now):
        raise ValueError('source observation time must be finite')
    key = ('source_turn_timeline', id(fleet))
    state = None if cache is None else cache.get(key)
    if state is None:
        metadata, size = fleet.metadata, len(fleet.count)
        sequences = metadata['turn_sequences']
        cadence = metadata.get('source_session_rps', 0.)
        phases = np.asarray(metadata.get('source_phase_s', np.zeros(size)), dtype=float)
        offsets = metadata.get('turn_offset', [0] * size)
        durations = metadata.get('turn_duration_s', [[] for _ in sequences])
        cycle = metadata.get('sequence_cycle', False)
        if (len(sequences) != size or len(durations) != size or len(offsets) != size
                or not math.isfinite(cadence) or cadence < 0 or (not cadence and any(sequences))):
            raise ValueError('invalid source trace pacing or cohort counts')
        if (phases.shape != (size,) or not np.isfinite(phases).all() or np.any(phases < 0)
                or np.any(phases * cadence >= 1)):
            raise ValueError('source phases must be within one offered period')
        ordered = []
        for sequence, values, offset in zip(sequences, durations, offsets):
            if (len(values) != len(sequence) or any(not math.isfinite(v) or v <= 0 for v in values)
                    or not isinstance(offset, (int, np.integer)) or isinstance(offset, bool) or offset < 0):
                raise ValueError('source turns require aligned positive durations and integer offsets')
            values = tuple(values)
            offset = offset % len(values) if cycle and values else 0
            ordered.append(values[offset:] + values[:offset])
        state = cadence, phases.copy(), ordered, cycle, [[] for _ in sequences], [[] for _ in sequences]
        if cache is not None:
            cache[key] = state
    cadence, phases, durations, cycle, starts, finishes = state
    started, completed, finish = np.zeros(len(durations), int), np.zeros(len(durations), int), np.full(len(durations), -np.inf)
    for i, values in enumerate(durations):
        while values and (cycle or len(starts[i]) < len(values)):
            n = len(starts[i])
            begin = max(n / cadence - phases[i], finishes[i][-1] if n else -np.inf)
            if begin > now:
                break
            end = begin + values[n % len(values)]
            if not math.isfinite(end) or end <= begin:
                raise ValueError('source finish must advance finite time')
            starts[i].append(begin)
            finishes[i].append(end)
        started[i], completed[i] = bisect_right(starts[i], now), bisect_right(finishes[i], now)
        if started[i]:
            finish[i] = finishes[i][started[i] - 1]
    return started, completed, finish


def simulate(requests, coefficients, until, token_budget=8192, max_sequences=256, endpoint_before_fraction=0.):
    """Run insertion-ordered partial prefills/decodes, with FIFO waiting requests.

    Coefficients price an iteration, not whole-request serialization. Client
    overhead is applied to delivery and does not occupy the GPU. Optional
    release_s delays eligibility without erasing original-arrival latency.
    Zero-output requests finish after prefill and have no first-token latency.
    """
    if (not math.isfinite(until) or until <= 0 or not isinstance(token_budget, int)
            or isinstance(token_budget, bool) or token_budget < 1 or not isinstance(max_sequences, int)
            or isinstance(max_sequences, bool) or max_sequences < 1
            or not math.isfinite(endpoint_before_fraction) or not 0 <= endpoint_before_fraction <= 1):
        raise ValueError("positive observation horizon and integer scheduler limits required")
    names = ("prefill_step_s", "prefill_token_s", "prefill_attention_s", "decode_step_s", "decode_attention_s", "endpoint_s")
    if (any(not math.isfinite(coefficients[k]) or coefficients[k] < 0 for k in names)
            or min(coefficients["prefill_step_s"], coefficients["decode_step_s"]) <= 0):
        raise ValueError("nonnegative timing coefficients and positive iteration time required")
    grouped, placement, identities = defaultdict(list), {}, set()
    for request in requests:
        r = dict(request)
        if r["request_id"] in identities:
            raise ValueError("duplicate request ID")
        identities.add(r["request_id"])
        if (not math.isfinite(r["arrival_s"]) or r["arrival_s"] < 0
                or not math.isfinite(r.get("release_s", 0.)) or r.get("release_s", 0.) < 0
                or any(not isinstance(r[k], int) or isinstance(r[k], bool) for k in ("prompt_tokens", "cached_tokens", "output_tokens"))
                or not 0 <= r["cached_tokens"] < r["prompt_tokens"] or r["output_tokens"] < 0):
            raise ValueError("request needs known supported prompt, cache, generation and arrival")
        if placement.setdefault(r["history"], r["gpu"]) != r["gpu"]:
            raise ValueError("resident history cannot change GPU without an explicit KV migration")
        grouped[r["gpu"]].append(r)
    output, gpu_stats = [], {}
    for gpu, rows in grouped.items():
        result, stats = _gpu(rows, coefficients, until, token_budget, max_sequences, endpoint_before_fraction)
        output.extend(result)
        gpu_stats[gpu] = stats
    return {"requests": output, "gpus": gpu_stats, "resident_latency_validated": False}


def _gpu(rows, c, until, token_budget, max_sequences, endpoint_before_fraction):
    rows = sorted(rows, key=lambda r: r["arrival_s"])
    following, last, pending = {}, {}, []
    before, after = c["endpoint_s"] * endpoint_before_fraction, c["endpoint_s"] * (1 - endpoint_before_fraction)
    for i, r in enumerate(rows):
        if r["history"] in last:
            following[last[r["history"]]] = i
        else:
            heapq.heappush(pending, (max(r["arrival_s"], r.get("release_s", 0.)) + before, i))
        last[r["history"]] = i
    result = [{**r, "eligible_s": None, "admitted_s": None, "first_s": None,
               "last_token_s": None, "end_s": None, "server_end_s": None,
               "generated_tokens": 0, "computed_prompt_tokens": 0, "done": False} for r in rows]
    for eligible, i in pending:
        result[i]["eligible_s"] = eligible
    remaining = [r["prompt_tokens"] - r["cached_tokens"] for r in rows]
    generated = [0] * len(rows)
    active, waiting, now, busy, steps = [], [], 0., 0., 0
    while (pending or active or waiting) and now < until:
        while pending and pending[0][0] <= now:
            eligible, i = heapq.heappop(pending)
            result[i]["eligible_s"] = eligible
            waiting.append(i)
        if not active and not waiting:
            now = pending[0][0]
            continue
        budget, scheduled = token_budget, []
        for i in active:
            if not budget:
                break
            q = min(remaining[i], budget) if remaining[i] else 1
            scheduled.append((i, q, bool(remaining[i])))
            budget -= q
        while waiting and budget and len(active) < max_sequences:
            i = waiting.pop(0)
            active.append(i)
            result[i]["admitted_s"] = now
            q = min(remaining[i], budget)
            scheduled.append((i, q, True))
            budget -= q
        prefill = sum(q for _, q, p in scheduled if p)
        attention = sum(q * (2 * (rows[i]["prompt_tokens"] - remaining[i] + q) - q)
                        for i, q, p in scheduled if p)
        decode = sum(rows[i]["prompt_tokens"] + generated[i] for i, _, p in scheduled if not p)
        duration = (max(c["prefill_step_s"] if prefill else 0., c["decode_step_s"] if decode else 0.)
                    + c["prefill_token_s"] * prefill + c["prefill_attention_s"] * attention + c["decode_attention_s"] * decode)
        busy += min(duration, until - now)
        now += duration
        steps += 1
        if now > until:
            break
        finished = set()
        for i, q, prefill_phase in scheduled:
            if prefill_phase:
                remaining[i] -= q
                result[i]["computed_prompt_tokens"] += q
            if not remaining[i]:
                delivered = now + after
                if rows[i]["output_tokens"]:
                    generated[i] += 1
                    if delivered <= until:
                        if result[i]["first_s"] is None:
                            result[i]["first_s"] = delivered
                        result[i]["last_token_s"] = delivered
                        result[i]["generated_tokens"] += 1
                if generated[i] == rows[i]["output_tokens"]:
                    finished.add(i)
                    result[i]["server_end_s"] = now
                    if delivered <= until:
                        result[i]["end_s"], result[i]["done"] = delivered, True
                    if i in following:
                        j = following[i]
                        result[j]["eligible_s"] = max(rows[j]["arrival_s"], rows[j].get("release_s", 0.), delivered) + before
                        heapq.heappush(pending, (result[j]["eligible_s"], j))
        active = [i for i in active if i not in finished]
    for r in result:
        r["ttft_s"] = None if r["first_s"] is None else r["first_s"] - r["arrival_s"]
        r["mean_tpot_s"] = ((r["last_token_s"] - r["first_s"]) / (r["output_tokens"] - 1)
                            if r["done"] and r["output_tokens"] > 1 else None)
    return result, {"busy_s": busy, "iterations": steps, "arrivals": sum(r["arrival_s"] < until for r in rows),
                    "unfinished": sum(r["arrival_s"] < until and not r["done"] for r in result)}
