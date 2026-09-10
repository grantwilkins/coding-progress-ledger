"""CPU pilot of GPU-local, cache-conditioned request scheduling; not a fleet SLO model."""

import heapq
import math
from collections import defaultdict


def simulate(requests, coefficients, until, token_budget=8192, max_sequences=256, endpoint_before_fraction=0.):
    """Run insertion-ordered partial prefills/decodes, with FIFO waiting requests.

    Coefficients price an iteration, not whole-request serialization. Client
    overhead is applied to token delivery and does not occupy the GPU.
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
                or any(not isinstance(r[k], int) or isinstance(r[k], bool) for k in ("prompt_tokens", "cached_tokens", "output_tokens"))
                or not 0 <= r["cached_tokens"] < r["prompt_tokens"] or r["output_tokens"] < 1):
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
            heapq.heappush(pending, (r["arrival_s"] + before, i))
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
                generated[i] += 1
                delivered = now + after
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
                        result[j]["eligible_s"] = max(rows[j]["arrival_s"], delivered) + before
                        heapq.heappush(pending, (result[j]["eligible_s"], j))
        active = [i for i in active if i not in finished]
    for r in result:
        r["ttft_s"] = None if r["first_s"] is None else r["first_s"] - r["arrival_s"]
        r["mean_tpot_s"] = ((r["last_token_s"] - r["first_s"]) / (r["output_tokens"] - 1)
                            if r["done"] and r["output_tokens"] > 1 else None)
    return result, {"busy_s": busy, "iterations": steps, "arrivals": sum(r["arrival_s"] < until for r in rows),
                    "unfinished": sum(r["arrival_s"] < until and not r["done"] for r in result)}
