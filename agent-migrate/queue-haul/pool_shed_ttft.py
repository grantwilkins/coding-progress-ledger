"""Fixed resident offers and a descriptive post-plan request-latency screen."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pool_shed_resident_queue import simulate


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def resident_templates(fleet, load, seed, until, baseline_s=60):
    """Generate one GPU's eight histories on [0, until), migration at baseline_s.

    The initial selected context is prewarmed; subsequent cache is only the
    same history's completed retained prefix. A reset or cycle wrap is cold.
    """
    if (not math.isfinite(load) or load < 0 or not math.isfinite(until) or until <= 0
            or not math.isfinite(baseline_s) or not 0 <= baseline_s < until
            or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
        raise ValueError("finite nonnegative load, valid baseline/horizon and integer seed required")
    counts = np.asarray(fleet.count, dtype=float)
    sequences, offsets = fleet.metadata["turn_sequences"], fleet.metadata["turn_offset"]
    reference = float(fleet.metadata["reference_rps"])
    cycle = fleet.metadata["sequence_cycle"]
    if (counts.ndim != 1 or len(counts) != len(sequences) or len(offsets) != len(counts)
            or not np.isfinite(counts).all() or np.any(counts < 0) or counts.sum() <= 0
            or not math.isfinite(reference) or reference <= 0 or not isinstance(cycle, bool)):
        raise ValueError("invalid resident fleet weights, histories or reference rate")
    normalized = []
    for sequence, offset in zip(sequences, offsets):
        if not sequence or not isinstance(offset, (int, np.integer)) or isinstance(offset, bool) or not 0 <= offset < len(sequence):
            raise ValueError("resident histories require nonempty sequences and valid offsets")
        turns = []
        for row in sequence:
            if (any(isinstance(row[k], bool) or not math.isfinite(row[k]) or row[k] < 0 or int(row[k]) != row[k]
                    for k in ("context", "prompt", "output")) or not isinstance(row["reset"], bool)
                    or row["context"] + row["prompt"] <= 0):
                raise ValueError("resident turns require supported integer token counts and explicit reset")
            turns.append({**{k: int(row[k]) for k in ("context", "prompt", "output")}, "reset": row["reset"]})
        normalized.append(turns)
    inputs = dict(count=counts.tolist(), turn_sequences=normalized, turn_offset=[int(v) for v in offsets],
                  sequence_cycle=cycle, reference_rps=reference)
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(counts), 8, p=counts / counts.sum()).tolist()
    cadence = load * reference / 8
    phases = ((rng.permutation(8) + rng.random(8)) / (8 * cadence)).tolist() if cadence else [None] * 8
    requests = []
    for history, (cohort, phase) in enumerate(zip(selected, phases)):
        sequence, offset, retained, n = normalized[cohort], int(offsets[cohort]), None, 0
        while cadence and phase + n / cadence < until and (cycle or offset + n < len(sequence)):
            index = (offset + n) % len(sequence)
            turn = sequence[index]
            prompt = turn["context"] + turn["prompt"]
            reset = bool(n and (index == 0 or turn["reset"]))
            prefix = turn["context"] if n == 0 else 0 if reset else retained
            if prefix > prompt:
                raise ValueError("retained history exceeds the next prompt without a reset")
            cached = (min(prefix, prompt - 1) // 16) * 16
            requests.append(dict(request_id=f"resident-{seed}-{history}-{n}", gpu="destination",
                history=f"resident-{seed}-{history}", cohort="resident", source_cohort=cohort,
                turn=n, recorded_turn=index, arrival_s=float(phase + n / cadence),
                prompt_tokens=prompt, cached_tokens=cached, output_tokens=turn["output"],
                reset=reset, initially_prewarmed=n == 0,
                cache_basis="initial_warm_prefix" if n == 0 else "reset_cold" if reset else "completed_self_history"))
            retained, n = prompt + turn["output"], n + 1
    requests.sort(key=lambda r: r["arrival_s"])
    metadata = dict(seed=seed, load=load, baseline_s=baseline_s, until_s=until,
        histories=8, reference_rps=reference, offered_rps=8 * cadence, per_history_rps=cadence,
        observed_calendar_rps=len(requests) / until, request_count=len(requests),
        zero_output_requests=sum(r["output_tokens"] == 0 for r in requests),
        baseline_request_count=sum(r["arrival_s"] < baseline_s for r in requests),
        selected_source_cohorts=selected, selected_turn_offsets=[int(offsets[i]) for i in selected],
        phase_s=phases, input_sha256={"fleet_contract": _sha(inputs)}, requests_sha256=_sha(requests),
        scope="One GPU, eight weighted recorded histories, independent seeded stratified offered phases and fixed per-history cadence. Starts idle with each initial prefix warm; a finite baseline develops causal queues and is not a stationary fleet-state sample. Self-prefix reuse is block16, without eviction or cross-history sharing; resets and cycle wraps are cold. Planned outputs determine service and future retained lengths, usable only after modeled predecessor completion. Nominal load uses the fleet reference-work denominator, not observed utilization. Zero-output turns are prefill-only and have no TTFT.")
    return dict(requests=requests, metadata=metadata)


def _window(rows, start, end, observation_end=None):
    observed = end if observation_end is None else observation_end
    selected = [r for r in rows if start <= r["arrival_s"] < end]
    output = [r for r in selected if r["output_tokens"] > 0]
    first = [r for r in output if r["first_s"] is not None and r["first_s"] <= observed]
    missing = [r for r in output if r["first_s"] is None or r["first_s"] > observed]
    complete = [r for r in selected if r["end_s"] is not None and r["end_s"] <= observed]
    tpot = [r["mean_tpot_s"] for r in complete if r["output_tokens"] > 1]
    violations = sum(r["ttft_s"] > 1 for r in first) + sum(observed - r["arrival_s"] > 1 for r in missing)
    unresolved = sum(observed - r["arrival_s"] <= 1 for r in missing)
    latency = [r["ttft_s"] for r in first]
    lower = latency + [observed - r["arrival_s"] for r in missing]
    multi = sum(r["output_tokens"] > 1 for r in selected)
    quantiles = lambda values: {f"p{int(p * 100)}": float(np.quantile(values, p)) if values else None for p in (.5, .9, .99)}
    return dict(arrivals=len(selected), output_requests=len(output), zero_output_requests=len(selected) - len(output),
        completed=len(complete), unfinished=len(selected) - len(complete),
        first_tokens_observed=len(first), ttft_right_censored=len(missing),
        ttft_s=quantiles(latency), ttft_lower_bound_s=quantiles(lower), known_ttft_violations=violations,
        ttft_unresolved=unresolved, ttft_violation_fraction_lower=violations / len(output) if output else None,
        ttft_violation_fraction_upper=(violations + unresolved) / len(output) if output else None,
        multi_output_requests=multi, tpot_completed_requests=len(tpot), tpot_right_censored=multi - len(tpot),
        tpot_s=quantiles(tpot), known_tpot_violations=sum(v > .1 for v in tpot),
        not_yet_eligible=sum(r["eligible_s"] is None or r["eligible_s"] > observed for r in selected),
        eligible_not_admitted=sum(r["eligible_s"] is not None and r["eligible_s"] <= observed
                                 and (r["admitted_s"] is None or r["admitted_s"] > observed) for r in selected))


def check_requests(requests, coefficients, windows, until):
    """Windows are (arrival_start, arrival_end[, observation_end])."""
    if not isinstance(windows, dict) or not windows:
        raise ValueError("named nonempty observation windows required")
    bounds = {}
    for name, values in windows.items():
        if len(values) not in (2, 3):
            raise ValueError("windows require arrival start/end and optional observation end")
        start, end = values[:2]
        observed = end if len(values) == 2 else values[2]
        if any(not math.isfinite(v) for v in (start, end, observed)) or not 0 <= start < end <= observed <= until:
            raise ValueError("windows must lie within the observation horizon")
        bounds[name] = start, end, observed
    rows = [dict(r) for r in requests]
    result = simulate(rows, coefficients, until)
    cohorts = sorted({r.get("cohort", "unspecified") for r in rows})
    result["windows"] = {name: dict(start_s=start, end_s=end, observation_end_s=observed,
        all=_window(result["requests"], start, end, observed),
        cohorts={cohort: _window([r for r in result["requests"] if r.get("cohort", "unspecified") == cohort], start, end, observed)
                 for cohort in cohorts}) for name, (start, end, observed) in bounds.items()}
    result["metadata"] = dict(until_s=until, request_count=len(rows), requests_sha256=_sha(rows),
        coefficients_sha256=_sha(coefficients), ttft_target_s=1., tpot_target_s=.1,
        input_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                      for path in (Path(__file__), Path(__file__).with_name("pool_shed_resident_queue.py"))},
        scope="Post-plan descriptive latency screen, not joint admission or a validated fleet SLO guarantee. Windows select original offered arrivals and censor at their explicit observation end (default: arrival-window end). Optional release_s and modeled predecessor completion delay eligibility without erasing offered-to-first-token waiting. TTFT quantiles use observed first tokens; separate lower bounds retain requests without a first token. TPOT uses complete multi-output requests only, with censor counts. Zero-output turns count as service but have no TTFT/TPOT. No measured future dispatch, output timing or cache observations are required.")
    return result
