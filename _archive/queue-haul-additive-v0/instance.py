"""Job generator (formulation.md §Job model; values from assumptions.md §1/§3).

Samples a session population: per-job context T, state, class, and the two load
components ℓ_pre, ℓ_dec kept separate, plus KV footprint m. Prefill load is normalized
per-job by ρ_dest(T_j); decode by the constant precision-keyed G.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from power import CAP_BF16_GB, CAP_FP8_GB, ETA_BYTES_PER_TOK, PoolPower, rho_dest


SESSION_CLASSES = ("ordinary_chat", "long_chat_code", "reasoning_chat", "agentic_tool_loop")


def _lognorm_mix_mean(t_mix: tuple) -> float:
    return sum(w * np.exp(mu + s**2 / 2) for w, mu, s in t_mix)


@dataclass(frozen=True)
class Workload:
    """The §1/§3 workload knobs; center defaults, every field a sweep axis."""

    state_mix: tuple = (0.30, 0.25, 0.45)  # active / idle-warm / cold
    class_mix: tuple = (0.0, 0.0, 0.0, 1.0)  # ordinary / long-chat / reasoning / agentic
    rate_means: tuple = (0.01, 0.01, 0.01, 0.15)
    rate_sigma: float = 0.3  # mean-preserving LogN spread on turn rate
    delta_medians: tuple = (150.0, 500.0, 500.0, 1800.0)
    delta_sigma: float = 1.0
    y_means: tuple = (300.0, 1000.0, 3000.0, 600.0)
    agentic_append_output_ratio: float = 3.0
    append_output_ratio_sigma: float = 0.5
    cache_hit: tuple = (0.99, 0.95, 0.90, 0.95)
    max_ell: float = 0.50  # per-session occupation cap; above this, turns queue behind themselves
    g_bf16: float = 4600.0  # first-order constant decode tok/s; sweep for long-T sensitivity
    g_fp8: float = 9200.0
    mfu: float = 0.35  # drives ρ_dest
    t_mixes: tuple = (
        ((1.0, 8.0, 0.5),),  # ordinary chat, E[T] ≈ 3.4k
        ((1.0, 9.5, 0.7),),  # long chat / code help, E[T] ≈ 17k
        ((1.0, 9.7, 0.8),),  # reasoning chat, E[T] ≈ 22k
        ((0.70, 10.07, 1.0), (0.30, 11.45, 0.8)),  # agentic, E[T] ≈ 66k
    )
    t_mix: tuple | None = None  # optional global override for context-sweep plots
    t_clip: tuple = (1e3, 1e6)
    occupancy: float = 1.2  # sessions held ÷ node memory capacity: n_jobs = occupancy·N_nodes·S_node


@dataclass(frozen=True)
class JobPopulation:
    """Columnar; parallel arrays of length N."""

    job_type: np.ndarray  # 'agentic' | 'chat'
    session_class: np.ndarray  # one of SESSION_CLASSES
    state: np.ndarray  # 'active' | 'idle' | 'cold'
    is_reasoning: np.ndarray  # bool
    T: np.ndarray  # context tokens
    turn_rate: np.ndarray  # active model calls/s; zero for idle/cold
    Delta: np.ndarray  # appended input tokens since cached prefix
    Y: np.ndarray  # generated tokens, including reasoning/tool-call text
    f: np.ndarray  # prefill tok/s in the current state
    g: np.ndarray  # decode tok/s in the current state
    cache_hit: np.ndarray  # whether active prefill uses Delta instead of full T
    ell_pre: np.ndarray  # f_j / ρ_dest(T_j)
    ell_dec: np.ndarray  # g_j / G
    m: np.ndarray  # η·T_j (bytes)
    precision: str
    mfu: float  # the MFU ρ_dest was built with; impact reads it so rebuild can't desync
    source_node: np.ndarray | None = None  # optional source-node placement for node-knee studies

    @property
    def ell(self) -> np.ndarray:
        return self.ell_pre + self.ell_dec

    def __len__(self) -> int:
        return len(self.T)


def _draw(rng, n: int, wl: Workload, precision: str) -> JobPopulation:
    """Sample n jobs (no sizing logic) — the testable unit."""
    state = rng.choice(("active", "idle", "cold"), size=n, p=_probs(wl.state_mix, "state_mix"))
    session_class = rng.choice(SESSION_CLASSES, size=n, p=_probs(wl.class_mix, "class_mix"))
    job_type = np.where(session_class == "agentic_tool_loop", "agentic", "chat")
    is_reasoning = session_class == "reasoning_chat"

    T = np.empty(n)
    for cls, mix in zip(SESSION_CLASSES, wl.t_mixes):
        sel = session_class == cls
        if sel.any():
            T[sel] = _draw_t(rng, sel.sum(), wl.t_mix or mix, wl.t_clip)

    idx = np.array([SESSION_CLASSES.index(c) for c in session_class])
    rate_mean = np.asarray(wl.rate_means)[idx]
    raw_rate = rng.lognormal(np.log(rate_mean) - wl.rate_sigma**2 / 2, wl.rate_sigma)
    y_mean = np.asarray(wl.y_means)[idx]
    Y = rng.geometric(1.0 / y_mean)
    delta = rng.lognormal(np.log(np.asarray(wl.delta_medians)[idx]), wl.delta_sigma)
    agentic = session_class == "agentic_tool_loop"
    delta[agentic] = Y[agentic] * rng.lognormal(
        np.log(wl.agentic_append_output_ratio), wl.append_output_ratio_sigma, agentic.sum()
    )
    cache_hit = rng.random(n) < np.asarray(wl.cache_hit)[idx]

    G = wl.g_fp8 if precision == "fp8" else wl.g_bf16
    work_per_turn = np.where(cache_hit, delta, T) / rho_dest(T, wl.mfu) + Y / G
    turn_rate = np.minimum(raw_rate, wl.max_ell / work_per_turn) * (state == "active")
    f = turn_rate * np.where(cache_hit, delta, T)
    g = turn_rate * Y
    ell_pre = f / rho_dest(T, wl.mfu)
    ell_dec = g / G
    m = ETA_BYTES_PER_TOK * T
    return JobPopulation(
        job_type, session_class, state, is_reasoning, T, turn_rate, delta, Y, f, g,
        cache_hit, ell_pre, ell_dec, m, precision, wl.mfu
    )


def _mean_T(wl: Workload) -> float:
    """Analytic (unclipped) E[T] of the §1 mixture."""
    if wl.t_mix is not None:
        return _lognorm_mix_mean(wl.t_mix)
    return sum(w * _lognorm_mix_mean(mix) for w, mix in zip(wl.class_mix, wl.t_mixes))


def _probs(p: tuple, name: str) -> np.ndarray:
    p = np.asarray(p, float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError(f"{name} must be nonnegative and sum to 1")
    return p


def _draw_t(rng, n: int, t_mix: tuple, t_clip: tuple) -> np.ndarray:
    comp = rng.choice(len(t_mix), size=n, p=_probs(tuple(c[0] for c in t_mix), "t_mix"))
    mu = np.array([t_mix[k][1] for k in comp])
    sig = np.array([t_mix[k][2] for k in comp])
    return np.clip(rng.lognormal(mu, sig), *t_clip)


def class_workload(session_class: str, **kwargs) -> Workload:
    """Single-class workload preset; pass state_mix=(1,0,0) for active-only plots."""
    mix = tuple(float(c == session_class) for c in SESSION_CLASSES)
    if not any(mix):
        raise ValueError(f"unknown session_class={session_class!r}")
    return replace(Workload(class_mix=mix), **kwargs)


def generate(
    pool: PoolPower, wl: Workload = Workload(), n_nodes: int = 32, seed: int = 42
) -> JobPopulation:
    """Draw exactly occupancy·N_nodes·S_node jobs against a fixed pool; regime is then measured.

    The caller keeps pool.mean_context_tokens in sync with wl's E[T] (both move together
    across the context sweep), so S_node and the drawn KV footprints stay consistent.
    """
    precision = {CAP_BF16_GB: "bf16", CAP_FP8_GB: "fp8"}.get(pool.cap_gb)
    if precision is None:
        raise ValueError(
            f"cap_gb={pool.cap_gb} is neither BF16 nor FP8; decode price G is undefined"
        )
    if not 0.5 < _mean_T(wl) / pool.mean_context_tokens < 2.0:
        raise ValueError(
            "pool.mean_context_tokens must track the workload's E[T] (sweep them together)"
        )
    n_jobs = round(wl.occupancy * n_nodes * pool.s_node)
    if n_jobs < 1:
        raise ValueError(f"n_jobs={n_jobs} < 1: occupancy·N·S_node too small")
    return _draw(np.random.default_rng(seed), n_jobs, wl, precision)
