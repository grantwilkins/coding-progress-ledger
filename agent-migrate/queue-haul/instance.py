"""Job generator (formulation.md §Job model; values from assumptions.md §1/§3).

Samples a session population: per-job context T, state, class, and the two load
components ℓ_pre, ℓ_dec kept separate, plus KV footprint m. Prefill load is normalized
per-job by ρ_dest(T_j); decode by the constant precision-keyed G.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from power import CAP_BF16_GB, CAP_FP8_GB, ETA_BYTES_PER_TOK, PoolPower, rho_dest


@dataclass(frozen=True)
class Workload:
    """The §1/§3 workload knobs; center defaults, every field a sweep axis."""

    state_mix: tuple = (0.30, 0.25, 0.45)  # active / idle-warm / cold
    agentic_frac: float = 0.5
    reasoning_frac: float = 0.3  # sub-fraction of agentic with elevated Y
    rate_agentic: float = 0.15
    rate_chat: float = 0.02
    rate_sigma: float = 0.3  # mean-preserving LogN spread on turn rate
    delta_agentic: tuple = (8.0, 1.0)  # input tok/turn LogN(μ,σ)
    delta_chat: tuple = (5.5, 1.0)
    y_agentic: float = 600.0  # output tok/turn Geometric means
    y_chat: float = 800.0
    y_reasoning: float = 4000.0
    g_bf16: float = 4600.0  # decode tok/s ceiling (precision-keyed)
    g_fp8: float = 9200.0
    mfu: float = 0.35  # drives ρ_dest
    t_mix: tuple = ((0.70, 10.07, 1.0), (0.30, 11.45, 0.8))  # (weight, μ, σ)
    t_clip: tuple = (1e3, 1e6)
    alpha: float = 1.2  # load factor: n_jobs = α·N_nodes·S_node


@dataclass(frozen=True)
class JobPopulation:
    """Columnar; parallel arrays of length N."""

    klass: np.ndarray  # 'agentic' | 'chat'
    state: np.ndarray  # 'active' | 'idle' | 'cold'
    is_reasoning: np.ndarray  # bool
    T: np.ndarray  # context tokens
    ell_pre: np.ndarray  # f_j / ρ_dest(T_j)
    ell_dec: np.ndarray  # g_j / G
    m: np.ndarray  # η·T_j (bytes)
    precision: str

    @property
    def ell(self) -> np.ndarray:
        return self.ell_pre + self.ell_dec

    def __len__(self) -> int:
        return len(self.T)


def _draw(rng, n: int, wl: Workload, precision: str) -> JobPopulation:
    """Sample n jobs (no sizing logic) — the testable unit."""
    agentic = rng.random(n) < wl.agentic_frac
    klass = np.where(agentic, "agentic", "chat")
    state = rng.choice(("active", "idle", "cold"), size=n, p=wl.state_mix)
    is_reasoning = agentic & (rng.random(n) < wl.reasoning_frac)

    comp = rng.choice(len(wl.t_mix), size=n, p=[c[0] for c in wl.t_mix])
    t_mu = np.array([wl.t_mix[k][1] for k in comp])
    t_sig = np.array([wl.t_mix[k][2] for k in comp])
    T = np.clip(rng.lognormal(t_mu, t_sig), *wl.t_clip)

    rate_mean = np.where(agentic, wl.rate_agentic, wl.rate_chat)
    rate = rng.lognormal(np.log(rate_mean) - wl.rate_sigma**2 / 2, wl.rate_sigma)
    d_mu = np.where(agentic, wl.delta_agentic[0], wl.delta_chat[0])
    d_sig = np.where(agentic, wl.delta_agentic[1], wl.delta_chat[1])
    delta = rng.lognormal(d_mu, d_sig)
    y_mean = np.where(is_reasoning, wl.y_reasoning, np.where(agentic, wl.y_agentic, wl.y_chat))
    Y = rng.geometric(1.0 / y_mean)

    active = state == "active"
    ell_pre = (rate * delta * active) / rho_dest(T, wl.mfu)
    ell_dec = (rate * Y * active) / (wl.g_fp8 if precision == "fp8" else wl.g_bf16)
    m = ETA_BYTES_PER_TOK * T
    return JobPopulation(klass, state, is_reasoning, T, ell_pre, ell_dec, m, precision)


def _mean_T(wl: Workload) -> float:
    """Analytic (unclipped) E[T] of the §1 mixture."""
    return sum(w * np.exp(mu + s**2 / 2) for w, mu, s in wl.t_mix)


def generate(pool: PoolPower, wl: Workload = Workload(), n_nodes: int = 32, seed: int = 42) -> JobPopulation:
    """Draw exactly α·N_nodes·S_node jobs against a fixed pool; regime is then measured.

    The caller keeps pool.mean_context_tokens in sync with wl's E[T] (both move together
    across the context sweep), so S_node and the drawn KV footprints stay consistent.
    """
    precision = {CAP_BF16_GB: "bf16", CAP_FP8_GB: "fp8"}.get(pool.cap_gb)
    if precision is None:
        raise ValueError(f"cap_gb={pool.cap_gb} is neither BF16 nor FP8; decode price G is undefined")
    if not 0.5 < _mean_T(wl) / pool.mean_context_tokens < 2.0:
        raise ValueError("pool.mean_context_tokens must track the workload's E[T] (sweep them together)")
    n_jobs = round(wl.alpha * n_nodes * pool.s_node)
    if n_jobs < 1:
        raise ValueError(f"n_jobs={n_jobs} < 1: α·N·S_node too small")
    return _draw(np.random.default_rng(seed), n_jobs, wl, precision)
