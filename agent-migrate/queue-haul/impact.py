"""Per-job impact & move costs (formulation.md §Per-job impact / §Dispatch; §6 movement).

Pure per-job calculator: turns T1 prices + T2 loads into the power freed by moving each
job (active-power certificate, future-node proxy, memory diagnostic) and the downtime of
each move primitive (replay c_j(R), KV transfer c_j(S)). Memory is a constraint/diagnostic,
not a certified-watt source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from instance import JobPopulation
from power import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, PoolPower, congestion, rho_replay


@dataclass(frozen=True)
class Movement:
    """§6 movement params. Both queue-wait utilizations are destination-side knobs (T7)."""

    lambda_src: float = 1e9  # WAN egress link, B/s
    mu_in: float = 512e9  # host-staged ingest, B/s
    dest_prefill_util: float = 0.6  # destination prefill load (§5: ~0.4 spare → ~0.6 used)
    dest_ingest_util: float = 0.0  # ingest uncontended by construction
    alpha_in: float = 0.10  # DES-only prefill slowdown per unit ingest-channel utilization.
    # Conservative center between ~0 (pure-DMA copy engines) and ~6% (SM-copy path, vLLM
    # KV-offloading blog 2026-01) / worse for sync loading; assumes async + uncompressed KV
    # (see formulation.md). Track 1 measures the real value.

    def __post_init__(self):
        if self.lambda_src <= 0:
            raise ValueError("lambda_src must be positive")
        if self.mu_in <= 0:
            raise ValueError("mu_in must be positive")
        for name in ("dest_prefill_util", "dest_ingest_util", "alpha_in"):
            v = getattr(self, name)
            if not 0 <= v < 1:
                raise ValueError(f"{name} must be in [0, 1)")


@dataclass(frozen=True)
class Impact:
    """Columnar; parallel arrays in pop order, plus the pool-level regime flag."""

    dp_guaranteed: np.ndarray  # s_plat·ℓ_j (guaranteed, single-price)
    dp_certified: np.ndarray  # fixed-node load slope + measured token work; no memory credit
    dp_expected: np.ndarray  # base·ℓ_j + c_pre·f_j + c_dec·g_j future-node estimate
    dp_expected_single: np.ndarray  # p̄·ℓ_j single-price comparison
    dp_memory: np.ndarray  # μ·T_j/E[T] (memory regime, watts)
    c_replay: np.ndarray  # c_j(R), seconds
    c_transfer: np.ndarray  # c_j(S), seconds
    b_replay: np.ndarray  # β·T_j egress bytes
    b_transfer: np.ndarray  # η·T_j egress bytes
    regime: str  # 'memory' if memory binds over the population else 'load'


def compute(pop: JobPopulation, pool: PoolPower, move: Movement = Movement()) -> Impact:
    ell = pop.ell
    active = pop.state == "active"
    cold_discount = np.where(pop.state == "cold", 1 / (1 + pool.gamma), 1.0)
    phi_pre = congestion(move.dest_prefill_util)  # replay queues against destination prefill
    phi_in = congestion(move.dest_ingest_util)  # transfer queues against destination ingest
    rho = rho_replay(pop.T, pop.mfu)
    eta_T = ETA_BYTES_PER_TOK * pop.T
    work_power = pool.c_prefill_j_per_tok * pop.f + pool.c_decode_j_per_tok * pop.g

    return Impact(
        dp_guaranteed=pool.s_plat * ell,
        dp_certified=pool.s_plat * ell + work_power,
        dp_expected=pool.base_w_per_load * ell + work_power,
        dp_expected_single=pool.p_bar * ell,
        dp_memory=pool.mu * cold_discount * pop.T / pool.mean_context_tokens,
        c_replay=active * (BETA_BYTES_PER_TOK * pop.T / move.lambda_src + (1 + phi_pre) * pop.T / rho),
        c_transfer=active * (eta_T / move.lambda_src + (1 + phi_in) * eta_T / move.mu_in),
        b_replay=BETA_BYTES_PER_TOK * pop.T,
        b_transfer=eta_T,
        regime="memory" if pool.memory_bound(ell.sum(), len(pop)) else "load",
    )


def move_costs(pop: JobPopulation, fleet, move: Movement = Movement()):
    """Per-(job, destination) move costs for the multi-dest LP: replay c_R, transfer c_S,
    prefill node-seconds reb — each (n, K). ρ_ℓ uses each destination's own MFU (decoupled
    from pop.mfu, the source's), and φ_pre,ℓ its own prefill load. Transfer is destination-
    independent (μ_in, φ_in shared) and just broadcasts over ℓ. K=1 from DestFleet.from_event
    reproduces compute()'s c_replay/c_transfer exactly."""
    T = pop.T[:, None]
    ones = np.ones(len(fleet))
    active = (pop.state == "active")[:, None]
    rho = rho_replay(T, np.asarray(fleet.mfu))  # (n, K)
    eta_T = ETA_BYTES_PER_TOK * T
    c_R = active * (BETA_BYTES_PER_TOK * T / move.lambda_src + (1 + congestion(np.asarray(fleet.prefill_util))) * T / rho)
    c_S = active * (eta_T / move.lambda_src + (1 + congestion(move.dest_ingest_util)) * eta_T / move.mu_in) * ones
    return c_R, c_S, T / rho
