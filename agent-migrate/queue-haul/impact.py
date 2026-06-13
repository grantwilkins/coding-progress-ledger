"""Per-job impact & move costs (formulation.md §Per-job impact / §Dispatch; §6 movement).

Pure per-job calculator: turns T1 prices + T2 loads into the power freed by moving each
job (ΔP_j bracket, plus the memory-regime value) and the downtime of each move primitive
(replay c_j(R), KV transfer c_j(S)). T4 selects the regime column and the action; the
one pool-level regime scalar is returned here since T3 holds both pop and pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from instance import JobPopulation
from power import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, PoolPower, congestion, rho_dest


@dataclass(frozen=True)
class Movement:
    """§6 movement params. Both queue-wait utilizations are destination-side knobs (T7)."""

    lambda_src: float = 1e9  # WAN egress link, B/s
    mu_in: float = 512e9  # host-staged ingest, B/s
    dest_prefill_util: float = 0.6  # destination prefill load (§5: ~0.4 spare → ~0.6 used)
    dest_ingest_util: float = 0.0  # ingest uncontended by construction


@dataclass(frozen=True)
class Impact:
    """Columnar; parallel arrays in pop order, plus the pool-level regime flag."""

    dp_guaranteed: np.ndarray  # s_plat·ℓ_j (guaranteed, single-price)
    dp_expected: np.ndarray  # p̄_pre·ℓ_pre + p̄_dec·ℓ_dec (expected, two-price)
    dp_expected_single: np.ndarray  # p̄·ℓ_j (the two-price-vs-single comparison)
    dp_memory: np.ndarray  # μ·T_j/E[T] (memory regime, watts)
    c_replay: np.ndarray  # c_j(R), seconds
    c_transfer: np.ndarray  # c_j(S), seconds
    b_replay: np.ndarray  # β·T_j egress bytes
    b_transfer: np.ndarray  # η·T_j egress bytes
    regime: str  # 'memory' if memory binds over the population else 'load'


def compute(pop: JobPopulation, pool: PoolPower, move: Movement = Movement()) -> Impact:
    ell = pop.ell
    phi_pre = congestion(move.dest_prefill_util)  # replay queues against destination prefill
    phi_in = congestion(move.dest_ingest_util)  # transfer queues against destination ingest
    rho = rho_dest(pop.T, pop.mfu)  # same MFU that built ℓ_pre — no desync
    eta_T = ETA_BYTES_PER_TOK * pop.T

    return Impact(
        dp_guaranteed=pool.s_plat * ell,
        dp_expected=pool.p_pre * pop.ell_pre + pool.p_dec * pop.ell_dec,
        dp_expected_single=pool.p_bar * ell,
        dp_memory=pool.mu * pop.T / pool.mean_context_tokens,
        c_replay=BETA_BYTES_PER_TOK * pop.T / move.lambda_src + (1 + phi_pre) * pop.T / rho,
        c_transfer=eta_T / move.lambda_src + (1 + phi_in) * eta_T / move.mu_in,
        b_replay=BETA_BYTES_PER_TOK * pop.T,
        b_transfer=eta_T,
        regime="memory" if pool.memory_bound(ell.sum(), len(pop)) else "load",
    )
