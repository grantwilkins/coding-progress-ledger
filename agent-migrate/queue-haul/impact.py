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
from power import BETA_BYTES_PER_TOK, ETA_BYTES_PER_TOK, PoolPower, congestion, rho_replay


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
    dp_expected: np.ndarray  # p̄·ℓ_j future-node proxy until raw f/g work power lands
    dp_expected_single: np.ndarray  # same single-price load proxy kept for comparisons
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

    return Impact(
        dp_guaranteed=pool.s_plat * ell,
        dp_expected=pool.p_bar * ell,
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
