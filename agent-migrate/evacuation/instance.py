"""Workload + destination instance for the staged evacuation program.

Constants are transcribed from `assumptions.md`. Each job is its own class
(`n_q = 1`); the model is carried as a per-job tag for downstream grouping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KIB = 1024.0
GB = 1e9


@dataclass(frozen=True)
class Model:
    name: str
    eta_bytes_per_tok: float
    beta_bytes_per_tok: float
    job_fraction: float
    lognormal_mu: float
    lognormal_sigma: float
    prefill_anchor_T: np.ndarray
    prefill_anchor_rho: np.ndarray


_ANCHOR_T = np.array([1_000.0, 10_000.0, 100_000.0, 1_000_000.0])


MODELS: tuple[Model, ...] = (
    Model("DeepSeek V4 Pro",  9.7 * KIB, 4.0, 0.25, float(np.log(8_000)),  1.5,
          _ANCHOR_T, np.array([ 28_000.0,  25_600.0,  13_900.0,  2_500.0])),
    Model("Kimi K2.6",       68.6 * KIB, 4.0, 0.25, float(np.log(12_000)), 1.8,
          _ANCHOR_T, np.array([ 42_500.0,  36_200.0,  14_700.0,  2_100.0])),
    Model("GLM 5",           87.8 * KIB, 4.0, 0.15, float(np.log(6_000)),  1.4,
          _ANCHOR_T, np.array([ 33_600.0,  26_200.0,   8_300.0,  1_100.0])),
    Model("Qwen3 235B",     188.0 * KIB, 4.0, 0.15, float(np.log(15_000)), 1.9,
          _ANCHOR_T, np.array([ 60_800.0,  46_600.0,  14_000.0,  1_700.0])),
    Model("Qwen3.5 397B",    30.0 * KIB, 4.0, 0.15, float(np.log(20_000)), 1.7,
          _ANCHOR_T, np.array([ 80_900.0,  76_000.0,  47_300.0,  9_900.0])),
    Model("Qwen3 Next 80B",  24.0 * KIB, 4.0, 0.05, float(np.log(5_000)),  1.3,
          _ANCHOR_T, np.array([454_300.0, 396_800.0, 175_000.0, 26_600.0])),
)


@dataclass(frozen=True)
class Destination:
    name: str
    lambda_bytes_per_s: float
    warm_instances: dict[str, int]


DESTINATIONS: tuple[Destination, ...] = (
    Destination("Site A", 25.0 * GB, {
        "DeepSeek V4 Pro": 2, "Kimi K2.6": 1, "GLM 5": 1,
        "Qwen3 235B": 2, "Qwen3.5 397B": 1, "Qwen3 Next 80B": 1,
    }),
    Destination("Site B", 12.5 * GB, {
        "DeepSeek V4 Pro": 1, "Kimi K2.6": 2, "GLM 5": 1,
        "Qwen3 235B": 1, "Qwen3.5 397B": 3, "Qwen3 Next 80B": 1,
    }),
    Destination("Site C", 50.0 * GB, {
        "DeepSeek V4 Pro": 1, "Kimi K2.6": 1, "GLM 5": 2,
        "Qwen3 235B": 1, "Qwen3.5 397B": 1, "Qwen3 Next 80B": 2,
    }),
)


MU_ING_BYTES_PER_S = 512.0 * GB
T_MIN, T_MAX = 1_000.0, 1_000_000.0
TOTAL_JOBS_DEFAULT = 10_000
SEED_DEFAULT = 42
D_DEFAULT_S = 300.0


@dataclass(frozen=True)
class ProblemInstance:
    model_idx: np.ndarray  # (Q,) int, model index per job
    T: np.ndarray          # (Q,) float, token length per job
    beta: np.ndarray       # (Q,) float, B/tok (context)
    eta: np.ndarray        # (Q,) float, B/tok (KV)
    rho: np.ndarray        # (Q,) float, prefill tok/s at T_q
    n: np.ndarray          # (Q,) float, jobs per class
    lambda_bps: np.ndarray # (L,) float
    W: np.ndarray          # (L, M) int, warm instances
    mu_ing: float          # bytes/s per warm instance
    D: float               # deadline seconds
    M_names: tuple[str, ...]
    L_names: tuple[str, ...]


def _log_interp(T: np.ndarray, anchor_T: np.ndarray, anchor_rho: np.ndarray) -> np.ndarray:
    return np.exp(np.interp(np.log(T), np.log(anchor_T), np.log(anchor_rho)))


def build_instance(D: float = D_DEFAULT_S,
                   total_jobs: int = TOTAL_JOBS_DEFAULT,
                   seed: int = SEED_DEFAULT) -> ProblemInstance:
    rng = np.random.default_rng(seed)
    counts = np.array([round(total_jobs * m.job_fraction) for m in MODELS], dtype=int)
    Q = int(counts.sum())
    model_idx = np.repeat(np.arange(len(MODELS)), counts)

    T = np.empty(Q)
    rho = np.empty(Q)
    for m_i, m in enumerate(MODELS):
        mask = model_idx == m_i
        T[mask] = np.clip(rng.lognormal(m.lognormal_mu, m.lognormal_sigma, counts[m_i]),
                          T_MIN, T_MAX)
        rho[mask] = _log_interp(T[mask], m.prefill_anchor_T, m.prefill_anchor_rho)

    eta = np.array([MODELS[i].eta_bytes_per_tok for i in model_idx])
    beta = np.array([MODELS[i].beta_bytes_per_tok for i in model_idx])

    M_names = tuple(m.name for m in MODELS)
    L_names = tuple(d.name for d in DESTINATIONS)
    lambda_bps = np.array([d.lambda_bytes_per_s for d in DESTINATIONS])
    W = np.array([[d.warm_instances[name] for name in M_names] for d in DESTINATIONS],
                 dtype=float)

    return ProblemInstance(
        model_idx=model_idx, T=T, beta=beta, eta=eta, rho=rho, n=np.ones(Q),
        lambda_bps=lambda_bps, W=W, mu_ing=MU_ING_BYTES_PER_S, D=float(D),
        M_names=M_names, L_names=L_names,
    )
