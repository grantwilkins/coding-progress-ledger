"""Workload + destination instance for the staged evacuation program.

The source site is a single provider serving one model family (Qwen3). The mix
is specified by *token share* (fraction of served tokens per model, flagship-
primary), not by job count: job counts are derived as share / mean-session-
length, so a flagship with long sessions owns most tokens but only a minority of
jobs. The fleet is sized in HBM: every warm instance is TP=8 (8x H100 = 640 GB),
holding the model's resident BF16 weights plus KV headroom.

Constants are transcribed from `assumptions.md`. KV size eta is exact from each
model's published attention config (layers x KV-heads x head_dim x 2(K,V) x
2 B, BF16); the linear-attention layers of Qwen3-Next carry no per-token KV.
Each job is its own class (n_q = 1) unless aggregated into log-T bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import prefill

KIB = 1024.0
GB = 1e9


@dataclass(frozen=True)
class Model:
    name: str
    eta_bytes_per_tok: float   # KV-cache HBM per context token (BF16), from attention config
    beta_bytes_per_tok: float  # context bytes/tok (uint32 token ids)
    token_share: float         # share of served *tokens* (flagship-primary), summed to 1
    weight_gb: float           # resident BF16 weight HBM (all experts), GB
    lognormal_mu: float
    lognormal_sigma: float
    prefill_anchor_T: np.ndarray
    prefill_anchor_rho: np.ndarray


_ANCHOR_T = prefill.ANCHOR_T


# One provider's Qwen3 suite, ordered small -> flagship. eta is architecture-exact:
#   235B-A22B: 94 layers x 4 KV-heads x 128 x 2(K,V) x 2 B = 188 KiB/tok
#   32B dense: 64 x 8 x 128 x 2 x 2                          = 256 KiB/tok
#   30B-A3B  : 48 x 4 x 128 x 2 x 2                          = 96 KiB/tok
#   Next-80B : 12 full-attn layers x 2 x 256 x 2 x 2         = 24 KiB/tok (rest linear)
# Prefill rho is the FLOP-roofline model of prefill.py (8x H100, BF16, MFU=0.35).
def _m(name, eta_kib, share, weight_gb, median_T, sigma):
    return Model(name, eta_kib * KIB, 4.0, share, weight_gb,
                 float(np.log(median_T)), sigma, _ANCHOR_T, prefill.anchors(name))


MODELS: tuple[Model, ...] = (
    _m("Qwen3-30B-A3B",       96.0, 0.08,  60.0,  5_000, 1.3),
    _m("Qwen3-Next-80B-A3B",  24.0, 0.25, 160.0,  8_000, 1.5),
    _m("Qwen3-32B",          256.0, 0.12,  64.0,  6_000, 1.4),
    _m("Qwen3-235B-A22B",    188.0, 0.55, 470.0, 15_000, 1.9),
)

FLAGSHIP_IDX = int(np.argmax([m.weight_gb for m in MODELS]))  # Qwen3-235B-A22B


@dataclass(frozen=True)
class Destination:
    name: str
    lambda_bytes_per_s: float
    warm_instances: dict[str, int]  # TP=8 instances per model (640 GB HBM each)


# Destination warm pools are sized for each site's own steady (flagship-primary)
# traffic, with site-specific specialization to keep routing non-trivial.
DESTINATIONS: tuple[Destination, ...] = (
    Destination("Site A", 25.0 * GB, {
        "Qwen3-30B-A3B": 1, "Qwen3-Next-80B-A3B": 2, "Qwen3-32B": 1, "Qwen3-235B-A22B": 3,
    }),
    Destination("Site B", 12.5 * GB, {
        "Qwen3-30B-A3B": 1, "Qwen3-Next-80B-A3B": 1, "Qwen3-32B": 2, "Qwen3-235B-A22B": 2,
    }),
    Destination("Site C", 50.0 * GB, {
        "Qwen3-30B-A3B": 2, "Qwen3-Next-80B-A3B": 2, "Qwen3-32B": 1, "Qwen3-235B-A22B": 4,
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
    d_miss: float = 0.0    # seconds, unmoved-job reconstruction penalty (Section 2.4)


def _log_interp(T: np.ndarray, anchor_T: np.ndarray, anchor_rho: np.ndarray) -> np.ndarray:
    return np.exp(np.interp(np.log(T), np.log(anchor_T), np.log(anchor_rho)))


def model_token_shares(flagship_share: float | None = None) -> np.ndarray:
    """Per-model token share. With flagship_share set, the flagship is pinned to
    it and the rest are rescaled proportionally (the sensitivity sweep axis)."""
    s = np.array([m.token_share for m in MODELS])
    if flagship_share is not None:
        rest = s.copy()
        rest[FLAGSHIP_IDX] = 0.0
        s = rest * (1.0 - flagship_share) / rest.sum()
        s[FLAGSHIP_IDX] = flagship_share
    return s / s.sum()


def _job_counts(total_jobs: int, shares: np.ndarray, sigma_scale: float, seed: int) -> np.ndarray:
    """Jobs per model from token share: n_m proportional to share_m / mean_T_m.
    mean_T_m is the *clipped* log-normal mean (estimated from a deterministic
    probe so the realized token share matches `shares`, not the analytic mean,
    whose heavy tail is truncated at T_MAX). Tokens, not jobs, are the mix."""
    prng = np.random.default_rng([seed, 7])
    mean_T = np.array([np.clip(prng.lognormal(m.lognormal_mu, m.lognormal_sigma * sigma_scale,
                                               100_000), T_MIN, T_MAX).mean()
                       for m in MODELS])
    w = shares / mean_T
    return np.array([round(total_jobs * wi / w.sum()) for wi in w], dtype=int)


def build_instance(D: float = D_DEFAULT_S,
                   total_jobs: int = TOTAL_JOBS_DEFAULT,
                   seed: int = SEED_DEFAULT,
                   d_miss: float | None = None,
                   rho_scale: float = 1.0,
                   sigma_scale: float = 1.0,
                   lambda_scale: float = 1.0,
                   flagship_share: float | None = None,
                   W: np.ndarray | None = None,
                   n_bins: int | None = None,
                   n_dest: int | None = None) -> ProblemInstance:
    rng = np.random.default_rng(seed)
    counts = _job_counts(total_jobs, model_token_shares(flagship_share), sigma_scale, seed)
    Q = int(counts.sum())
    model_idx = np.repeat(np.arange(len(MODELS)), counts)

    T = np.empty(Q)
    for m_i, m in enumerate(MODELS):
        mask = model_idx == m_i
        T[mask] = np.clip(rng.lognormal(m.lognormal_mu, m.lognormal_sigma * sigma_scale,
                                        counts[m_i]), T_MIN, T_MAX)

    # Optional aggregation: merge jobs of one model in a log-T bin into a class
    # with n_q = count and T = bin mean (exact for T-linear loads). Decouples
    # solve cost from total_jobs, since Q <= len(MODELS) * n_bins.
    if n_bins:
        edges = np.logspace(np.log10(T_MIN), np.log10(T_MAX), n_bins + 1)
        mi, Tc, nc = [], [], []
        for m_i in range(len(MODELS)):
            Tm = T[model_idx == m_i]
            b = np.clip(np.digitize(Tm, edges) - 1, 0, n_bins - 1)
            for k in range(n_bins):
                sel = Tm[b == k]
                if sel.size:
                    mi.append(m_i); Tc.append(sel.mean()); nc.append(sel.size)
        model_idx, T, n = np.array(mi), np.array(Tc), np.array(nc, float)
    else:
        n = np.ones(Q)

    rho = np.empty(T.size)
    for m_i, m in enumerate(MODELS):
        mask = model_idx == m_i
        rho[mask] = _log_interp(T[mask], m.prefill_anchor_T, m.prefill_anchor_rho)
    rho *= rho_scale

    eta = np.array([MODELS[i].eta_bytes_per_tok for i in model_idx])
    beta = np.array([MODELS[i].beta_bytes_per_tok for i in model_idx])

    M_names = tuple(m.name for m in MODELS)
    real_lambda = np.array([d.lambda_bytes_per_s for d in DESTINATIONS])
    real_W = np.array([[d.warm_instances[name] for name in M_names] for d in DESTINATIONS],
                      dtype=float)

    if n_dest is None:
        L_names = tuple(d.name for d in DESTINATIONS)
        lambda_bps = real_lambda * lambda_scale
        W = real_W if W is None else np.asarray(W, dtype=float)
    else:
        # Append synthetic sites drawn (independent stream) from the empirical
        # distribution of the 3 real ones. Adds capacity, so Z* (ADMM's
        # precondition) is preserved. Workload sampling above is untouched, so
        # instances at different n_dest share the same jobs for a given seed.
        assert n_dest >= len(DESTINATIONS) and W is None
        drng = np.random.default_rng([seed, 99])
        k = n_dest - len(DESTINATIONS)
        syn_lambda = drng.choice(real_lambda, k) * drng.lognormal(0.0, 0.25, k)
        syn_W = drng.poisson(real_W.mean(axis=0), (k, len(MODELS))).astype(float)
        L_names = tuple(d.name for d in DESTINATIONS) + tuple(f"Synthetic {i+1}" for i in range(k))
        lambda_bps = np.concatenate([real_lambda, syn_lambda]) * lambda_scale
        W = np.vstack([real_W, syn_W])

    return ProblemInstance(
        model_idx=model_idx, T=T, beta=beta, eta=eta, rho=rho, n=n,
        lambda_bps=lambda_bps, W=W, mu_ing=MU_ING_BYTES_PER_S, D=float(D),
        M_names=M_names, L_names=L_names,
        d_miss=2.0 * float(D) if d_miss is None else float(d_miss),
    )
