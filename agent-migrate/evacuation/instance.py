"""Workload + destination instance for the staged evacuation program.

Single-provider, single-model scenario: one flagship MoE (Qwen3-235B-A22B)
serving agentic/coding sessions. The source is a pod of racks under
prefill/decode (PD) disaggregation; deadline-elastic batch work is paused
first at zero cost, so the evacuated population is the interactive sessions
resident in decode HBM. The destination is a single remote site whose
headroom exists because it paused its own batch work; it is reached over a
WAN-class link.

Rack model (1P3D): a rack is 4 TP=8 nodes (8x H100 = 640 GB HBM each), one
serving prefill and three serving decode. Each decode node holds resident
BF16 weights plus KV headroom; active sessions per rack follow from that
headroom divided by the mean in-flight KV footprint. eta is exact from the
published attention config (94 layers x 4 KV-heads x 128 head-dim x 2(K,V) x
2 B BF16 = 188 KiB/tok). Each job is its own class (n_q = 1) unless
aggregated into log-T bins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import prefill

KIB = 1024.0
GB = 1e9

MODEL = "Qwen3-235B-A22B"
ETA_BYTES_PER_TOK = 188.0 * KIB
BETA_BYTES_PER_TOK = 4.0
WEIGHT_GB = 470.0

NODE_HBM_GB = 640.0  # TP=8 node, 8x H100
KV_FREE_PER_NODE_B = (NODE_HBM_GB - WEIGHT_GB) * GB
PREFILL_NODES_PER_RACK = 1  # PD disaggregation, 1P3D
DECODE_NODES_PER_RACK = 3
SRC_RACKS = 8
DST_RACKS = 8

LAMBDA_BPS = 1.0 * GB  # 8 Gbps WAN-class inter-site link
MU_ING_BYTES_PER_S = 512.0 * GB
T_MIN, T_MAX = 1_000.0, 1_000_000.0
SEED_DEFAULT = 42
D_DEFAULT_S = 600.0


@dataclass(frozen=True)
class ContextDist:
    """In-flight snapshot context length of an active session (tokens),
    clipped to [T_MIN, T_MAX]. Lognormal params are natural-log; a mixture is
    a tuple of (weight, mu_ln, sigma_ln) components."""
    family: str
    params: tuple

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        if self.family == "lognormal":
            mu, sigma = self.params
            t = rng.lognormal(mu, sigma, size)
        elif self.family == "lognormal_mixture":
            w = np.array([c[0] for c in self.params])
            comp = rng.choice(len(self.params), size, p=w / w.sum())
            mu = np.array([c[1] for c in self.params])[comp]
            sigma = np.array([c[2] for c in self.params])[comp]
            t = rng.lognormal(mu, sigma)
        else:
            raise ValueError(f"unknown family {self.family!r}")
        return np.clip(t, T_MIN, T_MAX)

    def mean(self) -> float:
        """Clipped mean from a deterministic probe (the analytic lognormal
        mean overstates the truncated-at-T_MAX heavy tail)."""
        return float(self.sample(np.random.default_rng(7), 200_000).mean())


# Placeholder pending the agentic-trace survey (see assumptions.md); replaced
# by the fitted snapshot distribution in Step 7 of the redesign.
CONTEXT_DIST = ContextDist("lognormal", (float(np.log(30_000.0)), 1.4))


def n_rack(dist: ContextDist = CONTEXT_DIST) -> int:
    """Active sessions per rack: decode-pool KV headroom / mean KV footprint."""
    return int(DECODE_NODES_PER_RACK * KV_FREE_PER_NODE_B
               // (ETA_BYTES_PER_TOK * dist.mean()))


@dataclass(frozen=True)
class ProblemInstance:
    model_idx: np.ndarray  # (Q,) int, model index per job (all 0: one model)
    T: np.ndarray          # (Q,) float, context tokens per job
    beta: np.ndarray       # (Q,) float, B/tok (context)
    eta: np.ndarray        # (Q,) float, B/tok (KV)
    rho: np.ndarray        # (Q,) float, prefill tok/s at T_q
    n: np.ndarray          # (Q,) float, jobs per class
    lambda_bps: np.ndarray # (L,) float
    W: np.ndarray          # (L, M) int, destination prefill nodes
    W_ing: np.ndarray      # (L, M) int, destination decode nodes (ingest)
    C_res: np.ndarray      # (L,) float, destination decode-HBM residency bytes
    mu_ing: float          # bytes/s per decode node
    D: float               # deadline seconds
    M_names: tuple[str, ...]
    L_names: tuple[str, ...]
    d_miss: float = 0.0    # seconds, unmoved-job reconstruction penalty (Section 2.4)


def _log_interp(T: np.ndarray, anchor_T: np.ndarray, anchor_rho: np.ndarray) -> np.ndarray:
    return np.exp(np.interp(np.log(T), np.log(anchor_T), np.log(anchor_rho)))


def build_instance(D: float = D_DEFAULT_S,
                   occupancy: float = 1.0,
                   seed: int = SEED_DEFAULT,
                   d_miss: float | None = None,
                   rho_scale: float = 1.0,
                   lambda_scale: float = 1.0,
                   n_bins: int | None = None,
                   dist: ContextDist = CONTEXT_DIST) -> ProblemInstance:
    rng = np.random.default_rng(seed)
    Q = round(occupancy * SRC_RACKS * n_rack(dist))
    T = dist.sample(rng, Q)

    # Optional aggregation: merge jobs in a log-T bin into a class with
    # n_q = count and T = bin mean (exact for T-linear loads).
    if n_bins:
        edges = np.logspace(np.log10(T_MIN), np.log10(T_MAX), n_bins + 1)
        b = np.clip(np.digitize(T, edges) - 1, 0, n_bins - 1)
        Tc = [(T[b == k].mean(), (b == k).sum()) for k in range(n_bins) if (b == k).any()]
        T = np.array([t for t, _ in Tc])
        n = np.array([c for _, c in Tc], float)
    else:
        n = np.ones(Q)

    rho = _log_interp(T, prefill.ANCHOR_T, prefill.anchors(MODEL)) * rho_scale

    return ProblemInstance(
        model_idx=np.zeros(T.size, dtype=int),
        T=T,
        beta=np.full(T.size, BETA_BYTES_PER_TOK),
        eta=np.full(T.size, ETA_BYTES_PER_TOK),
        rho=rho,
        n=n,
        lambda_bps=np.array([LAMBDA_BPS * lambda_scale]),
        W=np.array([[DST_RACKS * PREFILL_NODES_PER_RACK]], float),
        W_ing=np.array([[DST_RACKS * DECODE_NODES_PER_RACK]], float),
        C_res=np.array([DST_RACKS * DECODE_NODES_PER_RACK * KV_FREE_PER_NODE_B]),
        mu_ing=MU_ING_BYTES_PER_S,
        D=float(D),
        M_names=(MODEL,),
        L_names=("Destination",),
        d_miss=2.0 * float(D) if d_miss is None else float(d_miss),
    )
