"""Pool & power model (formulation.md §Pool power model; values from assumptions.md §2/§4).

Scalar prices the canonical dispatch consumes; node_knee.py also uses the node curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GB = 1e9
ETA_BYTES_PER_TOK = 188 * 1024  # KV bytes/tok, exact from attention config
BETA_BYTES_PER_TOK = 4  # context bytes/tok, uint32 token IDs
CAP_BF16_GB = 130.0  # 640 - 470 weights - 40
CAP_FP8_GB = 365.0  # 640 - 235 weights - 40


def congestion(u: float) -> float:
    """M/M/1 queue-wait multiplier φ(u)=u/(1−u) at utilization u∈[0,1)."""
    return u / (1 - u)

# Prefill roofline (formulation.md §6). C uses QUERY heads (compute scales with H_q);
# η above uses KV heads (cache size scales with H_kv). H_q gives T* ≈ 29k as stated.
N_ACT = 22e9  # A22B active params
PEAK_FLOPS = 8 * 989.5e12  # 8×H100 SXM, BF16 dense
L_ATTN, H_Q, D_HEAD = 94, 64, 128  # Qwen3-235B-A22B
C_ATTN = 2 * L_ATTN * H_Q * D_HEAD  # per-token attention coefficient


def rho_dest(T, mfu: float = 0.35):
    """Per-session prefill rate (tok/s) at context T: flat ≈63k below T*≈29k, ~1/T above."""
    return PEAK_FLOPS * mfu / (2 * N_ACT + C_ATTN * np.asarray(T, dtype=float))


def rho_replay(T, mfu: float = 0.35):
    """Average full-context replay rate; rho_dest(T) is the marginal append rate."""
    return rho_dest(np.asarray(T, dtype=float) / 2, mfu)


@dataclass(frozen=True)
class PoolPower:
    p_idle_w: float = 3200.0  # §2 center
    p_busy_w: float = 8400.0  # §2 center, 0.8x TDP
    power_knee: float = 0.10  # §2 center
    latency_knee: float = 0.85  # §2 center
    rho_star: float = 0.80  # §2 center
    bracket_ratio: float = 30.0  # §2 center, p̄/s_plat
    gamma: float = 0.5  # §4 center, paged-out uplift
    cap_gb: float = CAP_BF16_GB  # §4, KV bytes/node after weights
    mean_context_tokens: float = 65800.0  # §1 center E[T]
    c_prefill_j_per_tok: float = 0.148  # measured H100 dense analog, J/token
    c_decode_j_per_tok: float = 1.76  # measured H100 dense analog, J/token

    @property
    def p_bar(self) -> float:
        """Single-price comparison, W per node-unit of load."""
        return self.p_busy_w / self.rho_star

    @property
    def s_plat(self) -> float:
        """Guaranteed plateau slope, W per node-unit."""
        return self.p_bar / self.bracket_ratio

    @property
    def base_w_per_load(self) -> float:
        """Static node power, W per node-unit of load."""
        return self.p_idle_w / self.rho_star

    @property
    def m_bar(self) -> float:
        """Mean KV bytes per session."""
        return ETA_BYTES_PER_TOK * self.mean_context_tokens

    @property
    def s_node_resident(self) -> float:
        """Resident (runnable) sessions per node, Cap/m̄."""
        return self.cap_gb * GB / self.m_bar

    @property
    def s_node(self) -> float:
        """Held sessions per node, (1+γ)·Cap/m̄."""
        return (1 + self.gamma) * self.cap_gb * GB / self.m_bar

    @property
    def mu(self) -> float:
        """Memory-regime marginal power, W per held session (node sits at idle)."""
        return self.p_idle_w / self.s_node

    @property
    def p_knee(self) -> float:
        """Node power at the power knee, anchoring pi(rho*) = P_busy."""
        return self.p_busy_w - self.s_plat * (self.rho_star - self.power_knee)

    @property
    def ramp_slope(self) -> float:
        """Ramp slope below the power knee, W per node-unit."""
        return (self.p_knee - self.p_idle_w) / self.power_knee

    def node_power(self, load):
        """Ramp-then-plateau node power at node load; plateau slope extends above rho*."""
        load = np.asarray(load, dtype=float)
        ramp = self.p_idle_w + self.ramp_slope * load
        plat = self.p_knee + self.s_plat * (load - self.power_knee)
        return np.where(load <= self.power_knee, ramp, plat)

    def node_power_slope(self, load):
        """A subgradient of node_power(load); choose the ramp slope at the knee."""
        return np.where(np.asarray(load, dtype=float) <= self.power_knee, self.ramp_slope, self.s_plat)

    def node_count(self, load: float, s_held: float) -> float:
        return max(load / self.rho_star, s_held / self.s_node)

    def memory_bound(self, load: float, s_held: float) -> bool:
        return s_held / self.s_node > load / self.rho_star
