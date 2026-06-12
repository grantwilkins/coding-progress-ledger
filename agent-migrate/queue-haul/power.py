"""Pool & power model (formulation.md §Pool power model; values from assumptions.md §2/§4).

Scalar prices the dispatch consumes — the solver never evaluates a node power curve.
"""

from __future__ import annotations

from dataclasses import dataclass

GB = 1e9
ETA_BYTES_PER_TOK = 188 * 1024  # exact from attention config
CAP_BF16_GB = 130.0  # 640 - 470 weights - 40
CAP_FP8_GB = 365.0  # 640 - 235 weights - 40


@dataclass(frozen=True)
class PoolPower:
    p_idle_w: float = 3200.0  # §2 center
    p_busy_w: float = 8400.0  # §2 center, 0.8x TDP
    rho_star: float = 0.80  # §2 center
    bracket_ratio: float = 30.0  # §2 center, p̄/s_plat
    gamma: float = 0.5  # §4 center, paged-out uplift
    cap_gb: float = CAP_BF16_GB  # §4, KV bytes/node after weights
    mean_context_tokens: float = 65800.0  # §1 center E[T]
    phase_ratio: float = 5.0  # §2, p̄_dec/p̄_pre per busy-second

    @property
    def p_bar(self) -> float:
        """Amortized price, W per node-unit of load."""
        return self.p_busy_w / self.rho_star

    @property
    def s_plat(self) -> float:
        """Guaranteed plateau slope, W per node-unit."""
        return self.p_bar / self.bracket_ratio

    @property
    def p_pre(self) -> float:
        """Prefill price; equal-phase closure (p_pre + p_dec)/2 = p_bar."""
        return 2 * self.p_bar / (1 + self.phase_ratio)

    @property
    def p_dec(self) -> float:
        return self.phase_ratio * self.p_pre

    @property
    def m_bar(self) -> float:
        """Mean KV bytes per session."""
        return ETA_BYTES_PER_TOK * self.mean_context_tokens

    @property
    def s_node(self) -> float:
        """Held sessions per node, (1+γ)·Cap/m̄."""
        return (1 + self.gamma) * self.cap_gb * GB / self.m_bar

    @property
    def mu(self) -> float:
        """Memory-regime marginal power, W per held session (node sits at idle)."""
        return self.p_idle_w / self.s_node

    def node_count(self, load: float, s_held: float) -> float:
        return max(load / self.rho_star, s_held / self.s_node)

    def memory_bound(self, load: float, s_held: float) -> bool:
        return s_held / self.s_node > load / self.rho_star
