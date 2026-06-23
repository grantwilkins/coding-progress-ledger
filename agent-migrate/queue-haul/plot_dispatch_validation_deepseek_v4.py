"""Agentic dispatch validation with a DeepSeek-V4-Flash architecture proxy.

DeepSeek-V4-Pro does not fit the current single-8xH100-node abstraction because
its 1.6T total weights require model-parallel placement. This plot keeps the
existing event and workload setup, but swaps the model constants to V4-Flash:
284B total params, 13B active params, compressed-attention KV, and FP8-sized
weights. Decode throughput is an active-parameter-scaled placeholder, not a
measured DeepSeek serving trace.
"""

import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import impact
import instance
import power
from dispatch import Event, bind_dp, greedy, solve
from impact import Movement, compute
from instance import _mean_T, class_workload, generate
from power import PoolPower

COMPRESS_RATIOS = np.array([
    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
    128, 4, 128, 4, 128, 4, 128, 4, 0,
], dtype=float)
COMPRESS = np.array([1.0 if r == 0 else 1.0 / r for r in COMPRESS_RATIOS])
N_ACT = 13e9
C_ATTN = 2 * 64 * 512 * COMPRESS.sum()
ETA_BYTES_PER_TOK = 2 * 2 * 1 * 512 * COMPRESS.sum()
CAP_FP8_GB = 640.0 - 284.0 - 40.0
G_SCALE = 22.0 / 13.0

power.N_ACT = N_ACT
power.C_ATTN = C_ATTN
power.ETA_BYTES_PER_TOK = ETA_BYTES_PER_TOK
power.CAP_FP8_GB = CAP_FP8_GB
impact.ETA_BYTES_PER_TOK = ETA_BYTES_PER_TOK
instance.ETA_BYTES_PER_TOK = ETA_BYTES_PER_TOK
instance.CAP_FP8_GB = CAP_FP8_GB

EVENT = Event(dest_nodes=48, W=16)
MOVE = Movement()
N_NODES = 4
S_FRACS = np.linspace(0.30, 0.95, 8)
CASES = (
    ("agentic tool loop", "agentic_tool_loop"),
)
COUNT_CASES = (
    ("ordinary chat", "ordinary_chat"),
    ("long chat / code", "long_chat_code"),
    ("reasoning chat", "reasoning_chat"),
    *CASES,
)


def case(label, cls):
    wl = class_workload(
        cls,
        state_mix=(1.0, 0.0, 0.0),
        cache_hit=(1.0, 1.0, 1.0, 1.0),
        g_bf16=4600.0 * G_SCALE,
        g_fp8=9200.0 * G_SCALE,
    )
    pool = replace(PoolPower(), cap_gb=CAP_FP8_GB, mean_context_tokens=_mean_T(wl))
    pop = generate(pool, wl, n_nodes=N_NODES)
    imp = compute(pop, pool)
    dp = bind_dp(imp)
    lp_ceil = solve(pop, pool, imp, 2 * dp.sum(), EVENT, MOVE).shed_guaranteed
    S = S_FRACS * lp_ceil
    lp = [solve(pop, pool, imp, s, EVENT, MOVE) for s in S]
    gr = [greedy(pop, pool, imp, s, EVENT, MOVE) for s in S]
    return label, pop, imp, S, lp, gr


results = [case(*c) for c in CASES]

kW = 1e3
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for label, pop, imp, S, lp, gr in results:
    serving_eq = pop.ell.sum() / PoolPower().rho_star
    lp_cost = np.array([p.cost if p.feasible else np.nan for p in lp])
    gr_cost = np.array([p.cost if p.feasible else np.nan for p in gr])
    lp_norm, gr_norm = lp_cost / (S / kW), gr_cost / (S / kW)
    save = np.divide(gr_norm - lp_norm, gr_norm, out=np.zeros_like(gr_norm), where=gr_norm > 0)
    ax.plot(S / kW, gr_norm, color="tab:orange", lw=2, label="greedy")
    ax.plot(S / kW, lp_norm, color="tab:blue", lw=2, label="LP")
    if np.nanmax(save) > 0.01:
        i = int(np.nanargmax(save))
        ax.annotate(f"LP cuts disruption {100 * save[i]:.0f}%",
                    xy=(S[i] / kW, lp_norm[i]), xytext=(S[i] / kW, 0.72 * gr_norm[i]),
                    arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=8, ha="center")
    else:
        ax.text(0.60 * S.max() / kW, 0.55 * np.nanmax(gr_norm), "same sorted plan",
                fontsize=8, ha="center", color="0.3")
    ax.set(title=f"{label}: {imp.regime}, {len(pop)} jobs, {serving_eq:.0f} serving-node eq.",
           xlabel="requested shed $S^\\star$ (kW)", ylabel="disruption intensity (s/kW)")
ax.legend(loc="upper left", fontsize=8)
fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.88)

os.makedirs("outputs", exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(f"outputs/dispatch_validation_deepseek_v4_flash.{ext}", dpi=150)

fig2, ax2 = plt.subplots(figsize=(7.2, 4.6))
for label, pop, imp, S, lp, gr in results:
    lp_expected = np.array([imp.dp_expected_single @ p.y for p in lp]) / kW
    gr_expected = np.array([imp.dp_expected_single @ p.y for p in gr]) / kW
    ax2.plot(S / kW, S / kW, "k--", lw=1.5, label="certified floor")
    ax2.plot(S / kW, gr_expected, color="tab:orange", lw=2, label="greedy future proxy")
    ax2.plot(S / kW, lp_expected, color="tab:blue", lw=2, label="LP future proxy")
    i = int(np.nanargmax(lp_expected))
    ax2.annotate(f"LP future proxy {lp_expected[i]:.0f} kW",
                 xy=(S[i] / kW, lp_expected[i]), xytext=(0.62 * S[i] / kW, 0.78 * lp_expected[i]),
                 arrowprops=dict(arrowstyle="->", color="0.3"), fontsize=8, ha="center")
ax2.set(title="DeepSeek-V4-Flash: certified floor vs future node proxy",
        xlabel="requested certified shed $S^\\star$ (kW)", ylabel="power impact proxy (kW)")
ax2.legend(loc="upper left", fontsize=8)
fig2.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.88)
for ext in ("pdf", "png"):
    fig2.savefig(f"outputs/dispatch_expected_deepseek_v4_flash.{ext}", dpi=150)

print(f"DeepSeek-V4-Flash proxy: eta={ETA_BYTES_PER_TOK/1024:.1f} KiB/tok, "
      f"C_attn={C_ATTN:.0f}, cap={CAP_FP8_GB:.0f} GB, G_fp8={9200.0 * G_SCALE:.0f} tok/s")
print("4 held-memory-node session counts:")
for label, cls in COUNT_CASES:
    wl = class_workload(
        cls,
        state_mix=(1.0, 0.0, 0.0),
        cache_hit=(1.0, 1.0, 1.0, 1.0),
        g_bf16=4600.0 * G_SCALE,
        g_fp8=9200.0 * G_SCALE,
    )
    pool = replace(PoolPower(), cap_gb=CAP_FP8_GB, mean_context_tokens=_mean_T(wl))
    print(f"  {label:16s} jobs={len(generate(pool, wl, n_nodes=N_NODES)):5d}")
for label, pop, imp, S, lp, gr in results:
    serving_eq = pop.ell.sum() / PoolPower().rho_star
    lp_cost = np.array([p.cost for p in lp])
    gr_cost = np.array([p.cost for p in gr])
    lp_norm, gr_norm = lp_cost / (S / kW), gr_cost / (S / kW)
    save = np.divide(gr_norm - lp_norm, gr_norm, out=np.zeros_like(gr_norm), where=gr_norm > 0)
    i = int(np.nanargmax(save))
    future = np.array([imp.dp_expected_single @ p.y for p in lp])
    j = int(np.nanargmax(future))
    print(f"{label:16s} regime={imp.regime:6s} jobs={len(pop):5d} serving_node_eq={serving_eq:5.1f} "
          f"ceiling={max(p.shed_guaranteed for p in lp)/kW:6.1f} kW "
          f"max_cut={100 * save[i]:4.1f}% at S*={S[i]/kW:6.1f} kW "
          f"({gr_norm[i]:.1f}->{lp_norm[i]:.1f} s/kW), "
          f"max_future_proxy={future[j]/kW:6.1f} kW")
