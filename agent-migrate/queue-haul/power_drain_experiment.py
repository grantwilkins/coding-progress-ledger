"""Offline 10k-session power-drain and network sensitivity experiment."""

from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from impact import Movement, compute
from instance import _draw, _mean_T, class_workload
from node_knee import evaluate_node_expected_w, node_loads, with_source_nodes
from power import PoolPower
from stage1c_controller import (
    LIVE_A100_F_PREFILL_TPS,
    LIVE_A100_G_DECODE_TPS,
    LIVE_A100_LOG_SHAPE,
    LIVE_A100_P_BUSY_W,
    LIVE_A100_P_IDLE_W,
    LIVE_A100_POWER_KNEE,
    LIVE_A100_RHO_STAR,
)

WORKLOADS = {
    "interactive_coding": "ordinary_chat",
    "coding": "long_chat_code",
    "agentic_tool_loop": "agentic_tool_loop",
}
POLICIES = ("node-drain", "power-unaware", "random")


def a100_population(workload: str, n_sessions: int, seed: int):
    wl = class_workload(WORKLOADS[workload], state_mix=(1.0, 0.0, 0.0))
    pool = PoolPower(
        p_idle_w=LIVE_A100_P_IDLE_W,
        p_busy_w=LIVE_A100_P_BUSY_W,
        power_knee=LIVE_A100_POWER_KNEE,
        rho_star=LIVE_A100_RHO_STAR,
        mean_context_tokens=_mean_T(wl),
        power_curve="saturating",
        log_shape=LIVE_A100_LOG_SHAPE,
    )
    pop = _draw(np.random.default_rng(seed), n_sessions, wl, "bf16")
    ell_pre, ell_dec = pop.f / LIVE_A100_F_PREFILL_TPS, pop.g / LIVE_A100_G_DECODE_TPS
    scale = np.minimum(1.0, 0.5 / np.maximum(ell_pre + ell_dec, 1e-12))
    pop = replace(pop, turn_rate=pop.turn_rate * scale, f=pop.f * scale, g=pop.g * scale,
                  ell_pre=ell_pre * scale, ell_dec=ell_dec * scale)
    n_nodes = max(1, int(np.ceil(pop.ell.sum() / pool.rho_star * 1.1)))
    while True:
        heap = [(0.0, i) for i in range(n_nodes)]
        source = np.empty(n_sessions, int)
        for j in np.argsort(-pop.ell, kind="stable"):
            load, i = heapq.heappop(heap)
            source[j] = i
            heapq.heappush(heap, (load + pop.ell[j], i))
        pop = with_source_nodes(pop, source)
        if node_loads(pop).max() <= pool.rho_star + 1e-9:
            return pool, pop
        n_nodes = int(np.ceil(n_nodes * 1.1))


def policy_order(policy: str, pop, pool, imp, target_w: float, seed: int) -> np.ndarray:
    cost = np.minimum(imp.c_replay, imp.c_transfer)
    if policy == "random":
        return np.random.default_rng(seed).permutation(len(pop))
    if policy == "power-unaware":
        return np.lexsort((np.arange(len(pop)), cost / np.maximum(pop.ell, 1e-12)))
    if policy != "node-drain":
        raise ValueError(f"unknown policy {policy!r}")
    node, load = pop.source_node, node_loads(pop)
    bundles = []
    for i in range(len(load)):
        jobs = np.flatnonzero(node == i)
        jobs = jobs[np.lexsort((jobs, cost[jobs] / np.maximum(pop.ell[jobs], 1e-12)))]
        gain = float(pool.node_power(load[i]) - pool.node_power(0.0))
        bundles.append((float(cost[jobs].sum()) / min(gain, target_w), i, jobs))
    return np.concatenate([jobs for _score, _i, jobs in sorted(bundles)])


def select(pop, pool, imp, order: np.ndarray, target_frac: float) -> tuple[np.ndarray, float, float]:
    residual = node_loads(pop).copy()
    full_w = float(np.sum(pool.node_power(residual) - pool.node_power(0.0)))
    target_w, shed_w = target_frac * full_w, 0.0
    chosen = []
    for j in order:
        if shed_w >= target_w:
            break
        i = int(pop.source_node[j])
        after = residual[i] - pop.ell[j]
        shed_w += float(pool.node_power(residual[i]) - pool.node_power(after))
        residual[i] = after
        chosen.append(int(j))
    return np.asarray(chosen, int), shed_w, target_w


def completion_times(pop, imp, chosen: np.ndarray, mbps: float, rtt_ms: float) -> np.ndarray:
    action_r = imp.c_replay <= imp.c_transfer
    byte_count = np.where(action_r, imp.b_replay, imp.b_transfer)
    nominal = byte_count / 125_000_000.0
    rebuild = np.maximum(0.0, np.where(action_r, imp.c_replay, imp.c_transfer) - nominal)
    clocks, done = {}, np.full(len(pop), np.nan)
    rate = mbps * 125_000.0
    for j in chosen:
        i = int(pop.source_node[j])
        clocks[i] = clocks.get(i, 0.0) + byte_count[j] / rate
        done[j] = clocks[i] + rebuild[j] + rtt_ms / 1000.0
    return done


def run(n_sessions: int = 10_000, seed: int = 0, target_fracs=(0.5, 0.9),
        bandwidths=(250.0, 1000.0, 10_000.0), rtts=(10.0, 40.0, 80.0),
        deadline_s: float = 120.0) -> list[dict]:
    rows = []
    move = replace(Movement(), lambda_src=125_000_000.0, dest_prefill_util=0.0)
    for workload in WORKLOADS:
        pool, pop = a100_population(workload, n_sessions, seed)
        imp = compute(pop, pool, move)
        full_w = evaluate_node_expected_w(pop, pool, np.ones(len(pop)))
        for frac in target_fracs:
            target_w = frac * full_w
            for policy in POLICIES:
                order = policy_order(policy, pop, pool, imp, target_w, seed)
                chosen, selected_w, target_w = select(pop, pool, imp, order, frac)
                selected = np.zeros(len(pop))
                selected[chosen] = 1.0
                for mbps in bandwidths:
                    for rtt_ms in rtts:
                        done = completion_times(pop, imp, chosen, mbps, rtt_ms)
                        realized = selected * (np.nan_to_num(done, nan=np.inf) <= deadline_s)
                        realized_w = evaluate_node_expected_w(pop, pool, realized)
                        waits = done[chosen]
                        rows.append({
                            "workload": workload, "policy": policy, "seed": seed,
                            "sessions": len(pop), "source_nodes": len(node_loads(pop)),
                            "target_frac": frac, "target_w": target_w,
                            "selected_sessions": len(chosen), "selected_w": selected_w,
                            "realized_sessions": int(realized.sum()), "realized_w": realized_w,
                            "deadline_s": deadline_s, "deadline_hit": realized_w >= target_w,
                            "mbps": mbps, "rtt_ms": rtt_ms,
                            "mean_interruption_s": float(waits.mean()),
                            "p95_interruption_s": float(np.quantile(waits, 0.95)),
                            "makespan_s": float(waits.max()),
                            "disruption_s_per_kw": float(waits.sum() / (realized_w / 1000.0)) if realized_w else np.inf,
                        })
    return rows


def write(rows: list[dict], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scale_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    canonical = [r for r in rows if r["mbps"] == 1000 and r["rtt_ms"] == 40]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for policy in POLICIES:
        part = [r for r in canonical if r["policy"] == policy]
        x = np.arange(len(part))
        axes[0].plot(x, [r["realized_w"] / r["target_w"] for r in part], "o-", label=policy)
        axes[1].plot(x, [r["disruption_s_per_kw"] for r in part], "o-", label=policy)
    axes[0].axhline(1, color="black", lw=1)
    axes[0].set(ylabel="realized / target power", xlabel="workload-target case")
    axes[1].set(ylabel="session interruption (s/kW)", xlabel="workload-target case")
    axes[0].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "scale_policy_comparison.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for policy in POLICIES:
        vals = []
        for mbps in sorted({r["mbps"] for r in rows}):
            part = [r["makespan_s"] for r in rows if r["policy"] == policy and r["mbps"] == mbps]
            vals.append(float(np.median(part)))
        ax.plot(sorted({r["mbps"] for r in rows}), vals, "o-", label=policy)
    ax.set(xscale="log", xlabel="per-source link (Mbps)", ylabel="median migration makespan (s)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "scale_network_sensitivity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("queue-haul/outputs/power_drain"))
    parser.add_argument("--sessions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = run(args.sessions, args.seed)
    write(rows, args.out)
    print(f"rows={len(rows)} output={args.out}")


if __name__ == "__main__":
    main()
