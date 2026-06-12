# Queue-Haul Assumptions

Fill every `??` before running experiments. Values marked *(evac)* are carried from
`../evacuation/assumptions.md` — confirm or override. Ranges in *Sweep bounds* come from
the formulation's measured-range quotes; they bound the sweep, they are not point values.

## 1. Model & workload constants

| Parameter | Value | Source |
|---|---:|---|
| Model | Qwen3-235B-A22B | *(evac)* |
| η (KV bytes/tok) | 188 KiB | *(evac)* exact arithmetic |
| β (context bytes/tok) | 4 B | *(evac)* uint32 token IDs |
| Weights (BF16) | 470 GB | *(evac)* |
| Context length T | 0.70·LogN(10.07, 1.0) + 0.30·LogN(11.45, 0.8), clip [1e3, 1e6] | *(evac)* snapshot mixture |
| Seed | 42 | *(evac)* |

## 2. Node power (absolute watts, swept)

| Parameter | Value / sweep bounds | Source |
|---|---:|---|
| P_idle (W/node) | ?? | |
| P_busy (W/node) | ?? | |
| Power knee ℓ | ?? — formulation quotes 0.03–0.15 (dense 70B+), 0.3–0.6 (8B/MoE) | |
| Latency knee ℓ | ?? — formulation quotes 0.73–1.0 | |
| s_plat (W per unit load) | ?? — or derive from bracket ratio | |
| Bracket ratio p̄/s_plat | ?? — quotes: 3–5× MoE, 17–44× dense 70B, ~58× dense 405B | |
| Phase price ratio (prefill/decode per token) | ?? — quotes c1/c2 ≈ 0.10; per busy-second decode 4–8× prefill | |
| F (prefill tok/s at latency knee) | ?? | |
| G (decode tok/s at latency knee) | ?? | |

## 3. Per-job rate distributions

| Parameter | Value | Source |
|---|---:|---|
| State mix (active / idle / cold fractions) | ?? / ?? / ?? | |
| Turn rate (turns/s), active jobs | ?? | |
| Input tokens per turn | ?? | |
| Output tokens per turn (incl. reasoning) | ?? | |
| Idle-job rate floor | ~0 by construction | formulation §2 |

## 4. Capacity & memory regime

| Parameter | Value | Source |
|---|---:|---|
| Cap (KV bytes/node after weights) | ?? — *(evac)* decode node had 170 GB | |
| m̄ = η·E[T] | derived (~12.7 GB at E[T]≈66k) | *(evac)* |
| S_node^resident = Cap/m̄ | derived | |
| γ (paged-out uplift) | ?? — formulation suggests 0.5–1, swept | |
| ρ_low (utilization when memory binds) | ?? | |

## 5. Pools & event

| Parameter | Value | Source |
|---|---:|---|
| Source pool size (nodes) | ?? | |
| ρ* setpoint sweep band | ?? — between power knee and latency knee | |
| Destination spare load L̄_dest | ?? | |
| Destination spare held sessions S̄_dest | ?? — counts (1+γ) uplift | |
| Shed target S* range | ?? | |
| Deadline D | ?? — *(evac)* swept 15–600 s | |
| Hold H | ?? | |
| Pinned job classes (y_j = 0) | ?? — none by default | |

## 6. Movement

| Parameter | Value | Source |
|---|---:|---|
| Λ_src (egress link) | 1 GB/s | *(evac)* WAN-class |
| W (destination prefill nodes) | ?? — *(evac)* had 8 | |
| ρ_dest (destination prefill tok/s) | ?? — *(evac)* FLOP roofline, MFU 0.35 | |
| μ_in (ingest, per node) | 512 GB/s | *(evac)* host-staged PCIe Gen5 |
| τ_src, τ_pre, τ_in (startup latencies) | ?? / ?? / ?? | |
| w_pre, w_in (destination queueing waits) | ?? / ?? | |
