# Scenario Parameters and Data Assumptions

This document describes the constants, distributions, and configurations used to generate experimental instances. The scenario is a **single provider serving one flagship model** — Qwen3-235B-A22B (MoE, 235B total / 22B active) — on agentic and coding workloads. Mid-tier and small-model tiers are out of scope; the deadline-elastic **batch tier is paused first at zero migration cost**, so the evacuated population is exactly the interactive sessions resident in decode HBM, and the regime studied begins where pausing batch alone cannot meet the target. These are benchmark-derived scenario parameters, not universal facts. The formulation is independent of these specific values.

---

## Model

| Parameter | Value | Source |
|-----------|------:|--------|
| Model | Qwen3-235B-A22B | flagship MoE, 22B active |
| $\eta$ (KV bytes/tok) | 188 KiB | exact: 94 layers × 4 KV heads × $d{=}128$ × 2$_{(K,V)}$ × 2 B (BF16) |
| $\beta$ (context bytes/tok) | 4 B | uint32 token IDs (lower bound; metadata excluded) |
| Weights (BF16) | 470 GB | resident per TP=8 instance |

---

## Rack and Pod Model (PD disaggregation)

A **rack** is 4 TP=8 nodes (8×H100 SXM = 640 GB HBM each), PD-disaggregated **1P3D**: 1 prefill node + 3 decode nodes. Prefill nodes hold no long-lived KV; each decode node holds the 470 GB weights plus **170 GB of KV headroom**.

Active sessions per rack follow from HBM arithmetic, using the mean *in-flight snapshot* context length $\mathbb{E}[T]$ (below):

$$ N_{\text{rack}} = \left\lfloor \frac{3 \times 170\,\text{GB}}{\eta \, \mathbb{E}[T]} \right\rfloor = 40 \quad (\mathbb{E}[T] \approx 66\text{k tok} \Rightarrow \sim 12.7\,\text{GB KV/session}). $$

- **Source** = a pod of 8 racks ⇒ $N = \text{round}(o \times 8 \times N_{\text{rack}}) = 320$ jobs at occupancy $o = 1$.
- **Occupancy $o \in \{0.5, 0.75, 1.0, 1.25, 1.5\}$** is the primary sensitivity axis (±50% around the fitted occupancy); it scales $N$ linearly.
- Each job is its own class ($n_q = 1$); optional log-$T$ binning is off by default.

---

## Context Length Distribution (in-flight snapshot)

$T$ is the context length of an **active session caught mid-flight at a random instant** — not the per-request input length. Snapshot sampling is length-biased: long-resident, long-context sessions are over-represented. Fitted from a three-way survey of 2024–26 traces (three independent research passes; chat-era data downweighted):

1. **Production serving traces** — Mooncake/Kimi FAST'25 trace (tool&agent avg input 8.6k, conversation 12k, tails to 128k), Together.ai production coding-agent benchmark (inputs 45k–200k), DeepSeek V3/R1 fleet disclosure. Recommended $\mathbb{E}[T] \approx 60$k, mass$_{[30k,1M]} \approx 0.47$.
2. **Cloud traces** — Azure LLM inference traces 2023/24 (chat-era, ≤8k window: downweighted), ServeGen 2025 (Pareto+lognormal family, reasoning outputs ~4× answers), DualPath 2026 production coding-agent traces (avg context 32.7k, 157 rounds, tails to 1M). Recommended $\mathbb{E}[T] \approx 35$k, mass ≈ 0.34.
3. **Agentic telemetry** — per-turn context growth (5k→35k over ~50 turns), SWE-agent peak prompts ~88k, LOCA-bench 100k–325k accumulation, Claude Code compaction at ~155k; step-duration weighting for the snapshot view. Recommended $\mathbb{E}[T] \approx 70$k, mass ≈ 0.60.

The three estimates agree within 2× and all recommend a body + long-tail mixture, so the scenario uses a **2-component lognormal mixture** (relevance-weighted blend of the three fits), clipped to $[10^3, 10^6]$:

$$ T \sim 0.70 \cdot \text{LogN}(\mu{=}10.07, \sigma{=}1.0) + 0.30 \cdot \text{LogN}(\mu{=}11.45, \sigma{=}0.8) $$

| Statistic | Value |
|-----------|------:|
| Median | ~36k tok |
| $\mathbb{E}[T]$ (clipped) | ~66k tok |
| p90 / p99 | ~156k / ~428k |
| Mass in [30k, 1M] | 0.56 |

Body = standard agentic-coding sessions (median ~24k); tail = long-resident large-context sessions (median ~94k). The tier table's nominal 30k–1M flagship regime holds for ~56% of snapshot mass; no trace supports hard truncation at 30k.

**Default seed:** 42. All randomness through `numpy.random.default_rng(seed)`.

---

## Prefill Rate Model

Unchanged FLOP roofline (`prefill.py`): one TP=8 instance prefills at
$$ \rho(T) = \frac{\mathrm{EFF}}{2 N_{\text{act}} + C\,T}, \qquad C = 2\,L_{\text{attn}} H_q d_{\text{head}}, \qquad \mathrm{EFF} = 8 \times 989.5\,\text{TFLOP/s} \times \text{MFU}(0.35) \approx 2.77\,\text{PFLOP/s}. $$

For Qwen3-235B-A22B: ceiling 63k tok/s, attention crossover $T^\star = 28.6$k tok, $\rho$@100k = 14.0k tok/s, $\rho$@1M = 1.75k tok/s. MFU=0.35 reproduces published 8×H100 rates for this checkpoint. Log-linear interpolation between the {1k, 10k, 100k, 1M} anchors.

---

## Destination Configuration

A **single destination site**, reached over a **WAN-class link $\Lambda = 1$ GB/s (8 Gbps)**. Its spare capacity is 8 racks (same 1P3D arithmetic), freed because the destination paused its own deadline-elastic batch work — batch carries no evacuation cost and no parameter.

| Resource | Value |
|----------|------:|
| Network $\Lambda$ | 1 GB/s (8 Gbps) |
| Prefill pool $W$ | 8 prefill nodes |
| Ingest pool $W^{\text{ing}}$ | 24 decode nodes × $\mu = 512$ GB/s (8 × PCIe Gen5 ×16, host-staged) |
| **Residency $C^{\text{res}}$** | 24 × 170 GB = **4.08 TB** decode-HBM (a stock, not deadline-scaled) |

The residency constraint $\sum_q \eta T_q (x^R_q + x^S_q) \le C^{\text{res}}$ is the decode-HBM wall: at $o > 1$ the pod's KV exceeds it and full evacuation is infeasible at any deadline; the optimum strands the largest-KV sessions.

**Regime.** At $\Lambda = 1$ GB/s the single-session replay/state crossover is $T^\star_{\text{mig}} = (\mathrm{EFF}(\eta - \beta)/\Lambda - 2N_{\text{act}})/C \approx 318$k tokens: the body of the snapshot distribution replays (context bytes are trivial; prefill GPU-seconds bind) and only the long tail ships KV (WAN bytes bind). Ingest is always slack. Jobs with $T/\rho(T) > D$ cannot solo-finish a replay within the window and are masked to state-transfer-or-stranded.

---

## Derived Quantities

At defaults ($o = 1$, seed 42, $D = 600$ s):

| Quantity | Value |
|----------|------:|
| Jobs $N$ (= classes, $n_q = 1$) | 320 |
| Pod KV mass | ~4.1 TB (≈ $o \times C^{\text{res}}$ by construction) |
| Min deadline to clear the pod ($o=1$, mean over seeds) | ~250 s |
| Min-D at $o = 0.5$ / $1.5$ | ~152 s / ~165 s (residency wall strands the heavy tail at $o>1$, shrinking the remaining work) |
| Replay share of evacuated KV at the frontier | ~0.92 |

---

## Default Sweep Ranges

| Sweep parameter | Default values |
|-----------------|---------------|
| **Occupancy $o$** | 0.5, 0.75, 1.0, 1.25, 1.5 |
| Deadline $D$ | 15–600 s (log-spaced; saturates ~240 s at $o=1$) |
| $\Lambda$ scale | 0.5, 1.0, 2.0, 4.0 (migration-ratio figure) |
| $\rho$ scale | 0.5, 0.75, 1.0, 1.5, 2.0 |

**$d^{\text{miss}}$** = $2D$ (next recovery window). This is a declared convention, not a tuning parameter.
