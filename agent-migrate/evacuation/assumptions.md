# Scenario Parameters and Data Assumptions

This document describes the constants, distributions, and configurations used to generate experimental instances. The evacuating site is a **single provider serving one model family** (Qwen3), so the suite, its traffic mix, and the fleet sizing are all that one provider's choices. These are benchmark-derived scenario parameters, not universal facts. The formulation is independent of these specific values.

---

## Model Table

One provider's Qwen3 suite, spanning a hybrid-linear efficiency tier up to the dense and MoE flagships. The $\eta$ values are **exact from each model's published attention config**: $\eta = (\text{layers}) \times (\text{KV heads}) \times (\text{head dim}) \times 2_{(K,V)} \times 2\text{ B}$ at BF16. The hybrid-linear layers of Qwen3-Next carry no per-token KV, so only its full-attention layers count.

| Model | $\eta$ (KiB/tok) | Attention config | Weights (GB, BF16) | Token share |
|-------|------------------:|------------------|-------------------:|------------:|
| Qwen3-30B-A3B | 96.0 | 48 layers, 4 KV heads, $d{=}128$ | 60 | 8% |
| Qwen3-Next-80B-A3B | 24.0 | 12 full-attn layers, 2 KV heads, $d{=}256$ (rest linear) | 160 | 25% |
| Qwen3-32B | 256.0 | 64 layers, 8 KV heads, $d{=}128$ | 64 | 12% |
| Qwen3-235B-A22B (flagship) | 188.0 | 94 layers, 4 KV heads, $d{=}128$ | 470 | 55% |

**$\beta = 4$ B/tok** for all models. This assumes uint32 token IDs as the compact context representation. The actual transmitted context may include metadata (role tags, tool schemas, position IDs), but 4 B/tok is a reasonable lower bound on the information-theoretic minimum.

**$\eta$** is the KV cache size per token aggregated across all layers and KV heads at the serving precision (BF16). The dense 32B is the KV-heaviest per token (8 KV heads, 64 layers), heavier even than the 235B flagship (4 KV heads), while the hybrid-linear Next-80B is the lightest. The spread is purely an artifact of architecture choices *within one lab*.

**Token share, not job count.** The mix is the fraction of served *tokens* each model handles, and the provider is **flagship-primary**: most tokens flow to Qwen3-235B. Job counts are *derived*, not declared: for token share $s_m$ and clipped mean session length $\bar T_m$,
$$ n_m \;\propto\; s_m / \bar T_m, $$
so the flagship's long agentic sessions make it a token *majority* (55%) but a job *minority* (~24%). This is the realistic, billing-denominated way to state a workload, and it is the sensitivity axis (see below).

---

## Prefill Rate Model

Prefill rate $\rho(T)$ is not hand-set; it is the **FLOP roofline** of one warm
instance ($8{\times}$H100, TP=8). A prefill of $T$ tokens is itself a token batch, so
prefill is compute-bound, and the per-instance rate (`prefill.py`) is
$$ \rho(T) = \frac{\mathrm{EFF}}{2 N_{\text{act}} + C\,T}, \qquad C = 2\,L_{\text{attn}} H_q d_{\text{head}}, $$
where $2 N_{\text{act}}$ FLOP/tok is the FFN/projection cost (active params, so MoE
counts only the routed experts) and $C\,T$ FLOP/tok is the causal self-attention
cost, which grows linearly in context because total attention is quadratic in $T$.
$\mathrm{EFF} = G \cdot \text{peak} \cdot \text{MFU}$ is the sustained compute.

**Hardware / calibration.** $G=8$ H100 SXM, BF16 dense peak $989.5$\,TFLOP/s, and
$\text{MFU}=0.35$, giving $\mathrm{EFF}\approx 2.77$\,PFLOP/s per instance. This MFU
reproduces the published $8{\times}$H100 prefill rates for Qwen3-235B and Qwen3-Next
(to within rounding) and is consistent with — slightly conservative against, since
TP=8 pays inter-GPU comms — the measured single-GPU A3B prefill peaks of
$45$--$57$k\,tok/s. Sources: gpustack Qwen3-235B/H100 and Qwen3-32B/H100 labs,
Millstone single-H100/H200 A3B context sweeps, and the NVIDIA H100 datasheet peak.
Measured isolated $8{\times}$H100 prefill-vs-context sweeps for these exact
checkpoints are not public, so the roofline (not a deployment trace) is the source
of truth; treat $\rho$ as an architecture-grounded scenario parameter.

| Model | $\rho$ @1k | $\rho$ @100k | $\rho$ @1M | Ceiling (tok/s) | $T^\star$ (tok) |
|-------|----------:|------------:|-----------:|----------------:|----------------:|
| Qwen3-30B-A3B | 433,000 | 61,100 | 6,940 | 461,800 | 15,300 |
| Qwen3-Next-80B-A3B | 454,300 | 175,000 | 26,600 | 461,800 | 61,000 |
| Qwen3-32B | 42,600 | 16,400 | 2,490 | 43,300 | 61,000 |
| Qwen3-235B-A22B | 60,800 | 14,000 | 1,749 | 63,000 | 28,600 |

**The prefill bound, per model.** Two regimes are separated by the crossover
$T^\star = 2 N_{\text{act}} / C$, the context at which the attention term overtakes
the FFN term:
- $T < T^\star$: **FFN/params-bound**, a flat ceiling $\rho_{\max}=\mathrm{EFF}/(2 N_{\text{act}})$ set purely by active params ($63$k\,tok/s for the 235B flagship, $462$k for the 3B-active tiers);
- $T > T^\star$: **attention-bound**, $\rho \approx \mathrm{EFF}/(C\,T)$, decaying as $1/T$.

The flagship's bound is the binding one: $63$k\,tok/s ceiling, crossover at only
$28.6$k tokens, so at the workload's long agentic contexts it sits deep in the
attention-bound regime. Qwen3-Next is the outlier — same $3$B active as Qwen3-30B
and the same $H_q d_{\text{head}}$, but the $T^2$ term rides on only $12$ of its $48$
layers, so its crossover is $4\times$ later and it prefills $\sim4\times$ faster at
long context. **Interpolation:** between the four anchors the instance generator uses
log-linear interpolation of $\rho$, which tracks the roofline closely.

---

## Context Length Distribution

Private suffix token lengths are drawn from a lognormal distribution per model, clipped to $[1{,}000, 1{,}000{,}000]$.

| Model | $\mu$ (of $\ln T$) | $\sigma$ (of $\ln T$) | Implied median | Implied 95th pctl |
|-------|-------------------:|---------------------:|---------------:|-----------------:|
| Qwen3-30B-A3B | $\ln(5{,}000)$ | 1.3 | 5,000 | ~27,000 |
| Qwen3-Next-80B-A3B | $\ln(8{,}000)$ | 1.5 | 8,000 | ~74,000 |
| Qwen3-32B | $\ln(6{,}000)$ | 1.4 | 6,000 | ~42,000 |
| Qwen3-235B-A22B | $\ln(15{,}000)$ | 1.9 | 15,000 | ~310,000 |

**Rationale:** The flagship serves the longest agentic sessions, so its $\mu$ and $\sigma$ are largest; the cheap small/efficiency tiers serve shorter, more transactional requests. Combined with token share, this is what makes the flagship a token majority but a job minority: counts are $n_m \propto s_m / \bar T_m$ using the *clipped* mean $\bar T_m$ (the analytic mean overstates it because the tail is truncated at $10^6$).

**Default seed:** 42. All randomness is through `numpy.random.default_rng(seed)`.

---

## Destination Configuration (all should be swept, changed, done for different sensitivities)

Three destination sites with heterogeneous resource profiles. Each is a peer region of the *same provider*, so it runs the same Qwen3 suite; its warm pool is sized for its own steady, flagship-primary traffic, which is why the evacuee's surge does not land on perfectly matched capacity.

| Parameter | Site A | Site B | Site C |
|-----------|-------:|-------:|-------:|
| $\Lambda_\ell$ (GB/s) | 25 | 12.5 | 50 |
| $\Lambda_\ell$ (Gbps equiv.) | 200 | 100 | 400 |

**Warm instances $W_{\ell m}$ (sized in HBM).** Every instance is one TP=8 group — 8×H100 = 640 GB HBM — holding the model's resident BF16 weights (Table 1) plus KV headroom. So one 235B instance commits 470 GB of weights and a 30B instance 60 GB; the flagship is the dominant HBM line item per site, matching its token majority.

| Model | Site A | Site B | Site C | Total |
|-------|-------:|-------:|-------:|------:|
| Qwen3-30B-A3B | 1 | 1 | 2 | 4 |
| Qwen3-Next-80B-A3B | 2 | 1 | 2 | 5 |
| Qwen3-32B | 1 | 2 | 1 | 4 |
| Qwen3-235B-A22B | 3 | 2 | 4 | 9 |

**Rationale:** Pools are flagship-primary (9 flagship instances vs 4–5 for the others) yet site-specialized, so no destination's spare capacity exactly matches the evacuee's mix. The flagship's huge $\eta$ against its limited warm pool keeps it the binding model, and the optimizer must still route low-$\eta$ tiers (Next-80B) to state-transfer to dodge the prefill ceiling.

**Per-instance ingest rate:** $\mu_{\ell m}^{\text{ing}} = \text{TP} \times \text{BW}_{\text{PCIe}} = 8 \times 64 \text{ GB/s} = 512 \text{ GB/s}$ for all models and destinations. This assumes H100 SXM with PCIe Gen5 x16 per GPU, TP=8, and host-staged (not GPUDirect RDMA) state transfer. This rate is much larger than any destination's network ingress, so the ingest constraint is typically slack.

---

## Hardware Assumptions (these are hard fixed)

| Parameter | Value | Notes |
|-----------|------:|-------|
| GPU | H100 SXM | 80 GB HBM3 |
| Tensor parallelism | 8 | All models, all destinations |
| PCIe bandwidth | 64 GB/s per GPU | Gen5 x16, host-to-device |
| Network ingress | 1–50 GB/s per site | Site-level, not per-NIC |

All destinations use the same hardware. If destinations had different GPUs (e.g., H100 vs H200), $\rho_q$ would become $\rho_{q\ell}$ and $\mu_{\ell m}^{\text{ing}}$ would vary by site. The formulation supports this; the current experiments assume homogeneous hardware.

---

## Derived Quantities (these are sensitivity parameters)

At default parameters (10,000 jobs, seed 42), the instance generator produces approximately:

| Quantity | Value |
|----------|------:|
| Total classes $|\mathcal{Q}|$ (with `n_bins=5`) | ~20 (4 models × 5 buckets) |
| Pressure constraints $|\mathcal{I}|$ | 27 (3 net + 24 pfill/ing) |
| Decision variables | $2 \times 20 \times 3 + 20 = 140$ |
| Total prefill demand $\Delta$ | ~41,000 GPU-seconds |
| Total prefill supply $\Sigma$ at $D=300$s | ~6,600 GPU-seconds |
| Aggregate $\Sigma/\Delta$ at $D=300$s | ~0.16 |

The aggregate supply/demand ratio of 0.16 means the system is heavily prefill-constrained: full evacuation via replay alone would need ~6× more warm instances or deadline. The optimizer compensates by routing low-$\eta$ models (Qwen3-Next-80B) to state-transfer, bypassing the prefill bottleneck at the cost of network bandwidth.

---

## Default Sweep Ranges

| Sweep parameter | Default values |
|-----------------|---------------|
| Deadline $D$ | 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900 s |
| **Flagship token share** | 0.3, 0.4, 0.5, 0.55, 0.7, 0.8, 0.9 |
| $W$ scale | 0.5, 1.0, 1.5, 2.0, 3.0, 5.0 |
| $\Lambda$ scale | 0.25, 0.5, 1.0, 2.0, 4.0 |
| $\rho$ scale | 0.5, 0.75, 1.0, 1.5, 2.0 |
| Number of destinations | 2, 3, 4, 6, 8 |

**Flagship token share** (`build_instance(flagship_share=s)`) is the primary mix-sensitivity axis: it pins Qwen3-235B to a fraction $s$ of served tokens and rescales the other three tiers proportionally. Sweeping $s$ shifts token mass between the light efficiency tier and the heavy-$\eta$ flagship, which is exactly what moves the binding resource between network and prefill. The base instance sits at $s = 0.55$.

**$d^{\text{miss}}$** = $2D$ (next recovery window). This is a declared convention, not a tuning parameter.

**Total jobs** = 10,000 unless otherwise specified.