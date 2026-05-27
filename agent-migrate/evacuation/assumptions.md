# Scenario Parameters and Data Assumptions

This document describes the constants, distributions, and configurations used to generate experimental instances. These are benchmark-derived scenario parameters, not universal facts. The formulation is independent of these specific values.

---

## Model Table

Six models spanning the current frontier of MoE and dense architectures. The $\eta$ values reflect architectural differences in attention (MLA, GQA, full MHA). All values are per-token, per-layer aggregated across the full model.

| Model | $\eta$ (KiB/tok) | $\beta$ (B/tok) | Job fraction |
|-------|------------------:|----------------:|-------------:|
| DeepSeek V4 Pro | 9.7 | 4 | 25% |
| Kimi K2.6 | 68.6 | 4 | 25% |
| GLM 5 | 87.8 | 4 | 15% |
| Qwen3 235B (A22B) | 188.0 | 4 | 15% |
| Qwen3.5 397B (A17B) | 30.0 | 4 | 15% |
| Qwen3 Next 80B (A3B) | 24.0 | 4 | 5% |

**$\beta = 4$ B/tok** for all models. This assumes uint32 token IDs as the compact context representation. The actual transmitted context may include metadata (role tags, tool schemas, position IDs), but 4 B/tok is a reasonable lower bound on the information-theoretic minimum.

**$\eta$** is the KV cache size per token aggregated across all layers and all KV heads, at the serving precision (typically BF16 or FP8). MLA models (DeepSeek) compress KV into a low-rank latent, giving ~10 KiB/tok. Full MHA models (Qwen3 235B) store separate K and V projections per head per layer, giving ~188 KiB/tok. GQA and hybrid architectures fall in between.

**Job fractions** are assumed proportions of the workload mix at the evacuating site. These are scenario parameters, not measured traffic distributions.

---

## Prefill Rate Table

Single warm-instance prefill throughput (tok/s) at four context-length anchors. These capture both the model size effect (larger models are slower) and the context-length degradation (attention cost grows with sequence length).

| Model | $\rho$ @1k | $\rho$ @10k | $\rho$ @100k | $\rho$ @1M |
|-------|----------:|-----------:|------------:|-----------:|
| DeepSeek V4 Pro | 28,000 | 25,600 | 13,900 | 2,500 |
| Kimi K2.6 | 42,500 | 36,200 | 14,700 | 2,100 |
| GLM 5 | 33,600 | 26,200 | 8,300 | 1,100 |
| Qwen3 235B | 60,800 | 46,600 | 14,000 | 1,700 |
| Qwen3.5 397B | 80,900 | 76,000 | 47,300 | 9,900 |
| Qwen3 Next 80B | 454,300 | 396,800 | 175,000 | 26,600 |

**Interpolation:** For a context length $T$ between anchors, $\rho(T)$ is computed by log-linear interpolation: $\log \rho = \text{interp}(\log T, \log T_{\text{anchors}}, \log \rho_{\text{anchors}})$. This assumes power-law degradation between anchors.

**Serving configuration assumed:** TP=8 on H100 SXM, single-instance (not pipeline-parallel), BF16 weights. These rates are per-instance, not per-GPU.

**Source:** These values are estimates informed by public benchmarks and scaling analysis, not measured on a specific deployment. They should be treated as scenario parameters.

---

## Context Length Distribution

Private suffix token lengths are drawn from a lognormal distribution per model, clipped to $[1{,}000, 1{,}000{,}000]$.

| Model | $\mu$ (of $\ln T$) | $\sigma$ (of $\ln T$) | Implied median | Implied 95th pctl |
|-------|-------------------:|---------------------:|---------------:|-----------------:|
| DeepSeek V4 Pro | $\ln(8{,}000)$ | 1.5 | 8,000 | ~74,000 |
| Kimi K2.6 | $\ln(12{,}000)$ | 1.8 | 12,000 | ~215,000 |
| GLM 5 | $\ln(6{,}000)$ | 1.4 | 6,000 | ~42,000 |
| Qwen3 235B | $\ln(15{,}000)$ | 1.9 | 15,000 | ~310,000 |
| Qwen3.5 397B | $\ln(20{,}000)$ | 1.7 | 20,000 | ~280,000 |
| Qwen3 Next 80B | $\ln(5{,}000)$ | 1.3 | 5,000 | ~27,000 |

**Rationale:** Heavier models (Qwen3 235B, Kimi K2.6) tend to serve longer agentic sessions, so their $\mu$ and $\sigma$ are larger. Smaller models (Qwen3 Next 80B) serve shorter, more transactional requests.

This clearly makes T_p in a class / model q of dimension R^n_q.

**Default seed:** 42. All randomness is through `numpy.random.default_rng(seed)`.

---

## Destination Configuration (all should be swept, changed, done for different sensitivities)

Three destination sites with heterogeneous resource profiles.

| Parameter | Site A | Site B | Site C |
|-----------|-------:|-------:|-------:|
| $\Lambda_\ell$ (GB/s) | 25 | 12.5 | 50 |
| $\Lambda_\ell$ (Gbps equiv.) | 200 | 100 | 400 |

**Warm instances $W_{\ell m}$:**

| Model | Site A | Site B | Site C | Total |
|-------|-------:|-------:|-------:|------:|
| DeepSeek V4 Pro | 2 | 1 | 1 | 4 |
| Kimi K2.6 | 1 | 2 | 1 | 4 |
| GLM 5 | 1 | 1 | 2 | 4 |
| Qwen3 235B | 2 | 1 | 1 | 4 |
| Qwen3.5 397B | 1 | 3 | 1 | 5 |
| Qwen3 Next 80B | 1 | 1 | 2 | 4 |

**Rationale:** Each site specializes slightly in different models. Total warm instances per model range from 4–5, giving supply/demand ratios between 0.008 (Qwen3 235B, severely constrained) and 13.9 (Qwen3 Next 80B, abundant) at $D = 300$s. This heterogeneity ensures the optimizer faces genuine routing decisions.

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
| Total classes $|\mathcal{Q}|$ | ~28–30 |
| Pressure constraints $|\mathcal{I}|$ | 39 (3 net + 36 pfill/ing) |
| Decision variables | $2 \times 30 \times 3 + 30 = 210$ |
| Total prefill demand $\Delta$ | ~56,000 GPU-seconds |
| Total prefill supply $\Sigma$ at $D=300$s | ~6,300 GPU-seconds |
| Aggregate $\Sigma/\Delta$ at $D=300$s | ~0.11 |

The aggregate supply/demand ratio of 0.11 means the system is heavily prefill-constrained. Full evacuation via replay alone would require ~9× more warm instances or ~9× longer deadline. The optimizer compensates by routing low-$\eta$ models (DeepSeek, Qwen3 Next 80B) to state-transfer, bypassing the prefill bottleneck at the cost of network bandwidth.

---

## Default Sweep Ranges

| Sweep parameter | Default values |
|-----------------|---------------|
| Deadline $D$ | 10, 20, 30, 45, 60, 90, 120, 180, 300, 600, 900 s |
| $W$ scale | 0.5, 1.0, 1.5, 2.0, 3.0, 5.0 |
| $\Lambda$ scale | 0.25, 0.5, 1.0, 2.0, 4.0 |
| $\rho$ scale | 0.5, 0.75, 1.0, 1.5, 2.0 |
| Number of destinations | 2, 3, 4, 6, 8 |

**$d^{\text{miss}}$** = $2D$ (next recovery window). This is a declared convention, not a tuning parameter.

**Total jobs** = 10,000 unless otherwise specified.