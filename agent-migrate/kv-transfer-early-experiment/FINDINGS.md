# KV transfer vs prompt replay

Question: when moving an in-flight request to another site, is it faster to ship
the KV cache or replay the prompt/context?

## Setup

- Hardware: 8x H100 SXM, dense bf16 peak = 7.92 PFLOP/s.
- Assumed MFU: 35%, so effective prefill compute = 2.77 PFLOP/s.
- KV state is bf16. Quantized KV and allocator padding are not modeled.
- Replay time is recomputed from prefill FLOPs at each context length; it is not a
  fixed tokens/sec benchmark.
- Reference table below is at 100,000 tokens.

## Result

Above the crossover bandwidth, ship KV. Below it, replay.

| Model | KV @ 100k | Replay | Prefill | Crossover |
|---|---:|---:|---:|---:|
| DeepSeek-V4-Pro | 0.99 GB | 7.20 s | 13.9k tok/s | 1.10 Gbps |
| Kimi-K2.6 | 7.03 GB | 6.82 s | 14.7k tok/s | 8.24 Gbps |
| GLM-5 | 8.99 GB | 12.11 s | 8.3k tok/s | 5.93 Gbps |
| Qwen3-235B-A22B | 19.25 GB | 7.15 s | 14.0k tok/s | 21.55 Gbps |
| Qwen3.5-397B-A17B | 3.07 GB | 2.11 s | 47.3k tok/s | 11.62 Gbps |
| Qwen3-Next-80B-A3B | 2.46 GB | 0.57 s | 175.0k tok/s | 34.41 Gbps |

## Takeaways

- Crossovers are modest: about 1-34 Gbps at 100k tokens, not 100+ Gbps.
- GLM-5 is no longer a heavy full-MHA outlier. The corrected model treats it as
  MLA + DSA compressed KV.
- DeepSeek-V4-Pro has the lowest crossover because CSA/HCA keeps KV small enough
  that transfer usually wins.
- Qwen3-Next has the highest crossover because replay is extremely fast.
- Context length matters: replay has a quadratic attention term, so long contexts
  generally make KV transfer more attractive.
- `glm5_context_ratio_bandwidths.{png,pdf}` shows the GLM-5 ratio from 1k to
  10M tokens across 500 linear-spaced 0.1-25 Gbps links.

## Files

- `prefill-breakeven.py`: main corrected sweep and plots.
- `migration_sweep_corrected.csv`: generated sweep data.
- `migration_times.{png,pdf}`: transfer vs replay time by model.
- `crossover_bandwidth.{png,pdf}`: crossover summary.
- `migration_ratio.py`: ratio plots for bandwidth and GLM-5 context sweeps.
- `glm5_context_ratio_bandwidths.{png,pdf}`: GLM-5 TTFT/KV-transfer ratio vs
  context size across fixed bandwidth lines.
- `appendix.{tex,bib}`: writeup appendix for `migration_ratio.py` — cost model,
  per-model architecture data with sources, assumptions, and the top-k sparse
  attention gap. `\input`-able; compile standalone with a minimal preamble.
